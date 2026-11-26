"""Forward-only backtest experiment driver (Stage 3 U04 part 2).

Part 1 established the contract: a decision dated at a cutoff may only
use model inputs strictly before that cutoff, every policy faces the
identical feasible constraints, unavailable players are counted and
penalized, failed matchups stay reported, and every decision record
carries a reproducible hash. This module is the bounded remaining
functionality: the experiment driver that turns per-match VAEP rows
into per-decision ratings and realized windows, plans retraining on
the expanding cache-key schedule, compares policies with nulls
preserved, and wraps the part 1 runner. It also holds the real-data
training and scoring paths the experiment script uses; those need
StatsBomb data and are exercised by the script run, not by CI tests.

Rules carried over from part 1, all fail-closed with
``PointInTimeError`` from the contract module: inputs are matches with
``match_date < cutoff``; wages are a snapshot dated at or before the
cutoff; scored matches are strictly after the cutoff; input and scored
match IDs are disjoint. Nothing here retrains the shipped FC6 models
or changes any published result.
"""

from __future__ import annotations

import hashlib
import pickle

import pandas as pd

from effc.evaluation import forward_backtest as fb
from effc.portfolio.portfolio import SQUAD_SIZE
from effc.ratings.ratings import build_ratings

POLICIES = fb.POLICIES


def build_forward_inputs(
    decisions, pm, mins, wmap, min_minutes: float = 60.0, shrink_k: float = 900.0
):
    """Per-decision ratings and realized windows from per-match VAEP rows.

    ``pm`` holds one row per player per match (match_id, match_date,
    player_id, player, team, vaep); ``mins`` holds minutes per player
    per match (match_id, player_id, minutes, position); ``wmap`` is the
    wage map (player, team, wage_eur, value_eur, pos_group, matched,
    wage_source). For each decision, ratings come from the pre-cutoff
    slice only (shrunk VAEP/90 via ``build_ratings``) and realized
    rows come from the forward window only. Returns
    (ratings_by_decision, realized_by_decision).
    """
    frame = pm.copy()
    frame["match_date"] = pd.to_datetime(frame["match_date"])
    ratings: dict = {}
    realized: dict = {}
    for decision in decisions:
        pre = frame[frame["match_date"] < decision.cutoff]
        ratings[decision.decision_id] = build_ratings(
            pre, mins, wmap, min_minutes=min_minutes, shrink_k=shrink_k
        )
        window = frame[
            (frame["match_date"] > decision.window_start)
            & (frame["match_date"] <= decision.window_end)
        ]
        realized[decision.decision_id] = window[
            ["match_id", "match_date", "player_id", "vaep"]
        ].reset_index(drop=True)
    return ratings, realized


def plan_retraining(decisions, known_cache_keys, content_digest: str | None = None):
    """Retraining schedule for expanding decisions.

    ``known_cache_keys`` is the set of cache keys already trained. For
    each decision the key binds its cutoff and exact input match IDs
    (plus the content digest when given). A decision retrains when its
    key is unknown and reuses the cache when its key is known. Returns
    (plan, updated_known_keys); each plan row records the decision,
    cutoff, key, retrain flag, reason, and input match count.
    """
    known = set(known_cache_keys)
    plan: list[dict] = []
    for decision in decisions:
        key = fb.decision_cache_key(
            decision.cutoff, decision.input_match_ids, content_digest
        )
        retrain = key not in known
        plan.append(
            {
                "decision_id": decision.decision_id,
                "cutoff": decision.cutoff.strftime("%Y-%m-%d"),
                "cache_key": key,
                "retrain": bool(retrain),
                "reason": (
                    "no cache entry for this cutoff and input set"
                    if retrain
                    else "cache hit"
                ),
                "n_input_matches": len(decision.input_match_ids),
            }
        )
        known.add(key)
    return plan, known


def compare_policies(results, failures):
    """Comparison summary from a forward run. Records observed gaps only.

    No winner is named: a gap is a difference on this run, not a win,
    and a null (the optimizer not ahead of random or value) is
    reported as the observed gap, never hidden. Coverage is complete
    only when every decision/budget/policy combination appears in the
    results or the failures.
    """
    combos = {
        (r["decision_id"], float(r["budget"]), r["policy"])
        for _, r in results.iterrows()
    } | {
        (r["decision_id"], float(r["budget"]), r["policy"])
        for _, r in failures.iterrows()
    }
    decisions = sorted({c[0] for c in combos})
    budgets = sorted({c[1] for c in combos})
    policies = sorted({c[2] for c in combos})
    expected = {
        (d, b, p) for d in decisions for b in budgets for p in POLICIES
    }
    complete = combos == expected
    by_decision: dict = {}
    for _, row in results.iterrows():
        by_decision.setdefault(row["decision_id"], {}).setdefault(
            f"{float(row['budget']):g}", {}
        )[row["policy"]] = {
            "score_raw": float(row["score_raw"]),
            "score_penalized": float(row["score_penalized"]),
            "n_unavailable": row["n_unavailable"],
            "wage_bill": float(row["wage_bill"]),
        }
    gaps: list[dict] = []
    oracle_bound_holds = True
    for (decision_id, budget) in sorted(
        {(r["decision_id"], float(r["budget"])) for _, r in results.iterrows()}
    ):
        sub = results[
            (results["decision_id"] == decision_id)
            & (results["budget"] == budget)
        ]
        pen = dict(zip(sub["policy"], sub["score_penalized"]))
        raw = dict(zip(sub["policy"], sub["score_raw"]))
        if set(POLICIES) <= set(pen):
            gaps.append(
                {
                    "decision_id": decision_id,
                    "budget": budget,
                    "optimizer_minus_random": float(
                        pen["optimizer"] - pen["random"]
                    ),
                    "optimizer_minus_value": float(
                        pen["optimizer"] - pen["value"]
                    ),
                    "value_minus_random": float(pen["value"] - pen["random"]),
                }
            )
        if "oracle" in raw:
            for policy, score in raw.items():
                if policy != "oracle" and raw["oracle"] < score - 1e-9:
                    oracle_bound_holds = False
    return {
        "coverage": {
            "n_decisions": len(decisions),
            "n_budgets": len(budgets),
            "n_policies": len(policies),
            "n_result_rows": int(len(results)),
            "n_failure_rows": int(len(failures)),
            "complete": bool(complete),
        },
        "by_decision": by_decision,
        "gaps": gaps,
        "oracle_bound_holds": bool(oracle_bound_holds),
    }


def run_experiment(
    decisions,
    ratings_by_decision,
    realized_by_decision,
    budgets,
    seed: int = 0,
    n_random: int = 20,
    squad_size: int = SQUAD_SIZE,
    known_cache_keys=None,
):
    """Run the forward experiment over expanding decisions.

    Plans retraining on the cache-key schedule, runs the part 1
    forward backtest, and compares policies. Returns a dict with the
    results and failures frames, the decision records, the comparison
    summary, and the retraining plan.
    """
    schedule, _ = plan_retraining(
        decisions, set() if known_cache_keys is None else known_cache_keys
    )
    results, failures, records = fb.run_forward_backtest(
        decisions,
        ratings_by_decision,
        realized_by_decision,
        budgets,
        seed=seed,
        n_random=n_random,
        squad_size=squad_size,
    )
    return {
        "results": results,
        "failures": failures,
        "records": records,
        "comparison": compare_policies(results, failures),
        "schedule": schedule,
    }


def frames_before_cutoff(frames_by_id, meta, cutoff):
    """Frames with match_date strictly before ``cutoff``.

    ``frames_by_id`` maps match ID to its featurized frame; ``meta``
    is the match list (match_id, match_date). Returns (frames, ids)
    with ids sorted.
    """
    dates = dict(
        zip(meta["match_id"].astype(int), pd.to_datetime(meta["match_date"]))
    )
    ids = sorted(
        int(mid)
        for mid, dt in dates.items()
        if dt < pd.Timestamp(cutoff) and int(mid) in frames_by_id
    )
    return [frames_by_id[mid] for mid in ids], ids


def train_models_for_cutoff(
    target_frames, pool_frames, meta_all, config_name, seed: int = 0,
    content_digest: str | None = None,
):
    """Train improved goal models on pre-cutoff frames only.

    ``target_frames`` are the target competition's pre-cutoff frames
    (the cross-fit calibration group); ``pool_frames`` are the other
    contemporaneous competitions' pre-cutoff frames. The caller is
    responsible for the cutoff filter: no frame dated at or after the
    cutoff may be passed in. Returns (models, model_hash).
    """
    from effc.vaep import improved as imp

    models, _report = imp.train_improved_models_cached(
        target_frames, pool_frames, meta_all, config_name,
        seed=seed, content_digest=content_digest,
    )
    model_hash = hashlib.sha256(pickle.dumps(models, protocol=4)).hexdigest()[:16]
    return models, model_hash


def score_target(models, frames_by_id, target_meta):
    """Score target matches with models: (pm, mins) with match dates.

    ``pm`` holds one row per player per match (match_id, match_date,
    player_id, player, team, vaep); ``mins`` holds minutes per player
    per match (match_id, player_id, minutes, position). Every scored
    match uses the same frozen models: the deployment simulation.
    """
    from effc.data.statsbomb import minutes_played
    from effc.vaep.values import add_vaep, player_match_vaep

    dates = dict(
        zip(
            target_meta["match_id"].astype(int),
            pd.to_datetime(target_meta["match_date"]),
        )
    )
    pm_rows: list[pd.DataFrame] = []
    min_rows: list[pd.DataFrame] = []
    for mid in sorted(int(m) for m in frames_by_id):
        frame = add_vaep(frames_by_id[mid], models)
        pm = player_match_vaep(frame)
        pm["match_date"] = dates[mid]
        pm_rows.append(pm)
        min_rows.append(minutes_played(mid))
    pm = pd.concat(pm_rows, ignore_index=True)
    mins = pd.concat(min_rows, ignore_index=True)
    return pm, mins
