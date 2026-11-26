"""Real-data forward-only backtest experiment (Stage 3 U04 part 2).

Premier League 2015/16, three expanding decisions (cutoffs 2015-11-01,
2016-01-01, 2016-03-01). Goal models are retrained per cutoff on
pre-cutoff frames only from contemporaneous 2015/16 competitions
(Premier League, La Liga, Serie A, Ligue 1, Bundesliga). UEFA Euro
2020 and Indian Super League 2021/22 are excluded from every training
pool: a model deciding in 2015/16 must not have seen 2021/22 events.

Writes runs/forward_backtest/results.json and
runs/forward_backtest/report.md. The shipped FC6 models and results
are not touched, and no published number is replaced.

Usage: python scripts/run_forward_backtest.py  (from the repo root)
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from effc.data.download import DATA_DIR  # noqa: E402
from effc.data.statsbomb import (  # noqa: E402
    event_content_digest,
    list_matches,
)
from effc.data.wages import load_wages, match_players  # noqa: E402
from effc.evaluation import forward_backtest as fb  # noqa: E402
from effc.evaluation import forward_experiment as fe  # noqa: E402
from effc.leagues import BACKTEST_BUDGETS  # noqa: E402
from effc.pipeline import featurize_matches  # noqa: E402
from effc.vaep.features import FEATURE_VERSION  # noqa: E402

TARGET_COMP = (2, 27)  # Premier League 2015/16
POOL_COMPS = [(11, 27), (12, 27), (7, 27), (9, 27)]  # La Liga, Serie A, Ligue 1, Bundesliga
CUTOFFS = ["2015-11-01", "2016-01-01", "2016-03-01"]
HORIZON_DAYS = 120
CONFIG_NAME = "fc6_pool"
SEED = 0
N_RANDOM = 20
OUTDIR = os.path.join("runs", "forward_backtest")

COMP_NAMES = {
    (2, 27): "Premier League 2015/16",
    (11, 27): "La Liga 2015/16",
    (12, 27): "Serie A 2015/16",
    (7, 27): "Ligue 1 2015/16",
    (9, 27): "Bundesliga 2015/16",
}


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def main() -> None:
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    all_comps = [TARGET_COMP] + POOL_COMPS
    meta = list_matches(all_comps)
    epl_meta = meta[
        (meta["competition_id"] == TARGET_COMP[0])
        & (meta["season_id"] == TARGET_COMP[1])
    ].reset_index(drop=True)
    if len(epl_meta) != 380:
        raise ValueError(
            f"expected 380 Premier League 2015/16 matches, found {len(epl_meta)}"
        )
    involved = sorted(set(meta["match_id"].astype(int)))
    print(f"[meta] {len(meta)} matches across {len(all_comps)} competitions")
    for comp in all_comps:
        n = len(meta[(meta["competition_id"] == comp[0]) & (meta["season_id"] == comp[1])])
        print(f"  {COMP_NAMES[comp]}: {n} matches")

    # Audit: every involved match needs events; every EPL match needs a lineup.
    missing_events = [
        mid for mid in involved
        if not os.path.exists(os.path.join(DATA_DIR, "events", f"{mid}.json"))
    ]
    missing_lineups = [
        int(mid) for mid in epl_meta["match_id"]
        if not os.path.exists(os.path.join(DATA_DIR, "lineups", f"{int(mid)}.json"))
    ]
    if missing_events or missing_lineups:
        raise ValueError(
            f"missing data: {len(missing_events)} event files, "
            f"{len(missing_lineups)} EPL lineup files"
        )
    digest = event_content_digest(involved)
    print(f"[digest] event content digest {digest}")

    # Featurize every involved match once; the cache is content-addressed.
    frames = featurize_matches(involved, "pool_frames_forward_u04.pkl")
    by_id = {int(f["match_id"].iloc[0]): f for f in frames}
    print(f"[featurize] {len(frames)}/{len(involved)} matches featurized")

    decisions = fb.build_decision_schedule(
        meta, CUTOFFS, horizon_days=HORIZON_DAYS
    )
    for d in decisions:
        print(
            f"[decision] {d.decision_id}: {len(d.input_match_ids)} input "
            f"matches, max input {d.max_input_timestamp.date()}, "
            f"window {d.window_start.date()} to {d.window_end.date()}"
        )

    # Per decision: retrain on pre-cutoff frames only, score the EPL.
    epl_by_id = {
        int(mid): by_id[int(mid)]
        for mid in epl_meta["match_id"]
        if int(mid) in by_id
    }
    mins = None
    pm_by_decision: dict = {}
    model_hashes: dict = {}
    train_counts: dict = {}
    for decision in decisions:
        cutoff = decision.cutoff
        target_frames, target_ids = fe.frames_before_cutoff(
            epl_by_id, epl_meta, cutoff
        )
        pool_frames: list = []
        pool_ids: list = []
        for comp in POOL_COMPS:
            comp_meta = meta[
                (meta["competition_id"] == comp[0])
                & (meta["season_id"] == comp[1])
            ]
            comp_frames = {
                int(mid): by_id[int(mid)]
                for mid in comp_meta["match_id"]
                if int(mid) in by_id
            }
            fr, ids = fe.frames_before_cutoff(comp_frames, comp_meta, cutoff)
            pool_frames.extend(fr)
            pool_ids.extend(ids)
        print(
            f"[train] {decision.decision_id}: {len(target_ids)} EPL + "
            f"{len(pool_ids)} pooled pre-cutoff frames"
        )
        models, model_hash = fe.train_models_for_cutoff(
            target_frames, pool_frames, meta, CONFIG_NAME,
            seed=SEED, content_digest=digest,
        )
        model_hashes[decision.decision_id] = model_hash
        train_counts[decision.decision_id] = len(target_ids) + len(pool_ids)
        print(f"[train] {decision.decision_id}: model hash {model_hash}")
        pm_d, mins_d = fe.score_target(models, epl_by_id, epl_meta)
        pm_by_decision[decision.decision_id] = pm_d
        mins = mins_d
        print(f"[score] {decision.decision_id}: {len(pm_d)} player-match rows")

    # Wages: the repo's FC24 snapshot, matched once for the EPL pool.
    wages = load_wages()
    sb_players = (
        pd.concat(pm_by_decision.values(), ignore_index=True)[["player", "team"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    wmap = match_players(sb_players, wages)
    print(f"[wages] match rate: {wmap['matched'].mean():.1%} "
          f"of {len(wmap)} EPL players")

    ratings_by_decision: dict = {}
    realized_by_decision: dict = {}
    for decision in decisions:
        r, w = fe.build_forward_inputs(
            [decision], pm_by_decision[decision.decision_id], mins, wmap
        )
        ratings_by_decision.update(r)
        realized_by_decision.update(w)
        print(
            f"[inputs] {decision.decision_id}: "
            f"{len(ratings_by_decision[decision.decision_id])} rated players, "
            f"{len(realized_by_decision[decision.decision_id])} realized rows"
        )

    out = fe.run_experiment(
        decisions, ratings_by_decision, realized_by_decision,
        [float(b) for b in BACKTEST_BUDGETS], seed=SEED, n_random=N_RANDOM,
    )
    results, failures, records = out["results"], out["failures"], out["records"]
    print(f"[run] {len(results)} result rows, {len(failures)} failure rows")
    print(f"[comparison] oracle bound holds: "
          f"{out['comparison']['oracle_bound_holds']}")
    for gap in out["comparison"]["gaps"]:
        print(
            f"  {gap['decision_id']} budget {gap['budget']:g}: "
            f"optimizer-random {gap['optimizer_minus_random']:+.2f}, "
            f"optimizer-value {gap['optimizer_minus_value']:+.2f}, "
            f"value-random {gap['value_minus_random']:+.2f} (penalized)"
        )

    # Enrich the retraining plan with model hashes and window match IDs.
    schedule = []
    for entry in out["schedule"]:
        decision = next(
            d for d in decisions if d.decision_id == entry["decision_id"]
        )
        window_ids = sorted(
            int(m) for m in epl_meta[
                (pd.to_datetime(epl_meta["match_date"]) > decision.window_start)
                & (pd.to_datetime(epl_meta["match_date"]) <= decision.window_end)
            ]["match_id"]
        )
        schedule.append(
            {
                **entry,
                "model_hash": model_hashes[decision.decision_id],
                "n_train_matches": train_counts[decision.decision_id],
                "train_competitions": [COMP_NAMES[c] for c in all_comps],
                "window_match_ids": window_ids,
            }
        )

    artifact = {
        "experiment": "u04-part2-epl-2015-16",
        "model_config": CONFIG_NAME,
        "seed": SEED,
        "budgets": [float(b) for b in BACKTEST_BUDGETS],
        "cutoffs": CUTOFFS,
        "target": {
            "competition": "Premier League",
            "season": "2015/2016",
            "competition_id": TARGET_COMP[0],
            "season_id": TARGET_COMP[1],
            "n_matches": len(epl_meta),
        },
        "training_pool": {
            "pool_competitions": [COMP_NAMES[c] for c in all_comps],
            "extra_pool_excluded": [
                "UEFA Euro 2020",
                "Indian Super League 2021/22",
            ],
            "note": (
                "Models train only on matches dated before each cutoff "
                "from contemporaneous 2015/16 competitions. UEFA Euro "
                "2020 and Indian Super League 2021/22 never enter any "
                "training pool."
            ),
        },
        "wages": {
            "snapshot": "FC24 (EA FC 24 wages via load_wages)",
            "note": (
                "The wage file is a single snapshot; it is recorded at "
                "each decision cutoff, not backdated to 2015. Wage "
                "levels are therefore the repo's existing FC24 snapshot "
                "applied to 2015/16 players, the same basis as the "
                "shipped backtest, and a stated limitation, not a "
                "point-in-time wage history."
            ),
        },
        "schedule": schedule,
        "decisions": records,
        "results": _clean(results.to_dict("records")),
        "failures": _clean(failures.to_dict("records")),
        "comparison": _clean(out["comparison"]),
        "data": {
            "event_digest": digest,
            "n_involved_matches": len(involved),
            "feature_version": FEATURE_VERSION,
        },
        "limitations": [
            "Wages are the repo's FC24 snapshot, not a 2015-dated wage history.",
            "Three cutoffs on one competition (Premier League 2015/16).",
            "Goal models retrain per cutoff on pre-cutoff frames only; the shipped FC6 models are untouched.",
            "No published number is replaced by this experiment.",
        ],
    }
    results_path = os.path.join(OUTDIR, "results.json")
    with open(results_path, "w") as fh:
        json.dump(_clean(artifact), fh, indent=2)
    print(f"[write] {results_path}")

    lines = [
        "# Forward-only backtest experiment (U04 part 2)",
        "",
        "Premier League 2015/16, three expanding decisions. Goal models "
        "retrain per cutoff on pre-cutoff frames only from contemporaneous "
        "2015/16 competitions; UEFA Euro 2020 and Indian Super League "
        "2021/22 never enter any training pool.",
        "",
        "## Decisions",
        "",
    ]
    for entry in schedule:
        lines.append(
            f"- {entry['decision_id']}: cutoff {entry['cutoff']}, "
            f"{entry['n_input_matches']} input matches, model "
            f"{entry['model_hash']}, retrain={entry['retrain']}"
        )
    lines += ["", "## Policy comparison (penalized realized VAEP)", ""]
    for gap in out["comparison"]["gaps"]:
        lines.append(
            f"- {gap['decision_id']} budget {gap['budget']:g}: "
            f"optimizer-random {gap['optimizer_minus_random']:+.2f}, "
            f"optimizer-value {gap['optimizer_minus_value']:+.2f}, "
            f"value-random {gap['value_minus_random']:+.2f}"
        )
    lines += [
        "",
        "Gaps are observed differences on this run, not wins; a null "
        "(the optimizer not ahead) is reported as the observed gap.",
        "",
        "## Limitations",
        "",
    ]
    for item in artifact["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    with open(os.path.join(OUTDIR, "report.md"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"[write] {os.path.join(OUTDIR, 'report.md')}")
    print(f"[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
