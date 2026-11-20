"""Forward-only, point-in-time backtest (Stage 3 U04 part 1).

Source contract. The existing backtest splits each season in half,
rates players on the first half, and scores squads on the second half.
That split is chronological for the target competition, but the goal
models behind the ratings train on a pooled corpus that includes
matches played after the decisions being scored: UEFA Euro 2020
(51 matches, played June-July 2021) and the Indian Super League
2021/22 (115 matches, played November 2021 to March 2022) are pooled
training data in ``effc.data.download.EXTRA_POOL`` for 2015/16
decisions. A decision dated anywhere in 2015/16 therefore uses a
model that has seen events from 2021 and 2022. This is not a strict
historical deployment simulation.

Part 1 establishes the contract, the baseline capture, the frozen
fixtures, and the independent oracle. It does not retrain any model,
does not change any existing backtest number, and does not promote any
performance claim. The remaining work (bounded implementation and
experiments on real data, closing the parent item) is part 2.

Rules enforced here, all fail-closed with ``PointInTimeError``:

- Every forecast carries a recorded cutoff and a max input timestamp
  strictly earlier than the cutoff. Inputs are matches with
  ``match_date < cutoff``; wages are a snapshot dated at or before the
  cutoff; scored matches are strictly after the cutoff.
- Input and scored match IDs are disjoint. A pooled training match
  dated at or after the cutoff is a violation, reported by match.
- Decisions expand chronologically: each later cutoff sees a superset
  of the earlier cutoff's inputs. Retraining happens at each cutoff on
  the expanding window, and the cache key binds the cutoff plus the
  exact input match IDs (plus a content digest when one is given).
- All policies face identical feasible constraints: the same player
  pool, the same position quotas, the same wage budget, the same squad
  size, solved by the same exact MILP selector. The policies are the
  optimizer (exposure-weighted expected total VAEP), value
  (value per euro), point (raw per-90 rate, no exposure weighting),
  random (seeded mean over random feasible squads), and oracle
  (hindsight realized totals, an upper bound by construction).
- Unavailable players are counted and penalized, never dropped
  silently. A selected player with no realized action in the forward
  window contributes 0, is counted, and costs the squad the mean
  realized VAEP of its available selected players per unavailable
  player (the opportunity cost of a dead roster slot). Both the raw
  realized total and the penalized total are recorded.
- Failed matchups stay reported. A budget no policy can field a squad
  under produces a failed row per policy with its reason; it is never
  silently skipped the way the existing runner skips infeasible
  budgets.

Every decision produces a JSON-serializable record with a SHA-256 of
its canonical form (floats rounded to 6 decimals). Regenerating the
run from the frozen fixture must reproduce the committed record
hashes in the real-data fixture ledger.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from effc.backtest.runner import (
    _top_with_quotas,
    random_squads,
    squad_realized_score,
)
from effc.portfolio.portfolio import SQUAD_SIZE, select_squad

POLICIES = ("optimizer", "value", "point", "random", "oracle")

#: Committed real artifacts the fixture ledger anchors on. Paths are
#: relative to the repository root.
LEDGER_ARTIFACTS = (
    "demo/backtest.csv",
    "demo/players.csv",
    "demo/squads.csv",
    "demo/meta.json",
    "runs/five_leagues/results.json",
    "protocol/fc6_manifest.json",
    "tests/fixtures/forward_backtest/forward_fixture.json",
)


class PointInTimeError(ValueError):
    """A forecast would use information from at or after its cutoff."""


@dataclass(frozen=True)
class ForwardDecision:
    """One point-in-time decision: what was known, and when.

    ``cutoff`` is the decision date. ``max_input_timestamp`` is the
    latest timestamp among all model inputs (events and wages) and
    must be strictly earlier than ``cutoff``. ``input_match_ids`` are
    the matches those inputs come from. ``wage_snapshot_date`` is the
    date of the wage snapshot used. The forward window is
    ``(window_start, window_end]`` and holds only matches strictly
    after the cutoff. ``cache_key`` binds the cutoff and the exact
    input set for the retraining schedule.
    """

    decision_id: str
    cutoff: pd.Timestamp
    max_input_timestamp: pd.Timestamp
    input_match_ids: tuple
    wage_snapshot_date: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    retrain: bool
    cache_key: str


def canonical_hash(value) -> str:
    """SHA-256 of the canonical JSON form of ``value``."""
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def file_sha256(path: str) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ledger_artifact_paths() -> tuple:
    """Repo-relative paths the fixture ledger anchors on."""
    return LEDGER_ARTIFACTS


def decision_cache_key(
    cutoff, input_match_ids, content_digest: str | None = None
) -> str:
    """Retraining cache key: cutoff plus the exact input match IDs.

    Two decisions share a key only if they decide at the same cutoff
    on the same inputs (and the same content digest, when given).
    """
    ids = ",".join(str(int(m)) for m in sorted(int(m) for m in input_match_ids))
    src = f"{pd.Timestamp(cutoff).isoformat()}|{ids}|{content_digest or ''}"
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def check_pool_point_in_time(pool: pd.DataFrame, cutoff) -> list[dict]:
    """Pool matches dated at or after ``cutoff`` (the point-in-time leak).

    ``pool`` is a frame with ``match_id`` and ``match_date`` columns
    (extra columns such as competition and season are carried through).
    Returns one dict per violating match, sorted by date then match ID.
    An empty list means the pool is clean for that cutoff.
    """
    cutoff = pd.Timestamp(cutoff)
    frame = pool.copy()
    frame["match_date"] = pd.to_datetime(frame["match_date"])
    bad = frame[frame["match_date"] >= cutoff].sort_values(
        ["match_date", "match_id"]
    )
    out = []
    for _, row in bad.iterrows():
        entry = {
            "match_id": int(row["match_id"]),
            "match_date": pd.Timestamp(row["match_date"]).strftime("%Y-%m-%d"),
        }
        for col in ("competition", "season"):
            if col in bad.columns:
                entry[col] = row[col]
        out.append(entry)
    return out


def assert_pool_point_in_time(pool: pd.DataFrame, cutoff) -> None:
    """Raise ``PointInTimeError`` unless the pool is clean for ``cutoff``."""
    violations = check_pool_point_in_time(pool, cutoff)
    if violations:
        detail = ", ".join(
            f"match {v['match_id']} on {v['match_date']}" for v in violations
        )
        raise PointInTimeError(
            f"pool contains {len(violations)} match(es) dated at or after "
            f"cutoff {pd.Timestamp(cutoff).strftime('%Y-%m-%d')}: {detail}"
        )


def build_decision_schedule(
    matches: pd.DataFrame,
    cutoffs,
    horizon_days: int = 120,
    retrain: bool = True,
) -> list[ForwardDecision]:
    """Expanding chronological decisions from a match calendar.

    ``matches`` has ``match_id`` and ``match_date``. For each cutoff,
    the inputs are the matches strictly before it and the forward
    window runs from the cutoff to the next cutoff (or
    ``cutoff + horizon_days`` for the last one). A cutoff with no prior
    match fails closed: a forecast with no history is not a decision.
    """
    frame = matches.copy()
    frame["match_date"] = pd.to_datetime(frame["match_date"])
    ordered = sorted(pd.Timestamp(c) for c in cutoffs)
    decisions = []
    for i, cutoff in enumerate(ordered):
        inputs = frame[frame["match_date"] < cutoff]
        if inputs.empty:
            raise PointInTimeError(
                f"no matches before cutoff {cutoff.strftime('%Y-%m-%d')}; "
                "a decision needs history"
            )
        window_end = (
            ordered[i + 1]
            if i + 1 < len(ordered)
            else cutoff + pd.Timedelta(days=horizon_days)
        )
        input_ids = tuple(sorted(int(m) for m in inputs["match_id"].tolist()))
        decisions.append(
            ForwardDecision(
                decision_id=f"fwd-{i + 1:02d}-{cutoff.strftime('%Y-%m-%d')}",
                cutoff=cutoff,
                max_input_timestamp=pd.Timestamp(inputs["match_date"].max()),
                input_match_ids=input_ids,
                wage_snapshot_date=cutoff,
                window_start=cutoff,
                window_end=window_end,
                retrain=retrain,
                cache_key=decision_cache_key(cutoff, input_ids),
            )
        )
    return decisions


def validate_decision(
    decision: ForwardDecision,
    input_timestamps,
    scored_match_ids,
    wage_snapshot_date=None,
) -> None:
    """Fail closed unless ``decision`` respects its own cutoff.

    ``input_timestamps`` are the timestamps of the model inputs; every
    one must precede the cutoff and their max must equal the recorded
    max input timestamp. ``scored_match_ids`` must be disjoint from
    the input match IDs. The effective wage snapshot date (the
    argument, else the decision's own) must be at or before the cutoff.
    """
    if decision.max_input_timestamp >= decision.cutoff:
        raise PointInTimeError(
            f"decision {decision.decision_id}: max input timestamp "
            f"{decision.max_input_timestamp} is not before cutoff "
            f"{decision.cutoff}"
        )
    stamps = [pd.Timestamp(t) for t in input_timestamps]
    if stamps:
        if max(stamps) != decision.max_input_timestamp:
            raise PointInTimeError(
                f"decision {decision.decision_id}: recorded max input "
                f"timestamp {decision.max_input_timestamp} does not match "
                f"the inputs' max {max(stamps)}"
            )
        if max(stamps) >= decision.cutoff:
            raise PointInTimeError(
                f"decision {decision.decision_id}: an input timestamp is "
                f"not before cutoff {decision.cutoff}"
            )
    overlap = set(int(m) for m in scored_match_ids) & set(
        int(m) for m in decision.input_match_ids
    )
    if overlap:
        raise PointInTimeError(
            f"decision {decision.decision_id}: scored matches overlap "
            f"inputs: {sorted(overlap)}"
        )
    wage_date = (
        pd.Timestamp(wage_snapshot_date)
        if wage_snapshot_date is not None
        else decision.wage_snapshot_date
    )
    if wage_date > decision.cutoff:
        raise PointInTimeError(
            f"decision {decision.decision_id}: wage snapshot {wage_date} "
            f"is after cutoff {decision.cutoff}"
        )


def score_squad_with_penalty(
    squad: pd.DataFrame, realized_pm: pd.DataFrame
) -> dict:
    """Realized score of one squad, with the unavailable-player penalty.

    Raw score is the summed realized VAEP over squad members; players
    with no realized rows contribute 0 and are counted. The penalty
    prices each unavailable player at the mean realized VAEP of the
    squad's available selected players (0 when none is available), so
    the penalized total charges a dead roster slot its opportunity
    cost. Both totals are returned.
    """
    raw, n_unavailable = squad_realized_score(squad, realized_pm)
    if realized_pm.empty:
        penalty = 0.0
    else:
        totals = realized_pm.groupby("player_id")["vaep"].sum()
        available = [
            float(totals[pid])
            for pid in squad["player_id"].tolist()
            if pid in totals.index
        ]
        penalty = float(np.mean(available)) if available else 0.0
    return {
        "score_raw": float(raw),
        "n_unavailable": int(n_unavailable),
        "penalty_per_unavailable": penalty,
        "score_penalized": float(raw) - penalty * int(n_unavailable),
    }


def _select_policy(
    policy: str,
    ratings: pd.DataFrame,
    realized_totals: pd.Series,
    budget: float,
    seed: int,
    n_random: int,
    squad_size: int,
) -> list[pd.DataFrame]:
    """Select squad(s) for one policy under the shared constraints.

    Every policy solves the identical feasible problem (same pool, same
    quotas, same wage budget, same squad size) through the same exact
    MILP selector; only the maximized signal differs. Returns a list of
    squads: one squad per policy, except random, which returns
    ``n_random`` squads. Raises RuntimeError when no full squad fits.
    """
    r1 = ratings.copy()
    r1["exp_contrib"] = r1["minutes"].fillna(0.0) * r1["vaep_per90"] / 90.0
    if policy == "optimizer":
        return [select_squad(r1, budget, squad_size=squad_size, value_col="exp_contrib")]
    if policy == "value":
        mb = r1.copy()
        mb["vpe"] = mb["exp_contrib"] / mb["wage_eur"].clip(lower=1.0)
        return [_top_with_quotas(mb, "vpe", budget, squad_size)]
    if policy == "point":
        return [_top_with_quotas(r1, "vaep_per90", budget, squad_size)]
    if policy == "random":
        return random_squads(r1, budget, n=n_random, seed=seed, squad_size=squad_size)
    if policy == "oracle":
        r2 = r1.copy()
        r2["realized_total"] = (
            realized_totals.reindex(r2["player_id"]).fillna(0.0).to_numpy()
        )
        return [select_squad(r2, budget, squad_size=squad_size, value_col="realized_total")]
    raise ValueError(f"unknown policy {policy!r}")


def _round6(value):
    """Round floats to 6 decimals recursively (stable record hashes)."""
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: _round6(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round6(v) for v in value]
    return value


def decision_record(decision: ForwardDecision, policies: dict, failures: list) -> dict:
    """JSON-serializable record of one decision, with its hash.

    ``policies`` maps policy name to budget key (``f"{budget:g}"``) to
    row dict; ``failures`` is the decision's list of failed rows. The
    record hash covers the whole body except the hash field itself.
    """
    body = {
        "decision_id": decision.decision_id,
        "cutoff": decision.cutoff.strftime("%Y-%m-%d"),
        "max_input_timestamp": decision.max_input_timestamp.strftime("%Y-%m-%d"),
        "input_match_ids": [int(m) for m in decision.input_match_ids],
        "wage_snapshot_date": decision.wage_snapshot_date.strftime("%Y-%m-%d"),
        "window_start": decision.window_start.strftime("%Y-%m-%d"),
        "window_end": decision.window_end.strftime("%Y-%m-%d"),
        "retrain": bool(decision.retrain),
        "cache_key": decision.cache_key,
        "policies": _round6(policies),
        "failures": _round6(failures),
    }
    body["record_sha256"] = canonical_hash(body)
    return body


def verify_record_hash(record: dict) -> bool:
    """Recompute a decision record's hash and compare it."""
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    return canonical_hash(body) == record.get("record_sha256")


def run_forward_backtest(
    decisions: list[ForwardDecision],
    ratings_by_decision: dict,
    realized_by_decision: dict,
    budgets: list[float],
    seed: int = 0,
    n_random: int = 20,
    squad_size: int = SQUAD_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Run the forward-only backtest over expanding decisions.

    ``ratings_by_decision`` maps each decision ID to a ratings frame
    built only from pre-cutoff inputs (player_id, pos_group, wage_eur,
    minutes, vaep_per90). ``realized_by_decision`` maps each decision
    ID to the realized per-player per-match VAEP frame for its forward
    window (match_id, match_date, player_id, vaep). Every realized row
    must fall inside the decision's window and its match IDs must be
    disjoint from the decision's inputs, else ``PointInTimeError``.

    Returns (results, failures, records). Results hold one row per
    decision, budget, and policy that fielded a squad; failures hold
    one row per decision, budget, and policy that could not, with its
    reason. Records hold the full per-decision JSON record including
    the selected squads.
    """
    result_rows: list[dict] = []
    failure_rows: list[dict] = []
    records: list[dict] = []
    for decision in decisions:
        if decision.max_input_timestamp >= decision.cutoff:
            raise PointInTimeError(
                f"decision {decision.decision_id}: max input timestamp "
                f"{decision.max_input_timestamp} is not before cutoff "
                f"{decision.cutoff}"
            )
        ratings = ratings_by_decision[decision.decision_id]
        realized = realized_by_decision[decision.decision_id]
        realized = realized.copy()
        realized["match_date"] = pd.to_datetime(realized["match_date"])
        if not realized.empty:
            if (realized["match_date"] <= decision.cutoff).any():
                raise PointInTimeError(
                    f"decision {decision.decision_id}: realized rows at or "
                    f"before the cutoff"
                )
            if (realized["match_date"] > decision.window_end).any():
                raise PointInTimeError(
                    f"decision {decision.decision_id}: realized rows after "
                    f"the forward window"
                )
            overlap = set(realized["match_id"].astype(int)) & set(
                int(m) for m in decision.input_match_ids
            )
            if overlap:
                raise PointInTimeError(
                    f"decision {decision.decision_id}: realized matches "
                    f"overlap inputs: {sorted(overlap)}"
                )
        realized_totals = (
            realized.groupby("player_id")["vaep"].sum()
            if not realized.empty
            else pd.Series(dtype=float)
        )
        policies: dict = {}
        decision_failures: list[dict] = []
        for budget in budgets:
            budget = float(budget)
            key = f"{budget:g}"
            for policy in POLICIES:
                try:
                    squads = _select_policy(
                        policy, ratings, realized_totals, budget,
                        seed, n_random, squad_size,
                    )
                except RuntimeError as exc:
                    failure = {
                        "decision_id": decision.decision_id,
                        "cutoff": decision.cutoff.strftime("%Y-%m-%d"),
                        "budget": budget,
                        "policy": policy,
                        "status": "failed",
                        "reason": str(exc),
                    }
                    failure_rows.append(failure)
                    decision_failures.append(failure)
                    continue
                scored = [
                    score_squad_with_penalty(squad, realized)
                    for squad in squads
                ]
                representative = squads[0]
                row = {
                    "squad": [int(p) for p in representative["player_id"].tolist()],
                    "score_raw": float(np.mean([s["score_raw"] for s in scored])),
                    "score_penalized": float(
                        np.mean([s["score_penalized"] for s in scored])
                    ),
                    "penalty_per_unavailable": float(
                        np.mean([s["penalty_per_unavailable"] for s in scored])
                    ),
                    "n_unavailable": float(
                        np.mean([s["n_unavailable"] for s in scored])
                    ),
                    "n_selected": int(len(representative)),
                    "wage_bill": float(representative["wage_eur"].sum()),
                }
                if policy != "random":
                    row["n_unavailable"] = int(row["n_unavailable"])
                policies.setdefault(policy, {})[key] = row
                result_rows.append(
                    {
                        "decision_id": decision.decision_id,
                        "cutoff": decision.cutoff.strftime("%Y-%m-%d"),
                        "budget": budget,
                        "policy": policy,
                        "score_raw": row["score_raw"],
                        "score_penalized": row["score_penalized"],
                        "penalty_per_unavailable": row["penalty_per_unavailable"],
                        "n_unavailable": row["n_unavailable"],
                        "n_selected": row["n_selected"],
                        "wage_bill": row["wage_bill"],
                    }
                )
        records.append(decision_record(decision, policies, decision_failures))
    results = pd.DataFrame(
        result_rows,
        columns=[
            "decision_id", "cutoff", "budget", "policy", "score_raw",
            "score_penalized", "penalty_per_unavailable", "n_unavailable",
            "n_selected", "wage_bill",
        ],
    )
    failures = pd.DataFrame(
        failure_rows,
        columns=["decision_id", "cutoff", "budget", "policy", "status", "reason"],
    )
    return results, failures, records


def verify_ledger(ledger: dict, repo_root: str) -> list[str]:
    """Recompute a fixture ledger from disk; return mismatch messages.

    Checks every anchored artifact's SHA-256, the frozen FC6 manifest
    hash, the verbatim five-league backtest table, and the verbatim
    demo backtest CSV. An empty list means the ledger matches the repo.
    """
    problems: list[str] = []
    for entry in ledger.get("artifacts", []):
        path = os.path.join(repo_root, entry["path"])
        if not os.path.exists(path):
            problems.append(f"missing artifact {entry['path']}")
            continue
        if file_sha256(path) != entry["sha256"]:
            problems.append(f"artifact hash mismatch {entry['path']}")
    fc6 = ledger.get("fc6", {})
    if fc6:
        manifest = os.path.join(repo_root, "protocol", "fc6_manifest.json")
        if os.path.exists(manifest):
            if file_sha256(manifest) != fc6.get("manifest_sha256"):
                problems.append("fc6 manifest hash mismatch")
    baseline = ledger.get("baseline", {})
    five_path = os.path.join(repo_root, "runs", "five_leagues", "results.json")
    if baseline.get("five_leagues") and os.path.exists(five_path):
        with open(five_path) as fh:
            five = json.load(fh)
        if five.get("model_hash") != baseline["five_leagues"].get("model_hash"):
            problems.append("five-league model hash mismatch")
        for league, rows in baseline["five_leagues"].get("backtest", {}).items():
            if five.get("leagues", {}).get(league, {}).get("backtest") != rows:
                problems.append(f"five-league backtest mismatch {league}")
    demo_csv = baseline.get("demo_backtest_csv")
    if demo_csv is not None:
        demo_path = os.path.join(repo_root, "demo", "backtest.csv")
        if os.path.exists(demo_path):
            with open(demo_path) as fh:
                if fh.read() != demo_csv:
                    problems.append("demo backtest csv mismatch")
    return problems
