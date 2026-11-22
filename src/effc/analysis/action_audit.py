"""Independent audit of repo VAEP values against reference SPADL/Atomic-SPADL.

The repo values on-ball actions with its own VAEP-style formula
(`effc.vaep.values.add_vaep`) on its own featurization
(`effc.vaep.features.featurize_match`). This module checks that pipeline
against an independent replication built on socceraction, the reference
implementation behind the VAEP paper (Decroos et al. 2019):

- `flatten_statsbomb_events` reshapes raw StatsBomb event dicts into the
  flat frame socceraction expects (the same flattening statsbombpy does).
- `spadl_actions` / `atomic_spadl_actions` run the reference
  StatsBomb-to-SPADL and SPADL-to-Atomic-SPADL conversions.
- `coverage_table` counts, per raw event type, which actions each
  pipeline values and which it misses, with reconciliation identities.
- `reference_vaep` re-implements the VAEP value formula from scratch as
  an explicit per-row loop (different code path from `add_vaep`) so the
  two implementations can be checked against each other on known-answer
  fixtures.
- `check_vaep_invariants` verifies the mathematical invariants of a
  valued frame: values add up, probabilities stay in range, a goal
  earns exactly +1 above expectation, and the goal-conceded debit lands
  exactly once per goal.
- `orientation_check` decides, from the raw events themselves, whether
  StatsBomb coordinates are in the acting team's attacking frame or in
  a fixed frame. This matters because the reference conversion flips
  away-team coordinates to a fixed frame, while the repo converts with
  scale only. The check is decisive on real data: across a possession
  change the next action starts at the mirror of the previous action's
  end, not at the shared-frame point.
- `audit_match` runs the whole comparison for one match and returns the
  coverage table, the orientation result, the per-action disagreement
  distribution and the per-player aggregate comparison.

The value comparison deliberately shares one probability source: both
pipelines use the repo goal-model probabilities carried over by matched
event id. The comparison therefore isolates the action transformation
(action set, coordinates, sequence membership), not the goal model. No
numeric equivalence with published VAEP values is claimed anywhere
here; the repo models are its own, trained on its own features.

Full-match numbers for the spot check (Bayer Leverkusen vs VfB
Stuttgart, 2024-04-27, match 3895320, StatsBomb open data) are published
in docs/action_value_audit.md. The committed test fixture is a 141-event
excerpt of that match, indices 1990-2130, covering one Stuttgart goal.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from effc.vaep.features import ACTION_TYPES, featurize_match

# SPADL pitch dimensions, in metres.
SPADL_LENGTH = 105.0
SPADL_WIDTH = 68.0

# Raw StatsBomb keys that are flattened into their own columns; everything
# else stays in the `extra` dict socceraction parses (pass, shot, carry,
# duel, goalkeeper, interception, clearance, foul_committed, ...).
_FLAT_KEYS = {
    "id", "index", "period", "timestamp", "minute", "second", "type",
    "possession", "possession_team", "play_pattern", "team", "duration",
    "tactics", "related_events", "player", "position", "location",
}

COVERAGE_COLUMNS = [
    "event_type", "raw_count", "repo_rows", "spadl_actions",
    "matched", "repo_only", "spadl_only",
]

_SPADL_COLUMNS = [
    "game_id", "original_event_id", "period_id", "time_seconds",
    "team_id", "player_id", "start_x", "start_y", "end_x", "end_y",
    "type_id", "result_id", "bodypart_id", "action_id",
    "type_name", "result_name", "bodypart_name",
]

_ATOMIC_COLUMNS = [
    "game_id", "original_event_id", "action_id", "period_id",
    "time_seconds", "team_id", "player_id", "x", "y", "dx", "dy",
    "type_id", "bodypart_id", "type_name",
]


def flatten_statsbomb_events(events: list[dict], match_id: int) -> pd.DataFrame:
    """Reshape raw StatsBomb event dicts into socceraction's flat frame."""
    rows = []
    for e in events:
        extra = {k: v for k, v in e.items() if k not in _FLAT_KEYS}
        rows.append(
            {
                "game_id": match_id,
                "event_id": e.get("id"),
                "period_id": e.get("period", 1),
                "timestamp": e.get("timestamp", "00:00:00.000"),
                "team_id": (e.get("team") or {}).get("id"),
                "player_id": (e.get("player") or {}).get("id"),
                "location": e.get("location"),
                "extra": extra,
                "type_name": (e.get("type") or {}).get("name"),
            }
        )
    cols = [
        "game_id", "event_id", "period_id", "timestamp", "team_id",
        "player_id", "location", "extra", "type_name",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


def spadl_actions(
    events: list[dict], match_id: int, home_team_id: int
) -> pd.DataFrame:
    """Reference SPADL actions for raw events, via socceraction.

    Returns one row per SPADL action with human-readable type, result
    and body-part names attached. Raises ImportError with a plain message
    when socceraction is not installed.
    """
    try:
        from socceraction.spadl import add_names
        from socceraction.spadl.statsbomb import convert_to_actions
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "socceraction is required for the reference SPADL conversion; "
            "install the dev dependencies (pip install -e '.[dev]')"
        ) from exc
    flat = flatten_statsbomb_events(events, match_id)
    if flat.empty:
        return pd.DataFrame(columns=_SPADL_COLUMNS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        actions = convert_to_actions(flat, home_team_id)
    actions = add_names(actions)
    return actions.reset_index(drop=True)


def atomic_spadl_actions(spadl_df: pd.DataFrame) -> pd.DataFrame:
    """Reference Atomic-SPADL actions for a SPADL frame, via socceraction.

    Atomic SPADL splits SPADL actions into atomic pieces: a completed
    pass gains a `receival` row for the receiver, a goal becomes its own
    `goal` row, and long carries split into dribble pieces. The returned
    frame carries `type_name` mapped from the atomic action-type list.
    """
    try:
        from socceraction.atomic.spadl import config as atomic_config
        from socceraction.atomic.spadl import convert_to_atomic
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "socceraction is required for the reference Atomic-SPADL "
            "conversion; install the dev dependencies (pip install -e '.[dev]')"
        ) from exc
    if spadl_df.empty:
        return pd.DataFrame(columns=_ATOMIC_COLUMNS)
    atomic = convert_to_atomic(spadl_df)
    names = atomic_config.actiontypes
    atomic = atomic.copy()
    atomic["type_name"] = [names[int(t)] for t in atomic["type_id"]]
    return atomic.reset_index(drop=True)


def repo_action_ids(events: list[dict]) -> list:
    """Raw event ids in repo-row order (sorted by index, action types only)."""
    ordered = sorted(events, key=lambda e: e.get("index", 0))
    return [
        e.get("id")
        for e in ordered
        if (e.get("type") or {}).get("name") in ACTION_TYPES
    ]


def attach_event_ids(repo_df: pd.DataFrame, events: list[dict]) -> pd.DataFrame:
    """Add the raw event id to each repo frame row, in row order."""
    ids = repo_action_ids(events)
    if len(ids) != len(repo_df):
        raise ValueError(
            f"repo frame has {len(repo_df)} rows but {len(ids)} action events"
        )
    out = repo_df.copy()
    out["event_id"] = ids
    return out


def coverage_table(
    events: list[dict], match_id: int, home_team_id: int
) -> pd.DataFrame:
    """Per raw event type: what each pipeline values, matched, and missed.

    `matched` counts raw events valued by both pipelines (by event id).
    `repo_only` and `spadl_only` count the rest. SPADL actions with no
    raw event behind them (synthesized dribbles) are listed under the
    `synthesized` pseudo-type so the reconciliation identities hold:
    matched + repo_only == repo_rows, matched + spadl_only == spadl_actions.
    """
    if not events:
        return pd.DataFrame(columns=COVERAGE_COLUMNS)
    ordered = sorted(events, key=lambda e: e.get("index", 0))
    raw_counts: dict[str, int] = {}
    id_to_type: dict[str, str] = {}
    for e in ordered:
        t = (e.get("type") or {}).get("name", "?")
        raw_counts[t] = raw_counts.get(t, 0) + 1
        if e.get("id") is not None:
            id_to_type[e["id"]] = t

    repo = featurize_match(ordered, match_id=match_id)
    repo_ids = set(repo_action_ids(ordered))
    repo_counts = repo["action"].value_counts().to_dict() if len(repo) else {}

    spadl = spadl_actions(ordered, match_id, home_team_id)
    spadl_ids = set(spadl["original_event_id"]) if len(spadl) else set()
    spadl_counts: dict[str, int] = {}
    synthesized = 0
    for eid in spadl["original_event_id"] if len(spadl) else []:
        t = id_to_type.get(eid)
        if t is None:
            synthesized += 1
        else:
            spadl_counts[t] = spadl_counts.get(t, 0) + 1

    rows = []
    for t in sorted(raw_counts, key=lambda k: -raw_counts[k]):
        matched = len(
            {
                eid
                for eid, tt in id_to_type.items()
                if tt == t and eid in repo_ids and eid in spadl_ids
            }
        )
        r = repo_counts.get(t, 0)
        s = spadl_counts.get(t, 0)
        rows.append(
            {
                "event_type": t,
                "raw_count": raw_counts[t],
                "repo_rows": r,
                "spadl_actions": s,
                "matched": matched,
                "repo_only": r - matched,
                "spadl_only": s - matched,
            }
        )
    if synthesized:
        rows.append(
            {
                "event_type": "synthesized",
                "raw_count": 0,
                "repo_rows": 0,
                "spadl_actions": synthesized,
                "matched": 0,
                "repo_only": 0,
                "spadl_only": synthesized,
            }
        )
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def orientation_check(events: list[dict]) -> dict:
    """Decide from the raw events which coordinate frame StatsBomb uses.

    Across a possession change, the next action starts either at the
    previous action's end (shared fixed frame) or at its mirror
    (120 - x, 80 - y, acting-team attacking frame). Also reports shot
    and goalkeeper locations by team, which sit at the attacked goal
    and the own goal respectively only in the acting-team frame.
    """
    result = {
        "turnover_pairs": 0,
        "median_shared_m": None,
        "median_mirror_m": None,
        "share_mirror_closer": None,
        "same_team_pairs": 0,
        "same_team_median_m": None,
        "shot_x_min": {},
        "keeper_median_x": {},
    }
    if not events:
        return result
    ordered = sorted(events, key=lambda e: e.get("index", 0))

    def end_loc(e):
        for key in ("pass", "carry", "shot"):
            block = e.get(key)
            if isinstance(block, dict) and block.get("end_location"):
                return block["end_location"]
        return e.get("location")

    shared, mirror, same = [], [], []
    for a, b in zip(ordered, ordered[1:]):
        ta = (a.get("type") or {}).get("name")
        tb = (b.get("type") or {}).get("name")
        if ta not in ACTION_TYPES or tb not in ACTION_TYPES:
            continue
        ea, lb = end_loc(a), b.get("location")
        if not ea or not lb:
            continue
        d = float(np.hypot(lb[0] - ea[0], lb[1] - ea[1]))
        if a.get("team", {}).get("name") == b.get("team", {}).get("name"):
            same.append(d)
        else:
            shared.append(d)
            mirror.append(
                float(np.hypot(lb[0] - (120.0 - ea[0]), lb[1] - (80.0 - ea[1])))
            )
    result["turnover_pairs"] = len(shared)
    if shared:
        result["median_shared_m"] = float(np.median(shared))
        result["median_mirror_m"] = float(np.median(mirror))
        result["share_mirror_closer"] = float(
            np.mean(np.array(mirror) < np.array(shared))
        )
    result["same_team_pairs"] = len(same)
    if same:
        result["same_team_median_m"] = float(np.median(same))
    for team in {e.get("team", {}).get("name") for e in ordered}:
        xs = [
            e["location"][0]
            for e in ordered
            if (e.get("type") or {}).get("name") == "Shot"
            and e.get("team", {}).get("name") == team
            and e.get("location")
        ]
        if xs:
            result["shot_x_min"][team] = float(min(xs))
        kx = [
            e["location"][0]
            for e in ordered
            if (e.get("type") or {}).get("name") == "Goal Keeper"
            and e.get("team", {}).get("name") == team
            and e.get("location")
        ]
        if kx:
            result["keeper_median_x"][team] = float(np.median(kx))
    return result


def reference_vaep(df: pd.DataFrame, apply_debit: bool = True) -> pd.DataFrame:
    """Re-implement the VAEP value formula as an explicit per-row loop.

    Input is a frame that already carries the goal-model probabilities
    (`ps_incl`, `ps_excl`, `pc_incl`, `pc_excl`) plus `team`, `period`,
    `match_index` and `is_goal`. The formula, from Decroos et al.:

        vaep_off = P(scores after) - P(scores before) + goal
        vaep_def = P(concedes before) - P(concedes after)

    where the before-state is read from the acting team's perspective,
    so on a turnover the opponent's "scores" is our "concedes". The
    first action of a period falls back to the frame mean probabilities.
    With `apply_debit`, a scored goal also debits -1 from the conceding
    team's first action after the goal (same window as `add_vaep`), and
    own goals from `df.attrs["own_goals"]` debit the same way with no
    +1 realization anywhere. Output adds `vaep_off_ref`, `vaep_def_ref`
    and `vaep_ref` columns.
    """
    if df.empty:
        out = df.copy()
        for col in ("vaep_off_ref", "vaep_def_ref", "vaep_ref"):
            out[col] = pd.Series(dtype=float)
        return out
    for col in ("ps_incl", "ps_excl", "pc_incl", "pc_excl",
                "team", "period", "match_index", "is_goal"):
        if col not in df.columns:
            raise ValueError(f"reference_vaep needs column {col!r}")
    out = df.copy()
    n = len(out)
    ps_before = np.zeros(n)
    pc_before = np.zeros(n)
    ps_mean = float(out["ps_incl"].mean())
    pc_mean = float(out["pc_incl"].mean())
    group_keys = (
        ["match_id", "period"] if "match_id" in out.columns else ["period"]
    )
    positions = list(range(n))
    by_group: dict[tuple, list[int]] = {}
    for i in positions:
        key = tuple(out.iloc[i][k] for k in group_keys)
        by_group.setdefault(key, []).append(i)
    for idxs in by_group.values():
        idxs.sort(key=lambda i: out.iloc[i]["match_index"])
        prev = None
        for i in idxs:
            row = out.iloc[i]
            if prev is None:
                ps_before[i] = ps_mean
                pc_before[i] = pc_mean
            elif row["team"] == out.iloc[prev]["team"]:
                ps_before[i] = out.iloc[prev]["ps_incl"]
                pc_before[i] = out.iloc[prev]["pc_incl"]
            else:
                ps_before[i] = out.iloc[prev]["pc_incl"]
                pc_before[i] = out.iloc[prev]["ps_incl"]
            prev = i
    out["vaep_off_ref"] = (
        out["ps_excl"].to_numpy(dtype=float) - ps_before
        + out["is_goal"].to_numpy(dtype=float)
    )
    out["vaep_def_ref"] = pc_before - out["pc_excl"].to_numpy(dtype=float)
    out["vaep_ref"] = out["vaep_off_ref"] + out["vaep_def_ref"]
    if apply_debit:
        debit_targets: dict[int, None] = {}

        def _first_after(pos_list, goal_pos, scorer, period_val, mid_val):
            for j in pos_list:
                r = out.iloc[j]
                if (
                    r["match_index"] > goal_pos
                    and r["match_index"] <= goal_pos + 3
                    and r["team"] != scorer
                    and r["period"] == period_val
                    and ("match_id" not in out.columns or r["match_id"] == mid_val)
                ):
                    return j
            return None

        for key, idxs in by_group.items():
            for i in idxs:
                if not bool(out.iloc[i]["is_goal"]):
                    continue
                r = out.iloc[i]
                mid_val = r["match_id"] if "match_id" in out.columns else None
                tgt = _first_after(
                    idxs, r["match_index"], r["team"], r["period"], mid_val
                )
                if tgt is not None:
                    debit_targets[tgt] = None
        for own_idx, benefit, own_period in out.attrs.get("own_goals", []):
            for key, idxs in by_group.items():
                if key[-1] != own_period:
                    continue
                mid_val = key[0] if "match_id" in out.columns else None
                tgt = _first_after(idxs, own_idx, benefit, own_period, mid_val)
                if tgt is not None:
                    debit_targets[tgt] = None
        for j in debit_targets:
            out.iloc[j, out.columns.get_loc("vaep_def_ref")] -= 1.0
            out.iloc[j, out.columns.get_loc("vaep_ref")] -= 1.0
    return out


def check_vaep_invariants(valued: pd.DataFrame) -> list[dict]:
    """Check the mathematical invariants of a valued frame.

    `valued` is an `add_vaep` output (probability columns, before-state
    columns, vaep_off/vaep_def/vaep, is_goal, team, period, match_index,
    own goals in attrs). Each result is {"name", "passed", "detail"}.
    """
    results: list[dict] = []

    def _check(name, passed, detail):
        results.append({"name": name, "passed": bool(passed), "detail": detail})

    if valued.empty:
        _check("adds_up", True, "empty frame: nothing to add")
        _check("probabilities_in_range", True, "empty frame")
        _check("goal_realization", True, "empty frame: no goal rows")
        _check("debit_exactly_once_per_goal", True, "empty frame: no goals")
        _check("no_nan_values", True, "empty frame")
        _check("debit_total", True, "empty frame")
        return results
    _check(
        "adds_up",
        bool(np.allclose(valued["vaep"], valued["vaep_off"] + valued["vaep_def"])),
        "max |vaep - (vaep_off + vaep_def)| = "
        f"{float(np.abs(valued['vaep'] - valued['vaep_off'] - valued['vaep_def']).max()):.2e}",
    )
    prob_cols = ["ps_incl", "ps_excl", "pc_incl", "pc_excl"]
    missing = [c for c in prob_cols if c not in valued.columns]
    if missing:
        _check("probabilities_in_range", False, f"missing columns: {missing}")
    else:
        lo = min(float(valued[c].min()) for c in prob_cols)
        hi = max(float(valued[c].max()) for c in prob_cols)
        _check(
            "probabilities_in_range",
            bool(0.0 <= lo and hi <= 1.0),
            f"probability range [{lo:.4f}, {hi:.4f}]",
        )
    goals = valued[valued["is_goal"]]
    if len(goals) == 0:
        _check("goal_realization", True, "no goal rows")
    else:
        expect = goals["ps_excl"] - goals["ps_before"] + 1.0
        err = float(np.abs(goals["vaep_off"] - expect).max())
        _check("goal_realization", err <= 1e-9, f"max error {err:.2e} over {len(goals)} goals")
    # The debit, recomputed without it, must be exactly -1 on exactly the
    # rows the rule names: the first conceding-team action within 3 of
    # each goal (regular goals and own goals alike).
    ref_plain = reference_vaep(valued, apply_debit=False)
    diff = (valued["vaep_def"] - ref_plain["vaep_def_ref"]).to_numpy(dtype=float)
    expected: set[int] = set()
    pos_of = {i: p for p, i in enumerate(valued.index)}
    for i, row in valued.iterrows():
        if not bool(row["is_goal"]):
            continue
        later = valued[
            (valued["match_id"] == row["match_id"])
            & (valued["period"] == row["period"])
            & (valued["match_index"] > row["match_index"])
            & (valued["match_index"] <= row["match_index"] + 3)
            & (valued["team"] != row["team"])
        ]
        if len(later):
            expected.add(pos_of[later.index[0]])
    for own_idx, benefit, own_period in valued.attrs.get("own_goals", []):
        mid = valued["match_id"].iloc[0] if "match_id" in valued.columns else None
        later = valued[
            (valued["period"] == own_period)
            & (valued["match_index"] > own_idx)
            & (valued["match_index"] <= own_idx + 3)
            & (valued["team"] != benefit)
        ]
        if "match_id" in valued.columns:
            later = later[later["match_id"] == mid]
        if len(later):
            expected.add(pos_of[later.index[0]])
    debited = {p for p, d in enumerate(diff) if d != 0.0}
    _check(
        "debit_exactly_once_per_goal",
        debited == expected
        and all(diff[p] == -1.0 for p in debited),
        f"debited rows {sorted(debited)}, expected {sorted(expected)}",
    )
    _check(
        "no_nan_values",
        bool(not valued[["vaep", "vaep_off", "vaep_def"]].isna().any().any()),
        "NaN present" if valued[["vaep", "vaep_off", "vaep_def"]].isna().any().any() else "no NaN",
    )
    total = float((valued["vaep"] - ref_plain["vaep_ref"]).sum())
    _check(
        "debit_total",
        abs(total + len(expected)) <= 1e-9,
        f"total debit {total:.1f} over {len(expected)} goals with a following action",
    )
    return results


def audit_match(
    events: list[dict],
    match_id: int,
    home_team_id: int,
    repo_valued: pd.DataFrame | None = None,
) -> dict:
    """Run the full audit for one match.

    `repo_valued` is the `add_vaep` output for the repo frame of the same
    events; when omitted, the coverage and orientation results are still
    produced and the value comparison is reported as skipped. The
    reference sequence is the SPADL-matched subsequence: SPADL actions
    with no repo row (fouls, synthesized dribbles) carry no repo
    probabilities and are excluded from the value comparison, counted
    in the coverage table instead. Own-goal debits are not applied in
    the reference sequence because own goals have no action rows there;
    that is a documented cause of disagreement when own goals occur.
    """
    coverage = coverage_table(events, match_id, home_team_id)
    orientation = orientation_check(events)
    empty_dis = {
        "n_matched": 0,
        "mean_abs_diff": None,
        "median_abs_diff": None,
        "p90_abs_diff": None,
        "max_abs_diff": None,
        "share_gt_0.01": None,
        "share_gt_0.05": None,
        "share_gt_0.10": None,
    }
    result = {
        "coverage": coverage,
        "orientation": orientation,
        "matched": pd.DataFrame(),
        "disagreement": empty_dis,
        "disagreement_by_action": pd.DataFrame(),
        "disagreement_by_transition": pd.DataFrame(),
        "player_aggregates": pd.DataFrame(),
        "invariants": [],
    }
    if repo_valued is None or repo_valued.empty or not events:
        return result
    result["invariants"] = check_vaep_invariants(repo_valued)
    repo = attach_event_ids(repo_valued, events)
    spadl = spadl_actions(events, match_id, home_team_id)
    if spadl.empty:
        return result
    # One row per matched event: a split interception-pass yields two
    # SPADL rows for one event, so keep the first by action order.
    spadl = spadl.sort_values("action_id").drop_duplicates(
        "original_event_id", keep="first"
    )
    matched = repo.merge(
        spadl, left_on="event_id", right_on="original_event_id",
        suffixes=("_repo", "_sa"),
    )
    if matched.empty:
        return result
    result["matched"] = matched
    # Reference sequence: SPADL order, probabilities carried from the repo
    # rows, team identity from the SPADL team id. Own goals are absent
    # from this sequence by construction (see docstring).
    ref_frame = pd.DataFrame(
        {
            "match_id": match_id,
            "period": matched["period_id"].to_numpy(),
            "match_index": np.arange(len(matched)),
            "team": matched["team_id"].to_numpy(),
            "is_goal": matched["is_goal"].to_numpy(dtype=bool),
            "ps_incl": matched["ps_incl"].to_numpy(dtype=float),
            "ps_excl": matched["ps_excl"].to_numpy(dtype=float),
            "pc_incl": matched["pc_incl"].to_numpy(dtype=float),
            "pc_excl": matched["pc_excl"].to_numpy(dtype=float),
        }
    )
    ref = reference_vaep(ref_frame, apply_debit=True)
    comp = pd.DataFrame(
        {
            "event_id": matched["event_id"],
            "action": matched["action"],
            "team": matched["team"],
            "player": matched["player"],
            "player_id": matched["player_id_repo"],
            "repo_vaep": matched["vaep"].to_numpy(dtype=float),
            "ref_vaep": ref["vaep_ref"].to_numpy(dtype=float),
        }
    )
    comp["diff"] = comp["repo_vaep"] - comp["ref_vaep"]
    comp["abs_diff"] = comp["diff"].abs()
    # Transition in the repo sequence: same team as the previous repo row.
    repo_order = repo.reset_index(drop=True)
    same_team = (repo_order["team"] == repo_order["team"].shift(1)).fillna(True)
    transition = same_team.loc[repo["event_id"].isin(set(comp["event_id"]))].to_numpy()
    comp["same_team_transition"] = transition
    absd = comp["abs_diff"].to_numpy(dtype=float)
    result["disagreement"] = {
        "n_matched": int(len(comp)),
        "mean_abs_diff": float(absd.mean()),
        "median_abs_diff": float(np.median(absd)),
        "p90_abs_diff": float(np.quantile(absd, 0.9)),
        "max_abs_diff": float(absd.max()),
        "share_gt_0.01": float((absd > 0.01).mean()),
        "share_gt_0.05": float((absd > 0.05).mean()),
        "share_gt_0.10": float((absd > 0.10).mean()),
    }
    by_action = (
        comp.groupby("action")["abs_diff"]
        .agg(["count", "mean", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    result["disagreement_by_action"] = by_action
    by_trans = (
        comp.groupby("same_team_transition")["abs_diff"]
        .agg(["count", "mean", "max"])
        .reset_index()
    )
    result["disagreement_by_transition"] = by_trans
    from effc.vaep.values import player_match_vaep

    repo_players = player_match_vaep(repo_valued)
    ref_players = (
        comp.groupby(["player_id", "player", "team"], dropna=False)
        .agg(ref_vaep=("ref_vaep", "sum"), ref_actions=("ref_vaep", "size"))
        .reset_index()
    )
    players = repo_players.merge(
        ref_players, on=["player_id", "player", "team"], how="outer"
    )
    players["vaep"] = players["vaep"].fillna(0.0)
    players["ref_vaep"] = players["ref_vaep"].fillna(0.0)
    players["diff"] = players["vaep"] - players["ref_vaep"]
    players["actions"] = players["actions"].fillna(0).astype(int)
    players["ref_actions"] = players["ref_actions"].fillna(0).astype(int)
    result["player_aggregates"] = players.sort_values(
        "diff", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)
    return result
