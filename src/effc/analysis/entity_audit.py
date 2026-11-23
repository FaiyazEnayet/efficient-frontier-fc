"""Real-player entity matching and selection coverage audit (Stage 3 U05).

This module is the part of the entity-matching story the C6 rebuild did
not cover. C6 rebuilt the matcher itself (club crosswalk, ambiguity
quarantine, aliases with evidence, deterministic crosswalk IDs, proxy
wages, the join audit and the gold-set scorer, all in
``effc.data.wages``), and U06 audited action values against a reference
replication (``effc.analysis.action_audit``). Neither of them did the
following, which is what lives here:

- a player ID crosswalk tying StatsBomb player IDs to wage rows, so a
  rating can be traced to a person rather than a name string;
- confidence categories for every match type, grounded in the wage
  uncertainty intervals;
- a reviewed mapping registry: a committed, human-reviewed record of the
  uncertain matches (``data/mapping_registry.csv``), applied as an
  override the matcher cannot silently undo;
- per-position coverage measured from the event side, so unmatched
  players get a position instead of disappearing into an unknown group;
- selection sensitivity: what the exclusions cost the optimizer, measured
  by re-solving squads under wage-interval and proxy-inclusion scenarios
  with the exact MILP selector.

Nothing here retrains a model, re-runs the five-league backtest, or
changes the demo rated output. The frozen FC6 results stay historical.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from effc.data.wages import (
    WAGE_INTERVAL_HALF_WIDTH,
    norm,
    wage_uncertainty_intervals,
)

# StatsBomb lineup/event position names mapped to the four position
# groups the optimizer uses. The names are the 23 distinct values in the
# StatsBomb open-data lineups for the Bundesliga target pool; a name
# outside this table maps to "unknown" instead of being guessed.
SB_POSITION_GROUPS = {
    "Goalkeeper": "GK",
    "Center Back": "DEF",
    "Left Center Back": "DEF",
    "Right Center Back": "DEF",
    "Left Back": "DEF",
    "Right Back": "DEF",
    "Left Wing Back": "DEF",
    "Right Wing Back": "DEF",
    "Center Defensive Midfield": "MID",
    "Left Defensive Midfield": "MID",
    "Right Defensive Midfield": "MID",
    "Center Midfield": "MID",
    "Left Center Midfield": "MID",
    "Right Center Midfield": "MID",
    "Left Midfield": "MID",
    "Right Midfield": "MID",
    "Center Attacking Midfield": "MID",
    "Left Attacking Midfield": "MID",
    "Right Attacking Midfield": "MID",
    "Left Wing": "FWD",
    "Right Wing": "FWD",
    "Center Forward": "FWD",
    "Left Center Forward": "FWD",
    "Right Center Forward": "FWD",
}

# Confidence in a match, from the match type. The wage is a second
# question from the identity, so a reviewed match stays medium: the
# identity is verified by a human, but the wage still comes from another
# club and season. An exact match is the only high-confidence row.
# Ambiguous rows are quarantined, so their confidence is "uncertain"
# rather than "none": the file holds candidate rows for them, and the
# registry is where that uncertainty gets resolved.
MATCH_CONFIDENCE = {
    "exact": "high",
    "exact_other_club": "medium",
    "reviewed": "medium",
    "alias": "low",
    "fuzzy": "low",
    "ambiguous": "uncertain",
    "proxy": "proxy",
    "unmatched": "none",
}
_CONFIDENCE_RANK = {
    "none": 0, "uncertain": 1, "proxy": 2, "low": 3, "medium": 4, "high": 5,
}

REGISTRY_COLUMNS = (
    "player", "team", "player_id", "prior_match_type", "decision",
    "wage_row_id", "wage_eur", "evidence", "reviewer", "reviewed_date",
)
_REGISTRY_DECISIONS = ("accept", "reject")


def statsbomb_position_group(position) -> str:
    """One StatsBomb position name to GK/DEF/MID/FWD, or "unknown"."""
    if position is None:
        return "unknown"
    try:
        if pd.isna(position):
            return "unknown"
    except (TypeError, ValueError):
        pass
    return SB_POSITION_GROUPS.get(str(position).strip(), "unknown")


def match_confidence(match_type: str) -> str:
    """Confidence category for one match type; unknown types fail closed."""
    try:
        return MATCH_CONFIDENCE[match_type]
    except KeyError:
        raise ValueError(
            f"unknown match_type {match_type!r}: expected one of "
            f"{sorted(MATCH_CONFIDENCE)}")


def with_match_confidence(matched: pd.DataFrame) -> pd.DataFrame:
    """Add a ``confidence`` column derived from ``match_type``."""
    out = matched.copy()
    if "match_type" not in out.columns:
        raise ValueError("with_match_confidence needs a match_type column")
    out["confidence"] = out["match_type"].map(match_confidence)
    return out


def build_player_crosswalk(matched: pd.DataFrame) -> pd.DataFrame:
    """One row per StatsBomb player ID that has a wage row.

    A player ID at two clubs (a transfer across the pooled seasons) keeps
    one row with both wage row IDs; the person key must be the same for
    both, because a transfer does not change the person. Fails closed
    when one player ID points at two different people (two person keys,
    or two normalized names), or when a matched row carries no wage row
    ID. Unmatched and quarantined rows have no wage row to point at, so
    they are excluded and counted in the report instead of the crosswalk.

    The returned frame carries ``attrs["player_crosswalk_report"]`` with
    the counts: player IDs in, player IDs with a wage row, excluded rows,
    and player IDs at more than one club.
    """
    for column in ("player_id", "player", "team", "person_key",
                   "wage_row_id", "match_type", "matched"):
        if column not in matched.columns:
            raise ValueError(f"build_player_crosswalk needs a {column} column")
    frame = matched.copy()
    n_in = len(frame)
    frame["person_key"] = frame["person_key"].fillna(frame["player"].map(norm))
    # A matched row without a wage row ID is a broken crosswalk, not an
    # exclusion: every matched row must carry its crosswalk ID.
    broken = frame.loc[frame["matched"] & frame["wage_row_id"].isna(), "player"]
    if len(broken):
        raise ValueError(
            f"matched rows without a wage row ID: {broken.tolist()[:5]}")
    has_wage = frame["matched"] & frame["wage_row_id"].notna()
    n_excluded = int((~has_wage).sum())
    cross = frame.loc[has_wage].copy()
    # One player ID is one person: two person keys or two names fail.
    by_id = cross.groupby("player_id")
    bad_keys = by_id["person_key"].nunique()
    bad_keys = bad_keys[bad_keys > 1]
    if len(bad_keys):
        raise ValueError(
            "player_id maps to more than one person_key: "
            f"{bad_keys.index.tolist()[:5]}")
    bad_names = by_id["player"].agg(lambda s: len({norm(v) for v in s}))
    bad_names = bad_names[bad_names > 1]
    if len(bad_names):
        raise ValueError(
            "player_id maps to more than one normalized name: "
            f"{bad_names.index.tolist()[:5]}")
    rows = []
    for pid, sub in by_id:
        rows.append({
            "player_id": pid,
            "player": sub["player"].iloc[0],
            "person_key": sub["person_key"].iloc[0],
            "teams": "; ".join(sorted({str(t) for t in sub["team"]})),
            "n_teams": int(sub["team"].nunique()),
            "wage_row_ids": "; ".join(sorted(
                {str(w) for w in sub["wage_row_id"]})),
            "match_types": "; ".join(sorted(
                {str(t) for t in sub["match_type"]})),
            "confidence": min(
                (match_confidence(t) for t in sub["match_type"]),
                key=lambda c: _CONFIDENCE_RANK[c]),
        })
    out = pd.DataFrame(rows).sort_values("player_id").reset_index(drop=True)
    out.attrs["player_crosswalk_report"] = {
        "n_player_rows": int(n_in),
        "n_player_ids": int(out["player_id"].nunique()),
        "n_excluded_no_wage_row": n_excluded,
        "n_player_ids_multi_club": int((out["n_teams"] > 1).sum()),
    }
    return out


def default_registry_path() -> str:
    """Committed registry path, resolved from this file's location."""
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "data", "mapping_registry.csv"))


def load_mapping_registry(path: str | None = None) -> pd.DataFrame:
    """Load and validate the reviewed mapping registry.

    Every row needs a player, team and player ID, a decision of accept
    or reject, and non-empty evidence: an accept also needs a wage row ID
    and a positive wage, and the wage itself is checked against the wage
    file when the registry is applied, not here. A duplicate (player,
    team) row is a conflict and fails, because two decisions for one
    player cannot both stand. Returns the registry with player_id as int.
    """
    path = path or default_registry_path()
    reg = pd.read_csv(path, dtype={"player_id": "Int64"},
                      keep_default_na=False)
    # An empty wage reads back as "": coerce it to NaN so a reject row
    # (which carries no wage) loads instead of failing the parse.
    reg["wage_eur"] = pd.to_numeric(reg["wage_eur"], errors="coerce")
    missing = [c for c in REGISTRY_COLUMNS if c not in reg.columns]
    if missing:
        raise ValueError(
            f"mapping registry {path} is missing columns: {missing}")
    if reg.duplicated(subset=["player", "team"]).any():
        dupes = reg.loc[reg.duplicated(subset=["player", "team"], keep=False),
                        ["player", "team"]].drop_duplicates()
        raise ValueError(
            "mapping registry has more than one decision for the same "
            f"player at the same team: {dupes.to_dict('records')[:5]}")
    for i, row in reg.iterrows():
        who = f"{row['player']} at {row['team']} (row {i})"
        if row["decision"] not in _REGISTRY_DECISIONS:
            raise ValueError(
                f"{who}: decision must be accept or reject, "
                f"got {row['decision']!r}")
        if not str(row["evidence"]).strip():
            raise ValueError(f"{who}: evidence is required")
        if not str(row["reviewer"]).strip():
            raise ValueError(f"{who}: reviewer is required")
        if not str(row["reviewed_date"]).strip():
            raise ValueError(f"{who}: reviewed_date is required")
        if pd.isna(row["player_id"]):
            raise ValueError(f"{who}: player_id is required")
        if row["decision"] == "accept":
            if not str(row["wage_row_id"]).strip():
                raise ValueError(
                    f"{who}: an accept needs a wage_row_id")
            if not pd.notna(row["wage_eur"]) or float(row["wage_eur"]) <= 0:
                raise ValueError(
                    f"{who}: an accept needs a positive wage_eur")
    reg["player_id"] = reg["player_id"].astype(int)
    return reg


def apply_mapping_registry(matched: pd.DataFrame, registry: pd.DataFrame,
                           wages: pd.DataFrame) -> pd.DataFrame:
    """Apply reviewed decisions to a matched frame.

    An accept resolves the row to the reviewed wage row: wage, value,
    position group, club, person key, wage row ID and wage source all
    come from that wage row, the match type becomes "reviewed" and the
    row counts as matched. The wage in the registry must agree with the
    wage file within 1 EUR/week, so a stale registry cannot silently
    reprice a player. A reject quarantines the row: the match type
    becomes "ambiguous", the row counts as unmatched, and any wage is
    cleared, even if the matcher had matched it. Every row carries
    ``prior_match_type`` so the registry's effect stays auditable. Fails
    closed on a registry row for a (player, team) pair that is not in the
    matched frame, on an accept whose wage row ID is not in the wage
    table, and on a wage disagreement.
    """
    for column in ("player", "team", "match_type"):
        if column not in matched.columns:
            raise ValueError(f"apply_mapping_registry needs a {column} column")
    if "wage_row_id" not in wages.columns:
        raise ValueError("apply_mapping_registry needs wage_row_id in wages")
    out = matched.copy()
    out["prior_match_type"] = out["match_type"]
    keyed = out.set_index(["player", "team"])
    wage_by_id = wages.set_index("wage_row_id")
    for row in registry.itertuples():
        key = (row.player, row.team)
        if key not in keyed.index:
            raise ValueError(
                f"registry row for {row.player} at {row.team} has no "
                f"matching (player, team) row in the matched frame")
        idx = keyed.index.get_loc(key)
        # get_loc returns a slice or mask when the pair repeats; the
        # matched frame holds one row per (player, team), so take it.
        if not isinstance(idx, (int, np.integer)):
            idx = out.index[
                (out["player"] == row.player) & (out["team"] == row.team)]
            if len(idx) != 1:
                raise ValueError(
                    f"registry row for {row.player} at {row.team} matches "
                    f"{len(idx)} matched rows, expected exactly one")
            idx = idx[0]
        if row.decision == "accept":
            if row.wage_row_id not in wage_by_id.index:
                raise ValueError(
                    f"registry accept for {row.player} at {row.team} "
                    f"names wage row {row.wage_row_id}, which is not in "
                    f"the wage table")
            wrow = wage_by_id.loc[row.wage_row_id]
            if isinstance(wrow, pd.DataFrame):
                wrow = wrow.iloc[0]
            if abs(float(wrow["wage_eur"]) - float(row.wage_eur)) > 1.0:
                raise ValueError(
                    f"registry wage for {row.player} at {row.team} "
                    f"({float(row.wage_eur):,.0f}) disagrees with the wage "
                    f"file ({float(wrow['wage_eur']):,.0f})")
            out.at[idx, "wage_eur"] = float(wrow["wage_eur"])
            # The wage row ID is the wage table's index, not a column,
            # so it is set from the registry value (already validated
            # against the wage file above), not copied from the row.
            out.at[idx, "wage_row_id"] = row.wage_row_id
            for src, dst in (
                    ("value_eur", "value_eur"),
                    ("pos_group", "pos_group"),
                    ("person_key", "person_key"),
                    ("wage_source", "wage_source"),
                    ("wage_source_sha256", "wage_source_sha256"),
                    ("wage_source_mtime", "wage_source_mtime")):
                if src in wrow.index and pd.notna(wrow[src]):
                    out.at[idx, dst] = wrow[src]
            if "club_name" in wrow.index:
                out.at[idx, "fc_club"] = wrow["club_name"]
            out.at[idx, "match_type"] = "reviewed"
            out.at[idx, "matched"] = True
            out.at[idx, "match_note"] = f"reviewed: {row.evidence}"
        else:  # reject
            out.at[idx, "match_type"] = "ambiguous"
            out.at[idx, "matched"] = False
            out.at[idx, "wage_eur"] = float("nan")
            out.at[idx, "value_eur"] = float("nan")
            out.at[idx, "match_note"] = (
                f"reviewed rejection: {row.evidence}")
    return out


def coverage_by_position(matched: pd.DataFrame,
                         positions) -> dict:
    """Coverage by position group, with positions from the event side.

    ``positions`` is a Series indexed by player_id holding one StatsBomb
    position name per player (the caller's modal position), or a
    DataFrame with player_id and position columns, in which case the mode
    is taken. Players with no recorded position land in "unknown" instead
    of inheriting the wage file's group, so the unmatched players are
    counted where they actually play. Returns per-group counts and match
    rates plus the overall rate, all plain Python types.
    """
    if "player_id" not in matched.columns or "matched" not in matched.columns:
        raise ValueError(
            "coverage_by_position needs player_id and matched columns")
    if isinstance(positions, pd.DataFrame):
        if not {"player_id", "position"}.issubset(positions.columns):
            raise ValueError(
                "coverage_by_position needs player_id and position columns")
        modal = positions.dropna(subset=["position"]).groupby("player_id")[
            "position"].agg(lambda s: s.mode().iloc[0])
    else:
        modal = positions
    frame = matched.copy()
    frame["event_pos_group"] = frame["player_id"].map(
        lambda pid: statsbomb_position_group(modal.get(pid)))
    result: dict = {"by_position": {}, "n_players": int(len(frame)),
                    "n_matched": int(frame["matched"].sum())}
    result["match_rate"] = (
        result["n_matched"] / result["n_players"]) if result["n_players"] else 0.0
    for group, sub in frame.groupby("event_pos_group"):
        n = len(sub)
        m = int(sub["matched"].sum())
        result["by_position"][str(group)] = {
            "n": int(n), "matched": m,
            "match_rate": (m / n) if n else 0.0,
        }
    return result


def _solve(ratings: pd.DataFrame, budget: float, value_col: str,
           squad_size: int | None, quota: dict | None):
    """One exact MILP re-solve, returning the selected frame."""
    from effc.portfolio.portfolio import SQUAD_SIZE, select_squad

    return select_squad(
        ratings, float(budget), value_col=value_col,
        squad_size=squad_size if squad_size is not None else SQUAD_SIZE,
        quota=quota)


def selection_sensitivity(base_ratings: pd.DataFrame,
                          alt_ratings: pd.DataFrame,
                          budgets, value_col: str = "vaep_per90",
                          squad_size: int | None = None,
                          quota: dict | None = None) -> dict:
    """How much a different rated set moves the selected squads.

    Re-solves both frames at each budget with the exact MILP selector and
    reports the Jaccard similarity of the selected player IDs, the summed
    value objective on each side, the wage bills, and the entering and
    leaving players by name. A budget that is infeasible on either side is
    reported with an error instead of being silently skipped, and the
    comparison never tunes or refits anything: both frames are scored
    exactly as given.
    """
    for name, frame in (("base", base_ratings), ("alt", alt_ratings)):
        for column in ("player_id", "player", "team", "pos_group",
                       "wage_eur", value_col):
            if column not in frame.columns:
                raise ValueError(
                    f"selection_sensitivity: {name} ratings need a "
                    f"{column} column")
    results: dict = {"budgets": {}, "value_col": value_col}
    for budget in budgets:
        budget = float(budget)
        try:
            base_sq = _solve(base_ratings, budget, value_col,
                             squad_size, quota)
            alt_sq = _solve(alt_ratings, budget, value_col,
                            squad_size, quota)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            results["budgets"][budget] = {"error": str(exc)}
            continue
        base_ids = set(base_sq["player_id"].tolist())
        alt_ids = set(alt_sq["player_id"].tolist())
        union = base_ids | alt_ids
        entering = alt_sq.loc[~alt_sq["player_id"].isin(base_ids),
                              ["player_id", "player", "team", "pos_group",
                               value_col, "wage_eur"]]
        leaving = base_sq.loc[~base_sq["player_id"].isin(alt_ids),
                              ["player_id", "player", "team", "pos_group",
                               value_col, "wage_eur"]]
        results["budgets"][budget] = {
            "jaccard": (len(base_ids & alt_ids) / len(union)) if union else 1.0,
            "base_objective": float(base_sq[value_col].sum()),
            "alt_objective": float(alt_sq[value_col].sum()),
            "objective_delta": float(
                alt_sq[value_col].sum() - base_sq[value_col].sum()),
            "base_wage_bill": float(base_sq["wage_eur"].sum()),
            "alt_wage_bill": float(alt_sq["wage_eur"].sum()),
            "n_base": int(len(base_sq)),
            "n_alt": int(len(alt_sq)),
            "entering": entering.to_dict("records"),
            "leaving": leaving.to_dict("records"),
        }
    return results


def wage_interval_sensitivity(ratings: pd.DataFrame, budgets,
                              value_col: str = "vaep_per90",
                              squad_size: int | None = None,
                              quota: dict | None = None) -> dict:
    """How much the wage uncertainty intervals move the selected squads.

    Re-solves the same rated set at the low and high wage of every
    player's match-type interval (``effc.data.wages``) and compares each
    against the point-wage squad at the same budget, using
    ``selection_sensitivity`` so the comparison is one exact re-solve per
    side. Players without an interval (no wage, or a type with no half
    width) fail closed instead of being silently dropped.
    """
    if "match_type" not in ratings.columns:
        raise ValueError("wage_interval_sensitivity needs a match_type column")
    unknown = sorted(set(ratings["match_type"].unique()) - set(WAGE_INTERVAL_HALF_WIDTH))
    if unknown:
        raise ValueError(
            f"match types without a wage interval half width: {unknown}")
    with_intervals = wage_uncertainty_intervals(ratings)
    if with_intervals[["wage_low", "wage_high"]].isna().any().any():
        bad = with_intervals.loc[
            with_intervals[["wage_low", "wage_high"]].isna().any(axis=1),
            "player"].tolist()
        raise ValueError(
            f"players without a wage interval: {bad[:5]}")
    low = with_intervals.copy()
    low["wage_eur"] = with_intervals["wage_low"]
    high = with_intervals.copy()
    high["wage_eur"] = with_intervals["wage_high"]
    return {
        "low": selection_sensitivity(ratings, low, budgets, value_col,
                                     squad_size, quota),
        "high": selection_sensitivity(ratings, high, budgets, value_col,
                                      squad_size, quota),
    }
