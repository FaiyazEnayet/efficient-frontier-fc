"""Player ratings: VAEP/90 with empirical-Bayes shrinkage.

Raw per-90 numbers lie for small samples: a striker who scores twice in
90 minutes is not a 2.0 VAEP/90 player. Shrinkage pulls everyone toward
their position-group mean, with the pull weakening as minutes accumulate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_ratings(
    player_match: pd.DataFrame,
    minutes: pd.DataFrame,
    wage_map: pd.DataFrame,
    min_minutes: float = 270.0,
    shrink_k: float = 900.0,
) -> pd.DataFrame:
    """One row per player: shrunk VAEP/90, wages, position group.

    wage_map: output of effc.data.wages.match_players aggregated to one row
    per player (columns player, team, wage_eur, value_eur, pos_group, matched).
    Every retained row must carry a wage and a wage source: a rated
    player without a traceable wage source fails closed instead of
    entering the optimizer. Eligibility is unchanged (matched players
    with enough minutes), but the drop is now measured: the returned
    frame carries the wage join audit in ``attrs["wage_join_audit"]``
    with the counts dropped as unmatched and dropped for minutes.
    """
    pm = player_match.merge(
        minutes[["match_id", "player_id", "minutes", "position"]],
        on=["match_id", "player_id"],
        how="left",
    )
    pm["minutes"] = pm["minutes"].fillna(0.0)

    tot = pm.groupby("player_id", dropna=False).agg(
        vaep=("vaep", "sum"),
        minutes=("minutes", "sum"),
        matches=("match_id", "nunique"),
        player=("player", lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]),
        team=("team", lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]),
    ).reset_index()
    tot["vaep_per90_raw"] = tot["vaep"] / tot["minutes"].clip(lower=1e-9) * 90.0

    if wage_map.duplicated(subset=["player", "team"]).any():
        dupes = wage_map.loc[
            wage_map.duplicated(subset=["player", "team"], keep=False),
            ["player", "team"]].drop_duplicates()
        raise ValueError(
            "duplicate wage_map rows for the same player at the same team: "
            f"{dupes.head().to_dict('records')} (one row per player per "
            f"team is required, or the merge multiplies ratings)")
    wage_cols = ["player", "team", "wage_eur", "value_eur", "pos_group", "matched"]
    for optional in ("match_type", "wage_source", "wage_row_id",
                     "wage_source_sha256", "wage_source_mtime"):
        if optional in wage_map.columns:
            wage_cols.append(optional)
    tot = tot.merge(
        wage_map[wage_cols],
        on=["player", "team"],
        how="left",
    )
    n_players = len(tot)
    by_type = (
        tot["match_type"].fillna("unmatched").value_counts().to_dict()
        if "match_type" in tot.columns else {}
    )
    n_dropped_unmatched = int((tot["matched"] != True).sum())  # noqa: E712
    matched_only = tot[tot["matched"] == True].copy()  # noqa: E712
    n_dropped_below_minutes = int(
        (matched_only["minutes"] < min_minutes).sum())
    tot = matched_only
    tot = tot[tot["minutes"] >= min_minutes].copy()
    tot["pos_group"] = tot["pos_group"].fillna("MID")
    # Fail closed: no rated player enters the optimizer without a wage
    # and a recorded wage source.
    if tot["wage_eur"].isna().any():
        missing = tot.loc[tot["wage_eur"].isna(), "player"].tolist()
        raise ValueError(
            f"rated players without a wage: {missing[:5]}")
    if "wage_source" in tot.columns:
        no_source = tot["wage_source"].isna() | (
            tot["wage_source"].astype(str).str.strip() == "")
        if no_source.any():
            missing = tot.loc[no_source, "player"].tolist()
            raise ValueError(
                f"rated players without a wage source: {missing[:5]} "
                f"(every optimizer row needs one traceable wage source)")

    # empirical-Bayes shrinkage toward position-group means
    group_mean = tot.groupby("pos_group")["vaep_per90_raw"].transform("mean")
    w = tot["minutes"] / (tot["minutes"] + shrink_k)
    tot["vaep_per90"] = w * tot["vaep_per90_raw"] + (1 - w) * group_mean
    tot["shrink_w"] = w
    out = tot.reset_index(drop=True)
    out.attrs["wage_join_audit"] = {
        "n_players": int(n_players),
        "n_dropped_unmatched": n_dropped_unmatched,
        "n_dropped_below_minutes": n_dropped_below_minutes,
        "n_retained": int(len(out)),
        "match_type_counts": {str(k): int(v) for k, v in by_type.items()},
    }
    return out
