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

    wage_cols = ["player", "team", "wage_eur", "value_eur", "pos_group", "matched"]
    if "match_type" in wage_map.columns:
        wage_cols.append("match_type")
    tot = tot.merge(
        wage_map[wage_cols],
        on=["player", "team"],
        how="left",
    )
    tot = tot[tot["matched"] == True].copy()  # noqa: E712
    tot = tot[tot["minutes"] >= min_minutes].copy()
    tot["pos_group"] = tot["pos_group"].fillna("MID")

    # empirical-Bayes shrinkage toward position-group means
    group_mean = tot.groupby("pos_group")["vaep_per90_raw"].transform("mean")
    w = tot["minutes"] / (tot["minutes"] + shrink_k)
    tot["vaep_per90"] = w * tot["vaep_per90_raw"] + (1 - w) * group_mean
    tot["shrink_w"] = w
    return tot.reset_index(drop=True)
