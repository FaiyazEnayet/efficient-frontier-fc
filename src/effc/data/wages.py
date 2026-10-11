"""EA FC 24 wages joined to StatsBomb players by normalized name matching."""

from __future__ import annotations

import os
import unicodedata

import pandas as pd

from effc.data.download import DATA_DIR

POSITION_GROUPS = {
    "GK": "GK",
    "RB": "DEF",
    "RWB": "DEF",
    "CB": "DEF",
    "LB": "DEF",
    "LWB": "DEF",
    "CDM": "MID",
    "CM": "MID",
    "CAM": "MID",
    "RM": "MID",
    "LM": "MID",
    "RW": "FWD",
    "LW": "FWD",
    "CF": "FWD",
    "ST": "FWD",
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = "".join(c.lower() if c.isalnum() or c == " " else " " for c in s)
    return " ".join(s.split())


def position_group(positions: str) -> str:
    """First listed FC position mapped to GK/DEF/MID/FWD."""
    first = str(positions).split(",")[0].strip().upper()
    return POSITION_GROUPS.get(first, "MID")


def load_wages() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "fc24_wages.csv"), usecols=[
        "long_name", "short_name", "club_name", "positions", "age", "overall",
        "value_eur", "wage_eur",
    ])
    df = df.dropna(subset=["long_name", "wage_eur"])
    df["key"] = df["long_name"].map(norm)
    df["club_key"] = df["club_name"].map(norm)
    df["pos_group"] = df["positions"].map(position_group)
    # keep the highest-rated row per name (dedupes across roster updates)
    df = df.sort_values("overall", ascending=False).drop_duplicates("key")
    return df.reset_index(drop=True)


def match_players(sb_players: pd.DataFrame, wages: pd.DataFrame) -> pd.DataFrame:
    """Left-join StatsBomb players to wages. Returns sb_players + wage columns.

    sb_players needs columns: player, team. Matching is exact on normalized
    full name, falling back to last-name + first-initial within the same club.
    """
    w = wages.copy()
    sb = sb_players.copy()
    sb["key"] = sb["player"].map(norm)
    sb["club_key"] = sb["team"].map(norm)

    exact = w.set_index("key")
    sb["wage_eur"] = sb["key"].map(exact["wage_eur"])
    sb["value_eur"] = sb["key"].map(exact["value_eur"])
    sb["pos_group"] = sb["key"].map(exact["pos_group"])
    sb["fc_club"] = sb["key"].map(exact["club_name"])

    missing = sb["wage_eur"].isna()
    # fallback: same club + last name + first initial
    w["last"] = w["key"].str.split().str[-1]
    w["initial"] = w["key"].str.split().str[0].str[0]
    sb["last"] = sb["key"].str.split().str[-1]
    sb["initial"] = sb["key"].str.split().str[0].str[0]
    widx = w.set_index(["club_key", "last", "initial"]).sort_index()
    for i in sb[missing].index:
        key = (sb.at[i, "club_key"], sb.at[i, "last"], sb.at[i, "initial"])
        if key in widx.index:
            row = widx.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            sb.at[i, "wage_eur"] = row["wage_eur"]
            sb.at[i, "value_eur"] = row["value_eur"]
            sb.at[i, "pos_group"] = row["pos_group"]
            sb.at[i, "fc_club"] = row["club_name"]
    sb["matched"] = sb["wage_eur"].notna()
    return sb.drop(columns=["key", "club_key", "last", "initial"])
