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
    The match_type column records how each row matched: "exact", "fuzzy"
    (the fallback), or "unmatched". Unmatched rows carry NaN wages; use
    assign_proxy_wages to give them an explicit, flagged proxy wage.
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
    sb["match_type"] = "unmatched"
    sb.loc[sb["wage_eur"].notna(), "match_type"] = "exact"

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
            sb.at[i, "match_type"] = "fuzzy"
    sb["matched"] = sb["wage_eur"].notna()
    return sb.drop(columns=["key", "club_key", "last", "initial"])


# Half-widths of the wage uncertainty interval by match type, as a
# fraction of the point wage. Exact matches are only exposed to
# reporting rounding; fuzzy matches may be the wrong player; proxy
# wages are a group median standing in for an unknown wage.
WAGE_INTERVAL_HALF_WIDTH = {"exact": 0.05, "fuzzy": 0.25, "proxy": 0.50}


def summarize_wage_matches(matched: pd.DataFrame) -> dict:
    """Counts by match_type: exact, fuzzy, proxy, unmatched."""
    counts = matched["match_type"].value_counts().to_dict()
    return {
        kind: int(counts.get(kind, 0))
        for kind in ("exact", "fuzzy", "proxy", "unmatched")
    }


def assign_proxy_wages(matched: pd.DataFrame) -> pd.DataFrame:
    """Fill unmatched rows with an explicit proxy wage, flagged as such.

    The proxy is the median wage of matched players in the same
    position group, falling back to the overall matched median when
    the player has no position group or the group has no matched wage.
    Proxy rows are marked match_type "proxy" and matched stays False:
    a proxy is a labeled stand-in, not a match.
    """
    out = matched.copy()
    if "match_type" not in out.columns:
        out["match_type"] = "unmatched"
    group_median = out.loc[out["matched"], :].groupby("pos_group")["wage_eur"].median()
    overall_median = float(out.loc[out["matched"], "wage_eur"].median())
    for i in out.loc[~out["matched"]].index:
        group = out.at[i, "pos_group"]
        fill = group_median.get(group, float("nan"))
        if not pd.notna(fill):
            fill = overall_median
        out.at[i, "wage_eur"] = fill
        out.at[i, "match_type"] = "proxy"
    return out


def wage_uncertainty_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Add wage_low / wage_high columns from the match type.

    Unmatched rows (no wage, no proxy assigned) get NaN intervals:
    call assign_proxy_wages first if an interval is needed for them.
    """
    out = df.copy()
    if "match_type" not in out.columns:
        raise ValueError("wage_uncertainty_intervals needs a match_type column")
    half = out["match_type"].map(WAGE_INTERVAL_HALF_WIDTH)
    out["wage_low"] = out["wage_eur"] * (1.0 - half)
    out["wage_high"] = out["wage_eur"] * (1.0 + half)
    return out
