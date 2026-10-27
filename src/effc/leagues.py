"""Five-league expansion: league table, splits, audit, squad checks.

The nominated set is the big five European leagues. StatsBomb
competition IDs: Premier League 2, Ligue 1 7, Bundesliga 9,
La Liga 11, Serie A 12. Seasons are StatsBomb season IDs; only
Bundesliga has two seasons in the open data used here, and both are
partial (34 matches each). The other four leagues use their full
2015/2016 seasons.

Coordinate note: all event locations reach this module already
converted to the 105x68 m SPADL frame by
``effc.vaep.features.featurize_match`` (FEATURE_VERSION 3); nothing
here rescales coordinates.
"""

from __future__ import annotations

import hashlib
import json
import os

import pandas as pd

from effc.portfolio.portfolio import POS_QUOTA, SQUAD_SIZE
from effc.vaep import calibration as cal

LEAGUES = [
    {
        "name": "English Premier League",
        "competition_id": 2,
        "season_ids": [27],
    },
    {
        "name": "La Liga",
        "competition_id": 11,
        "season_ids": [27],
    },
    {
        "name": "Serie A",
        "competition_id": 12,
        "season_ids": [27],
    },
    {
        "name": "Bundesliga",
        "competition_id": 9,
        "season_ids": [27, 281],
    },
    {
        "name": "Ligue 1",
        "competition_id": 7,
        "season_ids": [27],
    },
]

FIVE_LEAGUE_IDS = {league["competition_id"] for league in LEAGUES}

# Backtest budgets shared with the v1.5.0 run (EUR weekly wages).
BACKTEST_BUDGETS = [300000.0, 500000.0, 700000.0]


def league_pools() -> dict[str, list[tuple[int, int]]]:
    """League name to its (competition_id, season_id) pairs."""
    return {
        league["name"]: [
            (league["competition_id"], season_id)
            for season_id in league["season_ids"]
        ]
        for league in LEAGUES
    }


def chronological_split(
    matches: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2
) -> tuple[set[int], set[int], set[int]]:
    """Chronological train/validation/test split of a league's matches.

    Matches are ordered by (match_date, match_id); the earliest
    ``train_frac`` become training, the next ``val_frac`` become
    validation, and the rest become the test set. The three ID sets are
    pairwise disjoint by construction, asserted before returning.
    Validation is reported only, never used to tune the test set.
    """
    ordered = matches.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    ids = [int(m) for m in ordered["match_id"].tolist()]
    n_train = int(len(ids) * train_frac)
    n_val = int(len(ids) * val_frac)
    train = set(ids[:n_train])
    val = set(ids[n_train : n_train + n_val])
    test = set(ids[n_train + n_val :])
    cal.assert_disjoint(train, val, "train/validation match")
    cal.assert_disjoint(train, test, "train/test match")
    cal.assert_disjoint(val, test, "validation/test match")
    return train, val, test


def audit_league(league: dict, data_dir: str) -> dict:
    """Count a league's available matches, events, and lineups.

    Reads the StatsBomb match lists for the league's seasons from
    ``data_dir`` and counts how many event and lineup files are present
    on disk. A missing match list counts as zero matches for that
    season; nothing is downloaded or generated here.
    """
    match_ids: list[int] = []
    for season_id in league["season_ids"]:
        path = os.path.join(
            data_dir, f"matches_{league['competition_id']}_{season_id}.json"
        )
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for m in json.load(f):
                match_ids.append(int(m["match_id"]))
    match_ids = sorted(set(match_ids))
    events_present = sum(
        os.path.exists(os.path.join(data_dir, "events", f"{mid}.json"))
        for mid in match_ids
    )
    lineups_present = sum(
        os.path.exists(os.path.join(data_dir, "lineups", f"{mid}.json"))
        for mid in match_ids
    )
    return {
        "league": league["name"],
        "matches": len(match_ids),
        "match_ids": match_ids,
        "events_present": events_present,
        "events_missing": len(match_ids) - events_present,
        "lineups_present": lineups_present,
        "lineups_missing": len(match_ids) - lineups_present,
    }


def squad_constraints_ok(selection: pd.DataFrame, budget: float) -> bool:
    """True when a squad obeys size, quota, and wage constraints."""
    if len(selection) != SQUAD_SIZE:
        return False
    counts = selection["pos_group"].value_counts().to_dict()
    for group, minimum in POS_QUOTA.items():
        if counts.get(group, 0) < minimum:
            return False
    return float(selection["wage_eur"].sum()) <= float(budget) + 1e-6


def selection_fingerprint(selection: pd.DataFrame) -> str:
    """Stable hash of a squad's sorted player IDs."""
    ids = ",".join(map(str, sorted(int(p) for p in selection["player_id"])))
    return hashlib.sha256(ids.encode()).hexdigest()
