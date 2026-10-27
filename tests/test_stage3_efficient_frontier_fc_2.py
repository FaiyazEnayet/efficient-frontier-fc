"""Stage 3 FIXED-v2 regression tests: five-league expansion.

The contract under test: the repo covers exactly the big five European
leagues (English Premier League, La Liga, Serie A, Bundesliga,
Ligue 1); every per-league chronological split keeps train, validation,
and test match IDs pairwise disjoint; every squad obeys size, quota,
and wage constraints; all coordinate distances match an independent
105x68 m SPADL fixture within 1e-8 m; and clean fixture regeneration
yields identical selected IDs and hashes.

Before the FIXED-v2 implementation these tests fail: there is no league
table, no chronological split helper, and no per-league audit, so the
imports below fail at collection. The SPADL distance fixture is a
known-answer guard: it failed before the C1 coordinate fix and must keep
passing after it.
"""

import hashlib
import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from effc.leagues import (
    FIVE_LEAGUE_IDS,
    LEAGUES,
    audit_league,
    chronological_split,
    selection_fingerprint,
    squad_constraints_ok,
)
from effc.portfolio.portfolio import POS_QUOTA, SQUAD_SIZE, select_squad
from effc.vaep.features import _dist_angle, _to_spadl


def test_stage3_efficient_frontier_fc_v2_league_set_exactly_five():
    """The declared set is exactly the five nominated leagues."""
    names = [league["name"] for league in LEAGUES]
    assert len(LEAGUES) == 5
    assert set(names) == {
        "English Premier League",
        "La Liga",
        "Serie A",
        "Bundesliga",
        "Ligue 1",
    }
    comp_ids = {league["competition_id"] for league in LEAGUES}
    assert comp_ids == FIVE_LEAGUE_IDS == {2, 7, 9, 11, 12}
    seasons = {league["name"]: league["season_ids"] for league in LEAGUES}
    assert seasons["Bundesliga"] == [27, 281]
    for name in ("English Premier League", "La Liga", "Serie A", "Ligue 1"):
        assert seasons[name] == [27]
    for league in LEAGUES:
        assert league["season_ids"], "every league needs at least one season"


def _fake_matches():
    """Ten matches on consecutive dates, two seasons, shuffled rows."""
    rows = []
    for i in range(10):
        rows.append(
            {
                "match_id": 100 + i,
                "match_date": f"2015-08-{10 + i:02d}",
                "season_id": 27 if i < 5 else 281,
            }
        )
    return pd.DataFrame(rows).sample(frac=1.0, random_state=7)


def test_stage3_efficient_frontier_fc_v2_chronological_split_disjoint_and_ordered():
    """Train/val/test IDs are pairwise disjoint and chronologically ordered."""
    matches = _fake_matches()
    train, val, test = chronological_split(matches, train_frac=0.6, val_frac=0.2)
    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(matches["match_id"])
    by_id = matches.set_index("match_id")["match_date"]
    assert max(by_id[i] for i in train) <= min(by_id[i] for i in val)
    assert max(by_id[i] for i in val) <= min(by_id[i] for i in test)


def test_stage3_efficient_frontier_fc_v2_spadl_distances_independent_fixture():
    """Converted distances match an independent 105x68 scaling within 1e-8."""
    raw_points = [(120.0, 40.0), (108.0, 40.0), (0.0, 0.0), (60.0, 40.0)]
    for x, y in raw_points:
        gx, gy = _to_spadl(x, y)
        ix, iy = x * 105.0 / 120.0, y * 68.0 / 80.0
        assert abs(gx - ix) <= 1e-8
        assert abs(gy - iy) <= 1e-8
    gx, gy = _to_spadl(120.0, 40.0)
    assert (gx, gy) == pytest.approx((105.0, 34.0), abs=1e-8)
    sx, sy = _to_spadl(108.0, 40.0)
    dist, _angle = _dist_angle(sx, sy)
    assert dist == pytest.approx(10.5, abs=1e-8)


def _fake_ratings():
    """Plenty of cheap players in every position group."""
    rng = np.random.default_rng(0)
    rows = []
    pid = 0
    for group, count in (("GK", 8), ("DEF", 14), ("MID", 14), ("FWD", 10)):
        for _ in range(count):
            pid += 1
            rows.append(
                {
                    "player_id": pid,
                    "player": f"P{pid}",
                    "team": "T",
                    "pos_group": group,
                    "vaep_per90": float(rng.uniform(0.0, 0.5)),
                    "wage_eur": 10000.0,
                }
            )
    return pd.DataFrame(rows)


def test_stage3_efficient_frontier_fc_v2_squad_constraints():
    """Every selected squad obeys size, quota, and wage constraints."""
    ratings = _fake_ratings()
    budget = 500000.0
    squad = select_squad(ratings, budget)
    assert squad_constraints_ok(squad, budget)
    assert len(squad) == SQUAD_SIZE
    counts = squad["pos_group"].value_counts().to_dict()
    for group, minimum in POS_QUOTA.items():
        assert counts.get(group, 0) >= minimum
    assert float(squad["wage_eur"].sum()) <= budget


def test_stage3_efficient_frontier_fc_v2_selection_regeneration_identical():
    """Clean regeneration yields identical selected IDs and hashes."""
    ratings = _fake_ratings()
    first = select_squad(ratings, 500000.0)
    second = select_squad(ratings, 500000.0)
    assert sorted(first["player_id"]) == sorted(second["player_id"])
    assert selection_fingerprint(first) == selection_fingerprint(second)
    assert selection_fingerprint(first) == hashlib.sha256(
        ",".join(map(str, sorted(first["player_id"]))).encode()
    ).hexdigest()


def test_stage3_efficient_frontier_fc_v2_audit_counts(tmp_path):
    """The audit counts matches, events, and lineups present per league."""
    data_dir = tmp_path / "data"
    (data_dir / "events").mkdir(parents=True)
    (data_dir / "lineups").mkdir(parents=True)
    matches = [
        {"match_id": 1, "match_date": "2015-08-10"},
        {"match_id": 2, "match_date": "2015-08-11"},
        {"match_id": 3, "match_date": "2015-08-12"},
    ]
    with open(data_dir / "matches_2_27.json", "w") as f:
        json.dump(matches, f)
    for mid in (1, 2):
        (data_dir / "events" / f"{mid}.json").write_text("[]")
    (data_dir / "lineups" / "1.json").write_text("[]")
    league = {"name": "English Premier League", "competition_id": 2, "season_ids": [27]}
    audit = audit_league(league, str(data_dir))
    assert audit["matches"] == 3
    assert audit["events_present"] == 2
    assert audit["lineups_present"] == 1
    assert audit["events_missing"] == 1
    assert audit["lineups_missing"] == 2
    assert audit["match_ids"] == [1, 2, 3]
