"""Per-league baseline ratings for the live refresh.

One versioned baseline artifact per top-five league, each snapshotted
from the fitted five-league run (config fc6_pool, FEATURE_VERSION 4,
seed 0) except the Bundesliga artifact, which is the original
bundesliga_2seasons snapshot and stays byte-identical. The refresh
selects the baseline matching the fixture league; a league with no
baseline falls back to the default artifact, so its players are
watchlisted exactly as before.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from effc import live

EXPECTED_COUNTS = {
    39: 179,   # English Premier League
    140: 146,  # La Liga
    78: 268,   # Bundesliga
    135: 143,  # Serie A
    61: 145,   # Ligue 1
}

LEAGUE_NAMES = {
    39: "English Premier League",
    140: "La Liga",
    78: "Bundesliga",
    135: "Serie A",
    61: "Ligue 1",
}


def test_league_baseline_table_covers_top_five():
    assert set(live.LEAGUE_BASELINES) == set(EXPECTED_COUNTS)
    for league_id, filename in live.LEAGUE_BASELINES.items():
        path = live.baseline_path_for_league(league_id)
        assert path is not None and os.path.exists(path)
        assert os.path.basename(path) == filename


def test_baseline_path_unknown_league_is_none():
    assert live.baseline_path_for_league(999) is None
    assert live.baseline_path_for_league(None) is None


@pytest.mark.parametrize("league_id,count", sorted(EXPECTED_COUNTS.items()))
def test_each_league_baseline_loads_and_verifies(league_id, count):
    path = live.baseline_path_for_league(league_id)
    artifact = live.load_baseline_artifact(path)
    assert artifact["version"] == live.BASELINE_VERSION
    assert artifact["feature_version"] == 4
    assert artifact["seed"] == 0
    assert artifact["model"] and artifact["data_snapshot"]
    assert len(artifact["ratings"]) == count
    ratings = live.load_baseline_ratings(league=league_id)
    assert len(ratings) == count
    assert all(isinstance(pid, str) for pid in ratings["player_id"])
    assert ratings["player_id"].is_unique
    if league_id != 78:
        assert artifact["league_id"] == league_id
        assert artifact["league"] == LEAGUE_NAMES[league_id]
        assert artifact["model_hash"] == "b86caaed47f745d4"


def test_league_argument_accepts_string_ids():
    ratings = live.load_baseline_ratings(league="39")
    assert len(ratings) == EXPECTED_COUNTS[39]


def test_bundesliga_league_baseline_is_the_default_artifact():
    by_league = live.load_baseline_artifact(live.baseline_path_for_league(78))
    default = live.load_baseline_artifact()
    assert by_league["ratings_sha256"] == default["ratings_sha256"]
    assert by_league["ratings"] == default["ratings"]


def test_unknown_league_falls_back_to_default_baseline():
    ratings = live.load_baseline_ratings(league=999)
    default = live.load_baseline_ratings()
    pd.testing.assert_frame_equal(ratings, default)


def test_league_ratings_match_published_five_league_squads():
    """Every squad row published in runs/five_leagues/results.json is
    in that league's baseline with the same rating, wage and group."""
    results_path = os.path.join(
        os.path.dirname(live.BASELINE_PATH), "..", "five_leagues", "results.json")
    with open(results_path) as f:
        results = json.load(f)
    name_to_id = {v: k for k, v in LEAGUE_NAMES.items()}
    for league_name, entry in results["leagues"].items():
        league_id = name_to_id[league_name]
        if league_id == 78:
            continue  # Bundesliga baseline is the 2-seasons snapshot
        ratings = live.load_baseline_ratings(league=league_id)
        by_player_team = {
            (r.player, r.team): r for r in ratings.itertuples()
        }
        assert len(ratings) == entry["n_rated"]
        for budget in entry["squads"].values():
            for row in budget.get("squad", []):
                got = by_player_team[(row["player"], row["team"])]
                assert got.vaep_per90 == pytest.approx(row["vaep_per90"])
                assert got.wage_eur == pytest.approx(row["wage_eur"])
                assert got.pos_group == row["pos_group"]


@pytest.mark.parametrize("league_id", sorted(EXPECTED_COUNTS))
def test_no_live_identity_holds_for_each_league(league_id):
    ratings = live.load_baseline_ratings(league=league_id)
    out = live.blend_ratings(ratings, {"live_minutes": {}, "live_contrib": {}})
    pd.testing.assert_series_equal(
        out["vaep_per90_live"], ratings["vaep_per90"], check_names=False)


def test_refresh_base_ratings_select_league():
    import effc.live_refresh as lr

    pl = lr.base_ratings(league=39)
    assert len(pl) == EXPECTED_COUNTS[39]
    default = lr.base_ratings()
    assert len(default) == EXPECTED_COUNTS[78]


def _bundle_payload(kind, fixture_id):
    path = os.path.join(
        os.path.dirname(live.BASELINE_PATH), "..", "..", "tests", "fixtures",
        "api_football", f"{kind}_{fixture_id}.json")
    with open(os.path.normpath(path)) as f:
        data = json.load(f)
    return data["response"] if isinstance(data, dict) else data


def test_premier_league_bundle_replay_against_pl_baseline(monkeypatch):
    """Replay of the captured 2026-11-07 Premier League bundle.

    Observed counts, recorded as they came out: the Premier League
    baseline is a 2015/16 ratings set, so none of the 93 players who
    played in these three 2026 fixtures has a baseline entry for his
    team and every one is watchlisted; no rating moves. The only two
    bundle players who are in the baseline at all (Ben Davies and
    Jack Butland) were unused substitutes, so they are skipped as
    unused, not counted. The fourth finished Premier League fixture
    that day (1557461) has no events or lineups in the bundle and is
    not part of this replay.
    """
    import effc.live_refresh as lr

    ratings = lr.base_ratings(league=39)
    current = {}
    monkeypatch.setattr(lr, "fetch_fixture_events",
                        lambda fid: current["events"])
    monkeypatch.setattr(lr, "fetch_fixture_lineups",
                        lambda fid: current["lineups"])

    def counted(fn, *a, **k):
        return fn(*a, **k)

    state = live.normalize_state({})
    seen = {}
    for fixture_id in (1557457, 1557463, 1557465):
        fixture = {"fixture": {"id": fixture_id,
                               "date": "2026-11-07T12:30:00+00:00"}}
        current["events"] = _bundle_payload("events", fixture_id)
        current["lineups"] = _bundle_payload("lineups", fixture_id)
        entry = lr._process_fixture(state, fixture, ratings, counted)
        state["processed_fixtures"][entry["fixture_id"]] = entry
        seen[fixture_id] = entry
    assert seen[1557457]["matched"] == 0
    assert seen[1557457]["unmatched"] == 32
    assert seen[1557463]["matched"] == 0
    assert seen[1557463]["unmatched"] == 32
    assert seen[1557465]["matched"] == 0
    assert seen[1557465]["unmatched"] == 29
    assert all(e["status"] == "ok" for e in seen.values())
    assert len(state["watchlist"]) == 93
    out = live.blend_ratings(ratings, state)
    pd.testing.assert_series_equal(
        out["vaep_per90_live"], ratings["vaep_per90"], check_names=False)
