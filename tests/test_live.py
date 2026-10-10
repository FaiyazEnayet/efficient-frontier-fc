"""Tests for the live data layer. No network: API calls are stubbed."""

import pandas as pd

from effc import live
from effc.live import (
    blend_ratings,
    fixture_contributions,
    match_player,
    rebalance_report,
)


def _ratings(n_per_group: int = 8) -> pd.DataFrame:
    rows = []
    pid = 0
    for g in ["GK", "DEF", "MID", "FWD"]:
        for i in range(n_per_group):
            rows.append({
                "player_id": f"p{pid}",
                "player": f"Test Player {pid}",
                "team": "Test FC",
                "minutes": 2000.0,
                "vaep": 5.0,
                "vaep_per90": 0.225,
                "pos_group": g,
                "wage_eur": 50_000.0,
            })
            pid += 1
    return pd.DataFrame(rows)


def test_fixture_contributions_weights():
    lineups = [{
        "team": {"name": "Test FC"},
        "startXI": [{"player": {"name": "Scorer Sam"}}],
        "substitutes": [{"player": {"name": "Sub Sid"}}],
    }]
    events = [
        {"type": "Goal", "detail": "Normal Goal", "time": {"elapsed": 23},
         "team": {"name": "Test FC"},
         "player": {"name": "Scorer Sam"}, "assist": {"name": "Assist Ann"}},
        {"type": "Card", "detail": "Yellow Card", "time": {"elapsed": 55},
         "team": {"name": "Test FC"},
         "player": {"name": "Scorer Sam"}, "assist": {"name": None}},
        {"type": "subst", "detail": "Substitution", "time": {"elapsed": 70},
         "team": {"name": "Test FC"},
         "player": {"name": "Scorer Sam"}, "assist": {"name": "Sub Sid"}},
    ]
    df = fixture_contributions(events, lineups)
    row = df.set_index("api_player").loc
    assert row["Scorer Sam"]["contrib"] == live.W_GOAL + live.W_YELLOW
    assert row["Scorer Sam"]["minutes"] == 70.0
    assert row["Sub Sid"]["contrib"] == 0.0
    assert row["Sub Sid"]["minutes"] == 20.0
    # assist credit lands on the assister even though she never appears in lineups
    assert ("Test FC", "Assist Ann") not in {
        (r["api_team"], r["api_player"]) for _, r in df.iterrows()}


def test_fixture_contributions_red_card():
    lineups = [{"team": {"name": "Test FC"},
                "startXI": [{"player": {"name": "Tough Tom"}}], "substitutes": []}]
    events = [{"type": "Card", "detail": "Red Card", "time": {"elapsed": 80},
               "team": {"name": "Test FC"},
               "player": {"name": "Tough Tom"}, "assist": {"name": None}}]
    df = fixture_contributions(events, lineups)
    assert df.iloc[0]["contrib"] == live.W_RED


def test_match_player_alias_and_fuzzy():
    ratings = _ratings(n_per_group=2)
    ratings.loc[0, "player"] = "Florian Wirtz"
    ratings.loc[0, "team"] = "Bayer Leverkusen"
    aliases: dict = {}
    # fuzzy match on a slightly different spelling
    pid = match_player("Florian Writz", "Bayer 04 Leverkusen", ratings, aliases)
    assert pid == "p0"
    assert aliases  # alias was persisted
    # second call resolves from the persisted map
    assert match_player("Florian Writz", "Bayer 04 Leverkusen", ratings, aliases) == "p0"
    # no match for a stranger
    assert match_player("Zzz Unknown", "Nowhere Rovers", ratings, {}) is None


def test_blend_ratings_no_live_is_stable():
    ratings = _ratings()
    out = blend_ratings(ratings, {"live_minutes": {}, "live_contrib": {}})
    # with no live data the blended rating equals the base rating
    pd.testing.assert_series_equal(out["vaep_per90_live"], out["vaep_per90"],
                                   check_names=False)


def test_blend_ratings_live_moves_needle():
    ratings = _ratings()
    pid = "p0"
    state = {"live_minutes": {pid: 900.0},
             "live_contrib": {pid: 900.0 / 90.0 * 1.5}}  # 1.5 proxy per90
    out = blend_ratings(ratings, state).set_index("player_id")
    assert out.loc[pid, "vaep_per90_live"] > out.loc[pid, "vaep_per90"]
    # a hot streak does not explode past the live observation
    assert out.loc[pid, "vaep_per90_live"] < 1.5


def test_rebalance_report():
    ratings = _ratings().set_index("player_id")
    ratings["vaep_per90_live"] = ratings["vaep_per90"]
    old = pd.DataFrame([{"player_id": "p0"}, {"player_id": "p1"}])
    new = pd.DataFrame([{"player_id": "p1"}, {"player_id": "p2"}])
    rep = rebalance_report(old, new, ratings.reset_index())
    assert set(rep["action"]) == {"OUT", "IN"}
    assert rep[rep["action"] == "OUT"]["player"].tolist() == ["Test Player 0"]
    assert rep[rep["action"] == "IN"]["player"].tolist() == ["Test Player 2"]


def test_live_refresh_dry_run(monkeypatch, capsys):
    import effc.live_refresh as lr

    fixtures = [
        {"fixture": {"id": 101, "date": "2026-10-09T18:30:00+00:00",
                     "status": {"short": "FT"}},
         "teams": {"home": {"name": "A"}, "away": {"name": "B"}},
         "goals": {"home": 2, "away": 1}},
        {"fixture": {"id": 102, "date": "2026-10-09T20:30:00+00:00",
                     "status": {"short": "FT"}},
         "teams": {"home": {"name": "C"}, "away": {"name": "D"}},
         "goals": {"home": 0, "away": 0}},
    ]
    monkeypatch.setattr(lr, "fetch_finished_fixtures", lambda *a, **k: fixtures)
    monkeypatch.setattr(lr, "load_state", lambda: {"last_fixture_id": 100})
    monkeypatch.setattr(lr, "base_ratings", lambda *a, **k: _ratings())
    monkeypatch.setattr("sys.argv", ["live_refresh", "--dry-run", "--days", "2"])

    lr.main()
    out = capsys.readouterr().out
    assert "2 new since last refresh" in out
    assert "Bayer" not in out  # no network, no optimizer run
