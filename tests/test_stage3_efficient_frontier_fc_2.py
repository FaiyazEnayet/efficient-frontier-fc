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


# ---------------------------------------------------------------------------
# Stage 3 C2: live-refresh defect and state semantics.
#
# Each test below targets one verified defect: the base-ratings rebuild
# crash, the double shrinkage in the blend, inconsistent state ID types,
# and the fixture-ID watermark that skipped out-of-order fixtures. The
# real provider bundle under tests/fixtures/api_football/ drives the
# full-refresh and replay tests; no network is used.
# ---------------------------------------------------------------------------

from effc import live  # noqa: E402
from effc.live import blend_ratings, fixture_contributions  # noqa: E402

_C2_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "api_football")
_C2_BUNDLE_IDS = [1557463, 1557457, 1557465]


def _c2_ratings(n_per_group: int = 8, numeric_ids: bool = False) -> pd.DataFrame:
    rows = []
    pid = 0
    for g in ["GK", "DEF", "MID", "FWD"]:
        for _ in range(n_per_group):
            rows.append({
                "player_id": pid if numeric_ids else f"p{pid}",
                "player": f"Test Player {pid}",
                "team": "Test FC",
                "minutes": 2000.0,
                "vaep": 5.0,
                # heterogeneous values inside each group: a uniform frame
                # hides the double-shrinkage defect
                "vaep_per90": 0.05 + 0.03 * pid,
                "pos_group": g,
                "wage_eur": 50_000.0,
            })
            pid += 1
    return pd.DataFrame(rows)


def _c2_bundle_fixtures():
    with open(os.path.join(_C2_FIXTURES, "fixtures_date_2026-11-07.json")) as f:
        payload = json.load(f)
    return [f for f in payload["response"]
            if f["league"]["id"] == 39
            and f["fixture"]["status"]["short"] == "FT"]


def _c2_bundle_payload(kind, fixture_id):
    with open(os.path.join(_C2_FIXTURES, f"{kind}_{fixture_id}.json")) as f:
        return json.load(f)["response"]


def test_stage3_efficient_frontier_fc_2_base_ratings_rebuild_supplies_models(monkeypatch):
    import effc.live_refresh as lr

    seen = {}

    def fake_compute(models, train_match_ids, config=None):
        seen["models"] = models
        seen["train_ids"] = train_match_ids
        pm = pd.DataFrame([{"player": "A", "team": "T", "match_id": 1,
                            "player_id": "p0", "vaep": 1.0}])
        mins = pd.DataFrame([{"match_id": 1, "player_id": "p0",
                              "minutes": 90.0, "position": "MID"}])
        return pm, mins, pd.DataFrame()

    monkeypatch.setattr(lr, "compute_target_vaep", fake_compute)
    monkeypatch.setattr(lr, "prepare_model_data",
                        lambda cfg: {"models": {"m": 1}, "train_ids": [10, 20]})
    monkeypatch.setattr(lr, "load_wages", lambda: pd.DataFrame())
    monkeypatch.setattr(lr, "match_players",
                        lambda sb, w: pd.DataFrame(
                            [{"player": "A", "team": "T", "wage_eur": 1.0,
                              "value_eur": 1.0, "pos_group": "MID",
                              "matched": True}]))
    out = lr.base_ratings(rebuild=True, min_minutes=0)
    assert seen["models"] == {"m": 1}
    assert seen["train_ids"] == [10, 20]
    assert len(out) == 1


def test_stage3_efficient_frontier_fc_2_base_ratings_loads_versioned_baseline():
    import effc.live_refresh as lr

    out = lr.base_ratings()
    assert len(out) > 100
    assert all(isinstance(pid, str) for pid in out["player_id"])
    meta = live.load_baseline_artifact()
    assert meta["version"] == live.BASELINE_VERSION
    assert meta["ratings_sha256"]
    assert meta["model"] and meta["data_snapshot"]


def test_stage3_efficient_frontier_fc_2_baseline_artifact_tamper_fails(tmp_path):
    src = live.load_baseline_artifact()
    path = tmp_path / "baseline.json"
    payload = dict(src)
    payload["ratings"] = [dict(r) for r in src["ratings"]]
    payload["ratings"][0]["vaep_per90"] = 99.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        live.load_baseline_ratings(str(path))


def test_stage3_efficient_frontier_fc_2_blend_no_live_identity_heterogeneous():
    ratings = _c2_ratings()
    out = blend_ratings(ratings, {"live_minutes": {}, "live_contrib": {}})
    pd.testing.assert_series_equal(out["vaep_per90_live"], ratings["vaep_per90"],
                                   check_names=False)
    pd.testing.assert_series_equal(out["event_proxy_blended_per90"],
                                   ratings["vaep_per90"], check_names=False)


def test_stage3_efficient_frontier_fc_2_blend_numeric_ids_match_string_state():
    ratings = _c2_ratings(numeric_ids=True)
    first = ratings.iloc[0]
    state = {"live_minutes": {str(first["player_id"]): 900.0},
             "live_contrib": {str(first["player_id"]): 15.0}}
    out = blend_ratings(ratings, state).set_index("player_id")
    assert out.loc[str(first["player_id"]), "vaep_per90_live"] > first["vaep_per90"]
    other = ratings.iloc[5]
    assert out.loc[str(other["player_id"]), "vaep_per90_live"] == other["vaep_per90"]


def test_stage3_efficient_frontier_fc_2_event_proxy_naming_and_time_decay():
    assert live.decay_factor(0.0) == 1.0
    assert live.decay_factor(live.EVENT_PROXY_HALF_LIFE_DAYS) == pytest.approx(0.5)
    assert live.decay_factor(2 * live.EVENT_PROXY_HALF_LIFE_DAYS) == pytest.approx(0.25)
    ratings = _c2_ratings()
    state = {"live_minutes": {"p0": 900.0}, "live_contrib": {"p0": 15.0},
             "live_last_seen": {"p0": "2026-11-07"}}
    fresh = blend_ratings(ratings, state, as_of="2026-11-07").set_index("player_id")
    stale = blend_ratings(
        ratings, state,
        as_of=pd.Timestamp("2026-11-07") + pd.Timedelta(
            days=live.EVENT_PROXY_HALF_LIFE_DAYS)).set_index("player_id")
    base = ratings.set_index("player_id").loc["p0", "vaep_per90"]
    assert fresh.loc["p0", "event_proxy_per90"] > 0
    assert base < stale.loc["p0", "vaep_per90_live"] < fresh.loc["p0", "vaep_per90_live"]


def test_stage3_efficient_frontier_fc_2_state_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "STATE_DIR", str(tmp_path))
    monkeypatch.setattr(live, "STATE_PATH", str(tmp_path / "state.json"))
    state = {
        "aliases": {"test fc||test player 0": 0},
        "live_minutes": {0: 90.0, "12": 45.0},
        "live_contrib": {0: 0.9},
        "processed_fixtures": {"101": {"fixture_id": 101, "status": "ok",
                                       "source_timestamp": "2026-11-07T12:30:00+00:00"}},
        "last_fixture_id": 101,
    }
    live.save_state(state)
    loaded = live.load_state()
    assert loaded["live_minutes"] == {"0": 90.0, "12": 45.0}
    assert loaded["live_contrib"] == {"0": 0.9}
    assert loaded["aliases"] == {"test fc||test player 0": "0"}
    assert loaded["processed_fixtures"]["101"]["fixture_id"] == "101"
    assert set(live.processed_fixture_ids(loaded)) == {"101"}
    live.save_state(loaded)
    assert live.load_state() == loaded


def test_stage3_efficient_frontier_fc_2_contributions_substitution_chain():
    lineups = [{"team": {"name": "Test FC"},
                "startXI": [{"player": {"name": "Starter Sam"}}],
                "substitutes": [{"player": {"name": "Sub One"}},
                                {"player": {"name": "Sub Two"}},
                                {"player": {"name": "Unused Uli"}}]}]
    events = [
        {"type": "subst", "detail": "Substitution 1", "time": {"elapsed": 60},
         "team": {"name": "Test FC"}, "player": {"name": "Starter Sam"},
         "assist": {"name": "Sub One"}},
        {"type": "subst", "detail": "Substitution 2", "time": {"elapsed": 75},
         "team": {"name": "Test FC"}, "player": {"name": "Sub One"},
         "assist": {"name": "Sub Two"}},
        {"type": "Card", "detail": "Red Card", "time": {"elapsed": 80},
         "team": {"name": "Test FC"}, "player": {"name": "Sub Two"},
         "assist": {"name": None}},
    ]
    df = fixture_contributions(events, lineups).set_index("api_player")
    assert df.loc["Starter Sam", "minutes"] == 60.0
    assert df.loc["Sub One", "minutes"] == 15.0
    assert df.loc["Sub Two", "minutes"] == 5.0
    assert df.loc["Unused Uli", "minutes"] == 0.0
    assert df.loc["Unused Uli", "stats_status"] == "unused"
    assert df.loc["Starter Sam", "stats_status"] == "observed"


def _c2_wire_refresh(monkeypatch, tmp_path, fixtures, fail_ids=(), empty_ids=()):
    import effc.live_refresh as lr

    saved = {}

    def save(state):
        saved["state"] = live.normalize_state(state)

    monkeypatch.setattr(lr, "load_state",
                        lambda: saved.get("state", live.normalize_state({})))
    monkeypatch.setattr(lr, "save_state", save)
    monkeypatch.setattr(lr, "fetch_finished_fixtures", lambda *a, **k: fixtures)
    monkeypatch.setattr(lr, "base_ratings", lambda *a, **k: _c2_ratings())

    def events(fid):
        if fid in fail_ids:
            raise live.ApiError("provider error (stubbed)")
        if fid in empty_ids:
            return []
        return _c2_bundle_payload("events", fid)

    def lineups(fid):
        if fid in fail_ids:
            raise live.ApiError("provider error (stubbed)")
        if fid in empty_ids:
            return []
        return _c2_bundle_payload("lineups", fid)

    monkeypatch.setattr(lr, "fetch_fixture_events", events)
    monkeypatch.setattr(lr, "fetch_fixture_lineups", lineups)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["live_refresh", "--budget", "2000000"])
    return lr, saved


def _c2_ratings_state(saved):
    s = saved["state"]
    return {k: s[k] for k in ("live_minutes", "live_contrib", "aliases",
                              "processed_fixtures", "watchlist")}


def test_stage3_efficient_frontier_fc_2_full_refresh_and_replay_idempotent(
        monkeypatch, tmp_path):
    fixtures = [f for f in _c2_bundle_fixtures()
                if f["fixture"]["id"] in _C2_BUNDLE_IDS]
    assert len(fixtures) == 3
    lr, saved = _c2_wire_refresh(monkeypatch, tmp_path, fixtures)
    lr.main()
    first = _c2_ratings_state(saved)
    ledger = first["processed_fixtures"]
    assert set(ledger) == {str(i) for i in _C2_BUNDLE_IDS}
    assert all(v["status"] == "ok" for v in ledger.values())
    assert all(v["source_timestamp"] for v in ledger.values())
    lr.main()
    assert _c2_ratings_state(saved) == first


def test_stage3_efficient_frontier_fc_2_out_of_order_discovery_not_skipped(
        monkeypatch, tmp_path):
    fixtures = [f for f in _c2_bundle_fixtures()
                if f["fixture"]["id"] in _C2_BUNDLE_IDS]
    lr, saved = _c2_wire_refresh(monkeypatch, tmp_path, fixtures)
    monkeypatch.setattr(lr, "fetch_finished_fixtures",
                        lambda *a, **k: [f for f in fixtures
                                          if f["fixture"]["id"] == max(_C2_BUNDLE_IDS)])
    lr.main()
    assert set(saved["state"]["processed_fixtures"]) == {str(max(_C2_BUNDLE_IDS))}
    monkeypatch.setattr(lr, "fetch_finished_fixtures", lambda *a, **k: fixtures)
    lr.main()
    assert set(saved["state"]["processed_fixtures"]) == {str(i) for i in _C2_BUNDLE_IDS}


def test_stage3_efficient_frontier_fc_2_partial_error_audited_not_counted(
        monkeypatch, tmp_path):
    fixtures = [f for f in _c2_bundle_fixtures()
                if f["fixture"]["id"] in _C2_BUNDLE_IDS]
    fail = _C2_BUNDLE_IDS[0]
    lr, saved = _c2_wire_refresh(monkeypatch, tmp_path, fixtures, fail_ids=(fail,))
    lr.main()
    ledger = saved["state"]["processed_fixtures"]
    assert ledger[str(fail)]["status"] == "error"
    assert ledger[str(fail)]["errors"]
    assert all(ledger[str(i)]["status"] == "ok" for i in _C2_BUNDLE_IDS[1:])
    assert str(fail) not in live.processed_fixture_ids(saved["state"])


def test_stage3_efficient_frontier_fc_2_missing_statistics_not_zero_production(
        monkeypatch, tmp_path):
    fixtures = [f for f in _c2_bundle_fixtures()
                if f["fixture"]["id"] in _C2_BUNDLE_IDS]
    empty = _C2_BUNDLE_IDS[1]
    lr, saved = _c2_wire_refresh(monkeypatch, tmp_path, fixtures, empty_ids=(empty,))
    lr.main()
    ledger = saved["state"]["processed_fixtures"]
    assert ledger[str(empty)]["status"] == "missing_statistics"
    assert str(empty) not in live.processed_fixture_ids(saved["state"])

