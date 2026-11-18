"""Stage 3 U03: make the live rebalancer operational, safe, and measurable.

The contract under test: the five baseline artifacts are tied together
in one verifiable bundle; the processing ledger records full event
availability timestamps and per-fixture contributions; state carries a
checksum that is verified on load and the live totals reconcile back
to the ledger; every refresh writes a reproducibility record and a
complete fixture and player audit, names the new-squad figure honestly
as a blended event-proxy value, and checks both squads are feasible;
and the event-summary proxy is measured against fitted VAEP on real
overlapping data in a separate study.

Before the U03 implementation these tests fail: there is no bundle
manifest or loader, no state digest, no fetch timestamps, no
contributions in the ledger, no reproducibility or audit block in the
refresh output, and no proxy study. The real provider fixture bundle
under tests/fixtures/api_football/ drives the end-to-end trace; no
network is used by these tests.
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest

from effc import live
from effc.live import blend_ratings

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "api_football")
_BUNDLE_IDS = [1557463, 1557457, 1557465]
EXPECTED_COUNTS = {39: 179, 140: 146, 78: 268, 135: 143, 61: 145}


def _canonical_sha256(value) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _c2_ratings(n_per_group: int = 8) -> pd.DataFrame:
    rows = []
    pid = 0
    for g in ["GK", "DEF", "MID", "FWD"]:
        for _ in range(n_per_group):
            rows.append({
                "player_id": f"p{pid}",
                "player": f"Test Player {pid}",
                "team": "Test FC",
                "minutes": 2000.0,
                "vaep": 5.0,
                "vaep_per90": 0.05 + 0.03 * pid,
                "pos_group": g,
                "wage_eur": 50_000.0,
            })
            pid += 1
    return pd.DataFrame(rows)


def _bundle_fixtures():
    with open(os.path.join(_FIXTURES, "fixtures_date_2026-11-07.json")) as f:
        payload = json.load(f)
    return [f for f in payload["response"]
            if f["league"]["id"] == 39
            and f["fixture"]["status"]["short"] == "FT"]


def _bundle_payload(kind, fixture_id):
    with open(os.path.join(_FIXTURES, f"{kind}_{fixture_id}.json")) as f:
        return json.load(f)["response"]


# ---------------------------------------------------------------------------
# Baseline bundle: one verifiable manifest for all five league artifacts.
# ---------------------------------------------------------------------------

def test_stage3_efficient_frontier_fc_03_bundle_manifest_verifies_all_leagues():
    manifest = live.load_baseline_bundle()
    assert manifest["version"] == live.BUNDLE_VERSION
    assert {e["league_id"] for e in manifest["entries"]} == set(EXPECTED_COUNTS)
    for entry in manifest["entries"]:
        assert entry["n_rated"] == EXPECTED_COUNTS[entry["league_id"]]
        artifact = live.load_baseline_artifact(
            live.baseline_path_for_league(entry["league_id"]))
        assert entry["ratings_sha256"] == artifact["ratings_sha256"]
        assert entry["file_sha256"] == live.file_sha256(
            live.baseline_path_for_league(entry["league_id"]))
        assert entry["model_hash"] == artifact.get("model_hash")
        assert entry["seed"] == artifact.get("seed")
        assert entry["feature_version"] == artifact.get("feature_version")


def _copy_bundle(tmp_path):
    """Copy the committed bundle manifest and its artifacts into a
    scratch dir; returns (manifest dict, manifest path)."""
    manifest = live.load_baseline_bundle()
    dest = tmp_path / "live"
    dest.mkdir()
    for entry in manifest["entries"]:
        src = live.baseline_path_for_league(entry["league_id"])
        (dest / os.path.basename(src)).write_bytes(open(src, "rb").read())
    manifest_path = dest / "baseline_bundle_v1.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest, manifest_path


def test_stage3_efficient_frontier_fc_03_bundle_tampered_artifact_fails(tmp_path):
    manifest, manifest_path = _copy_bundle(tmp_path)
    victim = manifest["entries"][0]
    path = manifest_path.parent / victim["artifact"]
    payload = json.loads(path.read_text())
    payload["ratings"][0]["vaep_per90"] = 99.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        live.load_baseline_bundle(str(manifest_path))


def test_stage3_efficient_frontier_fc_03_bundle_tampered_manifest_fails(tmp_path):
    manifest, manifest_path = _copy_bundle(tmp_path)
    copied = json.loads(manifest_path.read_text())
    copied["entries"][0]["ratings_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(copied))
    with pytest.raises(ValueError):
        live.load_baseline_bundle(str(manifest_path))


def test_stage3_efficient_frontier_fc_03_no_live_identity_through_bundle():
    for league_id in sorted(EXPECTED_COUNTS):
        ratings = live.load_baseline_ratings(league=league_id)
        out = blend_ratings(ratings, {"live_minutes": {}, "live_contrib": {}})
        pd.testing.assert_series_equal(
            out["vaep_per90_live"], ratings["vaep_per90"], check_names=False)


# ---------------------------------------------------------------------------
# State integrity: checksum on save, verified on load, totals reconcile.
# ---------------------------------------------------------------------------

def test_stage3_efficient_frontier_fc_03_state_digest_roundtrip_and_tamper(
        tmp_path, monkeypatch):
    monkeypatch.setattr(live, "STATE_DIR", str(tmp_path))
    monkeypatch.setattr(live, "STATE_PATH", str(tmp_path / "state.json"))
    state = {
        "aliases": {"test fc||test player 0": "0"},
        "live_minutes": {"0": 90.0, "12": 45.0},
        "live_contrib": {"0": 0.9},
        "processed_fixtures": {"101": {"fixture_id": "101", "status": "ok",
                                       "source_timestamp": "2026-11-07T12:30:00+00:00",
                                       "processed_at": "2026-11-18T04:00:00+00:00",
                                       "matched": 2, "unmatched": 0,
                                       "errors": [],
                                       "contributions": [
                                           {"player_id": "0", "minutes": 90.0,
                                            "contrib": 0.9},
                                           {"player_id": "12", "minutes": 45.0,
                                            "contrib": 0.0}],
                                       "contributions_sha256": _canonical_sha256(
                                           [{"player_id": "0", "minutes": 90.0,
                                             "contrib": 0.9},
                                            {"player_id": "12", "minutes": 45.0,
                                             "contrib": 0.0}])}},
    }
    live.save_state(state)
    loaded = live.load_state()
    assert loaded["state_sha256"] == live.state_digest(loaded)
    assert live.verify_state(loaded)["ok"]

    # tamper with a live total in the saved file: load must refuse it
    payload = json.loads((tmp_path / "state.json").read_text())
    payload["live_minutes"]["0"] = 900.0
    (tmp_path / "state.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        live.load_state()


def test_stage3_efficient_frontier_fc_03_legacy_state_without_digest_loads(
        tmp_path, monkeypatch):
    monkeypatch.setattr(live, "STATE_DIR", str(tmp_path))
    monkeypatch.setattr(live, "STATE_PATH", str(tmp_path / "state.json"))
    legacy = {"live_minutes": {"0": 90.0}, "live_contrib": {"0": 0.9},
              "processed_fixtures": {"101": {"fixture_id": "101",
                                             "status": "ok"}}}
    (tmp_path / "state.json").write_text(json.dumps(legacy))
    loaded = live.load_state()
    assert loaded["live_minutes"] == {"0": 90.0}
    # saving a legacy state gains a digest; the legacy ledger entry is
    # a warning, not an error: it cannot be reconciled, but it is not
    # treated as corruption
    live.save_state(loaded)
    assert live.load_state()["state_sha256"]
    report = live.verify_state(live.load_state())
    assert report["ok"]
    assert report["warnings"]


def test_stage3_efficient_frontier_fc_03_verify_state_catches_resum_mismatch():
    state = live.normalize_state({
        "live_minutes": {"0": 90.0, "12": 45.0},
        "live_contrib": {"0": 0.9, "12": 0.0},
        "processed_fixtures": {"101": {
            "fixture_id": "101", "status": "ok",
            "source_timestamp": "2026-11-07T12:30:00+00:00",
            "processed_at": "2026-11-18T04:00:00+00:00",
            "matched": 2, "unmatched": 0, "errors": [],
            "contributions": [
                {"player_id": "0", "minutes": 90.0, "contrib": 0.9},
                {"player_id": "12", "minutes": 45.0, "contrib": 0.0}],
            "contributions_sha256": _canonical_sha256([
                {"player_id": "0", "minutes": 90.0, "contrib": 0.9},
                {"player_id": "12", "minutes": 45.0, "contrib": 0.0}])}},
    })
    assert live.verify_state(state)["ok"]
    tampered = live.normalize_state(state)
    tampered["live_minutes"]["0"] = 180.0
    report = live.verify_state(tampered)
    assert not report["ok"]
    assert report["errors"]


# ---------------------------------------------------------------------------
# Wired end-to-end trace on the real captured fixture bundle.
# ---------------------------------------------------------------------------

def _wire_refresh(monkeypatch, tmp_path, fixtures, fail_ids=(), empty_ids=()):
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
        return _bundle_payload("events", fid)

    def lineups(fid):
        if fid in fail_ids:
            raise live.ApiError("provider error (stubbed)")
        if fid in empty_ids:
            return []
        return _bundle_payload("lineups", fid)

    monkeypatch.setattr(lr, "fetch_fixture_events", events)
    monkeypatch.setattr(lr, "fetch_fixture_lineups", lineups)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["live_refresh", "--budget", "2000000"])
    return lr, saved


def _ratings_state(saved):
    s = saved["state"]
    return {k: s[k] for k in ("live_minutes", "live_contrib", "aliases",
                              "processed_fixtures", "watchlist")}


def test_stage3_efficient_frontier_fc_03_ledger_availability_timestamps(
        monkeypatch, tmp_path):
    fixtures = [f for f in _bundle_fixtures()
                if f["fixture"]["id"] in _BUNDLE_IDS]
    lr, saved = _wire_refresh(monkeypatch, tmp_path, fixtures)
    lr.main()
    ledger = saved["state"]["processed_fixtures"]
    assert set(ledger) == {str(i) for i in _BUNDLE_IDS}
    for fid, entry in ledger.items():
        assert entry["status"] == "ok"
        # full UTC timestamps, not date-only
        for key in ("processed_at", "events_fetched_at", "lineups_fetched_at"):
            assert "T" in entry[key], key
        assert entry["availability_latency_hours"] >= 0.0
        assert entry["source_timestamp"]
        # per-fixture contributions recorded, hash-verified, reconciling
        assert entry["contributions"] == []
        assert entry["contributions_sha256"] == _canonical_sha256([])
    assert live.verify_state(saved["state"])["ok"]


def test_stage3_efficient_frontier_fc_03_replay_idempotent_with_new_ledger(
        monkeypatch, tmp_path):
    fixtures = [f for f in _bundle_fixtures()
                if f["fixture"]["id"] in _BUNDLE_IDS]
    lr, saved = _wire_refresh(monkeypatch, tmp_path, fixtures)
    lr.main()
    first = _ratings_state(saved)
    lr.main()
    assert _ratings_state(saved) == first
    # the call counter legitimately advances on the replay, so the
    # whole-state digest moves; what must not move is the ratings state
    assert live.verify_state(saved["state"])["ok"]


def test_stage3_efficient_frontier_fc_03_out_of_order_not_skipped_with_timestamps(
        monkeypatch, tmp_path):
    fixtures = [f for f in _bundle_fixtures()
                if f["fixture"]["id"] in _BUNDLE_IDS]
    lr, saved = _wire_refresh(monkeypatch, tmp_path, fixtures)
    monkeypatch.setattr(lr, "fetch_finished_fixtures",
                        lambda *a, **k: [f for f in fixtures
                                          if f["fixture"]["id"] == max(_BUNDLE_IDS)])
    lr.main()
    assert set(saved["state"]["processed_fixtures"]) == {str(max(_BUNDLE_IDS))}
    monkeypatch.setattr(lr, "fetch_finished_fixtures", lambda *a, **k: fixtures)
    lr.main()
    assert set(saved["state"]["processed_fixtures"]) == {str(i) for i in _BUNDLE_IDS}
    assert live.verify_state(saved["state"])["ok"]


def test_stage3_efficient_frontier_fc_03_refresh_output_reproducibility_and_audit(
        monkeypatch, tmp_path, capsys):
    fixtures = [f for f in _bundle_fixtures()
                if f["fixture"]["id"] in _BUNDLE_IDS]
    lr, saved = _wire_refresh(monkeypatch, tmp_path, fixtures)
    lr.main()
    out = capsys.readouterr().out
    assert "blended event-proxy" in out
    assert "new_exp_vaep" not in out

    reports = list((tmp_path / "runs" / "live").glob("refresh_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())

    # honest naming: the new squad is scored on proxy-blended ratings
    assert "new_exp_vaep" not in report
    assert "new_exp_blended_per90" in report
    assert "old_exp_vaep_per90" in report

    # reproducibility: everything needed to replay this refresh
    repro = report["reproducibility"]
    assert repro["baseline"]["league_id"] == 78
    assert repro["baseline"]["ratings_sha256"] == live.load_baseline_artifact(
        )["ratings_sha256"]
    assert repro["baseline"]["file_sha256"] == live.file_sha256(
        live.BASELINE_PATH)
    assert repro["state_sha256_before"]
    assert repro["state_sha256_after"]
    assert repro["state_sha256_after"] == live.state_digest(saved["state"])
    assert repro["blend"]["shrink_k"] == 900.0
    assert repro["blend"]["event_proxy_half_life_days"] == 45.0
    assert repro["fixture_sha256"] == {
        str(f["fixture"]["id"]): _canonical_sha256(f) for f in fixtures}

    # complete fixture and player audit
    audit = report["audit"]
    assert audit["fixtures_seen"] == 3
    assert audit["fixtures_processed_ok"] == 3
    assert audit["fixtures_error"] == 0
    assert audit["fixtures_missing_statistics"] == 0
    assert audit["fixtures_already_processed"] == 0
    assert audit["player_fixtures_matched"] == 0
    assert audit["player_fixtures_unmatched"] == 93
    assert audit["watchlist_size"] == 93
    assert audit["state_verification"]["ok"]
    assert audit["squads"]["old_feasible"] is True
    assert audit["squads"]["new_feasible"] is True
    assert len(audit["fixtures"]) == 3
    assert {f["fixture_id"] for f in audit["fixtures"]} == {
        str(i) for i in _BUNDLE_IDS}
    assert all(f["status"] == "ok" for f in audit["fixtures"])

    # history carries the baseline identifier for the run
    history = saved["state"]["history"]
    assert history[-1]["baseline_sha256"] == repro["baseline"]["ratings_sha256"]


def test_stage3_efficient_frontier_fc_03_provider_rejection_fails_cleanly(
        monkeypatch, tmp_path):
    """A provider-level rejection (suspended account, bad key) fails
    the run before any state is written, instead of recording
    invented fixtures or an empty refresh."""
    import effc.live_refresh as lr

    monkeypatch.setattr(lr, "load_state", lambda: live.normalize_state({}))
    saved = {}

    def save(state):
        saved["state"] = state

    monkeypatch.setattr(lr, "save_state", save)
    monkeypatch.setattr(
        lr, "fetch_finished_fixtures",
        lambda *a, **k: (_ for _ in ()).throw(
            live.ApiError("API errors: {'access': 'suspended (stubbed)'}")))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["live_refresh"])
    with pytest.raises(SystemExit):
        lr.main()
    assert saved == {}
    assert not list((tmp_path / "runs").rglob("refresh_*.json"))


# ---------------------------------------------------------------------------
# Proxy-performance study: known-answer fixtures for the study functions.
# ---------------------------------------------------------------------------

def _statsbomb_events():
    def ev(etype, player, team, minute, **kw):
        d = {"type": {"name": etype}, "player": {"name": player},
             "team": {"name": team}, "minute": minute}
        d.update(kw)
        return d

    return [
        ev("Shot", "Scorer Sam", "Test FC", 23,
           shot={"outcome": {"name": "Goal"}, "type": {"name": "Open Play"}}),
        {"type": {"name": "Pass"}, "player": {"name": "Assist Ann"},
         "team": {"name": "Test FC"}, "minute": 23,
         "pass": {"goal_assist": True}},
        ev("Foul Committed", "Carded Carl", "Test FC", 55,
           foul_committed={"card": {"name": "Yellow Card"}}),
        ev("Foul Committed", "Sentoff Sid", "Test FC", 80,
           foul_committed={"card": {"name": "Second Yellow"}}),
        ev("Shot", "Penalty Pete", "Test FC", 70,
           shot={"outcome": {"name": "Goal"}, "type": {"name": "Penalty"}}),
        # own goals contribute nothing to the proxy
        ev("Own Goal For", "Unlucky Uli", "Test FC", 60),
        ev("Shot", "Saved Steve", "Test FC", 10,
           shot={"outcome": {"name": "Saved"}, "type": {"name": "Open Play"}}),
    ]


def _statsbomb_lineups():
    def player(name, spans):
        return {"player_name": name,
                "positions": [{"position": "Midfield", "from": a, "to": b}
                              for a, b in spans]}

    return [{"team_name": "Test FC", "lineup": [
        player("Scorer Sam", [("00:00", "90:00")]),
        player("Assist Ann", [("00:00", "90:00")]),
        player("Carded Carl", [("00:00", "90:00")]),
        player("Sentoff Sid", [("00:00", "90:00")]),
        player("Penalty Pete", [("00:00", "90:00")]),
        player("Unlucky Uli", [("00:00", "90:00")]),
        player("Saved Steve", [("00:00", "90:00")]),
        player("Unused Uli", []),
    ]}]


def test_stage3_efficient_frontier_fc_03_statsbomb_proxy_known_answer():
    from effc.proxy_study import statsbomb_match_proxy

    df = statsbomb_match_proxy(_statsbomb_events(), _statsbomb_lineups())
    by_player = df.set_index("player")
    assert by_player.loc["Scorer Sam", "contrib"] == pytest.approx(0.9)
    assert by_player.loc["Assist Ann", "contrib"] == pytest.approx(0.4)
    assert by_player.loc["Carded Carl", "contrib"] == pytest.approx(-0.05)
    assert by_player.loc["Sentoff Sid", "contrib"] == pytest.approx(-0.30)
    assert by_player.loc["Penalty Pete", "contrib"] == pytest.approx(0.7)
    assert by_player.loc["Unlucky Uli", "contrib"] == pytest.approx(0.0)
    assert by_player.loc["Saved Steve", "contrib"] == pytest.approx(0.0)
    assert by_player.loc["Unused Uli", "minutes"] == 0.0
    assert by_player.loc["Scorer Sam", "minutes"] == 90.0
    assert by_player.loc["Scorer Sam", "proxy_per90"] == pytest.approx(0.9)
    assert by_player.loc["Penalty Pete", "proxy_per90"] == pytest.approx(0.7)


def test_stage3_efficient_frontier_fc_03_proxy_comparison_known_answer():
    from effc.proxy_study import compare_proxy_to_baseline

    proxy = pd.DataFrame([
        {"player": "A", "team": "T", "minutes": 900.0, "contrib": 9.0,
         "proxy_per90": 0.9},
        {"player": "B", "team": "T", "minutes": 900.0, "contrib": 4.5,
         "proxy_per90": 0.45},
        {"player": "C", "team": "T", "minutes": 900.0, "contrib": 0.0,
         "proxy_per90": 0.0},
    ])
    ratings = pd.DataFrame([
        {"player_id": "1", "player": "A", "team": "T", "pos_group": "FWD",
         "vaep_per90": 0.45, "wage_eur": 10000.0, "minutes": 900.0},
        {"player_id": "2", "player": "B", "team": "T", "pos_group": "FWD",
         "vaep_per90": 0.225, "wage_eur": 10000.0, "minutes": 900.0},
        {"player_id": "3", "player": "C", "team": "T", "pos_group": "FWD",
         "vaep_per90": 0.0, "wage_eur": 10000.0, "minutes": 900.0},
    ])
    stats, matched = compare_proxy_to_baseline(proxy, ratings, min_minutes=90.0)
    assert stats["n"] == 3
    # VAEP is exactly half the proxy here: perfect linear relation
    assert stats["pearson"] == pytest.approx(1.0)
    assert stats["spearman"] == pytest.approx(1.0)
    assert stats["slope"] == pytest.approx(0.5)
    assert stats["intercept"] == pytest.approx(0.0)
    assert stats["mae"] == pytest.approx(0.0)
    assert matched["player"].tolist() == ["A", "B", "C"]

    # a constant proxy has no defined correlation: reported, not crashed
    flat = proxy.copy()
    flat["proxy_per90"] = 0.5
    stats_flat, _ = compare_proxy_to_baseline(flat, ratings, min_minutes=90.0)
    assert stats_flat["pearson"] is None
    assert stats_flat["slope"] is None


def test_stage3_efficient_frontier_fc_03_proxy_accumulate_merges_name_variants():
    """One player spelled two ways across matches is one proxy row.

    StatsBomb spells N'Golo Kante with a doubled apostrophe in some
    matches; without normalized grouping the same player is two rows
    and can match the same baseline player twice.
    """
    from effc.proxy_study import accumulate_proxy

    frames = [
        pd.DataFrame([{"team": "Leicester City", "player": "N'Golo Kanté",
                       "minutes": 90.0, "goals": 0, "penalty_goals": 0,
                       "assists": 0, "yellow": 0, "red": 0, "contrib": 0.0,
                       "proxy_per90": 0.0}]),
        pd.DataFrame([{"team": "Leicester City", "player": "N''Golo Kanté",
                       "minutes": 90.0, "goals": 1, "penalty_goals": 0,
                       "assists": 0, "yellow": 0, "red": 0, "contrib": 0.9,
                       "proxy_per90": 0.9}]),
    ]
    out = accumulate_proxy(frames)
    assert len(out) == 1
    assert out.iloc[0]["minutes"] == 180.0
    assert out.iloc[0]["contrib"] == pytest.approx(0.9)
    assert out.iloc[0]["proxy_per90"] == pytest.approx(0.45)
    assert out.iloc[0]["matches"] == 2


def test_stage3_efficient_frontier_fc_03_committed_proxy_study_results():
    """The committed study ran on real overlapping data: the numbers
    below are the recorded outcome, checked for sanity, not tuned."""
    results_path = os.path.join(
        os.path.dirname(__file__), "..", "runs", "live", "proxy_study",
        "results.json")
    with open(os.path.normpath(results_path)) as f:
        results = json.load(f)
    assert results["league"] == "English Premier League"
    assert results["season"] == "2015/2016"
    assert results["matches"] == 380
    assert results["baseline"]["league_id"] == 39
    assert results["baseline"]["ratings_sha256"] == live.load_baseline_artifact(
        live.baseline_path_for_league(39))["ratings_sha256"]
    stats = results["comparison"]
    assert stats["n"] >= 100
    assert stats["n"] <= 179
    assert -1.0 <= stats["pearson"] <= 1.0
    assert -1.0 <= stats["spearman"] <= 1.0
    assert np.isfinite(stats["slope"]) and np.isfinite(stats["mae"])
    assert stats["mean_proxy_per90"] != stats["mean_vaep_per90"]
    report_path = os.path.join(
        os.path.dirname(__file__), "..", "runs", "live", "proxy_study",
        "report.md")
    report = open(os.path.normpath(report_path)).read()
    assert "Pearson" in report
    assert "not calibrated" in report
