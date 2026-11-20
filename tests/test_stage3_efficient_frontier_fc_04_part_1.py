"""Stage 3 U04 part 1: forward-only, point-in-time backtest contract.

The contract under test: a decision dated at a cutoff may only use model
inputs (events, wages) strictly before that cutoff; decisions expand
chronologically; every policy (optimizer, value, point estimate, random,
oracle) faces the identical feasible constraints; unavailable players are
counted and penalized, never dropped silently; failed matchups stay
reported; and a regenerated forward run matches the committed ledger
hashes. The existing FC6 results and the existing backtest numbers are
captured as baseline and must not move.

Before the U04 part 1 implementation these tests fail: there is no
evaluation/forward_backtest module, no frozen forward fixture, and no
real-data fixture ledger. No network is used by these tests; the only
real data involved is already committed in the repo.
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest

from effc.backtest.runner import run_backtest
from effc.data.download import EXTRA_POOL
from effc.evaluation import forward_backtest as fb

_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "forward_backtest")
_FIXTURE_PATH = os.path.join(_FIXTURE_DIR, "forward_fixture.json")
_LEDGER_PATH = os.path.join(_FIXTURE_DIR, "real_data_ledger.json")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_fixture():
    with open(_FIXTURE_PATH) as fh:
        return json.load(fh)


def _oracle_hash(value) -> str:
    """Independent canonical hash: recomputed here, not via the module."""
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _oracle_file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ratings_before(fixture, cutoff):
    """Ratings from pre-cutoff VAEP rows only, built here from raw rows."""
    rows = [r for r in fixture["vaep_rows"] if r["match_date"] < cutoff]
    by_player = {}
    for r in rows:
        agg = by_player.setdefault(r["player_id"], {"minutes": 0.0, "vaep": 0.0})
        agg["minutes"] += float(r["minutes"])
        agg["vaep"] += float(r["vaep"])
    wages = {p["player_id"]: p for p in fixture["players"]}
    out = []
    for pid in sorted(wages):
        agg = by_player.get(pid, {"minutes": 0.0, "vaep": 0.0})
        minutes = agg["minutes"]
        out.append(
            {
                "player_id": pid,
                "pos_group": wages[pid]["pos_group"],
                "wage_eur": float(wages[pid]["wage_eur"]),
                "minutes": minutes,
                "vaep_per90": (agg["vaep"] / minutes * 90.0) if minutes > 0 else 0.0,
                "vaep": agg["vaep"],
            }
        )
    return pd.DataFrame(out)


def _realized_in_window(fixture, start, end):
    rows = [
        r
        for r in fixture["vaep_rows"]
        if start < r["match_date"] <= end
    ]
    return pd.DataFrame(rows, columns=["match_id", "match_date", "player_id", "vaep", "minutes"])


def _run_on_fixture(fixture):
    matches = pd.DataFrame(fixture["matches"])
    matches["match_date"] = pd.to_datetime(matches["match_date"])
    decisions = fb.build_decision_schedule(
        matches, fixture["cutoffs"], horizon_days=fixture["horizon_days"]
    )
    ratings = {}
    realized = {}
    for d in decisions:
        ratings[d.decision_id] = _ratings_before(fixture, d.cutoff.strftime("%Y-%m-%d"))
        realized[d.decision_id] = _realized_in_window(
            fixture,
            d.window_start.strftime("%Y-%m-%d"),
            d.window_end.strftime("%Y-%m-%d"),
        )
    return fb.run_forward_backtest(
        decisions, ratings, realized, fixture["budgets"], seed=fixture["seed"]
    )


def test_stage3_efficient_frontier_fc_04_part_1_pool_leak_flagged():
    # The defect, in miniature: a pool with one contemporaneous match and
    # two later matches. Only the later matches violate a 2016-01-01 cutoff.
    pool = pd.DataFrame(
        [
            {"match_id": 9003, "competition": "Premier League", "match_date": "2015-08-15"},
            {"match_id": 9001, "competition": "UEFA Euro 2020", "match_date": "2021-06-11"},
            {"match_id": 9002, "competition": "Indian Super League", "match_date": "2021-11-19"},
        ]
    )
    pool["match_date"] = pd.to_datetime(pool["match_date"])
    violations = fb.check_pool_point_in_time(pool, "2016-01-01")
    assert [int(v["match_id"]) for v in violations] == [9001, 9002]
    with pytest.raises(fb.PointInTimeError):
        fb.assert_pool_point_in_time(pool, "2016-01-01")
    # The contemporaneous match alone passes.
    assert fb.check_pool_point_in_time(pool.iloc[[0]], "2016-01-01") == []


def test_stage3_efficient_frontier_fc_04_part_1_committed_pool_contains_later_competitions():
    # The committed pool constants are the leak: Euro 2020 (played 2021)
    # and the Indian Super League 2021/22 both postdate every 2015/16
    # decision cutoff, yet they train the goal models behind the backtest.
    assert (55, 43) in EXTRA_POOL  # UEFA Euro 2020
    assert (1238, 108) in EXTRA_POOL  # Indian Super League 2021/2022
    assert len(EXTRA_POOL) == 2
    # The era pool is the contemporaneous control: all 2015/16 seasons.
    from effc.data.download import ERA_2015_POOL

    assert ERA_2015_POOL == [(11, 27), (12, 27), (7, 27), (2, 27)]


def test_stage3_efficient_frontier_fc_04_part_1_schedule_expands_chronologically():
    fixture = _load_fixture()
    matches = pd.DataFrame(fixture["matches"])
    matches["match_date"] = pd.to_datetime(matches["match_date"])
    decisions = fb.build_decision_schedule(
        matches, fixture["cutoffs"], horizon_days=fixture["horizon_days"]
    )
    assert [d.decision_id for d in decisions] == [
        "fwd-01-2015-11-01",
        "fwd-02-2016-02-01",
    ]
    # Independent oracle: rebuild the input/window sets from the raw
    # fixture rows here, without the module's schedule code.
    for i, d in enumerate(decisions):
        cutoff = pd.Timestamp(fixture["cutoffs"][i])
        raw_inputs = matches[matches["match_date"] < cutoff]
        assert d.cutoff == cutoff
        assert d.max_input_timestamp < cutoff
        assert d.max_input_timestamp == raw_inputs["match_date"].max()
        assert sorted(d.input_match_ids) == sorted(raw_inputs["match_id"].tolist())
        next_cutoff = (
            pd.Timestamp(fixture["cutoffs"][i + 1])
            if i + 1 < len(fixture["cutoffs"])
            else cutoff + pd.Timedelta(days=fixture["horizon_days"])
        )
        raw_window = matches[
            (matches["match_date"] > cutoff) & (matches["match_date"] <= next_cutoff)
        ]
        assert d.window_start == cutoff
        assert d.window_end == next_cutoff
        assert not set(raw_window["match_id"]) & set(d.input_match_ids)
        assert d.wage_snapshot_date <= cutoff
        assert d.retrain is True
    # Expanding window: the second decision sees strictly more history.
    assert set(decisions[0].input_match_ids) < set(decisions[1].input_match_ids)
    # Cache keys bind the cutoff and the exact input set.
    assert decisions[0].cache_key != decisions[1].cache_key
    assert fb.decision_cache_key(
        decisions[0].cutoff, sorted(decisions[0].input_match_ids)
    ) == decisions[0].cache_key
    # A cutoff with no prior matches fails closed instead of deciding on air.
    with pytest.raises(fb.PointInTimeError):
        fb.build_decision_schedule(matches, ["2015-01-01"])


def test_stage3_efficient_frontier_fc_04_part_1_decision_rejects_future_input():
    fixture = _load_fixture()
    matches = pd.DataFrame(fixture["matches"])
    matches["match_date"] = pd.to_datetime(matches["match_date"])
    decisions = fb.build_decision_schedule(
        matches, fixture["cutoffs"], horizon_days=fixture["horizon_days"]
    )
    d = decisions[0]
    stamps = matches[matches["match_id"].isin(d.input_match_ids)]["match_date"]
    fb.validate_decision(d, stamps, scored_match_ids=[6, 7, 8])
    # A max input timestamp at or after the cutoff is a leak, not a decision.
    bad = fb.ForwardDecision(
        decision_id="bad-01",
        cutoff=d.cutoff,
        max_input_timestamp=d.cutoff,
        input_match_ids=d.input_match_ids,
        wage_snapshot_date=d.wage_snapshot_date,
        window_start=d.window_start,
        window_end=d.window_end,
        retrain=True,
        cache_key=d.cache_key,
    )
    with pytest.raises(fb.PointInTimeError):
        fb.validate_decision(bad, stamps, scored_match_ids=[6, 7, 8])
    # Scored matches overlapping the input set are a leak.
    with pytest.raises(fb.PointInTimeError):
        fb.validate_decision(d, stamps, scored_match_ids=[5, 6])
    # Wages dated after the cutoff are a leak.
    with pytest.raises(fb.PointInTimeError):
        fb.validate_decision(
            d, stamps, scored_match_ids=[6, 7, 8],
            wage_snapshot_date=d.cutoff + pd.Timedelta(days=1),
        )


def test_stage3_efficient_frontier_fc_04_part_1_run_reports_all_policies_same_constraints():
    fixture = _load_fixture()
    results, failures, records = _run_on_fixture(fixture)
    # Every feasible decision/budget/policy is reported exactly once.
    assert set(results["policy"].unique()) == {
        "optimizer", "value", "point", "random", "oracle",
    }
    assert len(results) == 2 * 2 * 5  # 2 decisions x 2 feasible budgets x 5 policies
    wages = {p["player_id"]: p["wage_eur"] for p in fixture["players"]}
    groups = {p["player_id"]: p["pos_group"] for p in fixture["players"]}
    from effc.portfolio.portfolio import POS_QUOTA

    for rec in records:
        for policy, by_budget in rec["policies"].items():
            for budget, row in by_budget.items():
                squad = row["squad"]
                # Independent feasibility check from the raw fixture wages.
                assert len(squad) == 23
                assert sum(wages[pid] for pid in squad) <= float(budget) + 1e-6
                counts = pd.Series([groups[pid] for pid in squad]).value_counts()
                for g, q in POS_QUOTA.items():
                    assert counts.get(g, 0) >= q
                assert row["n_selected"] == 23
                assert row["wage_bill"] == pytest.approx(
                    sum(wages[pid] for pid in squad)
                )
    # The point-estimate policy is a genuinely different rule: it picks by
    # raw per-90 rate with no exposure weighting and no value-per-euro term.
    point_squads = {
        (rec["decision_id"], b): rec["policies"]["point"][b]["squad"]
        for rec in records
        for b in rec["policies"]["point"]
    }
    value_squads = {
        (rec["decision_id"], b): rec["policies"]["value"][b]["squad"]
        for rec in records
        for b in rec["policies"]["value"]
    }
    assert any(
        set(point_squads[k]) != set(value_squads[k]) for k in point_squads
    )


def test_stage3_efficient_frontier_fc_04_part_1_unavailable_players_penalized_and_counted():
    fixture = _load_fixture()
    results, failures, records = _run_on_fixture(fixture)
    # Planted stars: player 0 is a top input rating for decision 1 but has
    # no realized rows in window 1; player 39 likewise for decision 2.
    assert 0 in records[0]["policies"]["optimizer"]["300000"]["squad"]
    assert 39 in records[1]["policies"]["optimizer"]["300000"]["squad"]
    for rec in records:
        for policy, by_budget in rec["policies"].items():
            for budget, row in by_budget.items():
                if policy == "random":
                    continue  # random squads are reported as means, not one squad
                # Independent recomputation from the raw fixture rows.
                window_rows = [
                    r
                    for r in fixture["vaep_rows"]
                    if rec["window_start"] < r["match_date"] <= rec["window_end"]
                ]
                realized = {}
                for r in window_rows:
                    realized[r["player_id"]] = realized.get(r["player_id"], 0.0) + r["vaep"]
                squad = row["squad"]
                raw = sum(realized.get(pid, 0.0) for pid in squad)
                n_unavail = sum(1 for pid in squad if pid not in realized)
                assert row["score_raw"] == pytest.approx(raw, abs=1e-6)
                assert row["n_unavailable"] == n_unavail
                available = [realized[pid] for pid in squad if pid in realized]
                penalty = (sum(available) / len(available)) if available else 0.0
                assert row["penalty_per_unavailable"] == pytest.approx(penalty, abs=1e-6)
                assert row["score_penalized"] == pytest.approx(
                    raw - penalty * n_unavail, abs=1e-6
                )
    # The planted stars are counted unavailable where they were selected.
    assert records[0]["policies"]["optimizer"]["300000"]["n_unavailable"] >= 1
    assert records[1]["policies"]["optimizer"]["300000"]["n_unavailable"] >= 1
    # The oracle never selects a player with zero realized production.
    assert records[0]["policies"]["oracle"]["300000"]["n_unavailable"] == 0
    assert records[1]["policies"]["oracle"]["300000"]["n_unavailable"] == 0


def test_stage3_efficient_frontier_fc_04_part_1_failed_matchups_reported_not_skipped():
    fixture = _load_fixture()
    results, failures, records = _run_on_fixture(fixture)
    # The 10,000 budget cannot field any 23-man squad here; unlike the
    # old runner, which silently skips the budget, every failed matchup
    # stays reported with its reason.
    assert not failures.empty
    assert set(failures["status"].unique()) == {"failed"}
    assert set(failures["budget"].unique()) == {10000.0}
    assert set(failures["policy"].unique()) == {
        "optimizer", "value", "point", "random", "oracle",
    }
    assert len(failures) == 2 * 5  # 2 decisions x 5 policies
    assert failures["reason"].notna().all()
    assert (failures["reason"].str.len() > 0).all()
    # Results plus failures cover every decision/budget/policy exactly once.
    combos = {
        (r["decision_id"], float(r["budget"]), r["policy"]) for _, r in results.iterrows()
    } | {
        (r["decision_id"], float(r["budget"]), r["policy"]) for _, r in failures.iterrows()
    }
    expected = {
        (f"fwd-0{i}-" + c, float(b), p)
        for i, c in enumerate(fixture["cutoffs"], start=1)
        for b in fixture["budgets"]
        for p in ("optimizer", "value", "point", "random", "oracle")
    }
    assert combos == expected


def test_stage3_efficient_frontier_fc_04_part_1_oracle_bounds_by_construction():
    fixture = _load_fixture()
    results, failures, records = _run_on_fixture(fixture)
    for _, row in results.iterrows():
        if row["policy"] == "oracle":
            continue
        oracle = results[
            (results["decision_id"] == row["decision_id"])
            & (results["budget"] == row["budget"])
            & (results["policy"] == "oracle")
        ].iloc[0]
        # The oracle maximizes realized totals in-sample, the exact
        # quantity scored, so it bounds every other rule on raw score.
        assert oracle["score_raw"] >= row["score_raw"] - 1e-9


def test_stage3_efficient_frontier_fc_04_part_1_regeneration_matches_committed_hashes():
    with open(_LEDGER_PATH) as fh:
        ledger = json.load(fh)
    fixture = _load_fixture()
    # The ledger's artifact hashes are recomputed from disk here.
    for entry in ledger["artifacts"]:
        path = os.path.join(_REPO_ROOT, entry["path"])
        assert _oracle_file_sha256(path) == entry["sha256"], entry["path"]
    # The forward run is regenerated from the frozen fixture and its
    # record hashes must match the committed ledger exactly.
    _, _, records = _run_on_fixture(fixture)
    assert fb.verify_ledger(ledger, _REPO_ROOT) == []
    for rec in records:
        assert fb.verify_record_hash(rec) is True
        assert ledger["forward_run"]["records"][rec["decision_id"]] == rec["record_sha256"]
        # The hash itself is recomputed independently in this test.
        body = {k: v for k, v in rec.items() if k != "record_sha256"}
        assert _oracle_hash(body) == rec["record_sha256"]


def test_stage3_efficient_frontier_fc_04_part_1_baseline_capture_existing_backtest_unchanged():
    # Frozen baseline: the existing runner on a fixed synthetic frame.
    # These numbers are the pre-U04 behavior; U04 must not move them.
    rng = np.random.default_rng(0)
    n = 48
    base = ["GK"] * 6 + ["DEF"] * 14 + ["MID"] * 14 + ["FWD"] * 14
    groups = (base * ((n // len(base)) + 1))[:n]
    ratings = pd.DataFrame(
        {
            "player_id": range(n),
            "pos_group": groups,
            "vaep_per90": rng.normal(0.3, 0.2, n),
            "wage_eur": rng.uniform(50_000, 400_000, n),
            "minutes": rng.uniform(300, 3000, n),
        }
    )
    ratings["vaep"] = ratings["vaep_per90"] * ratings["minutes"] / 90.0
    realized = 0.1 + np.arange(n) * 0.07
    pm2 = pd.DataFrame({"match_id": [1] * n, "player_id": list(range(n)), "vaep": realized})
    minutes = pd.DataFrame({"match_id": [1] * n, "player_id": list(range(n)), "minutes": [90] * n})
    r2 = ratings.copy()
    r2["vaep"] = realized
    bt = run_backtest(ratings, r2, pm2, minutes, [6_000_000], seed=0)
    assert len(bt) == 1
    row = bt.iloc[0]
    assert row["optimizer"] == pytest.approx(41.08, abs=1e-9)
    assert row["value"] == pytest.approx(34.08, abs=1e-9)
    assert row["random_mean"] == pytest.approx(38.959, abs=1e-9)
    assert row["oracle"] == pytest.approx(49.69, abs=1e-9)
    assert row["optimizer_unavail"] == 0
    assert row["oracle_unavail"] == 0


def test_stage3_efficient_frontier_fc_04_part_1_fc6_results_untouched():
    with open(_LEDGER_PATH) as fh:
        ledger = json.load(fh)
    # The frozen FC6 manifest and the published five-league run are
    # anchors, not inputs to rewrite: their hashes and model identity
    # must match the ledger exactly.
    manifest_hash = _oracle_file_sha256(
        os.path.join(_REPO_ROOT, "protocol", "fc6_manifest.json")
    )
    assert manifest_hash == ledger["fc6"]["manifest_sha256"]
    with open(os.path.join(_REPO_ROOT, "runs", "five_leagues", "results.json")) as fh:
        five = json.load(fh)
    assert five["model_hash"] == ledger["baseline"]["five_leagues"]["model_hash"]
    assert five["model_hash"] == "b86caaed47f745d4"
    assert five["model_config"] == "fc6_pool"
    assert five["seed"] == 0
    # The published backtest table in the ledger is a verbatim copy.
    for league, rows in ledger["baseline"]["five_leagues"]["backtest"].items():
        assert five["leagues"][league]["backtest"] == rows
    wins = sum(
        1
        for rows in ledger["baseline"]["five_leagues"]["backtest"].values()
        for r in rows
        if r["optimizer"] > r["value"]
    )
    assert wins == 14
    assert ledger["baseline"]["five_leagues"]["optimizer_wins"] == "14 of 15"
    # The demo backtest table in the ledger is a verbatim copy.
    with open(os.path.join(_REPO_ROOT, "demo", "backtest.csv")) as fh:
        demo_text = fh.read()
    assert demo_text == ledger["baseline"]["demo_backtest_csv"]
