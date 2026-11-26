"""Stage 3 U04 part 2: forward-only backtest experiment driver and real run.

Part 2 builds on the part 1 contract module. The experiment driver
turns per-match VAEP rows into per-decision ratings and realized
windows, plans retraining on the expanding cache-key schedule, compares
policies with nulls preserved, and wraps the part 1 runner. The
committed real-data run (Premier League 2015/16, era-only training
pools, no Euro 2020 or ISL 2021/22) is verified from its artifact:
every record hash must verify and every decision must respect its
cutoff.

Before the part 2 implementation these tests fail: there is no
evaluation/forward_experiment module, no part 2 fixture, and no
runs/forward_backtest/results.json. No network is used; the only real
data involved is already committed in the repo (the experiment
artifact) plus the frozen fixtures.
"""

import json
import os

import pandas as pd
import pytest

from effc.evaluation import forward_backtest as fb
from effc.evaluation import forward_experiment as fe
from effc.portfolio.portfolio import POS_QUOTA

_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "forward_backtest")
_FIXTURE_PATH = os.path.join(_FIXTURE_DIR, "forward_fixture_part2.json")
_LEDGER_PATH = os.path.join(_FIXTURE_DIR, "real_data_ledger.json")
_RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "runs", "forward_backtest", "results.json",
)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_fixture():
    with open(_FIXTURE_PATH) as fh:
        return json.load(fh)


def _run_on_fixture(fixture):
    pm = pd.DataFrame(fixture["pm_rows"])
    mins = pd.DataFrame(fixture["minutes_rows"])
    wmap = pd.DataFrame(fixture["players"])[
        ["player", "team", "wage_eur", "value_eur", "pos_group",
         "matched", "wage_source"]
    ]
    matches = pd.DataFrame(fixture["matches"])
    matches["match_date"] = pd.to_datetime(matches["match_date"])
    decisions = fb.build_decision_schedule(
        matches, fixture["cutoffs"], horizon_days=fixture["horizon_days"]
    )
    ratings, realized = fe.build_forward_inputs(decisions, pm, mins, wmap)
    return decisions, fe.run_experiment(
        decisions, ratings, realized, fixture["budgets"],
        seed=fixture["seed"], n_random=fixture["n_random"],
    )


def test_stage3_efficient_frontier_fc_04_part_2_driver_end_to_end():
    fixture = _load_fixture()
    decisions, out = _run_on_fixture(fixture)
    results, failures, records = out["results"], out["failures"], out["records"]
    assert [d.decision_id for d in decisions] == [
        "fwd-01-2015-11-01", "fwd-02-2016-01-15", "fwd-03-2016-03-01",
    ]
    # 3 decisions x 2 feasible budgets x 5 policies reported.
    assert len(results) == 3 * 2 * 5
    assert set(results["policy"].unique()) == {
        "optimizer", "value", "point", "random", "oracle",
    }
    # 3 decisions x 1 infeasible budget x 5 policies failed, not skipped.
    assert len(failures) == 3 * 5
    assert set(failures["budget"].unique()) == {10000.0}
    # Every record hash verifies, and regeneration is deterministic:
    # running the fixture again reproduces the same record hashes.
    for rec in records:
        assert fb.verify_record_hash(rec) is True
    _, out2 = _run_on_fixture(fixture)
    assert [r["record_sha256"] for r in out2["records"]] == [
        r["record_sha256"] for r in records
    ]
    # Every reported squad is feasible under the shared constraints.
    wages = {p["player_id"]: p["wage_eur"] for p in fixture["players"]}
    groups = {p["player_id"]: p["pos_group"] for p in fixture["players"]}
    for rec in records:
        for policy, by_budget in rec["policies"].items():
            for budget, row in by_budget.items():
                squad = row["squad"]
                assert len(squad) == 23
                assert sum(wages[pid] for pid in squad) <= float(budget) + 1e-6
                counts = pd.Series([groups[pid] for pid in squad]).value_counts()
                for g, q in POS_QUOTA.items():
                    assert counts.get(g, 0) >= q
    # The oracle bounds every other policy on raw score, by construction.
    for _, row in results.iterrows():
        if row["policy"] == "oracle":
            continue
        oracle = results[
            (results["decision_id"] == row["decision_id"])
            & (results["budget"] == row["budget"])
            & (results["policy"] == "oracle")
        ].iloc[0]
        assert oracle["score_raw"] >= row["score_raw"] - 1e-9


def test_stage3_efficient_frontier_fc_04_part_2_retraining_schedule():
    fixture = _load_fixture()
    matches = pd.DataFrame(fixture["matches"])
    matches["match_date"] = pd.to_datetime(matches["match_date"])
    decisions = fb.build_decision_schedule(
        matches, fixture["cutoffs"], horizon_days=fixture["horizon_days"]
    )
    # First plan: nothing cached, every expanding decision retrains.
    plan, known = fe.plan_retraining(decisions, set())
    assert [p["retrain"] for p in plan] == [True, True, True]
    assert len({p["cache_key"] for p in plan}) == 3
    assert [p["n_input_matches"] for p in plan] == [5, 9, 12]
    assert plan[0]["reason"] == "no cache entry for this cutoff and input set"
    # Input sets nest: each later decision sees a superset of the earlier.
    assert set(decisions[0].input_match_ids) < set(decisions[1].input_match_ids)
    assert set(decisions[1].input_match_ids) < set(decisions[2].input_match_ids)
    # Second plan with the first plan's keys: every decision is a hit.
    plan2, _ = fe.plan_retraining(decisions, known)
    assert [p["retrain"] for p in plan2] == [False, False, False]
    assert all(p["reason"] == "cache hit" for p in plan2)
    assert [p["cache_key"] for p in plan2] == [p["cache_key"] for p in plan]
    # Same cutoff, different inputs: a different key, so it retrains.
    changed = fb.ForwardDecision(
        decision_id="fwd-01b-2015-11-01",
        cutoff=decisions[0].cutoff,
        max_input_timestamp=decisions[0].max_input_timestamp,
        input_match_ids=tuple(sorted(set(decisions[0].input_match_ids) | {999})),
        wage_snapshot_date=decisions[0].wage_snapshot_date,
        window_start=decisions[0].window_start,
        window_end=decisions[0].window_end,
        retrain=True,
        cache_key="",
    )
    plan3, _ = fe.plan_retraining([changed], known)
    assert plan3[0]["retrain"] is True
    assert plan3[0]["cache_key"] != plan[0]["cache_key"]


def test_stage3_efficient_frontier_fc_04_part_2_policy_comparison():
    fixture = _load_fixture()
    _, out = _run_on_fixture(fixture)
    comparison = out["comparison"]
    assert comparison["coverage"]["complete"] is True
    assert comparison["coverage"]["n_result_rows"] == 30
    assert comparison["coverage"]["n_failure_rows"] == 15
    assert comparison["oracle_bound_holds"] is True
    # Gaps are observed differences, recomputed independently here from
    # the results table; the summary claims no winner.
    assert "winner" not in comparison
    for gap in comparison["gaps"]:
        sub = out["results"][
            (out["results"]["decision_id"] == gap["decision_id"])
            & (out["results"]["budget"] == gap["budget"])
        ]
        pen = dict(zip(sub["policy"], sub["score_penalized"]))
        assert gap["optimizer_minus_random"] == pytest.approx(
            pen["optimizer"] - pen["random"]
        )
        assert gap["optimizer_minus_value"] == pytest.approx(
            pen["optimizer"] - pen["value"]
        )
        assert gap["value_minus_random"] == pytest.approx(
            pen["value"] - pen["random"]
        )
    # Point and value are genuinely different rules on this fixture.
    _, out_records = _run_on_fixture(fixture)
    records = out_records["records"]
    assert any(
        set(rec["policies"]["point"][b]["squad"])
        != set(rec["policies"]["value"][b]["squad"])
        for rec in records
        for b in rec["policies"]["point"]
    )


def test_stage3_efficient_frontier_fc_04_part_2_unavailable_penalty():
    fixture = _load_fixture()
    _, out = _run_on_fixture(fixture)
    records = out["records"]
    # Planted stars: 40 top-rated for decision 1, 41 for decision 2,
    # 20 for decision 3; each has no realized rows in its own window.
    stars = {"fwd-01-2015-11-01": 40, "fwd-02-2016-01-15": 41, "fwd-03-2016-03-01": 20}
    for rec in records:
        star = stars[rec["decision_id"]]
        assert star in rec["policies"]["optimizer"]["500000"]["squad"]
        assert rec["policies"]["optimizer"]["500000"]["n_unavailable"] >= 1
        assert rec["policies"]["oracle"]["500000"]["n_unavailable"] == 0
        # Independent recomputation of the penalized score.
        window_rows = [
            r for r in fixture["pm_rows"]
            if rec["window_start"] < r["match_date"] <= rec["window_end"]
        ]
        realized = {}
        for r in window_rows:
            realized[r["player_id"]] = realized.get(r["player_id"], 0.0) + r["vaep"]
        squad = rec["policies"]["optimizer"]["500000"]["squad"]
        raw = sum(realized.get(pid, 0.0) for pid in squad)
        n_unavail = sum(1 for pid in squad if pid not in realized)
        available = [realized[pid] for pid in squad if pid in realized]
        penalty = (sum(available) / len(available)) if available else 0.0
        row = rec["policies"]["optimizer"]["500000"]
        assert row["score_raw"] == pytest.approx(raw, abs=1e-6)
        assert row["n_unavailable"] == n_unavail
        assert row["penalty_per_unavailable"] == pytest.approx(penalty, abs=1e-6)
        assert row["score_penalized"] == pytest.approx(
            raw - penalty * n_unavail, abs=1e-6
        )


def test_stage3_efficient_frontier_fc_04_part_2_failed_matchups():
    fixture = _load_fixture()
    _, out = _run_on_fixture(fixture)
    results, failures = out["results"], out["failures"]
    assert set(failures["status"].unique()) == {"failed"}
    assert set(failures["budget"].unique()) == {10000.0}
    assert set(failures["policy"].unique()) == {
        "optimizer", "value", "point", "random", "oracle",
    }
    assert failures["reason"].notna().all()
    assert (failures["reason"].str.len() > 0).all()
    combos = {
        (r["decision_id"], float(r["budget"]), r["policy"])
        for _, r in results.iterrows()
    } | {
        (r["decision_id"], float(r["budget"]), r["policy"])
        for _, r in failures.iterrows()
    }
    expected = {
        (f"fwd-0{i}-" + c, float(b), p)
        for i, c in enumerate(fixture["cutoffs"], start=1)
        for b in fixture["budgets"]
        for p in ("optimizer", "value", "point", "random", "oracle")
    }
    assert combos == expected


def test_stage3_efficient_frontier_fc_04_part_2_committed_real_run():
    with open(_RESULTS_PATH) as fh:
        artifact = json.load(fh)
    assert artifact["experiment"] == "u04-part2-epl-2015-16"
    assert artifact["model_config"] == "fc6_pool"
    assert artifact["seed"] == 0
    # The training pool is era-only: no Euro 2020 or ISL 2021/22 match
    # may appear in any decision's training set.
    assert artifact["training_pool"]["extra_pool_excluded"] == [
        "UEFA Euro 2020",
        "Indian Super League 2021/22",
    ]
    for entry in artifact["schedule"]:
        assert entry["retrain"] is True
        assert entry["train_competitions"] == [
            "Premier League 2015/16",
            "La Liga 2015/16",
            "Serie A 2015/16",
            "Ligue 1 2015/16",
            "Bundesliga 2015/16",
        ]
    # Every decision record hash verifies, and every decision respects
    # its cutoff: max input timestamp strictly before the cutoff, input
    # and window match IDs disjoint, wage snapshot at or before cutoff.
    assert len(artifact["decisions"]) == 3
    for rec in artifact["decisions"]:
        assert fb.verify_record_hash(rec) is True
        assert rec["max_input_timestamp"] < rec["cutoff"]
        assert rec["wage_snapshot_date"] <= rec["cutoff"]
        sched = next(
            s for s in artifact["schedule"]
            if s["decision_id"] == rec["decision_id"]
        )
        assert not set(rec["input_match_ids"]) & set(sched["window_match_ids"])
        assert sched["cache_key"] == rec["cache_key"]
    # Policy coverage is complete across decisions and budgets.
    seen = {
        (r["decision_id"], float(r["budget"]), r["policy"])
        for r in artifact["results"]
    } | {
        (r["decision_id"], float(r["budget"]), r["policy"])
        for r in artifact["failures"]
    }
    expected = {
        (d["decision_id"], float(b), p)
        for d in artifact["decisions"]
        for b in artifact["budgets"]
        for p in ("optimizer", "value", "point", "random", "oracle")
    }
    assert seen == expected
    # The comparison summary is present and claims no winner.
    assert "winner" not in artifact["comparison"]
    assert artifact["comparison"]["coverage"]["complete"] is True
    # Wages are the repo's FC24 snapshot, recorded honestly, not a
    # 2015-dated snapshot.
    assert "FC24" in artifact["wages"]["snapshot"]


def test_stage3_efficient_frontier_fc_04_part_2_fc6_untouched():
    with open(_LEDGER_PATH) as fh:
        ledger = json.load(fh)
    assert fb.verify_ledger(ledger, _REPO_ROOT) == []
    with open(os.path.join(_REPO_ROOT, "runs", "five_leagues", "results.json")) as fh:
        five = json.load(fh)
    assert five["model_hash"] == ledger["baseline"]["five_leagues"]["model_hash"]
    assert five["model_hash"] == "b86caaed47f745d4"
    with open(os.path.join(_REPO_ROOT, "demo", "backtest.csv")) as fh:
        assert fh.read() == ledger["baseline"]["demo_backtest_csv"]
