"""Tests for the dashboard demo tables. Fast: reads demo/ CSVs only."""

import json
import os

import pytest

from effc.demo import load_demo
from effc.portfolio.portfolio import POS_QUOTA

METRICS = os.path.join("runs", "bundesliga_2seasons", "metrics.json")
HEADLINE_BUDGETS = [300_000.0, 500_000.0, 700_000.0, 943_000.0, 1_500_000.0]
RULES = ["optimizer", "stars", "value", "random_mean", "oracle"]


@pytest.fixture(scope="module")
def demo():
    return load_demo()


@pytest.fixture(scope="module")
def metrics():
    with open(METRICS) as f:
        return json.load(f)


def test_demo_tables_load(demo):
    for name in ("players", "frontier", "squads", "xi", "backtest"):
        assert name in demo
        assert len(demo[name]) > 0
    assert demo["meta"]["run_name"] == "bundesliga_2seasons"


def test_players_schema(demo, metrics):
    p = demo["players"]
    assert len(p) == metrics["n_rated"] == 268
    assert {
        "player_id", "player", "team", "pos_group",
        "vaep_per90", "wage_eur", "minutes",
    } <= set(p.columns)
    assert (p["wage_eur"] > 0).all()


def test_frontier_matches_pipeline(demo, metrics):
    fr = demo["frontier"]
    exp = {r["budget"]: r["exp_vaep_per90"] for r in metrics["frontier"]}
    for b in HEADLINE_BUDGETS:
        row = fr[fr["budget"] == b]
        assert len(row) == 1
        assert row["exp_vaep_per90"].iloc[0] == pytest.approx(exp[b], rel=1e-9)
        assert row["wage_bill"].iloc[0] <= b + 1e-6
    # more budget cannot hurt the optimum
    assert fr.sort_values("budget")["exp_vaep_per90"].is_monotonic_increasing


def test_squads_valid(demo):
    sq = demo["squads"]
    fr = demo["frontier"]
    assert set(fr["budget"]) == set(sq["budget"])
    for b, g in sq.groupby("budget"):
        assert len(g) == 23
        assert g["wage_eur"].sum() <= b + 1e-6
        counts = g["pos_group"].value_counts()
        for pos, q in POS_QUOTA.items():
            assert counts.get(pos, 0) >= q


def test_xi_formation(demo):
    xi = demo["xi"]
    assert len(xi) == 11
    counts = xi["pos_group"].value_counts()
    assert counts["GK"] == 1
    assert counts["DEF"] == 4
    assert counts["MID"] == 3
    assert counts["FWD"] == 3


def test_backtest_matches_metrics(demo, metrics):
    bt = demo["backtest"]
    assert set(bt["rule"]) == set(RULES)
    wide = bt.pivot(index="budget", columns="rule", values="score")
    for row in metrics["backtest"]:
        b = row["budget"]
        for rule in RULES:
            assert wide.loc[b, rule] == pytest.approx(row[rule], rel=1e-9)


def test_top10_matches_metrics(demo, metrics):
    top10 = demo["players"].nlargest(10, "vaep_per90")["player"].tolist()
    expected = [t["player"] for t in metrics["top10"]]
    assert top10 == expected


def test_load_demo_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_demo(tmp_path / "nope")
