"""Stage 3 U12 regression tests: frontier stability, marginal value of budget
and interpretable tradeoffs.

The contract under test: a new diagnostics module over the exact MILP
squad selector must report exact budget breakpoints and roster turnover,
marginal integer value from finite adjacent-budget re-solves (never LP
relaxation duals), selection Jaccard similarity, player entry/exit, budget
slack, binding positional minima, and cheap versus high-impact value at
each point, plus selection frequency across training-side rating bootstrap
variants.

Before the U12 implementation these tests fail: the module does not exist
yet, so collection fails. After it, every assertion below must pass,
including the exact reconciliation of re-solved selections against the
derived demo squads (demo/*.csv, read only; scripts/make_demo.py itself
is never modified by this unit).
"""

import os

import numpy as np
import pandas as pd
import pytest

from effc.analysis.frontier_diagnostics import (
    binding_minima,
    breakpoint_intervals,
    breakpoints,
    check_feasibility,
    check_monotonicity,
    cheap_vs_high_impact,
    jaccard,
    marginal_value_table,
    reconcile_breakpoints,
    selection_frequency,
    solve_frontier,
)

QUOTA_11 = {"GK": 1, "DEF": 4, "MID": 3, "FWD": 2}


def _micro_pool():
    """Six players, squad of 3, one slot per position: optima hand-computed.

    Cheap trio costs 3000 (value 1.8); stars cost 9000/8000/7000.
    Budget 11000 -> g_star + d_cheap + f_cheap (value 2.8, unique optimum:
    the 2.6 and 2.4 alternatives lose). Budget 19000 -> g_star + d_star +
    f_cheap (value 3.6). Budget 27000 -> all stars (value 4.2).
    """
    return pd.DataFrame(
        [
            {"player_id": "g_cheap", "pos_group": "GK", "vaep_per90": 0.5, "wage_eur": 1000.0},
            {"player_id": "g_star", "pos_group": "GK", "vaep_per90": 1.5, "wage_eur": 9000.0},
            {"player_id": "d_cheap", "pos_group": "DEF", "vaep_per90": 0.6, "wage_eur": 1000.0},
            {"player_id": "d_star", "pos_group": "DEF", "vaep_per90": 1.4, "wage_eur": 8000.0},
            {"player_id": "f_cheap", "pos_group": "FWD", "vaep_per90": 0.7, "wage_eur": 1000.0},
            {"player_id": "f_star", "pos_group": "FWD", "vaep_per90": 1.3, "wage_eur": 7000.0},
        ]
    )


def _seeded_pool(seed=0, n_gk=6, n_def=20, n_mid=20, n_fwd=14):
    rng = np.random.RandomState(seed)
    rows = []
    for prefix, n, v_lo, v_hi, w_lo, w_hi in (
        ("g", n_gk, 0.2, 1.2, 4000, 20000),
        ("d", n_def, 0.2, 2.5, 5000, 60000),
        ("m", n_mid, 0.2, 2.5, 5000, 60000),
        ("f", n_fwd, 0.2, 2.5, 5000, 60000),
    ):
        group = {"g": "GK", "d": "DEF", "m": "MID", "f": "FWD"}[prefix]
        for i in range(n):
            rows.append(
                {
                    "player_id": f"{prefix}{i}",
                    "pos_group": group,
                    "vaep_per90": float(rng.uniform(v_lo, v_hi)),
                    "wage_eur": float(rng.uniform(w_lo, w_hi)),
                }
            )
    return pd.DataFrame(rows)


def test_stage3_efficient_frontier_fc_12_micro_exact_selections():
    pool = _micro_pool()
    res = solve_frontier(
        pool, [3000.0, 11000.0, 19000.0, 27000.0],
        squad_size=3, quota={"GK": 1, "DEF": 1, "FWD": 1},
    )
    assert res.infeasible_budgets == []
    assert [p.player_ids for p in res.points] == [
        ("d_cheap", "f_cheap", "g_cheap"),
        ("d_cheap", "f_cheap", "g_star"),
        ("d_star", "f_cheap", "g_star"),
        ("d_star", "f_star", "g_star"),
    ]
    assert [p.value for p in res.points] == pytest.approx([1.8, 2.8, 3.6, 4.2])
    assert [p.wage_bill for p in res.points] == pytest.approx(
        [3000.0, 11000.0, 18000.0, 24000.0]
    )
    assert [p.slack for p in res.points] == pytest.approx(
        [0.0, 0.0, 1000.0, 3000.0]
    )


def test_stage3_efficient_frontier_fc_12_micro_breakpoints_and_marginals():
    pool = _micro_pool()
    res = solve_frontier(
        pool, [3000.0, 11000.0, 19000.0, 27000.0],
        squad_size=3, quota={"GK": 1, "DEF": 1, "FWD": 1},
    )
    bps = breakpoints(res.points)
    assert len(bps) == 3
    assert bps[0]["entering"] == ("g_star",)
    assert bps[0]["leaving"] == ("g_cheap",)
    assert bps[1]["entering"] == ("d_star",)
    assert bps[1]["leaving"] == ("d_cheap",)
    assert bps[2]["entering"] == ("f_star",)
    assert bps[2]["leaving"] == ("f_cheap",)
    assert [b["jaccard"] for b in bps] == pytest.approx([0.5, 0.5, 0.5])
    # Marginal value is the finite difference of exact re-solves, never a
    # dual: (2.8-1.8)/8000, (3.6-2.8)/8000, (4.2-3.6)/8000.
    table = marginal_value_table(res.points)
    assert [t["marginal_per_eur"] for t in table] == pytest.approx(
        [0.000125, 0.0001, 0.000075]
    )
    assert [t["value_change"] for t in table] == pytest.approx([1.0, 0.8, 0.6])


def test_stage3_efficient_frontier_fc_12_monotonicity_and_feasibility():
    pool = _seeded_pool()
    res = solve_frontier(
        pool, [50000.0, 100000.0, 150000.0, 200000.0, 300000.0, 500000.0],
        squad_size=11, quota=QUOTA_11,
    )
    assert res.points, "expected feasible points on this grid"
    assert check_monotonicity(res.points) == []
    assert check_feasibility(res.points, QUOTA_11, 11) == []
    # Infeasible budgets are reported, not silently dropped from the record.
    res2 = solve_frontier(pool, [1000.0, 500000.0], squad_size=11, quota=QUOTA_11)
    assert res2.infeasible_budgets == [1000.0]
    assert len(res2.points) == 1


def test_stage3_efficient_frontier_fc_12_reconcile_exact():
    pool = _seeded_pool()
    res = solve_frontier(
        pool, [100000.0, 200000.0, 300000.0],
        squad_size=11, quota=QUOTA_11,
    )
    # Re-solving at every breakpoint budget reproduces the selection set
    # exactly; any mismatch is reported, never papered over.
    assert reconcile_breakpoints(pool, res.points, squad_size=11, quota=QUOTA_11) == []
    # Breakpoint intervals tile the grid: each distinct selection owns a
    # contiguous run of budgets, runs are in order, and together they
    # cover every point exactly once.
    intervals = breakpoint_intervals(res.points)
    assert sum(i["n_budgets"] for i in intervals) == len(res.points)
    runs = []
    for p in res.points:
        if not runs or runs[-1] != p.player_ids:
            runs.append(p.player_ids)
    assert [i["player_ids"] for i in intervals] == runs


def test_stage3_efficient_frontier_fc_12_jaccard_identities():
    assert jaccard(("a", "b"), ("a", "b")) == 1.0
    assert jaccard(("a",), ("b",)) == 0.0
    assert jaccard(("a", "b", "c"), ("b", "c", "d")) == pytest.approx(0.5)
    assert jaccard((), ()) == 1.0


def test_stage3_efficient_frontier_fc_12_binding_minima_and_cheap_high():
    pool = _micro_pool()
    res = solve_frontier(
        pool, [27000.0], squad_size=3, quota={"GK": 1, "DEF": 1, "FWD": 1}
    )
    point = res.points[0]
    # Every position sits exactly at its quota of 1: all binding.
    assert binding_minima(point, {"GK": 1, "DEF": 1, "FWD": 1}) == ["DEF", "FWD", "GK"]
    sel = pool[pool["player_id"].isin(point.player_ids)]
    stats = cheap_vs_high_impact(sel)
    assert stats["cheapest_id"] == "f_star"
    assert stats["cheapest_wage"] == pytest.approx(7000.0)
    assert stats["highest_value_id"] == "g_star"
    assert stats["highest_value"] == pytest.approx(1.5)
    # Best value per wage among the stars is the cheapest star per value.
    assert stats["best_value_per_wage_id"] == "f_star"


def test_stage3_efficient_frontier_fc_12_selection_frequency_sums():
    base = _seeded_pool()
    rng = np.random.RandomState(7)
    variants = []
    for _ in range(5):
        v = base.copy()
        v["vaep_per90"] = v["vaep_per90"] + rng.normal(0.0, 0.05, size=len(v))
        variants.append(v)
    freq = selection_frequency(
        variants, [200000.0, 500000.0], squad_size=11, quota=QUOTA_11
    )
    for budget in (200000.0, 500000.0):
        freqs = freq.frequencies[budget]
        # Frequencies are fractions over 5 variants; every variant selects
        # exactly 11 players, so the fractions sum to the squad size.
        assert sum(freqs.values()) == pytest.approx(11)
        assert all(abs(f * 5 - round(f * 5)) < 1e-9 for f in freqs.values())
        assert all(0.0 <= f <= 1.0 for f in freqs.values())
        assert 0.0 <= freq.mean_jaccard[budget] <= 1.0


def test_stage3_efficient_frontier_fc_12_demo_reconciliation():
    """Independent oracle on derived demo data (read only).

    Re-solving on demo/players.csv must reproduce demo/squads.csv
    selections exactly at every demo grid budget, and the summed values
    must match demo/frontier.csv. This is the pipeline's own output, not
    a fixture the unit controls.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    players = pd.read_csv(os.path.join(here, "demo", "players.csv"))
    frontier = pd.read_csv(os.path.join(here, "demo", "frontier.csv"))
    squads = pd.read_csv(os.path.join(here, "demo", "squads.csv"))
    budgets = [float(b) for b in frontier["budget"]]
    res = solve_frontier(players, budgets)
    assert res.infeasible_budgets == []
    assert len(res.points) == len(budgets)
    expected = {
        float(b): tuple(sorted(str(x) for x in g["player_id"]))
        for b, g in squads.groupby("budget")
    }
    for point, (_, frow) in zip(res.points, frontier.iterrows()):
        assert point.player_ids == expected[point.budget]
        assert point.value == pytest.approx(float(frow["exp_vaep_per90"]), abs=1e-6)
        assert point.wage_bill == pytest.approx(float(frow["wage_bill"]), abs=1e-6)
    assert check_monotonicity(res.points) == []
    assert check_feasibility(res.points, None, 23) == []
