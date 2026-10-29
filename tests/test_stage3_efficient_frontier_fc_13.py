"""Stage 3 U13#1/2 regression tests: team synergy as a disciplined optional
research hypothesis, Part 1 (source contract, baseline capture, fixture and
independent oracle; no performance promotion).

The contract under test: a new pair-synergy module must first verify enough
repeated real coappearances to estimate out-of-sample pair effects after
controlling for team and player context. If coverage supports it, it estimates
heavily regularized pair contributions and linearizes chosen pair indicators
y_ij with y <= x_i, y <= x_j, y >= x_i + x_j - 1, caps interaction count and
solver complexity, and compares against a zero-synergy additive MILP. Only
synergy stable across independent periods may be retained; if no identifiable
signal exists, that is the reported result.

Before the U13#1/2 implementation these tests fail: the module does not exist
yet, so collection fails. After it, every assertion below must pass, including
the exact brute-force reconciliation of the synergy MILP and the exact
equality of the synergy-off selector with the additive selector on the demo
pool (demo/*.csv, read only).
"""

import itertools

import numpy as np
import pandas as pd
import pytest

from effc.models.pair_synergy import (
    chronological_periods,
    coappearance_counts,
    compare_zero_synergy,
    coverage_audit,
    enumerate_optimum,
    estimate_pair_effects,
    linearization_rows,
    run_synergy_experiment,
    select_squad_with_synergy,
    stability_check,
)
from effc.portfolio.portfolio import (
    POS_QUOTA,
    SQUAD_SIZE,
    _milp_select,
    select_squad,
)


def _panel():
    """Four matches x three players; hand-computed pair counts are all 2.

    m1: A, B. m2: A, B, C. m3: A, C. m4: B, C. NaN means did not play.
    """
    return pd.DataFrame(
        {
            "A": [1.0, 1.0, 1.0, np.nan],
            "B": [2.0, 2.0, np.nan, 2.0],
            "C": [np.nan, 3.0, 3.0, 3.0],
        },
        index=["m1", "m2", "m3", "m4"],
    )


def _counts_dict(df):
    return {(r.player_a, r.player_b): int(r.coappearances) for r in df.itertuples()}


def test_stage3_efficient_frontier_fc_13_part_1_coappearance_counts_known():
    counts = _counts_dict(coappearance_counts(_panel()))
    assert counts == {("A", "B"): 2, ("A", "C"): 2, ("B", "C"): 2}


def test_stage3_efficient_frontier_fc_13_part_1_coappearance_ignores_missing():
    panel = pd.DataFrame(
        {"A": [1.0, np.nan], "B": [np.nan, 2.0]},
        index=["m1", "m2"],
    )
    assert _counts_dict(coappearance_counts(panel)) == {("A", "B"): 0}


def test_stage3_efficient_frontier_fc_13_part_1_coverage_audit_requires_all_periods():
    # Period 1 = m1, m2; period 2 = m3, m4. At threshold 2 no pair qualifies
    # in both periods (AB is 2 then 0; AC and BC are 1 then 1).
    audit = coverage_audit(_panel(), min_coappear=2, n_periods=2)
    assert audit["decision"] == "insufficient"
    assert audit["qualifying_pairs"] == []
    # At threshold 1, AC and BC qualify in both periods; AB does not.
    audit = coverage_audit(_panel(), min_coappear=1, n_periods=2)
    assert audit["decision"] == "sufficient"
    assert sorted(audit["qualifying_pairs"]) == [("A", "C"), ("B", "C")]


def test_stage3_efficient_frontier_fc_13_part_1_coverage_audit_empty_fail_closed():
    audit = coverage_audit(_panel().iloc[0:0], min_coappear=1, n_periods=2)
    assert audit["decision"] == "insufficient"
    assert audit["qualifying_pairs"] == []
    assert audit["reason"] != ""


def test_stage3_efficient_frontier_fc_13_part_1_chronological_periods_split_in_order():
    periods = chronological_periods(_panel(), n_periods=2)
    assert [list(p.index) for p in periods] == [["m1", "m2"], ["m3", "m4"]]


def test_stage3_efficient_frontier_fc_13_part_1_estimate_pair_effects_shrinkage_and_cap():
    fit = pd.DataFrame(
        {"A": [1.0, 3.0], "B": [2.0, 4.0]}, index=["f1", "f2"]
    )
    ev = pd.DataFrame(
        {"A": [3.0, 3.0], "B": [5.0, 5.0]}, index=["e1", "e2"]
    )
    # Fit means: A 2.0, B 3.0, baseline 5.0. Eval joints are 8.0, raw +3.0,
    # n = 2, shrink 10 -> effect 3.0 * 2 / 12 = 0.5.
    eff = estimate_pair_effects(fit, ev, min_coappear=2, shrink=10.0)
    assert len(eff) == 1
    row = eff.iloc[0]
    assert (row.player_a, row.player_b) == ("A", "B")
    assert row.n_coappear == 2
    assert row.raw_effect == pytest.approx(3.0)
    assert row.effect == pytest.approx(0.5)
    # A cap binds: same inputs with cap 0.4.
    eff = estimate_pair_effects(fit, ev, min_coappear=2, shrink=10.0, cap=0.4)
    assert eff.iloc[0].effect == pytest.approx(0.4)


def test_stage3_efficient_frontier_fc_13_part_1_estimate_pair_effects_min_coappear_gate():
    fit = pd.DataFrame({"A": [1.0], "B": [2.0]}, index=["f1"])
    ev = pd.DataFrame({"A": [9.0], "B": [9.0]}, index=["e1"])
    eff = estimate_pair_effects(fit, ev, min_coappear=2, shrink=10.0)
    assert len(eff) == 0


def test_stage3_efficient_frontier_fc_13_part_1_stability_sign_agreement_and_floor():
    eff_a = pd.DataFrame(
        {
            "player_a": ["A", "A", "A"],
            "player_b": ["B", "C", "D"],
            "effect": [0.5, -0.3, 0.05],
        }
    )
    eff_b = pd.DataFrame(
        {
            "player_a": ["A", "A", "A"],
            "player_b": ["B", "C", "D"],
            "effect": [0.4, 0.2, 0.06],
        }
    )
    check = stability_check(eff_a, eff_b, floor=0.1)
    assert check["n_common"] == 3
    assert check["n_same_sign"] == 2
    assert check["sign_agreement"] == pytest.approx(2 / 3)
    # AB is stable (same sign, above floor in both). AC flipped sign, AD is
    # below the floor in the first period.
    assert check["stable_pairs"] == [("A", "B")]


def test_stage3_efficient_frontier_fc_13_part_1_linearization_rows_enforce_and_only():
    a, lb, ub = linearization_rows(2, [(0, 1)])
    # Variables are [x0, x1, y]. Feasible exactly when y == x0 AND x1.
    for x0 in (0, 1):
        for x1 in (0, 1):
            for y in (0, 1):
                vals = a @ np.array([x0, x1, y], dtype=float)
                feasible = bool(np.all(vals >= lb - 1e-9) and np.all(vals <= ub + 1e-9))
                assert feasible == (y == (x0 and x1))


def _micro_instance():
    """Six players, pick 3, one per position group; hand-computed optima.

    Additive optimum is players 1, 3, 5 (value 6). A +5.0 bonus on pair
    (0, 2) moves the synergy optimum to players 0, 2, 5 (value 9).
    """
    mu = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    wage = np.full(6, 100.0)
    groups = np.array(["G0", "G0", "G1", "G1", "G2", "G2"])
    quota = {"G0": 1, "G1": 1, "G2": 1}
    return mu, wage, groups, 1000.0, 3, quota


def test_stage3_efficient_frontier_fc_13_part_1_synergy_selector_matches_enumeration():
    mu, wage, groups, budget, n_pick, quota = _micro_instance()
    pairs = [(0, 2)]
    effects = np.array([5.0])
    sel = select_squad_with_synergy(
        mu, wage, groups, budget, n_pick, quota, pairs, effects
    )
    value, enum_sel = enumerate_optimum(
        mu, wage, groups, budget, n_pick, quota, pairs, effects
    )
    assert list(np.flatnonzero(sel)) == [0, 2, 5]
    assert list(np.flatnonzero(enum_sel)) == [0, 2, 5]
    assert value == pytest.approx(9.0)


def test_stage3_efficient_frontier_fc_13_part_1_synergy_zero_effects_equals_additive():
    mu, wage, groups, budget, n_pick, quota = _micro_instance()
    sel = select_squad_with_synergy(
        mu, wage, groups, budget, n_pick, quota, [], np.array([])
    )
    base = _milp_select(
        mu, wage, groups, budget, n_pick, quota, 0.0, np.zeros(6)
    )
    assert list(np.flatnonzero(sel)) == list(np.flatnonzero(base))


def test_stage3_efficient_frontier_fc_13_part_1_synergy_selector_rejects_bad_inputs():
    mu, wage, groups, budget, n_pick, quota = _micro_instance()
    with pytest.raises(ValueError):
        select_squad_with_synergy(
            mu, wage, groups, budget, n_pick, quota, [(0, 1)], np.array([np.nan])
        )
    with pytest.raises(ValueError):
        select_squad_with_synergy(
            mu, wage, groups, budget, n_pick, quota, [(0, 0)], np.array([1.0])
        )
    with pytest.raises(ValueError):
        select_squad_with_synergy(
            mu, wage, groups, budget, n_pick, quota,
            [(0, 1), (0, 1)], np.array([1.0, 1.0]),
        )
    with pytest.raises(ValueError):
        select_squad_with_synergy(
            mu, wage, groups, budget, n_pick, quota, [(0, 6)], np.array([1.0])
        )


def test_stage3_efficient_frontier_fc_13_part_1_compare_zero_synergy_reports_both():
    mu, wage, groups, budget, n_pick, quota = _micro_instance()
    out = compare_zero_synergy(
        mu, wage, groups, budget, n_pick, quota, [(0, 2)], np.array([5.0])
    )
    assert list(np.flatnonzero(out["additive_selection"])) == [1, 3, 5]
    assert list(np.flatnonzero(out["synergy_selection"])) == [0, 2, 5]
    assert out["additive_value"] == pytest.approx(6.0)
    assert out["synergy_value"] == pytest.approx(9.0)
    assert out["synergy_value"] >= out["additive_value"]


def test_stage3_efficient_frontier_fc_13_part_1_demo_baseline_unchanged_by_synergy_off():
    """Baseline capture: with no pair effects, the synergy selector must
    reproduce the additive selector exactly on the real demo pool."""
    players = pd.read_csv("demo/players.csv")
    budget = float(pd.read_csv("demo/frontier.csv")["budget"].iloc[0])
    base = select_squad(players, budget, SQUAD_SIZE, POS_QUOTA)
    mu = players["vaep_per90"].to_numpy(dtype=float)
    wage = players["wage_eur"].to_numpy(dtype=float)
    groups = players["pos_group"].to_numpy()
    sel = select_squad_with_synergy(
        mu, wage, groups, budget, SQUAD_SIZE, POS_QUOTA, [], np.array([])
    )
    assert sorted(players["player_id"].to_numpy()[sel]) == sorted(
        base["player_id"].to_numpy()
    )


# ---------------------------------------------------------------------------
# Part 2 (U13#2/2): bounded remaining functionality and experiments.
# The experiment runner ties the Part 1 pieces together end to end: coverage
# audit, cross-period effect estimates in both directions, the stability
# retention rule, and the zero-synergy comparison on identical selection
# inputs. The honest real-data result is the null: no match-level panel is
# stored in this repo, so nothing is retained from real data.
# ---------------------------------------------------------------------------


def _synergy_panel(bump):
    """Twelve matches x three players, six per period, identical structure.

    Per period: m1, m2: A+C; m3, m4: B+C; m5, m6: A+B+C. Alone values are
    A 1.0, B 2.0, C 3.0. When A and B play together each gains ``bump``
    (A 1.0 + bump, B 2.0 + bump). With bump 5.0 the cross-period raw AB
    effect is +5.0 in both directions; with bump 0.0 every effect is 0.
    """
    rows = []
    for _ in range(2):
        rows.append({"A": 1.0, "B": np.nan, "C": 3.0})
        rows.append({"A": 1.0, "B": np.nan, "C": 3.0})
        rows.append({"A": np.nan, "B": 2.0, "C": 3.0})
        rows.append({"A": np.nan, "B": 2.0, "C": 3.0})
        rows.append({"A": 1.0 + bump, "B": 2.0 + bump, "C": 3.0})
        rows.append({"A": 1.0 + bump, "B": 2.0 + bump, "C": 3.0})
    return pd.DataFrame(rows, index=[f"m{i + 1}" for i in range(12)])


def _selection_inputs():
    """Micro selection where pair (0, 2) flips the pick when its bonus is
    large: additive optimum is [1, 3, 5] (value 6); [0, 2, 5] needs a bonus
    above 2.0 to win. Player names map A -> 0, B -> 2."""
    mu, wage, groups, budget, n_pick, quota = _micro_instance()
    return {
        "players": ["A", "X", "B", "Y", "Z", "W"],
        "mu": mu,
        "wage": wage,
        "groups": groups,
        "budget": budget,
        "n_pick": n_pick,
        "quota": quota,
    }


def test_stage3_efficient_frontier_fc_13_part_2_experiment_retains_stable_pair():
    out = run_synergy_experiment(
        _synergy_panel(5.0),
        min_coappear=2,
        shrink=0.5,
        floor=0.1,
        selection=_selection_inputs(),
    )
    assert out["coverage"]["decision"] == "sufficient"
    assert out["retained_pairs"] == [("A", "B")]
    # Raw AB effect +5.0, n = 2, shrink 0.5 -> 5.0 * 2 / 2.5 = 4.0 in both
    # directions, so the retained mean effect is 4.0.
    assert out["retained_effects"][("A", "B")] == pytest.approx(4.0)
    comp = out["comparison"]
    assert list(np.flatnonzero(comp["additive_selection"])) == [1, 3, 5]
    assert list(np.flatnonzero(comp["synergy_selection"])) == [0, 2, 5]
    assert comp["additive_value"] == pytest.approx(6.0)
    assert comp["synergy_value"] == pytest.approx(8.0)
    assert "retained" in out["verdict"]


def test_stage3_efficient_frontier_fc_13_part_2_experiment_no_signal_retains_nothing():
    out = run_synergy_experiment(
        _synergy_panel(0.0),
        min_coappear=2,
        shrink=10.0,
        floor=0.1,
        selection=_selection_inputs(),
    )
    assert out["coverage"]["decision"] == "sufficient"
    assert out["retained_pairs"] == []
    assert out["retained_effects"] == {}
    comp = out["comparison"]
    # Nothing retained, so the synergy selection is exactly the additive one.
    assert list(np.flatnonzero(comp["synergy_selection"])) == list(
        np.flatnonzero(comp["additive_selection"])
    )
    assert comp["synergy_value"] == pytest.approx(comp["additive_value"])
    assert "nothing retained" in out["verdict"]


def test_stage3_efficient_frontier_fc_13_part_2_experiment_sign_flip_not_retained():
    # Period 1 has no bump, period 2 has the bump. Coverage is sufficient
    # (AB coappears twice per period), but the directional effects have
    # opposite signs, so stability rejects the pair.
    p1 = _synergy_panel(0.0).iloc[:6]
    p2 = _synergy_panel(5.0).iloc[6:]
    panel = pd.concat([p1, p2], ignore_index=True)
    out = run_synergy_experiment(
        panel, min_coappear=2, shrink=10.0, floor=0.1
    )
    assert out["coverage"]["decision"] == "sufficient"
    fwd = out["effects_forward"]
    rev = out["effects_reverse"]
    ab_fwd = fwd[(fwd.player_a == "A") & (fwd.player_b == "B")].iloc[0].effect
    ab_rev = rev[(rev.player_a == "A") & (rev.player_b == "B")].iloc[0].effect
    assert ab_fwd > 0 and ab_rev < 0
    assert out["retained_pairs"] == []
    assert out["comparison"] is None


def test_stage3_efficient_frontier_fc_13_part_2_experiment_insufficient_coverage():
    panel = pd.DataFrame(
        {"A": [1.0, 1.0], "B": [2.0, 2.0]}, index=["m1", "m2"]
    )
    out = run_synergy_experiment(panel, min_coappear=2, floor=0.1)
    assert out["coverage"]["decision"] == "insufficient"
    assert out["retained_pairs"] == []
    assert out["comparison"] is None
    # With selection inputs the comparison still runs, with no pairs, so it
    # reproduces the additive selection exactly.
    out = run_synergy_experiment(
        panel, min_coappear=2, floor=0.1, selection=_selection_inputs()
    )
    comp = out["comparison"]
    assert list(np.flatnonzero(comp["synergy_selection"])) == list(
        np.flatnonzero(comp["additive_selection"])
    )


def test_stage3_efficient_frontier_fc_13_part_2_experiment_empty_panel_fail_closed():
    out = run_synergy_experiment(pd.DataFrame(), min_coappear=2, floor=0.1)
    assert out["coverage"]["decision"] == "insufficient"
    assert out["retained_pairs"] == []
    assert out["retained_effects"] == {}
    assert out["comparison"] is None


def test_stage3_efficient_frontier_fc_13_part_2_experiment_rejects_bad_inputs():
    panel = _synergy_panel(5.0)
    with pytest.raises(ValueError):
        run_synergy_experiment(panel, n_periods=3)
    with pytest.raises(ValueError):
        run_synergy_experiment(panel, floor=-0.1)
    with pytest.raises(ValueError):
        run_synergy_experiment(panel, min_coappear=0)
    with pytest.raises(ValueError):
        run_synergy_experiment(panel, shrink=0.0)


def test_stage3_efficient_frontier_fc_13_part_2_experiment_skips_unmapped_pairs():
    sel = _selection_inputs()
    sel["players"] = ["A", "X", "Y", "Z", "W", "V"]  # no "B" in the pool
    out = run_synergy_experiment(
        _synergy_panel(5.0),
        min_coappear=2,
        shrink=0.5,
        floor=0.1,
        selection=sel,
    )
    assert out["retained_pairs"] == [("A", "B")]
    assert out["skipped_pairs"] == [("A", "B")]
    comp = out["comparison"]
    assert list(np.flatnonzero(comp["synergy_selection"])) == list(
        np.flatnonzero(comp["additive_selection"])
    )


def test_stage3_efficient_frontier_fc_13_part_2_real_data_audit_finds_no_panel():
    """The honest real-data audit: no match-level player panel is stored in
    this repo, so no real-data experiment can run. runs/five_leagues/
    results.json carries counts, scoreboards, frontiers, backtests and
    per-budget squad selections, never per-match per-player values."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    data = json.loads((root / "runs" / "five_leagues" / "results.json").read_text())
    assert set(data.keys()) == {
        "feature_version",
        "model_config",
        "seed",
        "budgets_eur",
        "leagues",
        "model_hash",
        "train_pool_matches",
        "elapsed_s",
    }
    found_match_id = []

    def walk(node):
        if isinstance(node, dict):
            if "match_id" in node:
                found_match_id.append(node["match_id"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    assert found_match_id == []
    for league in data["leagues"].values():
        assert set(league.keys()) == {
            "audit",
            "split",
            "scoreboard_test",
            "scoreboard_val_report_only",
            "wage_match_rate",
            "n_rated",
            "pm_matches",
            "frontier",
            "backtest",
            "squad_checks",
            "squads",
        }
        assert isinstance(league["audit"]["matches"], int)
        for squad in league["squads"].values():
            for row in squad["squad"]:
                assert set(row.keys()) == {
                    "player",
                    "team",
                    "pos_group",
                    "vaep_per90",
                    "wage_eur",
                }
