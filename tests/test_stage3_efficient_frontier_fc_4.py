"""Stage 3 efficient-frontier-fc FC4: roster risk and wages robust.

Before this item, pipeline.main passed the covariance diagonal as
idio_var, conflating total variance with idiosyncratic variance, and
wage matching recorded no match type, so roster feasibility was never
checked against wage uncertainty. These tests pin the fix:

- the factor fit returns actual per-player idiosyncratic variance, and
  the returned covariance decomposes exactly as B Var(F) B^T + D;
- wage matches split into exact, fuzzy, proxy, and unmatched;
- wage uncertainty intervals widen from exact to fuzzy to proxy;
- rosters selected under high-wage stress meet every wage and position
  constraint under each predeclared scenario.
"""

import os

import numpy as np
import pandas as pd
import pytest

from effc.data.wages import (
    assign_proxy_wages,
    match_players,
    summarize_wage_matches,
    wage_uncertainty_intervals,
)
from effc.portfolio.portfolio import (
    build_wage_scenarios,
    check_roster_feasibility,
    describe_risk_objective,
    select_squad,
)
from effc.ratings.covariance import (
    fit_factor_model,
    team_factor,
    variance_decomposition,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _covariance_fixture():
    """Two teams, two players each, 20 matches, seeded noise.

    Player values are a team-factor loading plus independent noise, so
    systematic variance is genuinely nonzero and the covariance
    diagonal genuinely exceeds the idiosyncratic variance.
    """
    rng = np.random.default_rng(0)
    matches = list(range(20))
    factor_a = rng.normal(0.0, 1.0, size=20)
    factor_b = rng.normal(0.0, 1.0, size=20)
    panel = pd.DataFrame(
        {
            11: 0.8 * factor_a + rng.normal(0.0, 0.5, size=20),
            12: 0.4 * factor_a + rng.normal(0.0, 0.5, size=20),
            21: 0.9 * factor_b + rng.normal(0.0, 0.5, size=20),
            22: 0.3 * factor_b + rng.normal(0.0, 0.5, size=20),
        },
        index=matches,
    )
    player_team = pd.Series(
        {11: "A", 12: "A", 21: "B", 22: "B"}, name="team"
    )
    return panel, player_team


def test_stage3_efficient_frontier_fc_4_idio_var_returned():
    panel, player_team = _covariance_fixture()
    factors = team_factor(panel, player_team)
    Sigma, info = fit_factor_model(panel, factors, player_team)
    assert "idio_var" in info, "factor fit must return actual idiosyncratic variance"
    idio = info["idio_var"]
    assert list(idio.index) == list(Sigma.index)
    assert np.all(np.isfinite(idio.to_numpy()))
    assert np.all(idio.to_numpy() >= 0.0)


def test_stage3_efficient_frontier_fc_4_variance_decomposition_psd():
    panel, player_team = _covariance_fixture()
    factors = team_factor(panel, player_team)
    Sigma, info = fit_factor_model(panel, factors, player_team)
    # Rebuild B Var(F) B^T + D from the returned fit pieces.
    players = list(Sigma.index)
    teams = list(info["factor_var"].index)
    B = np.zeros((len(players), len(teams)))
    for j, pid in enumerate(players):
        B[j, teams.index(player_team[pid])] = info["beta"][pid]
    var_f = np.diag(info["factor_var"].to_numpy())
    rebuilt = B @ var_f @ B.T + np.diag(info["idio_var"].to_numpy())
    err = np.max(np.abs(Sigma.to_numpy() - rebuilt))
    assert err <= 1e-8, f"covariance does not decompose, max err {err}"
    eig = np.linalg.eigvalsh(Sigma.to_numpy())
    assert np.all(eig >= -1e-10), f"covariance not PSD: min eig {eig.min()}"
    # The old bug: diagonal-as-idio overstates idiosyncratic risk.
    diag = np.diag(Sigma.to_numpy())
    assert np.all(info["idio_var"].to_numpy() <= diag + 1e-12)
    assert np.any(info["idio_var"].to_numpy() < diag - 1e-6), (
        "fixture must separate total from idiosyncratic variance")


def test_stage3_efficient_frontier_fc_4_variance_decomposition_table():
    panel, player_team = _covariance_fixture()
    factors = team_factor(panel, player_team)
    Sigma, info = fit_factor_model(panel, factors, player_team)
    table = variance_decomposition(Sigma, info)
    for col in ("total_var", "systematic_var", "idio_var"):
        assert col in table.columns
    assert np.allclose(
        table["total_var"].to_numpy(),
        table["systematic_var"].to_numpy() + table["idio_var"].to_numpy(),
        atol=1e-10,
    )
    assert np.allclose(
        table["total_var"].to_numpy(), np.diag(Sigma.to_numpy()), atol=1e-12
    )


def _wage_frames():
    wages = pd.DataFrame(
        {
            "long_name": ["Alice Smith", "Bob Jones", "Carol White", "Dan Brown"],
            "short_name": ["A. Smith", "B. Jones", "C. White", "D. Brown"],
            "club_name": ["Red FC", "Red FC", "Blue FC", "Blue FC"],
            "positions": ["ST", "CM", "CB", "GK"],
            "age": [24, 27, 25, 30],
            "overall": [80, 78, 76, 74],
            "value_eur": [10_000_000.0, 8_000_000.0, 6_000_000.0, 4_000_000.0],
            "wage_eur": [100_000.0, 80_000.0, 60_000.0, 40_000.0],
        }
    )
    from effc.data.wages import norm, position_group

    wages["key"] = wages["long_name"].map(norm)
    wages["club_key"] = wages["club_name"].map(norm)
    wages["pos_group"] = wages["positions"].map(position_group)
    sb_players = pd.DataFrame(
        {
            "player": ["Alice Smith", "Bobby Jones", "Carol White", "Eve Green"],
            "team": ["Red FC", "Red FC", "Blue FC", "Blue FC"],
        }
    )
    return sb_players, wages


def test_stage3_efficient_frontier_fc_4_wage_match_types():
    sb_players, wages = _wage_frames()
    matched = match_players(sb_players, wages)
    by_player = dict(zip(matched["player"], matched["match_type"]))
    assert by_player["Alice Smith"] == "exact"
    assert by_player["Bobby Jones"] == "fuzzy"
    assert by_player["Carol White"] == "exact"
    assert by_player["Eve Green"] == "unmatched"
    split = summarize_wage_matches(matched)
    assert split["exact"] == 2
    assert split["fuzzy"] == 1
    assert split["unmatched"] == 1


def test_stage3_efficient_frontier_fc_4_proxy_and_intervals():
    sb_players, wages = _wage_frames()
    matched = match_players(sb_players, wages)
    proxied = assign_proxy_wages(matched)
    eve = proxied.loc[proxied["player"] == "Eve Green"].iloc[0]
    assert eve["match_type"] == "proxy"
    # Eve has no position group, so her proxy wage is the overall
    # matched median; the point is it is filled, flagged, and uncertain.
    assert np.isfinite(eve["wage_eur"])
    intervals = wage_uncertainty_intervals(proxied)
    by_player = intervals.set_index("player")
    for player in ("Alice Smith", "Bobby Jones", "Eve Green"):
        row = by_player.loc[player]
        assert row["wage_low"] <= row["wage_eur"] <= row["wage_high"]
    exact_width = (
        by_player.loc["Alice Smith", "wage_high"]
        - by_player.loc["Alice Smith", "wage_low"]
    ) / by_player.loc["Alice Smith", "wage_eur"]
    fuzzy_width = (
        by_player.loc["Bobby Jones", "wage_high"]
        - by_player.loc["Bobby Jones", "wage_low"]
    ) / by_player.loc["Bobby Jones", "wage_eur"]
    proxy_width = (
        by_player.loc["Eve Green", "wage_high"]
        - by_player.loc["Eve Green", "wage_low"]
    ) / by_player.loc["Eve Green", "wage_eur"]
    assert exact_width < fuzzy_width < proxy_width
    split = summarize_wage_matches(proxied)
    assert split["proxy"] == 1
    assert split["unmatched"] == 0


def _ratings_for_stress():
    """30 players, point wage 50,000, budget 1,250,000.

    Groups: GK 3, DEF 9, MID 10, FWD 8. Value falls with index, so the
    base selection takes indices 0..18 plus four FWD to meet the FWD
    minimum. Fuzzy matches are the indices divisible by 3: eight of
    them land in the base selection, pushing its high-wage bill to
    1,287,500, over budget. A robust selection under high wages can
    still fit: all 20 exact players (1,050,000) plus 3 fuzzy (187,500)
    cover every positional minimum within budget.
    """
    groups = ["GK"] * 3 + ["DEF"] * 9 + ["MID"] * 10 + ["FWD"] * 8
    rows = []
    for i, g in enumerate(groups):
        rows.append(
            {
                "player_id": 100 + i,
                "player": f"P{i}",
                "team": "T",
                "pos_group": g,
                "vaep_per90": 1.0 - 0.01 * i,
                "wage_eur": 50_000.0,
                "match_type": "fuzzy" if i % 3 == 0 else "exact",
            }
        )
    ratings = pd.DataFrame(rows)
    intervals = wage_uncertainty_intervals(ratings)
    return ratings.merge(
        intervals[["player_id", "wage_low", "wage_high"]],
        on="player_id",
        how="left",
    )


def test_stage3_efficient_frontier_fc_4_stress_scenarios():
    ratings = _ratings_for_stress()
    scenarios = build_wage_scenarios(ratings)
    assert set(scenarios) == {"base", "high", "fuzzy_high", "proxy_high"}
    budget = 1_250_000.0
    # Base selection fits the point-wage budget, but the high-wage
    # scenario breaks it: the stress check must say so.
    base_squad = select_squad(ratings, budget)
    base_check = check_roster_feasibility(base_squad, budget, scenarios)
    assert base_check["base"]["feasible"] is True
    assert base_check["high"]["feasible"] is False
    # Robust selection under high wages meets every scenario.
    high_ratings = ratings.copy()
    high_ratings["wage_eur"] = high_ratings["wage_high"]
    robust = select_squad(high_ratings, budget)
    robust_check = check_roster_feasibility(robust, budget, scenarios)
    assert len(robust) == 23
    for name, result in robust_check.items():
        assert result["feasible"] is True, f"robust roster fails {name}"
        assert result["meets_quotas"] is True
        assert result["within_budget"] is True


def test_stage3_efficient_frontier_fc_4_risk_objective_reported():
    desc = describe_risk_objective(gamma=0.5)
    assert "idiosyncratic" in desc["objective"]
    assert "ex post" in desc["objective"] or "ex-post" in desc["objective"]
    assert desc["risk_term"] == "linear_idio_var_penalty"
    assert desc["full_covariance"] == "ex_post_squad_risk"


def test_stage3_efficient_frontier_fc_4_pipeline_uses_idio_not_diag():
    text = open(os.path.join(REPO_ROOT, "src", "effc", "pipeline.py")).read()
    assert 'cov_info["idio_var"]' in text
    assert "np.diag(Sigma.to_numpy())" not in text
    five = open(
        os.path.join(REPO_ROOT, "scripts", "run_five_leagues.py")).read()
    assert 'cov_info["idio_var"]' in five
    assert "np.diag(sigma.to_numpy())" not in five
