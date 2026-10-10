"""Tests for Efficient Frontier FC. Fast, no data downloads."""

import numpy as np
import pandas as pd
import pytest

from effc.data.wages import norm, position_group
from effc.portfolio.portfolio import efficient_frontier, select_squad, select_xi
from effc.ratings.ratings import build_ratings
from effc.vaep.features import design_matrix, featurize_match


def _ev(idx, type_name, team="A", player="P1", pid=1, loc=(50.0, 34.0), minute=10):
    return {
        "id": str(idx),
        "index": idx,
        "period": 1,
        "timestamp": "00:10:00.000",
        "minute": minute,
        "second": 0,
        "type": {"id": 1, "name": type_name},
        "possession": 1,
        "possession_team": {"id": 1, "name": team},
        "team": {"id": 1, "name": team},
        "player": {"id": pid, "name": player},
        "location": list(loc),
    }


def test_featurize_labels_goal_within_k():
    # pass, then a shot that scores 3 actions later.
    # incl labels count the action's own goal; excl labels do not.
    events = [
        _ev(1, "Pass", loc=(60.0, 30.0)),
        _ev(2, "Carry", loc=(70.0, 30.0)),
        _ev(3, "Dribble", loc=(80.0, 34.0)),
    ]
    shot = _ev(4, "Shot", loc=(85.0, 34.0))
    shot["shot"] = {"outcome": {"name": "Goal"}, "end_location": [105.0, 34.0]}
    events.append(shot)
    df = featurize_match(events, match_id=1)
    assert len(df) == 4
    assert df.loc[0, "scores_incl"] == 1.0
    assert df.loc[0, "scores_excl"] == 1.0
    assert df.loc[3, "scores_incl"] == 1.0  # the goal counts for its own state
    assert df.loc[3, "scores_excl"] == 0.0  # but not for the after-state
    assert df.loc[0, "concedes_incl"] == 0.0


def test_featurize_concedes_opponent_goal():
    events = [_ev(1, "Pass", team="A", loc=(60.0, 30.0))]
    shot = _ev(2, "Shot", team="B", player="P9", pid=9, loc=(85.0, 34.0))
    shot["shot"] = {"outcome": {"name": "Goal"}, "end_location": [105.0, 34.0]}
    events.append(shot)
    df = featurize_match(events, match_id=1)
    assert df.loc[0, "concedes_incl"] == 1.0
    assert df.loc[0, "scores_incl"] == 0.0


def test_design_matrix_numeric():
    events = [_ev(1, "Pass"), _ev(2, "Carry")]
    df = featurize_match(events, match_id=1)
    X = design_matrix(df)
    assert X.shape[0] == 2
    assert X.dtypes.apply(lambda d: np.issubdtype(d, np.number)).all()


def test_vaep_values_shift():
    from effc.vaep.values import add_vaep

    events = [_ev(i, "Pass", loc=(50 + i, 34.0)) for i in range(1, 6)]
    df = featurize_match(events, match_id=7)

    class FakeModels(dict):
        pass

    # constant models: ps_after = 0.02 for all -> vaep_off == 0 except first row
    import effc.vaep.models as M

    orig = M.predict_proba
    M.predict_proba = lambda models, d: {k: np.full(len(d), v) for k, v in
        [("scores_incl", 0.02), ("scores_excl", 0.02),
         ("concedes_incl", 0.01), ("concedes_excl", 0.01)]}
    try:
        out = add_vaep(df, {})
    finally:
        M.predict_proba = orig
    assert np.allclose(out["vaep_off"].iloc[1:], 0.0)
    assert "vaep" in out.columns


def test_ratings_shrinkage():
    pm = pd.DataFrame(
        {
            "match_id": [1, 1, 1, 2, 2, 2],
            "team": ["A"] * 6,
            "player_id": [1, 2, 3, 1, 2, 3],
            "player": ["P1", "P2", "P3", "P1", "P2", "P3"],
            "vaep": [0.5, -0.1, -0.4, 0.3, 0.0, -0.2],
            "vaep_off": [0.4, 0.0, -0.3, 0.2, 0.0, -0.1],
            "vaep_def": [0.1, -0.1, -0.1, 0.1, 0.0, -0.1],
            "actions": [50, 40, 30, 45, 42, 33],
        }
    )
    mins = pd.DataFrame(
        {
            "match_id": [1, 1, 1, 2, 2, 2],
            "player_id": [1, 2, 3, 1, 2, 3],
            "minutes": [90.0] * 6,
            "position": ["Striker", "Midfielder", "Striker"] * 2,
        }
    )
    wmap = pd.DataFrame(
        {
            "player": ["P1", "P2", "P3"],
            "team": ["A"] * 3,
            "wage_eur": [100000.0, 50000.0, 40000.0],
            "value_eur": [50e6, 20e6, 10e6],
            "pos_group": ["FWD", "MID", "FWD"],
            "matched": [True] * 3,
        }
    )
    r = build_ratings(pm, mins, wmap, min_minutes=100, shrink_k=900)
    assert len(r) == 3
    # P1 raw = 0.4/90; group (FWD) mean is lower -> shrinkage pulls P1 down
    p1 = r[r["player"] == "P1"].iloc[0]
    assert p1["vaep_per90"] < p1["vaep_per90_raw"]
    assert p1["vaep_per90"] > r[r["player"] == "P3"]["vaep_per90"].iloc[0]
    assert p1["shrink_w"] == pytest.approx(180 / (180 + 900))


def _fake_ratings(n=30, seed=0):
    rng = np.random.default_rng(seed)
    groups = (["GK"] * 3 + ["DEF"] * 10 + ["MID"] * 10 + ["FWD"] * 7)[:n]
    return pd.DataFrame(
        {
            "player_id": np.arange(n),
            "player": [f"P{i}" for i in range(n)],
            "team": ["A"] * n,
            "pos_group": groups,
            "vaep_per90": rng.normal(0.05, 0.15, n),
            "wage_eur": rng.uniform(20_000, 300_000, n),
            "minutes": np.full(n, 900.0),
        }
    )


def test_select_squad_respects_budget_and_size():
    ratings = _fake_ratings()
    squad = select_squad(ratings, budget=5_000_000, squad_size=23)
    assert len(squad) == 23
    assert squad["wage_eur"].sum() <= 5_000_000 + 1e-6
    assert (squad["pos_group"] == "GK").sum() >= 2


def test_select_xi_formation():
    ratings = _fake_ratings()
    xi = select_xi(ratings, formation="4-3-3")
    counts = xi["pos_group"].value_counts()
    assert len(xi) == 11
    assert counts["GK"] == 1 and counts["DEF"] == 4
    assert counts["MID"] == 3 and counts["FWD"] == 3


def test_frontier_monotone_budgets():
    ratings = _fake_ratings()
    fr = efficient_frontier(ratings, np.array([4_000_000.0, 6_000_000.0, 9_000_000.0]))
    assert len(fr) == 3
    # more budget cannot hurt the optimum
    assert fr["exp_vaep_per90"].is_monotonic_increasing


def test_norm_and_positions():
    assert norm("Kylian Mbappé Lottin") == "kylian mbappe lottin"
    assert norm("E. Haaland") == "e haaland"
    assert position_group("ST, LW") == "FWD"
    assert position_group("CDM") == "MID"
    assert position_group("GK") == "GK"
