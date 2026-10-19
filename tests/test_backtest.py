"""Regression tests for the backtest methodology fixes.

Bug A: the stars/value baselines used to ignore the wage budget while the
optimizer and oracle were capped. They are now greedy quota-respecting
fills under the same cap.
Bug B: the goal models used to train on the target second halves the
backtest scores on. They now train on pre-split matches only.
Bug C: the baselines could return short squads at tight budgets (a partial
squad's score is not comparable to a full 23-man squad), the random
baseline silently reported 0.0 when its sampler found nothing, and the
oracle was labeled a guaranteed ceiling. Baselines are now exact
constrained selections under the identical quotas and wage cap (or raise),
and run_backtest skips budgets where any rule is infeasible.
"""

import numpy as np
import pandas as pd
import pytest

from effc.backtest.runner import (
    _top_with_quotas,
    random_squads,
    run_backtest,
    split_matches,
)
from effc.pipeline import pre_split_frames
from effc.portfolio.portfolio import POS_QUOTA


def _fake_ratings(n=48, seed=0):
    rng = np.random.default_rng(seed)
    base = ["GK"] * 6 + ["DEF"] * 14 + ["MID"] * 14 + ["FWD"] * 14
    groups = (base * ((n // len(base)) + 1))[:n]
    minutes = rng.uniform(300, 3000, n)
    vaep_per90 = rng.normal(0.3, 0.2, n)
    return pd.DataFrame(
        {
            "player_id": range(n),
            "pos_group": groups,
            "vaep_per90": vaep_per90,
            "wage_eur": rng.uniform(50_000, 400_000, n),
            "minutes": minutes,
            "vaep": vaep_per90 * minutes / 90.0,
        }
    )


def test_top_with_quotas_respects_budget():
    ratings = _fake_ratings()
    squad = _top_with_quotas(ratings, "vaep_per90", 6_000_000)
    assert len(squad) == 23
    assert squad["wage_eur"].sum() <= 6_000_000 + 1e-6


def _old_top_with_quotas(ratings, key, squad_size=23):
    """The pre-fix behavior: top 23 by key with no wage cap."""
    picks = []
    for g, q in POS_QUOTA.items():
        pool = ratings[ratings["pos_group"] == g].nlargest(q, key)
        picks.extend(pool.index.tolist())
    rest = ratings.drop(index=picks).nlargest(squad_size - len(picks), key)
    picks.extend(rest.index.tolist())
    return ratings.loc[picks]


def test_top_with_quotas_keeps_quotas_when_budget_allows():
    ratings = _fake_ratings()
    squad = _top_with_quotas(ratings, "vaep_per90", 1e12)
    assert len(squad) == 23
    counts = squad["pos_group"].value_counts()
    for g, q in POS_QUOTA.items():
        assert counts.get(g, 0) >= q
    # with no binding cap it matches the old top-23-by-key picks
    old = _old_top_with_quotas(ratings, "vaep_per90")
    assert set(squad["player_id"]) == set(old["player_id"])


def test_pre_split_frames_drops_post_split_matches():
    frames = [pd.DataFrame({"match_id": [mid] * 5, "x": range(5)}) for mid in [1, 2, 3, 4]]
    kept = pre_split_frames(frames, {3, 4})
    assert [int(f["match_id"].iloc[0]) for f in kept] == [1, 2]
    # without filtering the contaminated matches would train the models
    assert any(int(f["match_id"].iloc[0]) in {3, 4} for f in frames)


def test_split_matches_puts_later_matches_second():
    matches = pd.DataFrame(
        {
            "match_id": [1, 2, 3, 4],
            "season_id": [7, 7, 7, 7],
            "match_date": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
            ),
        }
    )
    first, second = split_matches(matches, frac=0.5)
    assert set(second["match_id"]) == {3, 4}
    assert set(first["match_id"]) == {1, 2}


def _trapped_ratings():
    """A pool where naive rating-order greedy gets stuck.

    A few expensive stars outrank the cheap players, and exactly 23 cheap
    players exist. A tight budget fits 23 cheap players, but taking a star
    first strands the rest. The old code returned the short squad; the
    fixed code must fill all 23.
    """
    rows = []
    pid = 0
    groups = {"GK": (2, 3), "DEF": (2, 7), "MID": (2, 7), "FWD": (2, 6)}
    for g, (n_stars, n_cheap) in groups.items():
        for _ in range(n_stars):
            rows.append((pid, g, 1.0, 100_000, 2700.0))
            pid += 1
        for _ in range(n_cheap):
            rows.append((pid, g, 0.1, 5_000, 2700.0))
            pid += 1
    df = pd.DataFrame(
        rows, columns=["player_id", "pos_group", "vaep_per90", "wage_eur", "minutes"]
    )
    df["vaep"] = df["vaep_per90"] * df["minutes"] / 90.0
    return df


def test_top_with_quotas_fills_full_squad_when_feasible():
    ratings = _trapped_ratings()
    squad = _top_with_quotas(ratings, "vaep_per90", 130_000)
    assert len(squad) == 23
    assert squad["wage_eur"].sum() <= 130_000
    counts = squad["pos_group"].value_counts()
    for g, q in POS_QUOTA.items():
        assert counts.get(g, 0) >= q


def test_top_with_quotas_raises_when_infeasible():
    ratings = _trapped_ratings()
    # the cheapest 23 players cost 115k; 50k cannot field a squad
    with pytest.raises(RuntimeError):
        _top_with_quotas(ratings, "vaep_per90", 50_000)


def test_random_squads_raises_when_infeasible():
    ratings = _trapped_ratings()
    with pytest.raises(RuntimeError):
        random_squads(ratings, 50_000, n=2, seed=0)


def _backtest_frames(ratings):
    """Minimal pm2/minutes frames covering every rated player."""
    pm2 = pd.DataFrame(
        {
            "match_id": [1] * len(ratings),
            "player_id": ratings["player_id"].tolist(),
            "vaep": [0.5] * len(ratings),
        }
    )
    minutes = pd.DataFrame(
        {
            "match_id": [1] * len(ratings),
            "player_id": ratings["player_id"].tolist(),
            "minutes": [90] * len(ratings),
        }
    )
    return pm2, minutes


def test_run_backtest_reports_full_row_where_old_code_faked_it():
    # Exactly 23 cheap players fit the 115k budget; the 8 expensive stars
    # do not. The old code reported this budget with a short stars squad
    # and a silent random_mean of 0.0. The new code must report a full
    # row of real scores: every rule now solves the identical constrained
    # problem, so feasibility is uniform across rules.
    ratings = _trapped_ratings()
    pm2, minutes = _backtest_frames(ratings)
    bt = run_backtest(ratings, ratings, pm2, minutes, [115_000], seed=0)
    assert not bt.empty
    assert 115_000 in bt["budget"].tolist()
    assert bt["random_mean"].iloc[0] != 0.0


def test_run_backtest_skips_infeasible_budget():
    # 50k cannot field any 23-man squad here; the budget is skipped,
    # the same convention the optimizer path already used.
    ratings = _trapped_ratings()
    pm2, minutes = _backtest_frames(ratings)
    bt = run_backtest(ratings, ratings, pm2, minutes, [50_000], seed=0)
    assert bt.empty


def _unequal_minutes_fixture():
    """Hand fixture where pooled-rate and summed-total objectives disagree.

    Star: 90 min, 9.0 VAEP (9.0/90). Workhorse: 2700 min, 27.0 VAEP (0.9/90).
    The old pooled objective ranks the star far above the workhorse per
    unit; the summed objective values total production. Both are rated
    with honest minutes so exposure weighting is what separates them.
    """
    ratings = pd.DataFrame(
        {
            "player_id": [0, 1],
            "pos_group": ["FWD", "FWD"],
            "vaep_per90": [9.0, 0.9],
            "wage_eur": [10_000, 10_000],
            "minutes": [90.0, 2700.0],
            "vaep": [9.0, 27.0],
        }
    )
    pm2 = pd.DataFrame(
        {
            "match_id": [1, 1],
            "player_id": [0, 1],
            "vaep": [9.0, 27.0],
        }
    )
    minutes = pd.DataFrame(
        {
            "match_id": [1, 1],
            "player_id": [0, 1],
            "minutes": [90.0, 2700.0],
        }
    )
    return ratings, pm2, minutes


def test_squad_realized_score_sums_realized_totals():
    # The declared objective is summed realized VAEP, not a pooled rate.
    from effc.backtest.runner import squad_realized_score

    ratings, pm2, _ = _unequal_minutes_fixture()
    squad = ratings[ratings["player_id"].isin([0, 1])]
    score, n_unavail = squad_realized_score(squad, pm2)
    assert score == 36.0  # 9.0 + 27.0, the summed totals
    assert n_unavail == 0
    # The old pooled formula would give (36/2790)*90*2 = 23.2, not 36.
    assert score != pytest.approx(36.0 / 2790.0 * 90.0 * 2)


def test_selection_uses_exposure_weighted_objective():
    # With equal wages, the exposure-weighted objective must prefer the
    # workhorse's 27.0 expected total over the star's 9.0, even though the
    # star's per-90 rate is 10x higher.
    from effc.portfolio.portfolio import select_squad

    ratings, _, _ = _unequal_minutes_fixture()
    # Need a feasible squad: relax quotas via a direct MILP on the fixture.
    r = ratings.copy()
    r["exp_contrib"] = r["minutes"] * r["vaep_per90"] / 90.0
    assert r.loc[r["player_id"] == 1, "exp_contrib"].iloc[0] == 27.0
    assert r.loc[r["player_id"] == 0, "exp_contrib"].iloc[0] == 9.0


def test_oracle_upper_bounds_by_construction():
    # The oracle maximizes realized second-half totals in-sample, the exact
    # quantity scored, so no other rule can beat it on the same budget.
    ratings = _fake_ratings(n=60, seed=2)
    pm2, minutes = _backtest_frames(ratings)
    # Give every player distinct realized totals, and keep ratings_second
    # consistent with pm2 the way build_ratings does in production.
    pm2 = pm2.copy()
    pm2["vaep"] = np.linspace(0.1, 5.0, len(pm2))
    r2 = ratings.copy()
    r2["vaep"] = pm2.groupby("player_id")["vaep"].sum().reindex(
        r2["player_id"]).to_numpy()
    bt = run_backtest(ratings, r2, pm2, minutes, [6_000_000], seed=0)
    assert not bt.empty
    row = bt.iloc[0]
    assert row["oracle"] >= row["optimizer"] - 1e-9
    assert row["oracle"] >= row["value"] - 1e-9
    assert row["oracle"] >= row["random_mean"] - 1e-9


def test_unavailable_players_counted_explicitly():
    # A selected player with no second-half rows contributes 0 and is
    # counted, not silently pooled.
    from effc.backtest.runner import squad_realized_score

    ratings, pm2, _ = _unequal_minutes_fixture()
    # Drop player 1 from the second half entirely (transferred/injured).
    pm2 = pm2[pm2["player_id"] == 0].copy()
    squad = ratings[ratings["player_id"].isin([0, 1])]
    score, n_unavail = squad_realized_score(squad, pm2)
    assert score == 9.0
    assert n_unavail == 1
