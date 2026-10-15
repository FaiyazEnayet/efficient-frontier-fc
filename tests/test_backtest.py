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
    groups = (["GK"] * 6 + ["DEF"] * 14 + ["MID"] * 14 + ["FWD"] * 14)[:n]
    return pd.DataFrame(
        {
            "player_id": range(n),
            "pos_group": groups,
            "vaep_per90": rng.normal(0.3, 0.2, n),
            "wage_eur": rng.uniform(50_000, 400_000, n),
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
            rows.append((pid, g, 1.0, 100_000))
            pid += 1
        for _ in range(n_cheap):
            rows.append((pid, g, 0.1, 5_000))
            pid += 1
    return pd.DataFrame(
        rows, columns=["player_id", "pos_group", "vaep_per90", "wage_eur"]
    )


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
