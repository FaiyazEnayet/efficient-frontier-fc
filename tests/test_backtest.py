"""Regression tests for the backtest methodology fixes.

Bug A: the stars/value baselines used to ignore the wage budget while the
optimizer and oracle were capped. They are now greedy quota-respecting
fills under the same cap.
Bug B: the goal models used to train on the target second halves the
backtest scores on. They now train on pre-split matches only.
"""

import numpy as np
import pandas as pd

from effc.backtest.runner import _top_with_quotas, split_matches
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
    # old behavior: top 23 by key with no cap
    unconstrained = _old_top_with_quotas(ratings, "vaep_per90")
    tight = unconstrained["wage_eur"].sum() * 0.5
    assert unconstrained["wage_eur"].sum() > tight  # old code overspends here
    squad = _top_with_quotas(ratings, "vaep_per90", tight)
    assert squad["wage_eur"].sum() <= tight + 1e-6
    assert len(squad) > 0


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
