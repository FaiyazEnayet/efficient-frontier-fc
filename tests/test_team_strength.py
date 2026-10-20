"""Dixon-Coles team strengths: fit sanity, feature attachment, and the
leakage property that strengths are fit only on the matches given."""

import numpy as np
import pandas as pd

from effc.vaep.features import FEATURE_COLS
from effc.vaep.team_strength import (
    STRENGTH_COLS,
    attach_strength_features,
    fit_dixon_coles,
    match_results,
)


def make_frame(mid, home, away, home_goals=1, away_goals=0, n=40, seed=0):
    """One match frame with both teams' actions and planted goals."""
    rng = np.random.default_rng(seed + mid)
    teams = [home if i % 2 == 0 else away for i in range(n)]
    is_goal = np.zeros(n, dtype=bool)
    home_idx = [i for i, t in enumerate(teams) if t == home]
    away_idx = [i for i, t in enumerate(teams) if t == away]
    for i in home_idx[:home_goals]:
        is_goal[i] = True
    for i in away_idx[:away_goals]:
        is_goal[i] = True
    df = pd.DataFrame(
        {
            "match_id": [mid] * n,
            "team": teams,
            "player_id": [1] * n,
            "player": ["p"] * n,
            "action": ["Pass"] * n,
            "prev_action": ["Pass"] * n,
            "is_goal": is_goal,
            "scores_incl": (rng.random(n) < 0.05).astype(float),
            "scores_excl": (rng.random(n) < 0.05).astype(float),
            "concedes_incl": (rng.random(n) < 0.05).astype(float),
            "concedes_excl": (rng.random(n) < 0.05).astype(float),
        }
    )
    for c in FEATURE_COLS:
        df[c] = rng.random(n)
    return df


def make_meta(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "match_id", "competition_id", "season_id", "season_name",
            "match_date", "home_team", "away_team",
        ],
    )


def league(seed=0):
    """Four teams, Strong beats Weak 3-0 repeatedly, home and away."""
    frames, rows = [], []
    mid = 0
    for rnd in range(6):
        for home, away, hg, ag in (
            ("Strong", "Weak", 3, 0),
            ("Weak", "Strong", 0, 3),
            ("Mid", "Weak", 1, 0),
            ("Strong", "Mid", 2, 0),
        ):
            mid += 1
            frames.append(make_frame(mid, home, away, hg, ag, seed=seed))
            rows.append(
                (mid, 9, 27, "2015/2016", f"2015-09-{mid:02d}", home, away)
            )
    return frames, make_meta(rows)


def test_match_results_counts_frame_goals():
    frames, meta = league()
    res = match_results(frames, meta)
    assert len(res) == len(frames)
    first = res.iloc[0]
    assert (first["home_team"], first["away_team"]) == ("Strong", "Weak")
    assert (first["home_goals"], first["away_goals"]) == (3, 0)
    assert res["home_goals"].sum() == 6 * (3 + 0 + 1 + 2)


def test_fit_ranks_strong_above_weak():
    frames, meta = league()
    dc = fit_dixon_coles(match_results(frames, meta), half_life_days=None)
    assert dc.att("Strong") > dc.att("Mid") > dc.att("Weak")
    lam_h, lam_a = dc.expected_goals("Strong", "Weak")
    assert lam_h > lam_a > 0
    # home advantage is positive in this league (home teams score more)
    lam_home_strong, _ = dc.expected_goals("Strong", "Mid")
    _, lam_away_strong = dc.expected_goals("Mid", "Strong")
    assert lam_home_strong > lam_away_strong


def test_unseen_team_is_league_average():
    frames, meta = league()
    dc = fit_dixon_coles(match_results(frames, meta), half_life_days=None)
    assert dc.att("NeverPlayed FC") == 0.0
    assert dc.dfn("NeverPlayed FC") == 0.0


def test_fit_is_deterministic():
    frames, meta = league()
    res = match_results(frames, meta)
    a = fit_dixon_coles(res, half_life_days=365.0)
    b = fit_dixon_coles(res, half_life_days=365.0)
    assert a.attack == b.attack and a.mu == b.mu


def test_attach_features_adds_columns_and_home_flag():
    frames, meta = league()
    dc = fit_dixon_coles(match_results(frames, meta), half_life_days=None)
    f = frames[0]  # Strong at home vs Weak
    out = attach_strength_features(f, dc, home_team="Strong")
    for c in STRENGTH_COLS:
        assert c in out.columns
    strong_rows = out[out["team"] == "Strong"]
    weak_rows = out[out["team"] == "Weak"]
    assert (strong_rows["st_home"] == 1.0).all()
    assert (weak_rows["st_home"] == 0.0).all()
    assert strong_rows["st_att_for"].iloc[0] > weak_rows["st_att_for"].iloc[0]
    assert (strong_rows["st_lam_for"] > strong_rows["st_lam_opp"]).all()
