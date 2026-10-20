"""Dixon-Coles style team strengths as goal-model features.

The VAEP game-state features describe where an action happens, not who
is playing. Team identity is the signal that survives roster turnover:
clubs persist even when most of their players change, which is exactly
what the season-transfer evaluation (train 2015/16, score 2023/24)
punishes. This module fits the classic Dixon-Coles attack/defence
model on match results,

    log lambda_home = mu + home_adv + attack[home] - defence[away]
    log lambda_away = mu + attack[away] - defence[home]

by penalized Poisson maximum likelihood with exponential recency decay
on match dates, and exposes the fitted strengths as per-action features
for the goal models.

Leakage discipline: a DixonColes model is only ever fit on matches
from the caller's training pool. Scoring matches are featurized with
the frozen strengths (inference), never fit on. Teams missing from the
fit get league-average strengths (0.0 after centering).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

# Feature columns added to action frames (see features.design_matrix).
STRENGTH_COLS = [
    "st_att_for",
    "st_def_for",
    "st_att_opp",
    "st_def_opp",
    "st_home",
    "st_lam_for",
    "st_lam_opp",
]


def match_results(
    frames: list[pd.DataFrame], meta: pd.DataFrame
) -> pd.DataFrame:
    """One row per match: date, home/away team, goals from the frames.

    Goals are counted from the frames' is_goal flags, so the results
    always describe exactly the matches the caller pooled. Matches
    missing from `meta` are skipped (no fixture identity, no result).
    """
    meta_idx = meta.set_index("match_id")
    rows = []
    for f in frames:
        mid = int(f["match_id"].iloc[0])
        if mid not in meta_idx.index:
            continue
        m = meta_idx.loc[mid]
        goals = f[f["is_goal"]].groupby("team").size()
        home, away = m["home_team"], m["away_team"]
        rows.append(
            {
                "match_id": mid,
                "match_date": str(m["match_date"]),
                "home_team": home,
                "away_team": away,
                "home_goals": int(goals.get(home, 0)),
                "away_goals": int(goals.get(away, 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("match_date").reset_index(drop=True)


class DixonColes:
    """Fitted attack/defence strengths (plain data, picklable)."""

    def __init__(self, attack: dict, defence: dict, mu: float, home_adv: float):
        self.attack = {str(k): float(v) for k, v in attack.items()}
        self.defence = {str(k): float(v) for k, v in defence.items()}
        self.mu = float(mu)
        self.home_adv = float(home_adv)

    def att(self, team: str) -> float:
        return self.attack.get(str(team), 0.0)

    def dfn(self, team: str) -> float:
        return self.defence.get(str(team), 0.0)

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        lam_home = np.exp(
            self.mu + self.home_adv + self.att(home_team) - self.dfn(away_team)
        )
        lam_away = np.exp(self.mu + self.att(away_team) - self.dfn(home_team))
        return float(lam_home), float(lam_away)


def fit_dixon_coles(
    results: pd.DataFrame,
    half_life_days: float | None = 365.0,
    l2: float = 0.01,
    ref_date: str | None = None,
) -> DixonColes:
    """Penalized Poisson fit with exponential recency decay.

    Weights are 0.5 ** (age_days / half_life_days) with ages measured
    back from `ref_date` (default: the latest match date in `results`),
    so the most recent match weighs 1. `l2` ridge-shrinks attack and
    defence parameters toward the league average.
    """
    if results.empty:
        raise ValueError("no match results to fit Dixon-Coles on")
    teams = sorted(set(results["home_team"]) | set(results["away_team"]))
    tix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    dates = pd.to_datetime(results["match_date"])
    ref = pd.to_datetime(ref_date) if ref_date else dates.max()
    age_days = (ref - dates).dt.days.to_numpy(dtype=float)
    if half_life_days:
        w = 0.5 ** (age_days / float(half_life_days))
    else:
        w = np.ones(len(results))
    hi = results["home_team"].map(tix).to_numpy()
    ai = results["away_team"].map(tix).to_numpy()
    hg = results["home_goals"].to_numpy(dtype=float)
    ag = results["away_goals"].to_numpy(dtype=float)

    def unpack(x):
        return x[0], x[1], x[2 : 2 + n], x[2 + n : 2 + 2 * n]

    def nll_and_grad(x):
        mu, home, att, dfn = unpack(x)
        lam_h = np.exp(mu + home + att[hi] - dfn[ai])
        lam_a = np.exp(mu + att[ai] - dfn[hi])
        ll = w * (
            poisson.logpmf(hg, lam_h) + poisson.logpmf(ag, lam_a)
        )
        pen = l2 * (np.sum(att**2) + np.sum(dfn**2))
        nll = -float(np.sum(ll)) + pen
        # gradient of the Poisson log-likelihood terms
        gh = w * (hg - lam_h)  # d ll / d log lam_home
        ga = w * (ag - lam_a)
        g = np.zeros_like(x)
        g[0] = -float(np.sum(gh + ga))
        g[1] = -float(np.sum(gh))
        g_att = np.zeros(n)
        g_dfn = np.zeros(n)
        np.add.at(g_att, hi, gh)
        np.add.at(g_att, ai, ga)
        np.add.at(g_dfn, ai, -gh)
        np.add.at(g_dfn, hi, -ga)
        g[2 : 2 + n] = -g_att + 2.0 * l2 * att
        g[2 + n : 2 + 2 * n] = -g_dfn + 2.0 * l2 * dfn
        return nll, g

    x0 = np.zeros(2 + 2 * n)
    x0[0] = np.log(max(0.1, float(np.mean(np.concatenate([hg, ag])))))
    res = minimize(
        nll_and_grad, x0, jac=True, method="L-BFGS-B",
        options={"maxiter": 500},
    )
    mu, home, att, dfn = unpack(res.x)
    # identifiability: center strengths, absorb the shift into mu
    ca, cd = float(np.mean(att)), float(np.mean(dfn))
    att = att - ca
    dfn = dfn - cd
    mu = mu + ca - cd
    return DixonColes(
        attack={t: att[tix[t]] for t in teams},
        defence={t: dfn[tix[t]] for t in teams},
        mu=mu,
        home_adv=home,
    )


def attach_strength_features(
    df: pd.DataFrame, dc: DixonColes, home_team: str
) -> pd.DataFrame:
    """Add the STRENGTH_COLS columns to one match's action frame.

    `df` holds both teams' actions (one match). The actor's team is in
    the frame's `team` column; the opponent is the other team present.
    Strengths are looked up from the frozen `dc` fit: pure inference,
    the frame's own result is never used.
    """
    out = df.copy()
    teams = list(pd.unique(out["team"]))
    if len(teams) != 2:
        # not a two-team frame (empty or malformed): neutral features
        for c in STRENGTH_COLS:
            out[c] = 0.0
        return out
    t0, t1 = teams
    if home_team == t0:
        lh, la = dc.expected_goals(t0, t1)
        lam = {t0: lh, t1: la}
    elif home_team == t1:
        lh, la = dc.expected_goals(t1, t0)
        lam = {t1: lh, t0: la}
    else:
        # home team unknown: neutral venue, no home advantage applied
        lam = {
            t: float(np.exp(dc.mu + dc.att(t) - dc.dfn(o)))
            for t, o in ((t0, t1), (t1, t0))
        }
    actor = out["team"].to_numpy()
    opp = np.where(actor == t0, t1, t0)
    att = dc.attack
    dfn = dc.defence
    lam_for = np.array([lam[t] for t in actor])
    lam_opp = np.array([lam[t] for t in opp])
    out["st_att_for"] = np.array([att.get(t, 0.0) for t in actor])
    out["st_def_for"] = np.array([dfn.get(t, 0.0) for t in actor])
    out["st_att_opp"] = np.array([att.get(t, 0.0) for t in opp])
    out["st_def_opp"] = np.array([dfn.get(t, 0.0) for t in opp])
    out["st_home"] = (actor == home_team).astype(float)
    out["st_lam_for"] = lam_for
    out["st_lam_opp"] = lam_opp
    return out


def attach_to_frames(
    frames: list[pd.DataFrame], dc: DixonColes, meta: pd.DataFrame
) -> list[pd.DataFrame]:
    """attach_strength_features for a list of frames, keyed by meta."""
    home_of = dict(
        zip(meta["match_id"].astype(int), meta["home_team"].astype(str))
    )
    out = []
    for f in frames:
        mid = int(f["match_id"].iloc[0])
        out.append(attach_strength_features(f, dc, home_of.get(mid, "")))
    return out
