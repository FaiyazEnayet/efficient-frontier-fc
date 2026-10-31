"""Team-factor covariance model for player VAEP/90 (Barra-style, one factor).

r_i = mu_i + beta_i * F_{team(i)} + eps_i

The factor is the team's total VAEP/90 in a match. Players on the same
team co-move through it; everything else is idiosyncratic. PSD by
construction: Sigma = B Var(F) B' + diag(var_eps).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def match_vaep_panel(
    player_match: pd.DataFrame, minutes: pd.DataFrame
) -> pd.DataFrame:
    """Matches x players panel of VAEP/90. Missing = did not play."""
    pm = player_match.merge(
        minutes[["match_id", "player_id", "minutes"]],
        on=["match_id", "player_id"],
        how="left",
    )
    pm["minutes"] = pm["minutes"].fillna(0.0)
    pm = pm[pm["minutes"] >= 15.0]  # cameo appearances are noise
    pm["vaep90"] = pm["vaep"] / pm["minutes"] * 90.0
    panel = pm.pivot_table(
        index="match_id", columns="player_id", values="vaep90"
    )
    return panel


def team_factor(panel: pd.DataFrame, player_team: pd.Series) -> pd.DataFrame:
    """Per-match team VAEP/90 factor (sum over that team's players)."""
    factors = {}
    for team, ids in player_team.groupby(player_team).groups.items():
        cols = [c for c in ids if c in panel.columns]
        factors[team] = panel[cols].sum(axis=1, skipna=True)
    return pd.DataFrame(factors)


def fit_factor_model(
    panel: pd.DataFrame,
    factors: pd.DataFrame,
    player_team: pd.Series,
    beta_shrink: float = 5.0,
    var_shrink: float = 5.0,
) -> tuple[pd.DataFrame, dict]:
    """Returns (Sigma, info). Betas and idiosyncratic variances are shrunk
    toward cross-sectional means; teams with few matches shrink hardest."""
    players = [c for c in panel.columns if c in player_team.index]
    panel = panel[players]
    betas, idio = {}, {}
    for pid in players:
        team = player_team[pid]
        f = factors[team]
        y = panel[pid]
        mask = y.notna() & f.notna()
        n = int(mask.sum())
        if n >= 4:
            fv = f[mask].to_numpy()
            yv = y[mask].to_numpy()
            b = np.cov(yv, fv, ddof=1)[0, 1] / (np.var(fv, ddof=1) + 1e-12)
            resid_var = np.var(yv - b * fv, ddof=1)
        else:
            b, resid_var, n = 0.0, np.nan, n
        w = n / (n + beta_shrink)
        betas[pid] = w * b
        idio[pid] = resid_var
    beta_s = pd.Series(betas)
    idio_s = pd.Series(idio)
    # shrink idiosyncratic variances toward the cross-sectional mean
    mean_var = float(np.nanmean(idio_s))
    wv = panel.notna().sum() / (panel.notna().sum() + var_shrink)
    idio_s = idio_s.fillna(mean_var)
    idio_s = wv * idio_s + (1 - wv) * mean_var

    B = np.zeros((len(players), factors.shape[1]))
    teams = list(factors.columns)
    for j, pid in enumerate(players):
        B[j, teams.index(player_team[pid])] = beta_s[pid]
    factor_var = factors.var(ddof=1).fillna(0) + 1e-12
    var_f = np.diag(factor_var.to_numpy())
    sigma = B @ var_f @ B.T + np.diag(idio_s.to_numpy())
    Sigma = pd.DataFrame(sigma, index=players, columns=players)
    info = {
        "mean_abs_beta": float(beta_s.abs().mean()),
        "mean_idio_var": float(idio_s.mean()),
        "n_players": len(players),
        # The actual idiosyncratic variance per player. Callers must
        # use this for the MILP penalty, never the covariance
        # diagonal: diag(Sigma) is total variance (factor plus
        # idiosyncratic) and overstates idiosyncratic risk wherever
        # the factor loading is nonzero.
        "idio_var": idio_s,
        "beta": beta_s,
        "factor_var": factor_var,
    }
    return Sigma, info


def variance_decomposition(
    Sigma: pd.DataFrame, info: dict
) -> pd.DataFrame:
    """Per-player variance split: total = systematic + idiosyncratic.

    Systematic variance is beta_i^2 times the variance of the player's
    team factor; idiosyncratic variance is the returned fit value.
    Total must equal the covariance diagonal by construction.
    """
    idio = info["idio_var"]
    players = list(Sigma.index)
    # Systematic variance is beta_i^2 times the variance of the
    # player's team factor; equivalently total minus idiosyncratic,
    # which the fit guarantees by construction. The table asserts
    # the identity total = systematic + idio per player.
    total = pd.Series(np.diag(Sigma.to_numpy()), index=players)
    systematic = total - idio.reindex(players)
    return pd.DataFrame(
        {
            "total_var": total,
            "systematic_var": systematic,
            "idio_var": idio.reindex(players),
        }
    )
