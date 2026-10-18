"""End-to-end pipeline: data -> VAEP -> ratings -> frontier -> backtest.

    python -m effc.pipeline [config]

Writes metrics.json + plots to runs/<name>/.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import yaml

from effc.data.download import DATA_DIR, MODEL_POOL, TARGET
from effc.data.statsbomb import list_matches, load_events, minutes_played
from effc.data.wages import load_wages, match_players
from effc.ratings.covariance import fit_factor_model, match_vaep_panel, team_factor
from effc.ratings.ratings import build_ratings
from effc.vaep.features import FEATURE_VERSION, featurize_match
from effc.vaep.models import brier_score, train_goal_models
from effc.vaep.values import add_vaep, player_match_vaep

CACHE = os.path.join(DATA_DIR, "cache")
# Bump to invalidate every VAEP cache when the cached schema changes.
CACHE_VERSION = 2
LEGACY_VAEP_CACHE = "target_vaep.pkl"


def _cache_get(name):
    p = os.path.join(CACHE, name)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def _cache_put(name, obj):
    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, name), "wb") as f:
        pickle.dump(obj, f)


def _model_hash(models) -> str:
    """Stable hash of the fitted goal models' pickled bytes.

    Different seeds, features or training data produce different fitted
    models, hence different hashes. A new library version may also
    change the bytes, which safely misses the cache instead of reusing
    stale values.
    """
    return hashlib.sha256(pickle.dumps(models, protocol=4)).hexdigest()[:16]


def vaep_cache_key(models, train_match_ids, config=None) -> str:
    """Versioned, content-addressed cache filename for target VAEP.

    The key binds the sorted training match IDs, the feature schema
    version, the run config and the fitted model hash, so changing any
    of them invalidates the cache while identical inputs reuse it.
    """
    h = hashlib.sha256()
    h.update(json.dumps(sorted(int(m) for m in train_match_ids)).encode())
    h.update(str(FEATURE_VERSION).encode())
    h.update(json.dumps(config or {}, sort_keys=True, default=str).encode())
    h.update(_model_hash(models).encode())
    return f"target_vaep_v{CACHE_VERSION}_{h.hexdigest()[:16]}.pkl"


def _reject_legacy_cache():
    """The old unversioned target_vaep.pkl is never trusted: it cannot
    say which model produced it. Leave the file alone, but refuse to
    read it."""
    legacy = os.path.join(CACHE, LEGACY_VAEP_CACHE)
    if os.path.exists(legacy):
        print(
            f"ignoring legacy unversioned cache {LEGACY_VAEP_CACHE}; "
            "recomputing under a versioned key"
        )


def pre_split_frames(
    frames: list[pd.DataFrame], post_split_ids: set[int]
) -> list[pd.DataFrame]:
    """Keep only frames from matches before the backtest split.

    The backtest scores squads on second-half matches, so the goal
    models must not train on them.
    """
    return [f for f in frames if int(f["match_id"].iloc[0]) not in post_split_ids]


def featurize_pool() -> list[pd.DataFrame]:
    cached = _cache_get("pool_frames.pkl")
    if cached is not None:
        print(f"loaded {len(cached)} cached action frames")
        return cached
    matches = list_matches(MODEL_POOL)
    frames = []
    for i, mid in enumerate(matches["match_id"].tolist()):
        try:
            events = load_events(int(mid))
        except FileNotFoundError:
            continue
        df = featurize_match(events, match_id=int(mid))
        if not df.empty:
            frames.append(df)
        if (i + 1) % 25 == 0:
            print(f"featurized {i + 1}/{len(matches)} matches")
    _cache_put("pool_frames.pkl", frames)
    total = sum(len(f) for f in frames)
    print(f"featurized {len(frames)} matches, {total} actions")
    return frames


def compute_target_vaep(models, train_match_ids, config=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (player_match vaep, minutes, matches) for the target competition.

    Results are cached under a versioned key derived from the training
    match IDs, feature version, config and model hash. A cache entry
    from a different model, seed, split or feature set is never reused.
    """
    _reject_legacy_cache()
    key = vaep_cache_key(models, train_match_ids, config)
    cached = _cache_get(key)
    if cached is not None:
        print(f"loaded cached target VAEP ({key})")
        return cached
    matches = list_matches(TARGET)
    pm_all, min_all = [], []
    for i, mid in enumerate(matches["match_id"].tolist()):
        try:
            events = load_events(int(mid))
        except FileNotFoundError:
            continue
        df = featurize_match(events, match_id=int(mid))
        if df.empty:
            continue
        df = add_vaep(df, models)
        pm_all.append(player_match_vaep(df))
        min_all.append(minutes_played(int(mid)))
        if (i + 1) % 10 == 0:
            print(f"valued {i + 1}/{len(matches)} target matches")
    pm = pd.concat(pm_all, ignore_index=True)
    mins = pd.concat(min_all, ignore_index=True)
    _cache_put(key, (pm, mins, matches))
    return pm, mins, matches


def main(config_path: str = "configs/bundesliga_2324.yaml"):
    t0 = time.time()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    outdir = os.path.join("runs", cfg.get("run_name", "run"))
    os.makedirs(outdir, exist_ok=True)

    # target matches and their chronological split, shared by model
    # training (pre-split only) and the backtest in step 7
    from effc.backtest.runner import split_matches

    target_matches = list_matches(TARGET)
    first, second = split_matches(target_matches, frac=cfg.get("backtest_split", 0.5))
    post_split_ids = set(second["match_id"].tolist())

    # 1. goal-probability models on pre-split matches only. Training on
    # the full pool would include the target second halves the backtest
    # scores on.
    frames = featurize_pool()
    frames = pre_split_frames(frames, post_split_ids)
    print(f"training goal models on {len(frames)} pre-split matches")
    models = train_goal_models(frames, seed=cfg.get("seed", 0))
    full = pd.concat(frames, ignore_index=True)
    from effc.vaep.models import predict_proba

    preds = predict_proba(models, full)
    brier_s = brier_score(full["scores_incl"].to_numpy(), preds["scores_incl"])
    brier_c = brier_score(full["concedes_incl"].to_numpy(), preds["concedes_incl"])
    print(f"goal models: Brier scores={brier_s:.4f} concedes={brier_c:.4f}")

    # 2. VAEP on the target competition. The cache key binds the training
    # matches, feature version, config and model hash.
    train_ids = sorted({int(f["match_id"].iloc[0]) for f in frames})
    pm, mins, _ = compute_target_vaep(models, train_ids, config=cfg)

    # 3. wages + ratings
    wages = load_wages()
    sb_players = (
        pm[["player", "team"]].drop_duplicates().reset_index(drop=True)
    )
    wmap = match_players(sb_players, wages)
    print(f"wage match rate: {wmap['matched'].mean():.1%}")
    ratings = build_ratings(
        pm, mins, wmap,
        min_minutes=cfg.get("min_minutes", 270),
        shrink_k=cfg.get("shrink_k", 900),
    )
    print(f"rated players: {len(ratings)}")

    # 4. factor covariance on target matches
    panel = match_vaep_panel(pm, mins)
    player_team = ratings.set_index("player_id")["team"]
    factors = team_factor(panel, player_team)
    Sigma, cov_info = fit_factor_model(panel, factors, player_team)
    print("covariance:", cov_info)

    # 5. efficient frontier
    from effc.portfolio.portfolio import (
        efficient_frontier,
        select_squad,
        select_xi,
        squad_risk,
    )

    budgets = np.array(cfg.get("budgets_eur", [50e6, 100e6, 200e6]), dtype=float)
    idio = pd.Series(np.diag(Sigma.to_numpy()), index=Sigma.index)
    frontier = efficient_frontier(
        ratings, budgets, gamma=cfg.get("gamma", 0.0), idio_var=idio
    )
    frontier["squad_risk"] = [
        squad_risk(
            select_squad(ratings, float(b), gamma=cfg.get("gamma", 0.0), idio_var=idio),
            Sigma,
        )
        for b in frontier["budget"]
    ]
    print(frontier[["budget", "exp_vaep_per90", "wage_bill"]].to_string(index=False))

    # 6. best XI at a headline budget
    xi_budget = float(cfg.get("xi_budget_eur", 150e6))
    xi_pool = select_squad(ratings, xi_budget, gamma=cfg.get("gamma", 0.0), idio_var=idio)
    xi = select_xi(xi_pool, formation=cfg.get("formation", "4-3-3"))
    print("best XI:\n", xi[["player", "team", "pos_group", "vaep_per90", "wage_eur"]].to_string(index=False))

    # 7. backtest on the date split from step 1
    from effc.backtest.runner import run_backtest

    pm1 = pm[pm["match_id"].isin(first["match_id"])]
    pm2 = pm[pm["match_id"].isin(second["match_id"])]
    r1 = build_ratings(pm1, mins, wmap, min_minutes=60, shrink_k=cfg.get("shrink_k", 900))
    r2 = build_ratings(pm2, mins, wmap, min_minutes=60, shrink_k=cfg.get("shrink_k", 900))
    bt = run_backtest(r1, r2, pm2, mins, [float(b) for b in budgets[:3]], seed=cfg.get("seed", 0))
    print(bt.to_string(index=False))

    # 8. plots + metrics
    from effc.viz.plots import plot_backtest, plot_frontier, plot_vaep_leaders

    try:
        plot_frontier(frontier, outdir)
        plot_backtest(bt, outdir)
        plot_vaep_leaders(ratings, outdir)
    except Exception as exc:
        print(f"[warn] plotting failed: {exc}")

    metrics = {
        "config": config_path,
        "train_time_s": time.time() - t0,
        "brier_scores": brier_s,
        "brier_concedes": brier_c,
        "wage_match_rate": float(wmap["matched"].mean()),
        "n_rated": len(ratings),
        "cov_info": cov_info,
        "frontier": frontier.to_dict("records"),
        "backtest": bt.to_dict("records"),
        "xi": xi[["player", "team", "pos_group", "vaep_per90", "wage_eur"]].to_dict("records"),
        "top10": ratings.nlargest(10, "vaep_per90")[
            ["player", "team", "pos_group", "vaep_per90", "wage_eur"]
        ].to_dict("records"),
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"\nsaved metrics + plots to {outdir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/bundesliga_2324.yaml")
