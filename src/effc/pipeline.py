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
from effc.vaep import calibration as cal
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


def _featurize_one(mid: int):
    try:
        events = load_events(int(mid))
    except Exception:
        return None
    df = featurize_match(events, match_id=int(mid))
    return df if not df.empty else None


def featurize_matches(match_ids, cache_name: str) -> list[pd.DataFrame]:
    """Featurize an arbitrary match list in parallel, cached as a whole.

    Used for the pooled training competitions: pooled matches are
    training data only, so a single cache entry per pool is enough.
    """
    cached = _cache_get(cache_name)
    if cached is not None:
        print(f"loaded {len(cached)} cached frames ({cache_name})")
        return cached
    import multiprocessing

    ids = sorted({int(m) for m in match_ids})
    with multiprocessing.Pool(processes=min(8, os.cpu_count() or 1)) as pool:
        results = pool.map(_featurize_one, ids, chunksize=8)
    frames = [r for r in results if r is not None]
    total = sum(len(f) for f in frames)
    print(f"featurized {len(frames)}/{len(ids)} pooled matches, {total} actions")
    _cache_put(cache_name, frames)
    return frames


_POOL_MEMO: dict = {}


def _memoized(key, loader):
    if key not in _POOL_MEMO:
        _POOL_MEMO[key] = loader()
    return _POOL_MEMO[key]


def extra_pool() -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """EXTRA_POOL + ERA_2015_POOL: goal-model training data only.

    The main models pool both; the season-transfer model uses only
    era2015_pool(). Union of the two caches, no re-featurization.
    Memoized per process: the era frames are shared, not reloaded.
    """
    return _memoized("extra_pool", _extra_pool_impl)


def _extra_pool_impl() -> tuple[list[pd.DataFrame], pd.DataFrame]:
    from effc.data.download import EXTRA_POOL

    era_frames, era_meta = era2015_pool()
    meta = list_matches(EXTRA_POOL)
    frames = featurize_matches(meta["match_id"].tolist(), "pool_frames_extra.pkl")
    meta_all = pd.concat([era_meta, meta], ignore_index=True)
    return era_frames + frames, meta_all


def era2015_pool() -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """ERA_2015_POOL competitions: contemporaneous 2015/2016 leagues.

    The only pooled data the season-transfer model may train on.
    Memoized per process (shared with extra_pool).
    """
    return _memoized("era2015_pool", _era2015_pool_impl)


def _era2015_pool_impl() -> tuple[list[pd.DataFrame], pd.DataFrame]:
    from effc.data.download import ERA_2015_POOL

    meta = list_matches(ERA_2015_POOL)
    frames = featurize_matches(meta["match_id"].tolist(), "pool_frames_era2015.pkl")
    return frames, meta


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


def prepare_model_data(cfg: dict) -> dict:
    """Model-training inputs shared by the pipeline and scripts/make_demo.py.

    Applies the backtest's chronological split, drops post-split frames,
    then splits the rest into a chronological train group and a held-out
    calibration group (FC6). The deployed goal models train on the train
    group only; the calibration group is scored but never trained on.

    Returns dict with first, second (the backtest split), frames, meta,
    train_frames, calib_frames, train_ids, calib_ids and models.
    """
    from effc.backtest.runner import split_matches

    target_matches = list_matches(TARGET)
    first, second = split_matches(target_matches, frac=cfg.get("backtest_split", 0.5))
    post_split_ids = set(second["match_id"].tolist())

    frames = featurize_pool()
    frames = pre_split_frames(frames, post_split_ids)
    meta = list_matches(MODEL_POOL)
    train_frames, calib_frames = cal.match_group_split(
        frames, meta, calib_frac=cfg.get("calib_frac", 0.2)
    )
    train_ids = sorted({int(f["match_id"].iloc[0]) for f in train_frames})
    calib_ids = sorted({int(f["match_id"].iloc[0]) for f in calib_frames})
    cal.assert_disjoint(train_ids, calib_ids)

    # Beat-baseline: the deployed goal models are the improved recipe
    # (pooled training data + Dixon-Coles strength features +
    # cross-fitted Platt calibration and base-rate blending, all fit
    # on training matches only). The FC6 split above is unchanged: the
    # same 130 train / 32 held-out match IDs.
    from effc.vaep import improved as imp

    pool_frames, pool_meta = extra_pool()
    meta_all = (
        pd.concat([meta, pool_meta], ignore_index=True)
        .drop_duplicates("match_id")
        .reset_index(drop=True)
    )
    pool_ids = {int(f["match_id"].iloc[0]) for f in pool_frames}
    target_2024_ids = set(
        target_matches.loc[target_matches["season_id"] == 281, "match_id"]
        .astype(int)
    )
    # Leakage guards, by match ID, on every run: pooled training data
    # must contain neither held-out matches nor any 2023/2024
    # Bundesliga match (the season-transfer test set).
    cal.assert_disjoint(pool_ids, calib_ids, "pooled/held-out match")
    cal.assert_disjoint(pool_ids, target_2024_ids, "pooled/2023-24 match")
    config_name = cfg.get("goal_model_config", "dc365")
    print(
        f"training improved goal models (config {config_name}) on "
        f"{len(train_frames)} train-group + {len(pool_frames)} pooled "
        f"matches; holding out {len(calib_frames)} for calibration"
    )
    models, calib_report = imp.train_improved_models_cached(
        train_frames, pool_frames, meta_all, config_name,
        seed=cfg.get("seed", 0),
    )
    improvement = {
        "config": config_name,
        "calibration": calib_report,
        "pool_counts": {
            "train_group_matches": len(train_ids),
            "held_out_matches": len(calib_ids),
            "pooled_training_matches": len(pool_ids),
        },
    }
    return {
        "first": first,
        "second": second,
        "frames": frames,
        "meta": meta,
        "meta_all": meta_all,
        "train_frames": train_frames,
        "calib_frames": calib_frames,
        "train_ids": train_ids,
        "calib_ids": calib_ids,
        "models": models,
        "improvement": improvement,
    }


def season_transfer_evaluation(
    frames: list[pd.DataFrame],
    meta: pd.DataFrame,
    seed: int = 0,
    config_name: str = "dc365",
) -> dict:
    """Train on the earlier season, score on the later season (FC6).

    `frames` are the pre-split action frames, so no model ever trains on
    the backtest's scored half. A genuine chronological deployment
    simulation: what the goal models would have scored had they been
    deployed a season later. The caller adds roster stability once the
    target VAEP tables exist.

    Beat-baseline: the transfer model uses the same improved recipe as
    the deployed models, but its pooled training data is restricted to
    contemporaneous 2015/2016 leagues (era2015_pool): a model trained
    in 2016 cannot use later data, and no 2023/2024 Bundesliga match
    (the scored set) appears in any training pool.
    """
    from effc.vaep import improved as imp

    tr_season, sc_season, season_names = cal.transfer_seasons(meta, competition_id=9)
    season_of = dict(
        zip(meta["match_id"].astype(int), meta["season_id"].astype(int))
    )
    tr_frames = [
        f
        for f in frames
        if season_of.get(int(f["match_id"].iloc[0])) == tr_season
    ]
    sc_frames = [
        f
        for f in frames
        if season_of.get(int(f["match_id"].iloc[0])) == sc_season
    ]
    tr_ids = sorted({int(f["match_id"].iloc[0]) for f in tr_frames})
    sc_ids = sorted({int(f["match_id"].iloc[0]) for f in sc_frames})
    cal.assert_disjoint(tr_ids, sc_ids)

    era_frames, era_meta = era2015_pool()
    era_ids = {int(f["match_id"].iloc[0]) for f in era_frames}
    cal.assert_disjoint(era_ids, sc_ids, "era-pooled/scored match")
    meta_all = (
        pd.concat([meta, era_meta], ignore_index=True)
        .drop_duplicates("match_id")
        .reset_index(drop=True)
    )
    print(
        f"season transfer: training improved models (config {config_name}) "
        f"on {len(tr_frames)} {season_names.get(tr_season, tr_season)} "
        f"matches + {len(era_frames)} pooled 2015/2016 league matches, "
        f"scoring on {len(sc_frames)} "
        f"{season_names.get(sc_season, sc_season)} matches"
    )
    tr_models, calib_report = imp.train_improved_models_cached(
        tr_frames, era_frames, meta_all, config_name, seed=seed,
    )
    tr_scored = cal.per_match_scores(tr_models, sc_frames)
    tr_base = cal.base_rate_scores(tr_frames, sc_frames)
    model_summary = cal.summarize(tr_scored)
    baseline_summary = cal.summarize(tr_base)
    return {
        "train_season": season_names.get(tr_season, str(tr_season)),
        "scored_season": season_names.get(sc_season, str(sc_season)),
        "train_season_id": tr_season,
        "scored_season_id": sc_season,
        "n_train_matches": len(tr_ids),
        "n_scored_matches": len(sc_ids),
        "improvement": {
            "config": config_name,
            "calibration": calib_report,
            "pooled_training_matches": len(era_ids),
        },
        "held_out": {
            label: {
                "model": model_summary[label],
                "base_rate_baseline": baseline_summary[label],
            }
            for label in cal.LABELS
        },
    }


def main(config_path: str = "configs/bundesliga_2324.yaml"):
    t0 = time.time()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    outdir = os.path.join("runs", cfg.get("run_name", "run"))
    os.makedirs(outdir, exist_ok=True)

    # model-training inputs, shared with scripts/make_demo.py
    data = prepare_model_data(cfg)
    first, second = data["first"], data["second"]
    train_frames, calib_frames = data["train_frames"], data["calib_frames"]
    train_ids, calib_ids, models = data["train_ids"], data["calib_ids"], data["models"]
    meta = data["meta"]

    # 1. held-out match-group calibration (FC6). The Brier scores the old
    # pipeline reported were computed on the training frames; the
    # in-sample values below are now explicitly labeled, and the
    # headline accuracy comes from the held-out match group, whose match
    # IDs are disjoint from the training set by construction.
    from effc.vaep.models import predict_proba

    train_full = pd.concat(train_frames, ignore_index=True)
    preds = predict_proba(models, train_full)
    in_sample_scores = {}
    for label in cal.LABELS:
        y = train_full[label].to_numpy()
        in_sample_scores[label] = {
            "brier": brier_score(y, preds[label]),
            "logloss": cal.log_loss(y, preds[label]),
        }
    print(
        "goal models in-sample: "
        + ", ".join(
            f"{label} Brier={in_sample_scores[label]['brier']:.4f}"
            for label in ("scores_incl", "concedes_incl")
        )
    )
    held_model = cal.per_match_scores(models, calib_frames)
    held_base = cal.base_rate_scores(train_frames, calib_frames)
    calibration_report = cal.build_report(
        train_ids=train_ids,
        calib_ids=calib_ids,
        in_sample_scores=in_sample_scores,
        model_summary=cal.summarize(held_model),
        baseline_summary=cal.summarize(held_base),
    )
    hb = calibration_report["held_out"]["scores_incl"]["model"]["brier"]
    print(
        f"held-out calibration ({hb['n_matches']} matches): "
        f"scores Brier={hb['mean']:.4f} 95% CI "
        f"[{hb['ci95'][0]:.4f}, {hb['ci95'][1]:.4f}]"
    )

    # 1b. genuine chronological season-transfer evaluation (FC6)
    transfer_report = season_transfer_evaluation(
        data["frames"],
        meta,
        seed=cfg.get("seed", 0),
        config_name=cfg.get("transfer_goal_model_config", "dc365"),
    )

    # 2. VAEP on the target competition. The cache key binds the training
    # matches, feature version, config and model hash.
    pm, mins, matches = compute_target_vaep(models, train_ids, config=cfg)

    # roster stability for the season transfer (FC6): how much of the
    # scored season's player pool the training season actually covers
    mins_season = mins.merge(
        matches[["match_id", "season_id"]], on="match_id", how="left"
    )
    tr_id, sc_id = (
        transfer_report["train_season_id"],
        transfer_report["scored_season_id"],
    )
    transfer_report["roster_stability"] = cal.roster_overlap(
        set(mins_season.loc[mins_season["season_id"] == tr_id, "player_id"]),
        set(mins_season.loc[mins_season["season_id"] == sc_id, "player_id"]),
        mins_season.loc[
            mins_season["season_id"] == sc_id, ["player_id", "minutes"]
        ],
    )

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

    # Preserve the FC6 run's numbers verbatim before overwriting: the
    # beat-baseline metrics below replace them as the headline, and the
    # before/after comparison stays in the file (and the README).
    fc6_reference = None
    metrics_path = os.path.join(outdir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            _old = json.load(f)
        if "calibration" in _old and "fc6_reference" not in _old:
            fc6_reference = {
                "calibration": _old.get("calibration"),
                "season_transfer": _old.get("season_transfer"),
                "backtest": _old.get("backtest"),
            }

    metrics = {
        "config": config_path,
        "train_time_s": time.time() - t0,
        # In-sample fit, explicitly labeled: measured on the training
        # frames, not a generalization claim (FC6). Held-out accuracy
        # lives under "calibration" / "season_transfer".
        "brier_scores_in_sample": in_sample_scores["scores_incl"]["brier"],
        "brier_concedes_in_sample": in_sample_scores["concedes_incl"]["brier"],
        "logloss_scores_in_sample": in_sample_scores["scores_incl"]["logloss"],
        "logloss_concedes_in_sample": in_sample_scores["concedes_incl"]["logloss"],
        "calibration": calibration_report,
        "season_transfer": transfer_report,
        "wage_match_rate": float(wmap["matched"].mean()),
        "n_rated": len(ratings),
        "cov_info": cov_info,
        "frontier": frontier.to_dict("records"),
        "backtest": bt.to_dict("records"),
        "xi": xi[["player", "team", "pos_group", "vaep_per90", "wage_eur"]].to_dict("records"),
        "top10": ratings.nlargest(10, "vaep_per90")[
            ["player", "team", "pos_group", "vaep_per90", "wage_eur"]
        ].to_dict("records"),
        "improvement": {
            "main": data.get("improvement"),
            "season_transfer": transfer_report.get("improvement"),
            "experiment": "runs/beat_baseline/results.json",
        },
    }
    if fc6_reference is not None:
        metrics["fc6_reference"] = fc6_reference
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"\nsaved metrics + plots to {outdir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/bundesliga_2324.yaml")
