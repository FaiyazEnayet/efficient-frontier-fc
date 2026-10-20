"""Improved goal models: pooled training data, Dixon-Coles team
strength features, cross-fitted Platt calibration and base-rate
blending.

Discipline (the frozen FC6 scoreboard):
- Test matches (the 32 held-out, all 2023/24 Bundesliga) are never in
  any training pool here: not for the GBMs, not for the Dixon-Coles
  fit, not for calibration. Callers pass explicit training pools and
  this module fits only on what it is given.
- Configuration selection uses an internal chronological validation
  split of the training group.
- Platt parameters and the blend weight toward the base rate are fit
  on out-of-fold (cross-fitted) predictions over training-group
  matches only: every match's OOF prediction comes from a model that
  never trained on it (the Dixon-Coles fit is also refit per fold).
- The frozen bundle is then retrained on the full training pool and
  scored on the test sets exactly once.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier

from effc.vaep.features import design_matrix
from effc.vaep.models import LABELS, brier_score
from effc.vaep.team_strength import (
    attach_to_frames,
    fit_dixon_coles,
    match_results,
)

FC6_GBM_PARAMS = dict(
    max_iter=200,
    learning_rate=0.06,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1.0,
)
BIG_GBM_PARAMS = dict(
    max_iter=400,
    learning_rate=0.04,
    max_leaf_nodes=63,
    min_samples_leaf=100,
    l2_regularization=1.0,
)

# Candidate configurations for internal-validation selection.
CONFIGS = {
    "fc6_pool": {  # FC6 recipe, pooled data (strengths stay zero)
        "use_strengths": False,
        "dc_half_life": None,
        "gbm": FC6_GBM_PARAMS,
        "recency_half_life": None,
    },
    "dc365": {
        "use_strengths": True,
        "dc_half_life": 365.0,
        "gbm": FC6_GBM_PARAMS,
        "recency_half_life": None,
    },
    "dc180_recency": {
        "use_strengths": True,
        "dc_half_life": 180.0,
        "gbm": FC6_GBM_PARAMS,
        "recency_half_life": 540.0,
    },
    "dc365_big": {
        "use_strengths": True,
        "dc_half_life": 365.0,
        "gbm": BIG_GBM_PARAMS,
        "recency_half_life": None,
    },
}


def _frame_id(frame: pd.DataFrame) -> int:
    return int(frame["match_id"].iloc[0])


def _recency_weights(frames, match_dates: dict, half_life_days: float):
    """Per-row sample weights from match dates (newest match weighs 1)."""
    dates = {mid: pd.to_datetime(d) for mid, d in match_dates.items()}
    ref = max(dates.values())
    per_frame = []
    for f in frames:
        age = (ref - dates[_frame_id(f)]).days
        per_frame.append(np.full(len(f), 0.5 ** (age / half_life_days)))
    return np.concatenate(per_frame)


class CalibratedGoalModel:
    """A fitted GBM plus frozen Platt scaling and base-rate blending.

    predict_proba applies, in order: the GBM probability, Platt scaling
    on the logit (sigmoid(a * logit(p) + b)), then shrinkage toward the
    training base rate with weight w: p' = (1 - w) * p + w * base_rate.
    Plain data + a sklearn model: picklable.
    """

    def __init__(self, clf, platt_a=1.0, platt_b=0.0, blend_w=0.0, base_rate=0.0):
        self.clf = clf
        self.platt_a = float(platt_a)
        self.platt_b = float(platt_b)
        self.blend_w = float(blend_w)
        self.base_rate = float(base_rate)

    def predict_proba(self, X):
        p = self.clf.predict_proba(X)[:, 1]
        p = np.clip(p, 1e-7, 1 - 1e-7)
        logit = np.log(p / (1 - p))
        p = 1.0 / (1.0 + np.exp(-(self.platt_a * logit + self.platt_b)))
        p = (1.0 - self.blend_w) * p + self.blend_w * self.base_rate
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


def train_bundle(
    train_frames: list[pd.DataFrame],
    meta: pd.DataFrame,
    config: dict,
    seed: int = 0,
    match_dates: dict | None = None,
    labels: tuple = tuple(LABELS),
) -> dict:
    """Train raw GBMs for all labels on `train_frames` only.

    If the config uses strengths, the Dixon-Coles model is fit on
    these frames' results and attached before training; the returned
    bundle carries the fit so predict_proba can attach the same frozen
    strengths to scoring frames. Nothing outside `train_frames` is fit
    on. Returns a models dict compatible with vaep.models.predict_proba
    (plus "strengths" / "home_team_by_match" entries when applicable).
    """
    frames = train_frames
    dc = None
    if config.get("use_strengths"):
        results = match_results(train_frames, meta)
        ref = str(results["match_date"].max()) if len(results) else None
        dc = fit_dixon_coles(
            results, half_life_days=config.get("dc_half_life"), ref_date=ref
        )
        frames = attach_to_frames(train_frames, dc, meta)
    # Build the design matrix match-by-match into float32 blocks and
    # stack them: concatenating ~4M rows of raw frames first would
    # roughly double the peak memory (small machines OOM there).
    blocks: list[np.ndarray] = []
    columns: list = []
    ys: dict = {label: [] for label in labels}
    for f in frames:
        Xi = design_matrix(f)
        for c in Xi.columns:
            if c not in columns:
                columns.append(c)
        blocks.append(Xi)
        for label in labels:
            ys[label].append(f[label].to_numpy())
    blocks = [
        b.reindex(columns=columns, fill_value=0.0).to_numpy(dtype=np.float32)
        for b in blocks
    ]
    X = np.concatenate(blocks) if blocks else np.empty((0, 0), dtype=np.float32)
    del blocks
    ys = {label: np.concatenate(v) for label, v in ys.items()}
    weights = None
    if config.get("recency_half_life") and match_dates:
        weights = _recency_weights(frames, match_dates, config["recency_half_life"])
    if frames is not train_frames:
        del frames
    import gc

    gc.collect()
    models: dict = {}
    for label in labels:
        clf = HistGradientBoostingClassifier(
            **config["gbm"], random_state=seed
        )
        clf.fit(X, ys[label], sample_weight=weights)
        models[label] = clf
    models["columns"] = columns
    if dc is not None:
        models["strengths"] = dc
        models["home_team_by_match"] = dict(
            zip(meta["match_id"].astype(int), meta["home_team"].astype(str))
        )
    return models


def raw_predict(models: dict, frame: pd.DataFrame) -> dict:
    """predict_proba for a bundle returned by train_bundle.

    Mirrors vaep.models.predict_proba (including strength attachment)
    but tolerates bundles trained on a subset of labels (used by
    config selection, which scores only the headline labels).
    """
    df = frame
    if "strengths" in models:
        from effc.vaep.team_strength import attach_strength_features

        home = models.get("home_team_by_match", {}).get(
            int(df["match_id"].iloc[0]), ""
        )
        df = attach_strength_features(df, models["strengths"], home)
    X = design_matrix(df).reindex(columns=models["columns"], fill_value=0.0)
    return {
        label: models[label].predict_proba(X)[:, 1]
        for label in LABELS
        if label in models
    }


def chronological_folds(
    frames: list[pd.DataFrame], match_dates: dict, n_folds: int = 4
) -> list[list[pd.DataFrame]]:
    """Split frames into chronological folds by match date."""
    ordered = sorted(frames, key=lambda f: (str(match_dates[_frame_id(f)]), _frame_id(f)))
    folds: list[list[pd.DataFrame]] = [[] for _ in range(n_folds)]
    for i, f in enumerate(ordered):
        folds[min(n_folds - 1, i * n_folds // max(1, len(ordered)))].append(f)
    return [fold for fold in folds if fold]


# Cross-fit bundles train on a deterministic subsample of the pool
# (the calibration signal comes from the group matches being
# predicted, not from pool size); the final bundle trains on the
# full pool.
CROSSFIT_POOL_CAP = 800
SELECTION_POOL_CAP = 600


def subsample_frames(frames: list[pd.DataFrame], cap: int) -> list[pd.DataFrame]:
    """Deterministic evenly-spaced subsample by match ID."""
    if len(frames) <= cap:
        return list(frames)
    ordered = sorted(frames, key=_frame_id)
    step = len(ordered) / cap
    return [ordered[int(i * step)] for i in range(cap)]


def crossfit_oof(
    group_frames: list[pd.DataFrame],
    pool_frames: list[pd.DataFrame],
    meta: pd.DataFrame,
    config: dict,
    seed: int = 0,
    n_folds: int = 4,
) -> dict:
    """Out-of-fold predictions for every match in `group_frames`.

    Fold k is predicted by a bundle trained on `pool_frames` plus the
    group matches in folds < k. Pool frames are training-only matches
    (never group members, never test matches); group matches are never
    trained on before their own fold is predicted. Returns
    {match_id: {label: preds}}.
    """
    match_dates = dict(
        zip(meta["match_id"].astype(int), meta["match_date"].astype(str))
    )
    folds = chronological_folds(group_frames, match_dates, n_folds)
    pool_capped = subsample_frames(pool_frames, CROSSFIT_POOL_CAP)

    # Disk cache: cross-fitting is the expensive middle step of the
    # training recipe; caching the OOF predictions (keyed by config,
    # seed, and the exact group/pool match-ID sets) makes interrupted
    # runs resumable without changing any result.
    import hashlib
    import json as _json
    import pickle

    from effc.data.download import DATA_DIR
    from effc.vaep.features import FEATURE_VERSION

    key_src = (
        f"{_json.dumps(config, sort_keys=True, default=str)}|{seed}|"
        f"{n_folds}|v{FEATURE_VERSION}|"
        + ",".join(map(str, sorted(_frame_id(f) for f in group_frames)))
        + "|"
        + ",".join(map(str, sorted(_frame_id(f) for f in pool_capped)))
    )
    key = hashlib.md5(key_src.encode()).hexdigest()[:16]
    cache_dir = os.path.join(DATA_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"oof_{key}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            print(f"loaded cached cross-fit OOF (key {key})")
            return pickle.load(fh)

    oof: dict = {}
    for k in range(n_folds):
        earlier = [f for fold in folds[:k] for f in fold]
        bundle = train_bundle(
            pool_capped + earlier, meta, config, seed=seed,
            match_dates=match_dates,
        )
        for f in folds[k]:
            oof[_frame_id(f)] = raw_predict(bundle, f)
    with open(path, "wb") as fh:
        pickle.dump(oof, fh)
    return oof


def _fit_platt(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Fit sigmoid(a * logit(p) + b) by penalized log-loss."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    logit = np.log(p / (1 - p))

    def nll(ab):
        a, b = ab
        q = 1.0 / (1.0 + np.exp(-(a * logit + b)))
        q = np.clip(q, 1e-12, 1 - 1e-12)
        ll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        return ll + 1e-3 * ((a - 1.0) ** 2 + b**2)

    res = minimize(nll, x0=np.array([1.0, 0.0]), method="Nelder-Mead")
    return float(res.x[0]), float(res.x[1])


def fit_calibration(
    oof: dict, group_frames: list[pd.DataFrame]
) -> dict:
    """Platt + blend per label from OOF predictions over group matches.

    Base rates are the group frames' own label means (the FC6 baseline
    definition: the training set the deployed models learn from).
    Returns {label: (platt_a, platt_b, blend_w, base_rate)}.
    """
    by_id = {_frame_id(f): f for f in group_frames}
    out: dict = {}
    for label in LABELS:
        ys, ps = [], []
        for mid, preds in oof.items():
            ys.append(by_id[mid][label].to_numpy())
            ps.append(preds[label])
        y = np.concatenate(ys)
        p = np.concatenate(ps)
        base_rate = float(y.mean())
        a, b = _fit_platt(y, p)
        pc = np.clip(p, 1e-7, 1 - 1e-7)
        logit = np.log(pc / (1 - pc))
        p_cal = 1.0 / (1.0 + np.exp(-(a * logit + b)))
        best_w, best_brier = 0.0, brier_score(y, p_cal)
        for w in [round(0.05 * i, 10) for i in range(1, 13)]:
            cand = (1 - w) * p_cal + w * base_rate
            val = brier_score(y, cand)
            if val < best_brier:
                best_w, best_brier = float(w), val
        out[label] = (a, b, best_w, base_rate)
    return out


def wrap_calibrated(bundle: dict, calibration: dict) -> dict:
    """Wrap a raw train_bundle with frozen per-label calibration."""
    models = {"columns": bundle["columns"]}
    for label in LABELS:
        a, b, w, base_rate = calibration[label]
        models[label] = CalibratedGoalModel(
            bundle[label], platt_a=a, platt_b=b, blend_w=w, base_rate=base_rate
        )
    for key in ("strengths", "home_team_by_match"):
        if key in bundle:
            models[key] = bundle[key]
    return models


def train_improved_models(
    group_frames: list[pd.DataFrame],
    pool_frames: list[pd.DataFrame],
    meta: pd.DataFrame,
    config: dict,
    seed: int = 0,
) -> tuple[dict, dict]:
    """Full recipe: cross-fit calibration on the group, then retrain on
    pool + group with the frozen calibration applied.

    Returns (models, calibration_report) where the report records the
    chosen config name (if set), per-label Platt/blend parameters and
    the OOF Brier of raw vs calibrated predictions.
    """
    match_dates = dict(
        zip(meta["match_id"].astype(int), meta["match_date"].astype(str))
    )
    oof = crossfit_oof(group_frames, pool_frames, meta, config, seed=seed)
    calibration = fit_calibration(oof, group_frames)
    bundle = train_bundle(
        pool_frames + group_frames, meta, config, seed=seed,
        match_dates=match_dates,
    )
    models = wrap_calibrated(bundle, calibration)
    by_id = {_frame_id(f): f for f in group_frames}
    report = {"labels": {}}
    for label in LABELS:
        y = np.concatenate([by_id[mid][label].to_numpy() for mid in oof])
        p_raw = np.concatenate([oof[mid][label] for mid in oof])
        a, b, w, base_rate = calibration[label]
        pc = np.clip(p_raw, 1e-7, 1 - 1e-7)
        logit = np.log(pc / (1 - pc))
        p_cal = 1.0 / (1.0 + np.exp(-(a * logit + b)))
        p_final = (1 - w) * p_cal + w * base_rate
        report["labels"][label] = {
            "platt_a": a,
            "platt_b": b,
            "blend_w": w,
            "base_rate": base_rate,
            "oof_brier_raw": brier_score(y, p_raw),
            "oof_brier_calibrated": brier_score(y, p_final),
        }
    return models, report


def select_config(
    group_frames: list[pd.DataFrame],
    pool_frames: list[pd.DataFrame],
    meta: pd.DataFrame,
    seed: int = 0,
    val_frac: float = 0.25,
    labels: tuple = ("scores_incl", "concedes_incl"),
) -> tuple[str, dict]:
    """Pick a config on an internal chronological validation split.

    The latest `val_frac` of the group (per the FC6 per-competition
    chronological convention, applied within the group) is scored by
    bundles trained on pool + the rest of the group. Raw (uncalibrated)
    Brier on the two headline labels decides. Returns (name, scores).
    """
    from effc.vaep import calibration as cal

    sub_train, val = cal.match_group_split(
        group_frames, meta, calib_frac=val_frac
    )
    match_dates = dict(
        zip(meta["match_id"].astype(int), meta["match_date"].astype(str))
    )
    pool_capped = subsample_frames(pool_frames, SELECTION_POOL_CAP)
    scores: dict = {}
    for name, config in CONFIGS.items():
        bundle = train_bundle(
            pool_capped + sub_train, meta, config, seed=seed,
            match_dates=match_dates, labels=tuple(labels),
        )
        per_label = {}
        for label in labels:
            ys, ps = [], []
            for f in val:
                ys.append(f[label].to_numpy())
                ps.append(raw_predict(bundle, f)[label])
            per_label[label] = brier_score(np.concatenate(ys), np.concatenate(ps))
        scores[name] = per_label
    best = min(
        scores,
        key=lambda n: scores[n]["scores_incl"] + scores[n]["concedes_incl"],
    )
    return best, scores


def train_improved_models_cached(
    group_frames: list[pd.DataFrame],
    pool_frames: list[pd.DataFrame],
    meta: pd.DataFrame,
    config_name: str,
    seed: int = 0,
):
    """train_improved_models with a disk cache (data/cache, gitignored).

    The pipeline, demo builder, and the experiment driver all train the
    identical recipe; the cache key covers the config name, seed,
    feature version, and the exact group/pool match-ID sets, so a hit
    is bit-identical training work already done, never stale data.
    """
    import hashlib
    import pickle

    from effc.data.download import DATA_DIR
    from effc.vaep.features import FEATURE_VERSION

    group_ids = sorted(_frame_id(f) for f in group_frames)
    pool_ids = sorted(_frame_id(f) for f in pool_frames)
    key_src = (
        f"{config_name}|{seed}|v{FEATURE_VERSION}|"
        + ",".join(map(str, group_ids))
        + "|"
        + ",".join(map(str, pool_ids))
    )
    key = hashlib.md5(key_src.encode()).hexdigest()[:16]
    cache_dir = os.path.join(DATA_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"improved_models_{key}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            models, report = pickle.load(fh)
        print(f"loaded cached improved models ({config_name}, key {key})")
        return models, report
    models, report = train_improved_models(
        group_frames, pool_frames, meta, CONFIGS[config_name], seed=seed
    )
    with open(path, "wb") as fh:
        pickle.dump((models, report), fh)
    return models, report
