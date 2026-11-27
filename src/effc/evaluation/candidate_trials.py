"""Candidate trials for U07 part 2.

Part 2 only. This module runs the candidate families (a) to (f) from
the frozen catalog as an ablation on the internal development split,
selects on that split only, and then scores the frozen separate
benchmark once after selection. It never touches the shipped goal
models: family (a) reuses the part-1 baseline capture, and families
(b) to (f) train trial estimators on trial matrices built by
``models.candidate_features``.

Selection rule, fixed before any number is read: the mean headline
Brier (scores_incl and concedes_incl) on the development split
decides. A challenger is selected only if it beats the baseline by
at least SELECTION_MARGIN. Promotion is stricter, because the
contract asks for a clear measured benefit, not a point-estimate
win. A selected challenger is promoted only if, on the frozen
benchmark, it (1) beats the baseline by at least that margin,
(2) has a paired per-match 95% interval for that difference that
excludes zero, and (3) beats the trivial base-rate predictor on
both headline labels (positive Brier skill). Gate (3) is what
stops a synthetic-fixture win from shipping: a goal model that
still loses to predicting the training base rate has not shown a
clear benefit, however it compares to the baseline GBM. A margin
cleared on two synthetic matches is a lead to confirm on real
data, not a promotion. Both margins are recorded in the ablation
artifact, and every failed candidate is recorded with its reason
instead of being dropped.

Leakage gate: selection uses the development split only, the frozen
test split is scored once after selection, and no candidate is ever
selected using either original FC6 scored arm.
"""

from __future__ import annotations

import importlib.util
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from effc.evaluation import development_trials as oracle
from effc.models import candidate_features as cand_feat
from effc.models import feature_catalog as catalog
from effc.vaep.features import design_matrix
from effc.vaep.improved import CONFIGS
from effc.vaep.models import LABELS

# A challenger must beat the baseline by at least this much mean
# headline Brier (development split for selection, frozen benchmark
# for promotion) to count as a clear measured benefit. Fixed before
# any number is read; with one 120-action development match the
# margin is deliberately small but nonzero, and the outcome is
# reported either way.
SELECTION_MARGIN = 0.002

HEADLINE_LABELS = ("scores_incl", "concedes_incl")

# Tuned histogram-GBM grid for family (c). Small and fixed: the
# development split has one 120-action match, so a wide grid would
# only fit noise. Every grid point is recorded in the artifact.
TUNED_GBM_GRID = [
    {"max_iter": 100, "learning_rate": 0.10, "max_leaf_nodes": 15,
     "min_samples_leaf": 20, "l2_regularization": 0.0},
    {"max_iter": 300, "learning_rate": 0.04, "max_leaf_nodes": 31,
     "min_samples_leaf": 50, "l2_regularization": 1.0},
    {"max_iter": 200, "learning_rate": 0.06, "max_leaf_nodes": 63,
     "min_samples_leaf": 30, "l2_regularization": 0.5},
    {"max_iter": 400, "learning_rate": 0.03, "max_leaf_nodes": 31,
     "min_samples_leaf": 100, "l2_regularization": 1.0},
]

LOGISTIC_CONFIG = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}

SEQUENCE_CONFIG = {"hidden_layer_sizes": (32,), "max_iter": 300}

# Trial matrix kind per family.
MATRIX_KIND = {
    "a": "design",
    "b": "rich",
    "c": "design",
    "e": "design",
    "f": "sequence",
}


def probe_external_library(name: str) -> dict:
    """Availability probe for an optional gradient-boosting library.

    Returns whether the import is available in this training
    environment and its version when present. Never installs or adds
    a dependency: an unavailable library is a recorded failed
    candidate, not a reason to change the environment.
    """
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"name": name, "available": False, "version": None,
                "reason": f"{name} is not installed in the training "
                          "environment"}
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
    except ImportError as exc:  # pragma: no cover - defensive
        return {"name": name, "available": False, "version": None,
                "reason": f"{name} is present but fails to import: {exc}"}
    return {"name": name, "available": True, "version": str(version),
            "reason": None}


def _matrix(frame: pd.DataFrame, events: list[dict] | None,
            kind: str) -> pd.DataFrame:
    if kind == "design":
        return design_matrix(frame)
    if kind == "rich":
        if events is None:
            raise ValueError(
                "richer-context matrix needs the raw events for the frame"
            )
        rich = cand_feat.extend_frame_with_richer_context(frame, events)
        return cand_feat.extended_design_matrix(rich, "b")
    if kind == "sequence":
        return cand_feat.sequence_window_matrix(frame)
    raise ValueError(f"unknown trial matrix kind {kind!r}")


def _stack_matrices(frames, events_by_id, kind):
    """Stack per-frame matrices with a union of columns.

    Mirrors the shipped train_bundle alignment: a frame that never
    saw a dummy value gets 0.0 there, so train and score frames
    always share one column set.
    """
    blocks, columns = [], []
    for f in frames:
        mid = int(f["match_id"].iloc[0])
        events = events_by_id.get(mid) if events_by_id is not None else None
        Xi = _matrix(f, events, kind)
        for c in Xi.columns:
            if c not in columns:
                columns.append(c)
        blocks.append(Xi)
    stacked = [
        b.reindex(columns=columns, fill_value=0.0).to_numpy(dtype=np.float32)
        for b in blocks
    ]
    X = np.concatenate(stacked) if stacked else np.empty((0, 0), dtype=np.float32)
    return X, columns


def _estimator_for(key: str, seed: int, gbm_params: dict | None = None):
    if key in ("a", "b", "c"):
        params = dict(gbm_params) if gbm_params is not None else dict(
            CONFIGS["fc6_pool"]["gbm"]
        )
        return HistGradientBoostingClassifier(**params, random_state=seed)
    if key == "e":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=seed, **LOGISTIC_CONFIG)),
        ])
    if key == "f":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(random_state=seed, **SEQUENCE_CONFIG)),
        ])
    raise ValueError(f"no trial estimator for candidate {key!r}")


def _fit_bundle(key, train_frames, train_events_by_id, seed,
                gbm_params=None):
    """Fit one trial estimator per label on the training frames."""
    kind = MATRIX_KIND[key]
    X, columns = _stack_matrices(train_frames, train_events_by_id, kind)
    models = {}
    for label in LABELS:
        y = np.concatenate([f[label].to_numpy() for f in train_frames])
        est = _estimator_for(key, seed, gbm_params)
        est.fit(X, y)
        models[label] = est
    models["columns"] = columns
    models["kind"] = kind
    return models


def _predict_bundle(bundle, frame, events):
    """Per-label probabilities for one frame from a fitted bundle."""
    kind = bundle["kind"]
    X = _matrix(frame, events, kind).reindex(
        columns=bundle["columns"], fill_value=0.0
    ).to_numpy(dtype=np.float32)
    return {
        label: bundle[label].predict_proba(X)[:, 1] for label in LABELS
    }


def _score_frames(bundle, frames, events_by_id):
    """Score frames with a bundle; returns per-label and per-match data."""
    preds_by_match: dict[int, dict] = {}
    for f in frames:
        mid = int(f["match_id"].iloc[0])
        events = events_by_id.get(mid) if events_by_id is not None else None
        preds_by_match[mid] = _predict_bundle(bundle, f, events)
    return preds_by_match


def _label_metrics(train_frames, scored_frames, preds_by_match):
    """Per-label Brier, log loss, prevalence and Brier skill."""
    train_pool = pd.concat(train_frames, ignore_index=True)
    out = {}
    for label in LABELS:
        y = np.concatenate([f[label].to_numpy() for f in scored_frames])
        p = np.concatenate([preds_by_match[int(f["match_id"].iloc[0])][label]
                            for f in scored_frames])
        base_rate = float(train_pool[label].mean())
        out[label] = {
            "brier": oracle.brier(y, p),
            "log_loss": oracle.log_loss(y, p),
            "prevalence": float(y.mean()),
            "n_positive": int(np.sum(y > 0.5)),
            "n_scored": int(len(y)),
            "train_base_rate": base_rate,
            "brier_skill_vs_base_rate": oracle.brier_skill_score(
                y, p, np.full(len(y), base_rate)),
        }
    return out


def _mean_headline_brier(label_metrics: dict) -> float:
    return float(np.mean([label_metrics[label]["brier"]
                          for label in HEADLINE_LABELS]))


def _reliability(frames, preds_by_match):
    return {
        label: oracle.reliability_bins(
            np.concatenate([f[label].to_numpy() for f in frames]),
            np.concatenate([preds_by_match[int(f["match_id"].iloc[0])][label]
                            for f in frames]),
        )
        for label in HEADLINE_LABELS
    }


def _paired_vs_baseline(frames, preds_by_match, baseline_preds):
    """Per-match paired Brier differences against the baseline.

    Positive means this candidate is worse than the baseline on that
    headline label; the interval is a bootstrap 95% interval of the
    mean over identical matches.
    """
    out = {}
    for label in HEADLINE_LABELS:
        a = oracle.per_match_metric(frames, preds_by_match, label, "brier")
        b = oracle.per_match_metric(frames, baseline_preds, label, "brier")
        col = f"brier_{label}"
        out[label] = oracle.paired_difference(a, b, col, seed=0)
    return out


def _candidate_result(key, name, status, config, label_metrics,
                      runtime_seconds, n_train_actions, n_scored_actions,
                      reliability=None, paired=None, failure_reason=None,
                      extra=None):
    result = {
        "key": key,
        "name": name,
        "status": status,
        "config": config,
        "labels": label_metrics,
        "mean_headline_brier": (
            _mean_headline_brier(label_metrics)
            if label_metrics else None
        ),
        "reliability_bins": reliability,
        "paired_vs_baseline": paired,
        "runtime_seconds": float(runtime_seconds),
        "n_train_actions": int(n_train_actions),
        "n_scored_actions": int(n_scored_actions),
        "selected_on_fc6": False,
        "failure_reason": failure_reason,
    }
    if extra:
        result.update(extra)
    return result


def _fit_and_score(key, train_frames, scored_frames, train_events_by_id,
                   scored_events_by_id, seed, gbm_params=None):
    """Fit a trial candidate on train frames and score scored frames."""
    started = time.perf_counter()
    bundle = _fit_bundle(key, train_frames, train_events_by_id, seed,
                         gbm_params)
    preds = _score_frames(bundle, scored_frames, scored_events_by_id)
    runtime = time.perf_counter() - started
    labels_out = _label_metrics(train_frames, scored_frames, preds)
    return labels_out, preds, runtime


def _family_name(key: str) -> str:
    return [fam["name"] for fam in catalog.CANDIDATE_FAMILIES
            if fam["key"] == key][0]


def run_ablation(train_frames, dev_frames, *, train_events_by_id,
                 dev_events_by_id, seed: int = 0) -> dict:
    """Run families (a) to (f) in order on the development split.

    Families are evaluated in increasing complexity; family (f) runs
    only after (b), (c) and (e) are evaluated. Returns the ablation
    matrix: one record per family with its exact config, per-label
    metrics, reliability bins, paired differences against the
    baseline over identical matches, runtime and, for failed
    candidates, the reason instead of metrics.
    """
    oracle.validate_no_fc6_selection()
    train_ids = sorted(int(f["match_id"].iloc[0]) for f in train_frames)
    dev_ids = sorted(int(f["match_id"].iloc[0]) for f in dev_frames)
    n_train = int(sum(len(f) for f in train_frames))
    n_dev = int(sum(len(f) for f in dev_frames))
    results: dict[str, dict] = {}

    # (a) corrected-geometry existing GBM: the frozen baseline, via
    # the part-1 capture so the numbers are identical to the frozen
    # artifact on the same fixture.
    started = time.perf_counter()
    captured = oracle.baseline_capture(train_frames, dev_frames, seed=seed)
    baseline_preds = {}
    bundle_a = _fit_bundle("a", train_frames, None, seed)
    baseline_preds = _score_frames(bundle_a, dev_frames, None)
    results["a"] = _candidate_result(
        "a", _family_name("a"), "evaluated",
        {"config_name": "fc6_pool",
         "gbm": dict(CONFIGS["fc6_pool"]["gbm"]),
         "feature_version": captured["feature_version"],
         "lookahead_k": 10},
        captured["labels"], time.perf_counter() - started,
        n_train, n_dev,
        reliability=_reliability(dev_frames, baseline_preds),
        paired=None,
    )

    # (b) richer event context on the same GBM recipe.
    cand_feat.validate_candidate_spec()
    labels_b, preds_b, runtime_b = _fit_and_score(
        "b", train_frames, dev_frames, train_events_by_id,
        dev_events_by_id, seed)
    results["b"] = _candidate_result(
        "b", _family_name("b"), "evaluated",
        {"gbm": dict(CONFIGS["fc6_pool"]["gbm"]),
         "features": "shipped design matrix plus richer context",
         "richer_columns": list(cand_feat.RICH_COLUMNS)},
        labels_b, runtime_b, n_train, n_dev,
        reliability=_reliability(dev_frames, preds_b),
        paired=_paired_vs_baseline(dev_frames, preds_b, baseline_preds),
    )

    # (c) tuned histogram GBM: every grid point is scored on the
    # development split and recorded; the best point is the family
    # result. The grid choice and the family comparison share the
    # development split; that is recorded as a limitation and the
    # frozen benchmark is the only untouched check.
    grid_results = []
    best_grid = None
    for params in TUNED_GBM_GRID:
        labels_c, preds_c, runtime_c = _fit_and_score(
            "c", train_frames, dev_frames, train_events_by_id,
            dev_events_by_id, seed, gbm_params=params)
        grid_results.append({
            "gbm": dict(params),
            "mean_headline_brier": _mean_headline_brier(labels_c),
            "labels": labels_c,
            "runtime_seconds": float(runtime_c),
        })
        if best_grid is None or (
                grid_results[-1]["mean_headline_brier"]
                < best_grid["mean_headline_brier"]):
            best_grid = grid_results[-1]
            best_preds_c = preds_c
            best_runtime_c = runtime_c
    results["c"] = _candidate_result(
        "c", _family_name("c"), "evaluated",
        {"gbm": dict(best_grid["gbm"]),
         "grid": "TUNED_GBM_GRID, best point by dev mean headline Brier"},
        best_grid["labels"], best_runtime_c, n_train, n_dev,
        reliability=_reliability(dev_frames, best_preds_c),
        paired=_paired_vs_baseline(dev_frames, best_preds_c, baseline_preds),
        extra={"grid_results": [
            {"gbm": g["gbm"],
             "mean_headline_brier": g["mean_headline_brier"],
             "runtime_seconds": g["runtime_seconds"]}
            for g in grid_results
        ]},
    )

    # (d) LightGBM / CatBoost where permitted and justified. Neither
    # library is installed in this training environment and this unit
    # adds no new dependency, so the family is recorded as failed
    # with its reason instead of being silently skipped.
    probes = {name: probe_external_library(name)
              for name in ("lightgbm", "catboost")}
    available = [name for name, probe in probes.items() if probe["available"]]
    if available:
        raise RuntimeError(
            "unexpected: an external boosting library is available "
            f"({available}); the (d) path below was written for the "
            "unavailable case and must be extended before use"
        )
    results["d"] = _candidate_result(
        "d", _family_name("d"), "failed",
        {"probes": probes},
        {}, 0.0, n_train, n_dev,
        failure_reason=(
            "neither LightGBM nor CatBoost is installed in the training "
            "environment, so no (d) model could be trained; no new "
            "dependency was added by this unit, and the family stays a "
            "failed candidate rather than an assumed improvement"
        ),
    )

    # (e) regularized logistic comparator on the shipped matrix.
    labels_e, preds_e, runtime_e = _fit_and_score(
        "e", train_frames, dev_frames, None, None, seed)
    results["e"] = _candidate_result(
        "e", _family_name("e"), "evaluated",
        {"logistic": dict(LOGISTIC_CONFIG),
         "features": "shipped design matrix, standardized"},
        labels_e, runtime_e, n_train, n_dev,
        reliability=_reliability(dev_frames, preds_e),
        paired=_paired_vs_baseline(dev_frames, preds_e, baseline_preds),
    )

    # (f) compact sequence model, only after the simpler candidates
    # are evaluated. Enforced, not just ordered: a missing simpler
    # result stops the run.
    for required in ("b", "c", "e"):
        if results.get(required, {}).get("status") != "evaluated":
            raise RuntimeError(
                f"family (f) requires family ({required}) evaluated first; "
                f"got status {results.get(required, {}).get('status')!r}"
            )
    labels_f, preds_f, runtime_f = _fit_and_score(
        "f", train_frames, dev_frames, None, None, seed)
    results["f"] = _candidate_result(
        "f", _family_name("f"), "evaluated",
        {"sequence": dict(SEQUENCE_CONFIG),
         "features": "shipped design matrix plus one- and two-action lags",
         "window": 3},
        labels_f, runtime_f, n_train, n_dev,
        reliability=_reliability(dev_frames, preds_f),
        paired=_paired_vs_baseline(dev_frames, preds_f, baseline_preds),
        extra={"evaluated_after": ["b", "c", "e"]},
    )

    return {
        "artifact": "u07 part 2 ablation",
        "seed": seed,
        "split_registry": {
            "train_ids": train_ids,
            "dev_ids": dev_ids,
            "test_ids": [],
        },
        "catalog_digest": catalog.catalog_digest(),
        "evaluation_order": ["a", "b", "c", "d", "e", "f"],
        "selection_split_ids": dev_ids,
        "candidates": results,
        "limitations": [
            "The fixture is synthetic (seed 7, 120 actions per match) "
            "with almost no learnable signal; it verifies the trial "
            "machinery, it does not evaluate the model on real data.",
            "The development split is a single match, so per-candidate "
            "differences on it are noisy and no narrow win is a real "
            "finding.",
            "Family (c) grid choice and family comparison share the "
            "development split; the frozen benchmark is the only "
            "untouched check.",
            "Family (f) windows may cross a period boundary.",
            "Downstream squad-regret impact is not measured here; that "
            "is the U09 remit.",
        ],
    }


def select_candidate(ablation: dict, margin: float = SELECTION_MARGIN) -> dict:
    """Select on the development split only, by the fixed rule.

    The best evaluated challenger is selected only when its mean
    headline Brier beats the baseline by at least `margin`. Otherwise
    the selection is none and the baseline is retained. The frozen
    test split plays no part in this decision.
    """
    candidates = ablation["candidates"]
    baseline_brier = candidates["a"]["mean_headline_brier"]
    challengers = [
        (key, rec) for key, rec in candidates.items()
        if key != "a" and rec["status"] == "evaluated"
    ]
    best_key, best_rec = min(
        challengers, key=lambda kv: kv[1]["mean_headline_brier"]
    )
    improvement = baseline_brier - best_rec["mean_headline_brier"]
    selected = best_key if improvement >= margin else None
    return {
        "selected": selected,
        "best_challenger": best_key,
        "best_mean_headline_brier": best_rec["mean_headline_brier"],
        "baseline_mean_headline_brier": baseline_brier,
        "improvement": float(improvement),
        "margin": float(margin),
        "selection_split_ids": list(ablation["selection_split_ids"]),
        "reason": (
            f"best challenger ({best_key}) beats the baseline by "
            f"{improvement:.6f} mean headline Brier on the development "
            f"split, clearing the {margin} margin"
            if selected is not None else
            f"best challenger ({best_key}) beats the baseline by only "
            f"{improvement:.6f} mean headline Brier on the development "
            f"split, below the {margin} margin; baseline retained"
        ),
    }


def validate_selection_not_on_fc6(ablation: dict, selection: dict) -> bool:
    """Leakage gate for part 2: selection used the dev split only.

    Returns True only when no candidate was selected on FC6, the
    selection split is exactly the recorded development split, and
    the frozen test split was never part of selection. Raises
    ValueError otherwise.
    """
    for key, rec in ablation["candidates"].items():
        if rec.get("selected_on_fc6"):
            raise ValueError(
                f"candidate {key} was selected on FC6; the leakage gate "
                "forbids it"
            )
    if list(selection["selection_split_ids"]) != list(
            ablation["selection_split_ids"]):
        raise ValueError(
            "selection split does not match the recorded development split"
        )
    return True


def _benchmark_model(key, train_frames, test_frames, train_events_by_id,
                     test_events_by_id, seed, gbm_params=None):
    """Train one candidate on train+dev and score the frozen test split."""
    started = time.perf_counter()
    bundle = _fit_bundle(key, train_frames, train_events_by_id, seed,
                         gbm_params)
    preds = _score_frames(bundle, test_frames, test_events_by_id)
    labels_out = _label_metrics(train_frames, test_frames, preds)
    return {
        "key": key,
        "name": _family_name(key),
        "labels": labels_out,
        "mean_headline_brier": _mean_headline_brier(labels_out),
        "reliability_bins": _reliability(test_frames, preds),
        "paired_vs_baseline": None,
        "runtime_seconds": float(time.perf_counter() - started),
        "n_train_actions": int(sum(len(f) for f in train_frames)),
        "n_scored_actions": int(sum(len(f) for f in test_frames)),
        "_preds_by_match": preds,
    }


def _headline_paired(frames, preds_a, preds_b):
    """Paired per-match difference of the mean headline Brier.

    One value per match (the mean of the two headline labels'
    Brier), then a bootstrap 95% interval of the mean difference
    over identical matches. Returns a - b, so with (challenger,
    baseline) a negative interval means the challenger is better on
    every resampled mean.
    """
    rows_a, rows_b = [], []
    for f in frames:
        mid = int(f["match_id"].iloc[0])
        for rows, preds in ((rows_a, preds_a), (rows_b, preds_b)):
            vals = [oracle.brier(f[label].to_numpy(), preds[mid][label])
                    for label in HEADLINE_LABELS]
            rows.append({"match_id": mid,
                         "mean_headline_brier": float(np.mean(vals))})
    a = pd.DataFrame(rows_a)
    b = pd.DataFrame(rows_b)
    return oracle.paired_difference(a, b, "mean_headline_brier", seed=0)


def decide_promotion(benchmark: dict, margin: float = SELECTION_MARGIN) -> dict:
    """Promotion decision from the frozen benchmark only.

    A selected challenger is promoted only when, on the frozen
    benchmark, it (1) beats the baseline by at least `margin` mean
    headline Brier, (2) has a paired per-match 95% interval for
    that difference that excludes zero, and (3) beats the trivial
    base-rate predictor on both headline labels (positive Brier
    skill). When nothing was selected, or any gate fails, nothing
    is promoted and the reason names the failing gate.
    """
    selected = benchmark["selected"]
    if selected is None:
        return {"promoted": None,
                "reason": "no challenger was selected on the development "
                          "split, so there is nothing to promote"}
    baseline = benchmark["models"]["a"]["mean_headline_brier"]
    challenger = benchmark["models"][selected]["mean_headline_brier"]
    improvement = baseline - challenger
    paired = benchmark["models"][selected].get("paired_headline_mean")
    skill = {
        label: benchmark["models"][selected]["labels"][label][
            "brier_skill_vs_base_rate"]
        for label in HEADLINE_LABELS
    }
    mean_skill = float(np.mean([skill[label] for label in HEADLINE_LABELS]))
    evidence = {
        "improvement": float(improvement),
        "margin": float(margin),
        "paired_headline_mean": paired,
        "headline_brier_skill": skill,
        "mean_headline_brier_skill": mean_skill,
    }
    if improvement < margin:
        return {"promoted": None, **evidence,
                "reason": (
                    f"selected challenger ({selected}) beats the baseline "
                    f"by only {improvement:.6f} mean headline Brier on "
                    f"the frozen benchmark, below the {margin} margin; "
                    "nothing is promoted"
                )}
    if paired is not None and paired["ci95"][1] >= 0.0:
        return {"promoted": None, **evidence,
                "reason": (
                    f"selected challenger ({selected}) beats the baseline "
                    f"by {improvement:.6f} mean headline Brier on the "
                    "frozen benchmark, clearing the point margin, but the "
                    "paired per-match 95% interval "
                    f"{[round(x, 6) for x in paired['ci95']]} covers zero "
                    "over only "
                    f"{paired['n_matches']} benchmark matches, so the "
                    "benefit is not clear and nothing is promoted"
                )}
    if mean_skill <= 0.0:
        return {"promoted": None, **evidence,
                "reason": (
                    f"selected challenger ({selected}) beats the baseline "
                    f"by {improvement:.6f} mean headline Brier on the "
                    "frozen benchmark, but its mean headline Brier skill "
                    f"against the training base rate is {mean_skill:.3f}: "
                    "it still loses to the trivial base-rate predictor, "
                    "so the win is not a clear measured benefit and "
                    "nothing is promoted. Confirm on real data first."
                )}
    return {"promoted": selected, **evidence,
            "reason": (
                f"selected challenger ({selected}) beats the baseline by "
                f"{improvement:.6f} mean headline Brier on the frozen "
                "benchmark, clearing the point margin, the paired "
                "per-match 95% interval excludes zero, and its mean "
                f"headline Brier skill is {mean_skill:.3f}"
            )}


def score_frozen_benchmark(ablation: dict, selection: dict,
                           train_frames, dev_frames, test_frames,
                           *, train_events_by_id, dev_events_by_id,
                           test_events_by_id, seed: int = 0) -> dict:
    """Score the frozen separate benchmark once after selection.

    The selected challenger (when one was selected) and the baseline
    are both retrained on train plus dev and scored on the frozen
    test split in this single benchmark event. Selection is not
    revisited here: the promotion decision only reads these numbers.
    """
    validate_selection_not_on_fc6(ablation, selection)
    all_train = list(train_frames) + list(dev_frames)
    all_events = dict(train_events_by_id or {})
    all_events.update(dev_events_by_id or {})
    test_ids = sorted(int(f["match_id"].iloc[0]) for f in test_frames)
    models: dict[str, dict] = {}
    models["a"] = _benchmark_model(
        "a", all_train, test_frames, None, None, seed)
    selected = selection["selected"]
    if selected is not None:
        gbm_params = None
        if selected == "c":
            gbm_params = ablation["candidates"]["c"]["config"]["gbm"]
        models[selected] = _benchmark_model(
            selected, all_train, test_frames, all_events,
            test_events_by_id, seed, gbm_params)
        models[selected]["paired_vs_baseline"] = _paired_vs_baseline(
            test_frames,
            models[selected]["_preds_by_match"],
            models["a"]["_preds_by_match"],
        )
        models[selected]["paired_headline_mean"] = _headline_paired(
            test_frames,
            models[selected]["_preds_by_match"],
            models["a"]["_preds_by_match"],
        )
    for model in models.values():
        model.pop("_preds_by_match", None)
    benchmark = {
        "artifact": "u07 part 2 frozen benchmark",
        "seed": seed,
        "selected": selected,
        "train_ids": sorted(int(f["match_id"].iloc[0]) for f in all_train),
        "test_ids": test_ids,
        "scored_once": True,
        "models": models,
    }
    benchmark["promotion"] = decide_promotion(benchmark)
    return benchmark
