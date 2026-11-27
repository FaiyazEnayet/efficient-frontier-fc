"""Independent oracle for U07 development trials.

Part 1 only. This module is the second, independent implementation
of the scores the shipped pipeline reports: Brier and log loss are
computed here from first principles (not by importing the shipped
calibration module), and the tests cross-check the two
implementations against each other. It also owns the frozen
train/dev/test split registry, the per-match aggregation with
paired differences across identical matches, calibration and
reliability bins, prevalence-adjusted performance, rare-event
stratification, and the fc6_pool baseline capture.

Leakage gate: nothing in this module selects a feature, cutoff,
architecture or model using either original FC6 scored arm. The
frozen test split is recorded here but never scored in part 1; part
2 scores the frozen separate benchmark once after selection.
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd

from effc.models import feature_catalog as catalog
from effc.vaep.features import FEATURE_VERSION
from effc.vaep.models import LABELS

LABELS_TUPLE = tuple(LABELS)

# A label counts as rare when its pooled positives fall below this
# count. The threshold is recorded, not tuned: with fewer positives
# no per-label claim is stable.
RARE_POSITIVE_THRESHOLD = 10


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared probability error, computed directly."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((y - p) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    """Natural-log log loss with clipped probabilities, computed directly."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def prevalence(y: np.ndarray) -> dict:
    """Class prevalence: counts and positive rate of one label."""
    y = np.asarray(y, dtype=float)
    n = int(len(y))
    n_pos = int(np.sum(y > 0.5))
    return {"n": n, "n_positive": n_pos,
            "rate": (n_pos / n) if n else 0.0}


def reliability_bins(y: np.ndarray, p: np.ndarray,
                     n_bins: int = 10) -> list[dict]:
    """Calibration bins: per predicted-probability decile, the count,
    mean predicted probability and observed positive rate."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        n = int(mask.sum())
        out.append({
            "bin": i,
            "lo": float(lo),
            "hi": float(hi),
            "n": n,
            "mean_predicted": float(p[mask].mean()) if n else 0.0,
            "observed_rate": float(y[mask].mean()) if n else 0.0,
        })
    return out


def brier_skill_score(y: np.ndarray, p: np.ndarray,
                      p_base: np.ndarray) -> float:
    """Prevalence-adjusted performance: 1 - Brier(model)/Brier(base).

    1.0 is perfect, 0.0 matches the base-rate predictor, negative is
    worse than predicting the training base rate. Returns 0.0 when
    both Briers are zero (identical perfect predictions).
    """
    base = brier(y, p_base)
    model = brier(y, p)
    if base == 0.0:
        return 0.0 if model == 0.0 else float("-inf")
    return float(1.0 - model / base)


def per_match_metric(frames: list[pd.DataFrame],
                     preds_by_match: dict[int, dict],
                     label: str, metric: str = "brier") -> pd.DataFrame:
    """One row per match: match_id, n_actions and the metric value.

    Aggregation is by match, never pooled across matches: paired
    comparisons below run over identical match sets.
    """
    fn = brier if metric == "brier" else log_loss
    rows = []
    for f in frames:
        mid = int(f["match_id"].iloc[0])
        rows.append({
            "match_id": mid,
            "n_actions": len(f),
            f"{metric}_{label}": fn(
                f[label].to_numpy(), preds_by_match[mid][label]),
        })
    return pd.DataFrame(rows).sort_values("match_id").reset_index(drop=True)


def paired_difference(a: pd.DataFrame, b: pd.DataFrame, value_col: str,
                      seed: int = 0, n_boot: int = 2000) -> dict:
    """Paired per-match difference of two score tables (a - b).

    Both tables must cover the identical match set, in any order.
    Returns the mean difference and a bootstrap 95% interval of the
    mean over matches, with a fixed seed.
    """
    merged = a[["match_id", value_col]].merge(
        b[["match_id", value_col]], on="match_id", suffixes=("_a", "_b")
    )
    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError(
            "paired difference needs identical match IDs in both tables: "
            f"got {len(a)} and {len(b)} rows, {len(merged)} matched"
        )
    diffs = (merged[f"{value_col}_a"] - merged[f"{value_col}_b"]
             ).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(diffs, size=(n_boot, len(diffs)),
                       replace=True).mean(axis=1)
    return {
        "mean_difference": float(np.mean(diffs)),
        "std_difference": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
        "n_matches": int(len(diffs)),
        "ci95": [float(np.quantile(boots, 0.025)),
                 float(np.quantile(boots, 0.975))],
    }


class SplitRegistry:
    """Frozen train/dev/test split, recorded by match ID.

    The three sets are disjoint by construction (overlap raises) and
    the digest binds the exact split: any change to membership
    changes the digest, so a later run cannot silently resplit.
    """

    def __init__(self, train_ids, dev_ids, test_ids):
        self.train_ids = sorted({int(i) for i in train_ids})
        self.dev_ids = sorted({int(i) for i in dev_ids})
        self.test_ids = sorted({int(i) for i in test_ids})
        train, dev, test = (set(self.train_ids), set(self.dev_ids),
                            set(self.test_ids))
        overlap = (train & dev) | (train & test) | (dev & test)
        if overlap:
            raise ValueError(
                f"split overlap on match IDs {sorted(overlap)}; train, "
                "dev and test must be disjoint"
            )

    @property
    def digest(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {"train_ids": self.train_ids, "dev_ids": self.dev_ids,
                "test_ids": self.test_ids}


def validate_no_fc6_selection() -> bool:
    """Leakage gate: no candidate is evaluated or selected on FC6.

    Returns True only when every candidate family in the catalog is
    still proposed, unevaluated, and never selected on an FC6 scored
    arm. Part 2 updates candidate statuses only after selection on
    the internal development split.
    """
    for fam in catalog.CANDIDATE_FAMILIES:
        if fam["evaluated"] or fam["selected_on_fc6"]:
            raise ValueError(
                f"candidate {fam['key']} ({fam['name']}) was evaluated or "
                "selected on FC6; the leakage gate forbids it"
            )
    return True


def stratify_rare_events(frames: list[pd.DataFrame],
                         labels: tuple = LABELS_TUPLE) -> dict:
    """Per-label prevalence over pooled frames, with a rare flag.

    Rare labels (fewer positives than RARE_POSITIVE_THRESHOLD) are
    flagged so no per-label claim is made on them without noting the
    thin support.
    """
    pooled = pd.concat(frames, ignore_index=True)
    out = {}
    for label in labels:
        prev = prevalence(pooled[label].to_numpy())
        out[label] = {
            **prev,
            "flag": ("rare" if prev["n_positive"] < RARE_POSITIVE_THRESHOLD
                     else "common"),
        }
    return out


def baseline_capture(train_frames: list[pd.DataFrame],
                     scored_frames: list[pd.DataFrame],
                     config_name: str = "fc6_pool",
                     seed: int = 0) -> dict:
    """Capture the frozen baseline on a train/score split.

    Trains the catalog baseline config on `train_frames` only and
    scores `scored_frames` (the development split in part 1; the
    frozen test split is never scored here). Records per-label Brier
    and log loss, prevalence, Brier skill against the training base
    rate, runtime, action counts and the catalog digest that binds
    the record to the contract it was captured under.
    """
    from effc.vaep.improved import CONFIGS, raw_predict, train_bundle

    validate_no_fc6_selection()
    config = CONFIGS[config_name]
    meta = pd.DataFrame([
        {"match_id": int(f["match_id"].iloc[0]),
         "home_team": str(f["team"].iloc[0]),
         "match_date": "2026-01-01"}
        for f in train_frames + scored_frames
    ])
    started = time.perf_counter()
    bundle = train_bundle(train_frames, meta, config, seed=seed)
    train_pool = pd.concat(train_frames, ignore_index=True)
    labels_out = {}
    for label in LABELS_TUPLE:
        ys, ps = [], []
        for f in scored_frames:
            ys.append(f[label].to_numpy())
            ps.append(raw_predict(bundle, f)[label])
        y = np.concatenate(ys)
        p = np.concatenate(ps)
        base_rate = float(train_pool[label].mean())
        labels_out[label] = {
            "brier": brier(y, p),
            "log_loss": log_loss(y, p),
            "prevalence": float(y.mean()),
            "n_positive": int(np.sum(y > 0.5)),
            "n_scored": int(len(y)),
            "train_base_rate": base_rate,
            "brier_skill_vs_base_rate": brier_skill_score(
                y, p, np.full(len(y), base_rate)),
        }
    return {
        "config_name": config_name,
        "feature_version": FEATURE_VERSION,
        "catalog_digest": catalog.catalog_digest(),
        "split_digest": SplitRegistry(
            [int(f["match_id"].iloc[0]) for f in train_frames],
            [int(f["match_id"].iloc[0]) for f in scored_frames],
            [],
        ).digest,
        "train_match_ids": sorted(
            int(f["match_id"].iloc[0]) for f in train_frames),
        "scored_match_ids": sorted(
            int(f["match_id"].iloc[0]) for f in scored_frames),
        "n_train_actions": int(sum(len(f) for f in train_frames)),
        "n_scored_actions": int(sum(len(f) for f in scored_frames)),
        "runtime_seconds": float(time.perf_counter() - started),
        "labels": labels_out,
    }
