"""Improved goal models: calibration/blend contract, cross-fitting
coverage, and the leakage guards (test matches can never enter a
training pool; calibration sees training-group matches only)."""

import numpy as np
import pandas as pd
import pytest

from effc.vaep import calibration as cal
from effc.vaep import improved as imp
from effc.vaep.models import LABELS, predict_proba
from tests.test_team_strength import league, make_frame, make_meta


class _ConstClf:
    """predict_proba stub returning a constant probability."""

    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        p = np.full(len(X), self.p)
        return np.column_stack([1 - p, p])


def test_calibrated_model_blend_extremes():
    X = np.zeros((5, 3))
    pure = imp.CalibratedGoalModel(_ConstClf(0.4), blend_w=0.0, base_rate=0.1)
    out = pure.predict_proba(X)[:, 1]
    assert np.allclose(out, 0.4)  # identity Platt, no blending
    shrunk = imp.CalibratedGoalModel(_ConstClf(0.4), blend_w=1.0, base_rate=0.1)
    out = shrunk.predict_proba(X)[:, 1]
    assert np.allclose(out, 0.1)  # full shrinkage -> the base rate itself
    mid = imp.CalibratedGoalModel(_ConstClf(0.4), blend_w=0.5, base_rate=0.1)
    out = mid.predict_proba(X)[:, 1]
    assert np.all((out >= 0.0) & (out <= 1.0))
    assert np.allclose(out, 0.25)


def test_train_bundle_and_predict_contract():
    frames, meta = league()
    bundle = imp.train_bundle(frames, meta, imp.CONFIGS["dc365"], seed=0)
    assert bundle["columns"]
    dc = bundle["strengths"]
    assert dc.att("Strong") > dc.att("Weak")
    preds = predict_proba(bundle, frames[0])
    for label in LABELS:
        p = preds[label]
        assert p.shape == (len(frames[0]),)
        assert np.all((p >= 0.0) & (p <= 1.0))


def test_crossfit_oof_covers_each_group_match_once():
    frames, meta = league()
    group = frames[:8]
    pool = frames[8:]
    oof = imp.crossfit_oof(group, pool, meta, imp.CONFIGS["fc6_pool"], seed=0)
    group_ids = {int(f["match_id"].iloc[0]) for f in group}
    assert set(oof) == group_ids
    for mid, preds in oof.items():
        for label in LABELS:
            assert len(preds[label]) == len(group[0])


def test_calibration_uses_group_frames_only():
    frames, meta = league()
    group = frames[:8]
    oof = {}
    by_id = {}
    for f in group:
        mid = int(f["match_id"].iloc[0])
        by_id[mid] = f
        oof[mid] = {
            label: np.full(len(f), 0.05) for label in LABELS
        }
    calib = imp.fit_calibration(oof, group)
    for label in LABELS:
        expected_base = float(
            np.concatenate([f[label].to_numpy() for f in group]).mean()
        )
        assert calib[label][3] == pytest.approx(expected_base)
        a, b, w, _ = calib[label]
        assert 0.0 <= w <= 0.6


def test_leakage_guard_rejects_planted_overlap():
    train_ids = {1, 2, 3, 4}
    held_out_ids = {5, 6}
    transfer_ids = {7, 8}
    cal.assert_disjoint(train_ids, held_out_ids)
    cal.assert_disjoint(train_ids, transfer_ids)
    cal.assert_disjoint(held_out_ids, transfer_ids)
    with pytest.raises(ValueError):
        cal.assert_disjoint(train_ids | {6}, held_out_ids)
    with pytest.raises(ValueError):
        cal.assert_disjoint(train_ids | {8}, transfer_ids)


def test_chronological_folds_partition_group():
    frames, meta = league()
    dates = dict(
        zip(meta["match_id"].astype(int), meta["match_date"].astype(str))
    )
    folds = imp.chronological_folds(frames, dates, n_folds=4)
    seen = [int(f["match_id"].iloc[0]) for fold in folds for f in fold]
    assert sorted(seen) == sorted(dates)
    assert len(seen) == len(set(seen))
    # chronological: fold match dates are non-decreasing across folds
    fold_max = [
        max(str(dates[int(f["match_id"].iloc[0])]) for f in fold) for fold in folds
    ]
    assert fold_max == sorted(fold_max)
