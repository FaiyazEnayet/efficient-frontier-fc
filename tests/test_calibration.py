"""FC6: goal-model calibration is measured on held-out matches.

The pipeline used to train the goal models on all pre-split action frames
and report Brier scores computed on those same frames: an in-sample fit
presented as the headline accuracy number. It now trains on a chronological
train group, scores a held-out match group whose match IDs are disjoint
from the training set, compares against a training-set base-rate
baseline, and runs a genuine chronological season-transfer evaluation
(train on the earlier season, score on the later one).
"""

import numpy as np
import pandas as pd
import pytest

from effc.vaep import calibration as cal


def _frame(mid, n=20, seed=0, pos_rate=0.1):
    rng = np.random.default_rng(seed + mid)
    return pd.DataFrame(
        {
            "match_id": [mid] * n,
            "scores_incl": rng.random(n) < pos_rate,
            "scores_excl": rng.random(n) < pos_rate,
            "concedes_incl": rng.random(n) < pos_rate,
            "concedes_excl": rng.random(n) < pos_rate,
        }
    )


def _meta(rows):
    # rows: (match_id, competition_id, season_id, season_name, match_date)
    return pd.DataFrame(
        rows,
        columns=[
            "match_id",
            "competition_id",
            "season_id",
            "season_name",
            "match_date",
        ],
    )


def test_split_is_chronological_and_disjoint():
    frames = [_frame(mid, seed=mid) for mid in range(1, 21)]
    meta = _meta(
        [(mid, 9, 27, "2015/2016", f"2015-09-{mid:02d}") for mid in range(1, 21)]
    )
    train, calib = cal.match_group_split(frames, meta, calib_frac=0.2)
    train_ids = {int(f["match_id"].iloc[0]) for f in train}
    calib_ids = {int(f["match_id"].iloc[0]) for f in calib}
    assert train_ids.isdisjoint(calib_ids)
    # latest 20% by date become the calibration group
    assert calib_ids == {17, 18, 19, 20}
    assert train_ids == set(range(1, 17))


def test_split_applies_per_competition():
    frames = [_frame(mid, seed=mid) for mid in range(1, 21)]
    rows = [
        (mid, 9, 27, "2015/2016", f"2015-09-{mid:02d}") for mid in range(1, 11)
    ] + [
        (mid, 43, 106, "2022", f"2022-11-{mid - 10:02d}") for mid in range(11, 21)
    ]
    train, calib = cal.match_group_split(frames, _meta(rows), calib_frac=0.2)
    calib_ids = {int(f["match_id"].iloc[0]) for f in calib}
    assert calib_ids == {9, 10, 19, 20}


def test_split_small_group_stays_in_train():
    train, calib = cal.match_group_split(
        [_frame(1, seed=1)],
        _meta([(1, 9, 27, "2015/2016", "2015-09-01")]),
        calib_frac=0.5,
    )
    assert len(train) == 1 and len(calib) == 0


def test_assert_disjoint_raises_on_overlap():
    with pytest.raises(ValueError, match="disjoint"):
        cal.assert_disjoint([1, 2, 3], [3, 4])
    cal.assert_disjoint([1, 2], [3, 4])  # no raise


def test_log_loss_known_values():
    y = np.array([1, 0, 1, 0])
    assert cal.log_loss(y, np.full(4, 0.5)) == pytest.approx(np.log(2))
    assert cal.log_loss(y, np.array([1.0, 0.0, 1.0, 0.0])) < 1e-6


def test_per_match_scores_reports_per_match(monkeypatch):
    def fake_predict(models, df):
        n = len(df)
        return {label: np.full(n, 0.1) for label in cal.LABELS}

    monkeypatch.setattr(cal, "predict_proba", fake_predict)
    frames = [_frame(7, n=10, seed=3), _frame(9, n=10, seed=4)]
    out = cal.per_match_scores({}, frames)
    assert set(out["match_id"]) == {7, 9}
    assert (out["n_actions"] == 10).all()
    y = frames[0]["scores_incl"].to_numpy()
    got = out.loc[out["match_id"] == 7, "brier_scores_incl"].iloc[0]
    assert got == pytest.approx(float(np.mean((y - 0.1) ** 2)))


def test_base_rate_scores_uses_train_mean():
    train = [
        _frame(1, n=50, seed=10, pos_rate=0.3),
        _frame(2, n=50, seed=11, pos_rate=0.3),
    ]
    scored = [_frame(3, n=20, seed=12, pos_rate=0.1)]
    out = cal.base_rate_scores(train, scored, labels=("scores_incl",))
    p = float(pd.concat(train, ignore_index=True)["scores_incl"].mean())
    y = scored[0]["scores_incl"].to_numpy()
    assert out["brier_scores_incl"].iloc[0] == pytest.approx(
        float(np.mean((y - p) ** 2))
    )


def test_build_report_names_in_sample_explicitly():
    in_sample = {label: {"brier": 0.05, "logloss": 0.2} for label in cal.LABELS}
    summ = {
        label: {"brier": {"mean": 0.06}, "logloss": {"mean": 0.21}}
        for label in cal.LABELS
    }
    rep = cal.build_report(
        train_ids=[1, 2],
        calib_ids=[3, 4],
        in_sample_scores=in_sample,
        model_summary=summ,
        baseline_summary=summ,
    )
    bare = [
        k for k in rep if k.startswith("brier_") and not k.endswith("_in_sample")
    ]
    assert bare == [], f"bare Brier keys read as generalization claims: {bare}"
    assert rep["brier_scores_incl_in_sample"] == pytest.approx(0.05)
    assert rep["n_train_matches"] == 2 and rep["n_calib_matches"] == 2
    assert set(rep["held_out"]) == set(cal.LABELS)
    assert set(rep["held_out"]["scores_incl"]) == {"model", "base_rate_baseline"}


def test_build_report_rejects_overlap():
    with pytest.raises(ValueError, match="disjoint"):
        cal.build_report(
            train_ids=[1, 2],
            calib_ids=[2, 3],
            in_sample_scores={},
            model_summary={},
            baseline_summary={},
        )


def test_transfer_seasons_picks_earliest_and_latest():
    meta = _meta(
        [
            (1, 9, 27, "2015/2016", "2015-08-01"),
            (2, 9, 27, "2015/2016", "2015-12-01"),
            (3, 9, 281, "2023/2024", "2023-08-01"),
            (4, 9, 281, "2023/2024", "2023-12-01"),
            (5, 43, 106, "2022", "2022-11-01"),
        ]
    )
    tr, sc, names = cal.transfer_seasons(meta, competition_id=9)
    assert (tr, sc) == (27, 281)
    assert names == {27: "2015/2016", 281: "2023/2024"}
    with pytest.raises(ValueError, match="two seasons"):
        cal.transfer_seasons(meta[meta["competition_id"] == 43], competition_id=43)


def test_roster_overlap():
    mins = pd.DataFrame(
        {"player_id": [1, 2, 3, 4], "minutes": [900.0, 800.0, 100.0, 50.0]}
    )
    out = cal.roster_overlap({1, 2, 5}, {1, 2, 3, 4}, mins)
    assert out["n_overlap_players"] == 2
    assert out["player_overlap_frac"] == pytest.approx(0.5)
    assert out["minutes_overlap_frac"] == pytest.approx(1700.0 / 1850.0)
    same = cal.roster_overlap({1, 2}, {1, 2}, mins.iloc[:2])
    assert same["player_overlap_frac"] == 1.0
    assert same["minutes_overlap_frac"] == 1.0
    disjoint = cal.roster_overlap({9}, {1, 2}, mins.iloc[:2])
    assert disjoint["player_overlap_frac"] == 0.0
    assert disjoint["minutes_overlap_frac"] == 0.0


def test_summarize_ci_brackets_mean():
    per_match = pd.DataFrame({"match_id": range(10)})
    for label in cal.LABELS:
        per_match[f"brier_{label}"] = np.linspace(0.04, 0.08, 10)
        per_match[f"logloss_{label}"] = np.linspace(0.15, 0.25, 10)
    out = cal.summarize(per_match, labels=("scores_incl",), n_boot=500, seed=0)
    s = out["scores_incl"]["brier"]
    assert s["n_matches"] == 10
    assert s["ci95"][0] <= s["mean"] <= s["ci95"][1]
    assert s["std"] > 0
