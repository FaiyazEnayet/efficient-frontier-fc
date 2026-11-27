"""Stage 3 efficient-frontier-fc U07 part 1: source contract, baseline
capture, fixture and independent oracle.

Part 1 only. No performance promotion, no model selection, no new
architecture ships here: the catalog freezes the shipped fc6_pool
baseline and records the candidate families (a) to (f) as proposed
specifications, and the oracle in evaluation/development_trials.py
scores predictions independently of the shipped calibration module.

The fixture is synthetic StatsBomb-style events, built
deterministically from a seed. It is not FC6 data: no tuned feature,
cutoff, architecture or model is selected using either original FC6
scored arm, and the frozen test split (matches 105, 106) is never
scored in part 1.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from effc.evaluation import development_trials as oracle
from effc.models import feature_catalog as catalog
from effc.vaep import calibration as cal
from effc.vaep.features import FEATURE_COLS, FEATURE_VERSION, featurize_match
from effc.vaep.improved import CONFIGS
from effc.vaep.models import LABELS, brier_score

TRAIN_IDS = (101, 102, 103)
DEV_IDS = (104,)
TEST_IDS = (105, 106)
N_ACTIONS = 120
FIXTURE_SEED = 7

BASELINE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "runs",
    "u07_part1_baseline.json",
)


def _synthetic_events(match_id, seed, n_actions=N_ACTIONS):
    """Deterministic StatsBomb-style events for one synthetic match."""
    rng = np.random.default_rng(seed)
    types = ["Pass", "Carry", "Dribble", "Shot", "Ball Recovery",
             "Interception", "Duel", "Clearance"]
    weights = [0.55, 0.12, 0.08, 0.12, 0.05, 0.04, 0.02, 0.02]
    events = []
    for i in range(n_actions):
        tname = str(rng.choice(types, p=weights))
        team = "Home" if rng.random() < 0.5 else "Away"
        x = float(rng.uniform(5, 115))
        y = float(rng.uniform(5, 75))
        ev = {
            "index": i,
            "period": 1 if i < n_actions // 2 else 2,
            "type": {"name": tname},
            "team": {"name": team},
            "player": {"id": int(rng.integers(1, 23)),
                       "name": f"Player {int(rng.integers(1, 23))}"},
            "location": [x, y],
            "minute": int(i * 0.75),
            "second": 0,
        }
        if rng.random() < 0.3:
            ev["under_pressure"] = True
        ex = float(np.clip(x + rng.normal(0, 12), 0, 120))
        ey = float(np.clip(y + rng.normal(0, 10), 0, 80))
        if tname == "Pass":
            ev["pass"] = {"end_location": [ex, ey]}
        elif tname == "Carry":
            ev["carry"] = {"end_location": [ex, ey]}
        elif tname == "Shot":
            # Learnable signal, as in real data: a shot close to the
            # goal scores far more often than a long-range one, so the
            # baseline has something to learn from the distance
            # features instead of pure noise.
            dist_to_goal = ((120.0 - x) ** 2 + (40.0 - y) ** 2) ** 0.5
            goal_prob = 0.45 if dist_to_goal < 25.0 else 0.08
            outcome = "Goal" if rng.random() < goal_prob else "Saved"
            ev["shot"] = {"end_location": [ex, ey],
                          "outcome": {"name": outcome}}
        events.append(ev)
    return events


def build_fixture_frames():
    """Featurized frames for the frozen train/dev/test split."""
    frames = {}
    for mid in TRAIN_IDS + DEV_IDS + TEST_IDS:
        frames[mid] = featurize_match(
            _synthetic_events(mid, FIXTURE_SEED + mid), match_id=mid
        )
    return frames


@pytest.fixture(scope="module")
def fixture_frames():
    return build_fixture_frames()


def test_stage3_efficient_frontier_fc_07_part_1_catalog_freezes_fc6_pool():
    base = catalog.FC6_BASELINE
    assert base["config_name"] == "fc6_pool"
    assert base["gbm_params"] == dict(CONFIGS["fc6_pool"]["gbm"])
    assert base["feature_version"] == FEATURE_VERSION
    assert base["lookahead_k"] == 10
    assert list(base["labels"]) == list(LABELS)
    assert base["seed"] == 0
    assert base["use_strengths"] is False
    assert catalog.CATALOG_VERSION == 1


def test_stage3_efficient_frontier_fc_07_part_1_catalog_lists_candidates_proposed_only():
    keys = [fam["key"] for fam in catalog.CANDIDATE_FAMILIES]
    assert keys == ["a", "b", "c", "d", "e", "f"]
    for fam in catalog.CANDIDATE_FAMILIES:
        assert fam["status"] == "proposed"
        assert fam["evaluated"] is False
        assert fam["selected_on_fc6"] is False
        assert fam["required_raw_fields"], fam["key"]
        assert fam["availability"], fam["key"]
    names = [fam["name"] for fam in catalog.CANDIDATE_FAMILIES]
    assert "corrected-geometry existing GBM" in names
    assert "regularized logistic comparator" in names
    assert "compact sequence model" in names


def test_stage3_efficient_frontier_fc_07_part_1_schema_accepts_shipped_matrix(
    fixture_frames,
):
    frame = fixture_frames[TRAIN_IDS[0]]
    from effc.vaep.features import design_matrix

    X = design_matrix(frame)
    catalog.check_schema(X)
    for col in FEATURE_COLS:
        assert col in X.columns
    extra = [c for c in X.columns if c not in FEATURE_COLS]
    assert extra, "fixture matrix carries no categorical dummies"
    assert all(c.startswith(("acti_", "prev_")) or c in
               catalog.STRENGTH_COLUMNS for c in extra)


def test_stage3_efficient_frontier_fc_07_part_1_schema_rejects_unknown_feature(
    fixture_frames,
):
    from effc.vaep.features import design_matrix

    X = design_matrix(fixture_frames[TRAIN_IDS[0]]).copy()
    X["scores_incl"] = fixture_frames[TRAIN_IDS[0]]["scores_incl"]
    with pytest.raises(ValueError, match="unexpected feature"):
        catalog.check_schema(X)
    X2 = design_matrix(fixture_frames[TRAIN_IDS[0]]).drop(
        columns=["start_x"]
    )
    with pytest.raises(ValueError, match="missing feature"):
        catalog.check_schema(X2)


def test_stage3_efficient_frontier_fc_07_part_1_causality_future_goal_keeps_past_features():
    base_events = _synthetic_events(201, FIXTURE_SEED + 201)
    changed_events = [dict(e) for e in base_events]
    # Flip one late shot from a goal to a save: only the label window
    # and later score_diff may change, never earlier features.
    late_shots = [i for i, e in enumerate(changed_events)
                  if e["type"]["name"] == "Shot"
                  and e.get("shot", {}).get("outcome", {}).get("name") == "Goal"]
    assert late_shots, "fixture has no late goal to flip"
    flip = late_shots[-1]
    changed_events[flip] = dict(changed_events[flip])
    changed_events[flip]["shot"] = dict(changed_events[flip]["shot"])
    changed_events[flip]["shot"]["outcome"] = {"name": "Saved"}
    df_base = featurize_match(base_events, match_id=201)
    df_changed = featurize_match(changed_events, match_id=201)
    assert len(df_base) == len(df_changed)
    feat_cols = [c for c in df_base.columns
                 if c not in ("scores_incl", "scores_excl",
                              "concedes_incl", "concedes_excl",
                              "is_goal", "score_diff")]
    flip_row = int(np.where(df_base["match_index"].to_numpy() == flip)[0][0])
    before = df_base.iloc[:flip_row][feat_cols].reset_index(drop=True)
    after = df_changed.iloc[:flip_row][feat_cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after)
    # The flip is real: some label in the lookahead window changes.
    assert not df_base["scores_incl"].equals(df_changed["scores_incl"])


def test_stage3_efficient_frontier_fc_07_part_1_availability_rejects_future_features():
    good = {"key": "b", "required_raw_fields": ["pass.outcome.name",
                                                "pass.height.name"]}
    catalog.validate_feature_availability(good)
    for bad_field in ("scores_incl", "next_goal", "future_window_outcome",
                      "concedes_excl", "lookahead_label"):
        with pytest.raises(ValueError, match="future"):
            catalog.validate_feature_availability(
                {"key": "x", "required_raw_fields": [bad_field]}
            )
    with pytest.raises(ValueError, match="known raw event field"):
        catalog.validate_feature_availability(
            {"key": "x", "required_raw_fields": ["not_a_statsbomb_field"]}
        )


def test_stage3_efficient_frontier_fc_07_part_1_oracle_matches_reference_implementation():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.1).astype(float)
    p = rng.random(500) * 0.3
    assert oracle.brier(y, p) == pytest.approx(brier_score(y, p), abs=1e-12)
    assert oracle.log_loss(y, p) == pytest.approx(cal.log_loss(y, p), abs=1e-12)
    assert oracle.brier(y, p) == pytest.approx(float(np.mean((y - p) ** 2)))


def test_stage3_efficient_frontier_fc_07_part_1_oracle_bins_prevalence_and_skill():
    rng = np.random.default_rng(1)
    y = (rng.random(1000) < 0.05).astype(float)
    p = np.clip(rng.random(1000) * 0.2, 0.001, 0.999)
    bins = oracle.reliability_bins(y, p, n_bins=10)
    assert sum(b["n"] for b in bins) == 1000
    assert all(0.0 <= b["observed_rate"] <= 1.0 for b in bins)
    assert all(0.0 <= b["mean_predicted"] <= 1.0 for b in bins)
    prev = oracle.prevalence(y)
    assert prev["n"] == 1000
    assert prev["n_positive"] == int(y.sum())
    assert prev["rate"] == pytest.approx(float(y.mean()))
    base = np.full(1000, float(y.mean()))
    assert oracle.brier_skill_score(y, base, base) == pytest.approx(0.0)
    assert oracle.brier_skill_score(y, p, base) <= 1.0
    # The skill score responds to real signal: an informative predictor
    # beats the base rate by a wide margin.
    p_good = np.where(y == 1.0, 0.8, 0.05)
    assert oracle.brier_skill_score(y, p_good, base) > 0.5
    # A label with no positives stays finite and reports zero prevalence.
    y_zero = np.zeros(50)
    assert oracle.prevalence(y_zero)["rate"] == 0.0
    assert np.isfinite(oracle.log_loss(y_zero, np.full(50, 0.02)))


def test_stage3_efficient_frontier_fc_07_part_1_split_registry_disjoint_and_frozen():
    reg = oracle.SplitRegistry(TRAIN_IDS, DEV_IDS, TEST_IDS)
    assert reg.train_ids == [101, 102, 103]
    assert reg.dev_ids == [104]
    assert reg.test_ids == [105, 106]
    assert reg.digest == oracle.SplitRegistry(
        TRAIN_IDS, DEV_IDS, TEST_IDS).digest
    assert len(reg.digest) == 16
    d = reg.to_dict()
    assert d["train_ids"] == [101, 102, 103]
    assert d["test_ids"] == [105, 106]
    with pytest.raises(ValueError, match="overlap"):
        oracle.SplitRegistry((101, 102), (102,), (105,))
    with pytest.raises(ValueError, match="overlap"):
        oracle.SplitRegistry((101,), (104,), (101,))


def test_stage3_efficient_frontier_fc_07_part_1_paired_difference_matches_manual():
    a = pd.DataFrame({"match_id": [1, 2, 3], "brier_scores_incl": [0.05, 0.06, 0.04]})
    b = pd.DataFrame({"match_id": [1, 2, 3], "brier_scores_incl": [0.07, 0.06, 0.05]})
    out = oracle.paired_difference(a, b, "brier_scores_incl", seed=0)
    manual = np.array([-0.02, 0.0, -0.01])
    assert out["mean_difference"] == pytest.approx(float(manual.mean()))
    assert out["n_matches"] == 3
    assert out["ci95"][0] <= out["mean_difference"] <= out["ci95"][1]
    same = oracle.paired_difference(a, a, "brier_scores_incl", seed=0)
    assert same["mean_difference"] == pytest.approx(0.0)
    assert same["ci95"] == [0.0, 0.0]
    with pytest.raises(ValueError, match="match IDs"):
        oracle.paired_difference(a, b.drop(index=[0]), "brier_scores_incl")


def test_stage3_efficient_frontier_fc_07_part_1_leakage_gate_no_fc6_selection():
    assert oracle.validate_no_fc6_selection() is True
    assert catalog.leakage_gate_statement().startswith("No tuned feature")


def test_stage3_efficient_frontier_fc_07_part_1_rare_event_stratification(
    fixture_frames,
):
    frames = [fixture_frames[m] for m in TRAIN_IDS + DEV_IDS]
    strata = oracle.stratify_rare_events(frames)
    for label in LABELS:
        assert strata[label]["n"] == sum(len(f) for f in frames)
        assert strata[label]["rate"] == pytest.approx(
            float(pd.concat(frames, ignore_index=True)[label].mean())
        )
        assert strata[label]["n_positive"] > 0
    assert all(s["flag"] in ("common", "rare") for s in strata.values())


def test_stage3_efficient_frontier_fc_07_part_1_baseline_capture_frozen(
    fixture_frames,
):
    train_frames = [fixture_frames[m] for m in TRAIN_IDS]
    scored_frames = [fixture_frames[m] for m in DEV_IDS]
    captured = oracle.baseline_capture(
        train_frames, scored_frames, seed=0
    )
    assert captured["config_name"] == "fc6_pool"
    assert captured["feature_version"] == FEATURE_VERSION
    assert captured["catalog_digest"] == catalog.catalog_digest()
    assert captured["train_match_ids"] == [101, 102, 103]
    assert captured["scored_match_ids"] == [104]
    assert captured["n_train_actions"] == sum(len(f) for f in train_frames)
    assert captured["n_scored_actions"] == sum(len(f) for f in scored_frames)
    assert captured["runtime_seconds"] >= 0.0
    for label in LABELS:
        entry = captured["labels"][label]
        assert 0.0 <= entry["brier"] <= 1.0
        assert np.isfinite(entry["log_loss"])
        assert 0.0 <= entry["prevalence"] <= 1.0
        assert entry["n_positive"] >= 0
        assert np.isfinite(entry["brier_skill_vs_base_rate"])
    # The frozen artifact records the same capture.
    assert os.path.exists(BASELINE_PATH), "runs/u07_part1_baseline.json missing"
    with open(BASELINE_PATH) as fh:
        frozen = json.load(fh)
    assert frozen["config_name"] == captured["config_name"]
    assert frozen["catalog_digest"] == captured["catalog_digest"]
    assert frozen["train_match_ids"] == captured["train_match_ids"]
    assert frozen["scored_match_ids"] == captured["scored_match_ids"]
    for label in LABELS:
        assert frozen["labels"][label]["brier"] == pytest.approx(
            captured["labels"][label]["brier"], abs=1e-12
        )
        assert frozen["labels"][label]["log_loss"] == pytest.approx(
            captured["labels"][label]["log_loss"], abs=1e-12
        )
    # The frozen test split is never scored in part 1.
    assert set(frozen["scored_match_ids"]).isdisjoint(TEST_IDS)
