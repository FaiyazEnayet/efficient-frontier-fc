"""Stage 3 efficient-frontier-fc U07 part 2: candidate trials and
frozen benchmark.

Part 2 only. The candidate families (a) to (f) from the frozen
catalog are implemented and evaluated as an ablation on the internal
development split (match 104); selection uses that split only, and
the frozen separate benchmark (matches 105, 106) is scored once
after selection. No candidate is selected using either original FC6
scored arm, and no newly selected model is judged by FC6.

The fixture is the part-1 synthetic fixture with extra StatsBomb
fields added deterministically (no random draw is consumed, so the
featurized frames are bit-identical to part 1 and the family (a)
numbers reproduce the frozen part-1 artifact exactly). The fixture
carries almost no learnable signal, so it verifies the trial
machinery; a null outcome here is a statement about the fixture,
not about the shipped model on real data.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from effc.evaluation import candidate_trials as trials
from effc.evaluation import development_trials as oracle
from effc.models import candidate_features as cand_feat
from effc.models import feature_catalog as catalog
from effc.vaep.features import featurize_match
from effc.vaep.models import LABELS
from tests.test_stage3_efficient_frontier_fc_07_part_1 import (
    DEV_IDS,
    FIXTURE_SEED,
    TEST_IDS,
    TRAIN_IDS,
    _synthetic_events,
    build_fixture_frames,
)

ARTIFACT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "runs",
    "u07_part2_ablation.json",
)
BASELINE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "runs",
    "u07_part1_baseline.json",
)


def _enrich_events(events, match_id):
    """Add family (b) raw fields to part-1 synthetic events.

    Deterministic in (match_id, position) only: no random draw is
    consumed, so featurize_match sees the same events part 1 saw
    plus extra fields it ignores, and the frames stay identical.
    """
    out = []
    for i, e in enumerate(events):
        e = dict(e)
        tname = e["type"]["name"]
        if tname == "Pass":
            p = dict(e.get("pass", {}))
            if (match_id * 7 + i * 13) % 10 == 8:
                p["outcome"] = {"name": "Incomplete"}
            p["height"] = {"name": "High" if i % 5 == 0 else "Ground"}
            p["type"] = {"name": "Through Ball" if i % 11 == 0 else "Normal"}
            p["body_part"] = {"name": "Head" if i % 7 == 0 else "Foot"}
            e["pass"] = p
        elif tname == "Shot":
            s = dict(e.get("shot", {}))
            s["body_part"] = {"name": "Head" if i % 3 == 0 else "Right Foot"}
            s["type"] = {"name": "Normal"}
            e["shot"] = s
        out.append(e)
    return out


def build_rich_fixture():
    """Featurized frames plus raw events for the frozen split."""
    frames, events = {}, {}
    for mid in TRAIN_IDS + DEV_IDS + TEST_IDS:
        evs = _enrich_events(_synthetic_events(mid, FIXTURE_SEED + mid), mid)
        events[mid] = evs
        frames[mid] = featurize_match(evs, match_id=mid)
    return frames, events


@pytest.fixture(scope="module")
def rich_fixture():
    return build_rich_fixture()


@pytest.fixture(scope="module")
def ablation(rich_fixture):
    frames, events = rich_fixture
    return trials.run_ablation(
        [frames[m] for m in TRAIN_IDS],
        [frames[m] for m in DEV_IDS],
        train_events_by_id={m: events[m] for m in TRAIN_IDS},
        dev_events_by_id={m: events[m] for m in DEV_IDS},
        seed=0,
    )


@pytest.fixture(scope="module")
def selection(ablation):
    return trials.select_candidate(ablation)


@pytest.fixture(scope="module")
def benchmark(ablation, selection, rich_fixture):
    frames, events = rich_fixture
    return trials.score_frozen_benchmark(
        ablation, selection,
        [frames[m] for m in TRAIN_IDS],
        [frames[m] for m in DEV_IDS],
        [frames[m] for m in TEST_IDS],
        train_events_by_id={m: events[m] for m in TRAIN_IDS},
        dev_events_by_id={m: events[m] for m in DEV_IDS},
        test_events_by_id={m: events[m] for m in TEST_IDS},
        seed=0,
    )


def test_stage3_efficient_frontier_fc_07_part_2_enriched_fixture_matches_part_1(
    rich_fixture,
):
    frames, _ = rich_fixture
    part1 = build_fixture_frames()
    for mid in TRAIN_IDS + DEV_IDS + TEST_IDS:
        pd.testing.assert_frame_equal(frames[mid], part1[mid])


def test_stage3_efficient_frontier_fc_07_part_2_richer_context_known_answer():
    events = [
        {"index": 0, "period": 1, "type": {"name": "Pass"},
         "team": {"name": "Home"}, "player": {"id": 1, "name": "A"},
         "location": [60.0, 40.0], "minute": 10, "second": 0,
         "pass": {"end_location": [80.0, 40.0],
                  "height": {"name": "High"},
                  "type": {"name": "Through Ball"},
                  "body_part": {"name": "Head"}}},
        {"index": 1, "period": 1, "type": {"name": "Carry"},
         "team": {"name": "Home"}, "player": {"id": 1, "name": "A"},
         "location": [80.0, 40.0], "minute": 10, "second": 5,
         "carry": {"end_location": [90.0, 40.0]}},
        {"index": 2, "period": 1, "type": {"name": "Shot"},
         "team": {"name": "Home"}, "player": {"id": 2, "name": "B"},
         "location": [100.0, 40.0], "minute": 11, "second": 0,
         "shot": {"end_location": [118.0, 40.0],
                  "outcome": {"name": "Goal"},
                  "body_part": {"name": "Head"},
                  "type": {"name": "Normal"}}},
        {"index": 3, "period": 1, "type": {"name": "Pass"},
         "team": {"name": "Away"}, "player": {"id": 3, "name": "C"},
         "location": [30.0, 40.0], "minute": 12, "second": 0,
         "pass": {"end_location": [40.0, 30.0],
                  "outcome": {"name": "Incomplete"},
                  "height": {"name": "Ground"},
                  "type": {"name": "Normal"},
                  "body_part": {"name": "Foot"}}},
    ]
    frame = featurize_match(events, match_id=301)
    rich = cand_feat.extend_frame_with_richer_context(frame, events)
    assert list(rich["rich_pass_height"]) == [
        "High", "NA", "NA", "Ground"]
    assert list(rich["rich_pass_type"]) == [
        "Through Ball", "NA", "NA", "Normal"]
    assert list(rich["rich_pass_body_part"]) == [
        "Head", "NA", "NA", "Foot"]
    assert list(rich["rich_pass_complete"]) == [1.0, 0.0, 0.0, 0.0]
    assert list(rich["rich_is_carry"]) == [0.0, 1.0, 0.0, 0.0]
    assert rich["rich_carry_length"].iloc[1] == pytest.approx(8.75)
    assert list(rich["rich_shot_body_part"]) == ["NA", "NA", "Head", "NA"]
    assert list(rich["rich_prev_prev_action"]) == [
        "None", "None", "Pass", "Carry"]
    assert list(rich["rich_prev_prev_same_team"]) == [0.0, 0.0, 1.0, 0.0]
    X = cand_feat.extended_design_matrix(rich, "b")
    assert "rich_pass_complete" in X.columns
    assert "rich_pass_height_High" in X.columns
    assert "rich_prev_prev_action_Pass" in X.columns
    # Family (a) is the shipped matrix unchanged.
    Xa = cand_feat.extended_design_matrix(frame, "a")
    from effc.vaep.features import design_matrix
    pd.testing.assert_frame_equal(Xa, design_matrix(frame))


def test_stage3_efficient_frontier_fc_07_part_2_richer_context_errors():
    frame = pd.DataFrame({"match_index": [0], "action": ["Pass"]})
    with pytest.raises(ValueError, match="match_index"):
        cand_feat.extend_frame_with_richer_context(
            frame.drop(columns=["match_index"]), [])
    with pytest.raises(ValueError, match="unknown trial family"):
        cand_feat.extended_design_matrix(frame, "z")
    with pytest.raises(ValueError, match="richer-context columns"):
        cand_feat.extended_design_matrix(frame, "b")
    with pytest.raises(ValueError, match="fixed at 3"):
        cand_feat.sequence_window_matrix(frame, window=2)


def test_stage3_efficient_frontier_fc_07_part_2_availability_validates_family_b():
    assert cand_feat.validate_candidate_spec() is True
    spec = [fam for fam in catalog.CANDIDATE_FAMILIES if fam["key"] == "b"][0]
    for field in spec["required_raw_fields"]:
        assert field in catalog.ALLOWED_RAW_FIELDS
    with pytest.raises(ValueError, match="future"):
        catalog.validate_feature_availability(
            {"key": "x", "required_raw_fields": ["scores_incl"]}
        )


def test_stage3_efficient_frontier_fc_07_part_2_causality_future_goal_keeps_past_rich(
    rich_fixture,
):
    _, events = rich_fixture
    base_events = _enrich_events(_synthetic_events(201, FIXTURE_SEED + 201), 201)
    changed_events = [dict(e) for e in base_events]
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
    rich_base = cand_feat.extend_frame_with_richer_context(df_base, base_events)
    rich_changed = cand_feat.extend_frame_with_richer_context(
        df_changed, changed_events)
    flip_row = int(np.where(
        df_base["match_index"].to_numpy() == flip)[0][0])
    before = rich_base.iloc[:flip_row][cand_feat.RICH_COLUMNS].reset_index(drop=True)
    after = rich_changed.iloc[:flip_row][cand_feat.RICH_COLUMNS].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after)
    assert not df_base["scores_incl"].equals(df_changed["scores_incl"])


def test_stage3_efficient_frontier_fc_07_part_2_ablation_covers_all_families(
    ablation,
):
    assert ablation["evaluation_order"] == ["a", "b", "c", "d", "e", "f"]
    assert list(ablation["candidates"]) == ["a", "b", "c", "d", "e", "f"]
    assert ablation["selection_split_ids"] == [104]
    assert ablation["catalog_digest"] == catalog.catalog_digest()
    for key in ("b", "c", "e", "f"):
        rec = ablation["candidates"][key]
        assert rec["status"] == "evaluated", key
        assert rec["selected_on_fc6"] is False
        assert rec["config"], key
        assert rec["n_train_actions"] == 360
        assert rec["n_scored_actions"] == 120
        for label in LABELS:
            entry = rec["labels"][label]
            assert 0.0 <= entry["brier"] <= 1.0
            assert np.isfinite(entry["log_loss"])
            assert np.isfinite(rec["mean_headline_brier"])
        assert rec["paired_vs_baseline"] is not None
        for label in trials.HEADLINE_LABELS:
            paired = rec["paired_vs_baseline"][label]
            assert paired["n_matches"] == 1
            assert paired["ci95"][0] <= paired["mean_difference"] \
                <= paired["ci95"][1]
        bins = rec["reliability_bins"]["scores_incl"]
        assert sum(b["n"] for b in bins) == 120


def test_stage3_efficient_frontier_fc_07_part_2_baseline_reproduces_part_1(
    ablation,
):
    rec_a = ablation["candidates"]["a"]
    with open(BASELINE_PATH) as fh:
        frozen = json.load(fh)
    assert rec_a["config"]["config_name"] == "fc6_pool"
    for label in LABELS:
        assert rec_a["labels"][label]["brier"] == pytest.approx(
            frozen["labels"][label]["brier"], abs=1e-12)
        assert rec_a["labels"][label]["log_loss"] == pytest.approx(
            frozen["labels"][label]["log_loss"], abs=1e-12)


def test_stage3_efficient_frontier_fc_07_part_2_external_library_recorded_failed(
    ablation,
):
    rec_d = ablation["candidates"]["d"]
    assert rec_d["status"] == "failed"
    assert rec_d["labels"] == {}
    assert "LightGBM" in rec_d["failure_reason"]
    assert "CatBoost" in rec_d["failure_reason"]
    probes = rec_d["config"]["probes"]
    assert probes["lightgbm"]["available"] is False
    assert probes["catboost"]["available"] is False
    assert trials.probe_external_library("lightgbm")["available"] is False
    assert trials.probe_external_library("catboost")["available"] is False


def test_stage3_efficient_frontier_fc_07_part_2_tuned_grid_and_sequence_order(
    ablation,
):
    rec_c = ablation["candidates"]["c"]
    grid = rec_c["grid_results"]
    assert len(grid) == len(trials.TUNED_GBM_GRID)
    assert [g["gbm"] for g in grid] == trials.TUNED_GBM_GRID
    best = min(grid, key=lambda g: g["mean_headline_brier"])
    assert rec_c["config"]["gbm"] == best["gbm"]
    assert rec_c["mean_headline_brier"] == pytest.approx(
        best["mean_headline_brier"])
    rec_f = ablation["candidates"]["f"]
    assert rec_f["evaluated_after"] == ["b", "c", "e"]
    assert rec_f["config"]["window"] == 3


def test_stage3_efficient_frontier_fc_07_part_2_rare_event_stratification(
    rich_fixture,
):
    frames, _ = rich_fixture
    strata = oracle.stratify_rare_events(
        [frames[m] for m in TRAIN_IDS + DEV_IDS])
    for label in LABELS:
        assert strata[label]["n"] == 480
        assert strata[label]["flag"] in ("common", "rare")
        assert strata[label]["n_positive"] > 0


def test_stage3_efficient_frontier_fc_07_part_2_selection_uses_dev_only(
    ablation, selection,
):
    assert selection["selection_split_ids"] == [104]
    assert set(selection["selection_split_ids"]).isdisjoint(TEST_IDS)
    assert trials.validate_selection_not_on_fc6(ablation, selection) is True
    best = selection["best_challenger"]
    assert best in ("b", "c", "e", "f")
    assert selection["best_mean_headline_brier"] == pytest.approx(
        ablation["candidates"][best]["mean_headline_brier"])
    if selection["selected"] is None:
        assert selection["improvement"] < trials.SELECTION_MARGIN
    else:
        assert selection["improvement"] >= trials.SELECTION_MARGIN
        assert selection["selected"] == best


def test_stage3_efficient_frontier_fc_07_part_2_frozen_benchmark_scored_once(
    ablation, selection, benchmark,
):
    assert benchmark["scored_once"] is True
    assert benchmark["test_ids"] == [105, 106]
    assert benchmark["selected"] == selection["selected"]
    assert set(benchmark["train_ids"]) == set(TRAIN_IDS + DEV_IDS)
    assert "a" in benchmark["models"]
    base = benchmark["models"]["a"]
    assert base["n_train_actions"] == 480
    assert base["n_scored_actions"] == 240
    for label in LABELS:
        assert 0.0 <= base["labels"][label]["brier"] <= 1.0
        assert np.isfinite(base["labels"][label]["log_loss"])
    if selection["selected"] is not None:
        chall = benchmark["models"][selection["selected"]]
        assert chall["paired_vs_baseline"] is not None
        for label in trials.HEADLINE_LABELS:
            assert chall["paired_vs_baseline"][label]["n_matches"] == 2
        paired = chall["paired_headline_mean"]
        assert paired["n_matches"] == 2
        assert paired["mean_difference"] == pytest.approx(
            chall["mean_headline_brier"]
            - base["mean_headline_brier"])
    promo = benchmark["promotion"]
    if selection["selected"] is None:
        assert promo["promoted"] is None
    elif promo["promoted"] is not None:
        assert promo["promoted"] == selection["selected"]
        assert promo["improvement"] >= trials.SELECTION_MARGIN
        assert promo["paired_headline_mean"]["ci95"][1] < 0.0
        assert promo["mean_headline_brier_skill"] > 0.0
    else:
        # Not promoted: the point margin was missed, the paired
        # interval covers zero, or the challenger still loses to the
        # base rate, and the reason says which.
        assert promo["improvement"] < trials.SELECTION_MARGIN or \
            promo["paired_headline_mean"]["ci95"][1] >= 0.0 or \
            promo["mean_headline_brier_skill"] <= 0.0


def test_stage3_efficient_frontier_fc_07_part_2_artifact_matches_trial(
    ablation, selection, benchmark, rich_fixture,
):
    assert os.path.exists(ARTIFACT_PATH), "runs/u07_part2_ablation.json missing"
    with open(ARTIFACT_PATH) as fh:
        frozen = json.load(fh)
    assert frozen["artifact"] == "u07 part 2 ablation"
    assert frozen["catalog_digest"] == catalog.catalog_digest()
    assert frozen["selection_split_ids"] == [104]
    assert frozen["split_registry"]["test_ids"] == [105, 106]
    assert frozen["selection"]["selected"] == selection["selected"]
    assert frozen["benchmark"]["test_ids"] == [105, 106]
    assert frozen["benchmark"]["selected"] == selection["selected"]
    # The promotion decision agrees up to floating-point noise, not
    # bit-exactly: the logistic comparator runs through BLAS, whose
    # summation order varies with thread count, so a recomputation
    # in a different thread environment differs by about 1e-7. The
    # 1e-6 tolerances below are far tighter than the 0.002 selection
    # margin, so they verify the numbers without pretending to a
    # bit-exactness the estimator does not have.
    frozen_promo = frozen["benchmark"]["promotion"]
    promo = benchmark["promotion"]
    assert frozen_promo["promoted"] == promo["promoted"]
    assert ("nothing is promoted" in frozen_promo["reason"]) == \
        ("nothing is promoted" in promo["reason"])
    if "improvement" in promo:
        assert frozen_promo["improvement"] == pytest.approx(
            promo["improvement"], abs=1e-6)
        assert frozen_promo["margin"] == promo["margin"]
        assert frozen_promo["mean_headline_brier_skill"] == pytest.approx(
            promo["mean_headline_brier_skill"], abs=1e-6)
        for label in trials.HEADLINE_LABELS:
            assert frozen_promo["headline_brier_skill"][label] == \
                pytest.approx(promo["headline_brier_skill"][label], abs=1e-6)
        paired_f = frozen_promo["paired_headline_mean"]
        paired = promo["paired_headline_mean"]
        assert paired_f["mean_difference"] == pytest.approx(
            paired["mean_difference"], abs=1e-6)
        assert paired_f["ci95"] == pytest.approx(paired["ci95"], abs=1e-6)
        assert paired_f["n_matches"] == paired["n_matches"]
    # The candidate scores are recomputed through the same BLAS path,
    # and CI showed thread-count noise of about 4.3e-6 on one candidate,
    # so this comparison allows 1e-5: still far tighter than the 0.002
    # selection margin, and still tight enough to catch a real change.
    for key in ("a", "b", "c", "d", "e", "f"):
        assert frozen["candidates"][key]["status"] == \
            ablation["candidates"][key]["status"]
        if ablation["candidates"][key]["status"] == "evaluated":
            assert frozen["candidates"][key]["mean_headline_brier"] == \
                pytest.approx(
                    ablation["candidates"][key]["mean_headline_brier"],
                    abs=1e-5)
    for key, model in benchmark["models"].items():
        assert frozen["benchmark"]["models"][key]["mean_headline_brier"] == \
            pytest.approx(model["mean_headline_brier"], abs=1e-6)
    # The anchor numbers are independently re-derived, not copied:
    # a fresh part-1 capture on the same fixture reproduces the
    # artifact's family (a) labels.
    frames, _ = rich_fixture
    captured = oracle.baseline_capture(
        [frames[m] for m in TRAIN_IDS],
        [frames[m] for m in DEV_IDS],
        seed=0,
    )
    for label in LABELS:
        assert frozen["candidates"]["a"]["labels"][label]["brier"] == \
            pytest.approx(captured["labels"][label]["brier"], abs=1e-12)


def test_stage3_efficient_frontier_fc_07_part_2_no_fc6_selection():
    assert oracle.validate_no_fc6_selection() is True
    assert catalog.leakage_gate_statement().startswith("No tuned feature")
