"""Stage 3 U01 tests: immutable FC6 + end-to-end leakage certification.

The manifest (protocol/fc6_manifest.json) freezes the FC6 scoreboard:
train 130, held-out 32, transfer-train 17, transfer-scored 17, era pool
1517, extra pool 1683 match IDs, derived from real StatsBomb open-data
match metadata with the repo split code. These tests certify that
manifest: seeded overlaps are rejected, same IDs with different raw
content invalidate the caches, published losses match the frozen
results file, frozen-file mutation is detected, and future tracks stay
separate from FC6. Fast, no data downloads (the manifest itself is the
frozen real-data artifact; digest tests use tmp fixture files only).
"""

import copy
import hashlib
import json

import pytest

from effc.fc6_manifest import (
    ManifestError,
    audit_report,
    load_manifest,
    validate_manifest,
)
from effc.pipeline import vaep_cache_key
from effc.vaep.improved import improved_cache_key


def _dummy_models(tag):
    return {"tag": tag, "columns": ["x"]}


CHECKED_PAIRS = [
    ("train", "held_out"),
    ("extra_pool", "held_out"),
    ("era_pool", "held_out"),
    ("transfer_train", "transfer_scored"),
    ("era_pool", "transfer_scored"),
    ("extra_pool", "transfer_scored"),
]


def test_stage3_efficient_frontier_fc_01_manifest_validates_and_reports_pass():
    manifest = load_manifest()
    checks = validate_manifest(manifest)
    assert all(c["passed"] for c in checks)
    report = audit_report(manifest)
    assert "RESULT: PASS" in report
    assert "FAIL" not in report


@pytest.mark.parametrize("group_a,group_b", CHECKED_PAIRS)
def test_stage3_efficient_frontier_fc_01_seeded_overlap_rejected(group_a, group_b):
    manifest = load_manifest()
    tampered = copy.deepcopy(manifest)
    # Plant one ID from group B into group A, keeping n consistent so
    # only the disjointness check can fail.
    planted = tampered["groups"][group_b]["match_ids"][0]
    tampered["groups"][group_a]["match_ids"].append(planted)
    tampered["groups"][group_a]["n"] = len(tampered["groups"][group_a]["match_ids"])
    with pytest.raises(ManifestError) as exc:
        validate_manifest(tampered)
    assert group_a in str(exc.value) and group_b in str(exc.value)


def test_stage3_efficient_frontier_fc_01_cross_arm_overlap_is_by_design():
    # transfer_train shares 14 matches with train and 3 with held_out:
    # different arms (transfer scores 2023/24, main scores held-out),
    # so this is not leakage, and validation must still pass.
    manifest = load_manifest()
    groups = manifest["groups"]
    tr = set(groups["transfer_train"]["match_ids"])
    assert len(tr & set(groups["train"]["match_ids"])) == 14
    assert len(tr & set(groups["held_out"]["match_ids"])) == 3
    validate_manifest(manifest)


def test_stage3_efficient_frontier_fc_01_same_ids_different_content_invalidates_vaep_key():
    models = _dummy_models(1)
    ids = [10, 20, 30]
    config = {"seed": 0}
    base = vaep_cache_key(models, ids, config)
    assert vaep_cache_key(models, ids, config) == base
    assert vaep_cache_key(models, ids, config, content_digest=None) == base
    key_a = vaep_cache_key(models, ids, config, content_digest="aaa")
    key_b = vaep_cache_key(models, ids, config, content_digest="bbb")
    assert key_a != base
    assert key_b != base
    assert key_a != key_b
    assert vaep_cache_key(models, ids, config, content_digest="aaa") == key_a


def test_stage3_efficient_frontier_fc_01_same_ids_different_content_invalidates_improved_key():
    group = [1, 2, 3]
    pool = [4, 5]
    # Backward compatible: no digest reproduces the historical derivation.
    key_src = "fc6_pool|0|v4|" + ",".join(map(str, group)) + "|" + ",".join(map(str, pool))
    assert improved_cache_key("fc6_pool", 0, group, pool) == hashlib.md5(
        key_src.encode()
    ).hexdigest()[:16]
    key_a = improved_cache_key("fc6_pool", 0, group, pool, content_digest="aaa")
    key_b = improved_cache_key("fc6_pool", 0, group, pool, content_digest="bbb")
    assert key_a != key_b
    assert key_a != improved_cache_key("fc6_pool", 0, group, pool)


def test_stage3_efficient_frontier_fc_01_manifest_losses_match_results():
    manifest = load_manifest()
    with open("runs/beat_baseline/results.json") as f:
        results = json.load(f)
    assert manifest["published_losses"]["held_out"] == results["held_out"]
    assert manifest["published_losses"]["season_transfer"] == results["season_transfer"]


def test_stage3_efficient_frontier_fc_01_frozen_results_mutation_detected(tmp_path):
    manifest = load_manifest()
    tampered_results = tmp_path / "results.json"
    with open("runs/beat_baseline/results.json") as f:
        results = json.load(f)
    results["held_out"]["scores_incl"]["model"]["mean"] += 0.001
    with open(tampered_results, "w") as f:
        json.dump(results, f)
    tampered = copy.deepcopy(manifest)
    tampered["frozen_artifacts"] = [
        {"path": str(tampered_results), "sha256": a["sha256"]}
        for a in manifest["frozen_artifacts"]
    ]
    with pytest.raises(ManifestError):
        validate_manifest(tampered, results_path=str(tampered_results))


def test_stage3_efficient_frontier_fc_01_future_track_cannot_claim_fc6_path():
    manifest = load_manifest()
    tampered = copy.deepcopy(manifest)
    frozen_path = manifest["frozen_artifacts"][0]["path"]
    tampered["future_tracks"][0]["results_path"] = frozen_path
    with pytest.raises(ManifestError):
        validate_manifest(tampered)


def test_stage3_efficient_frontier_fc_01_event_content_digest(tmp_path):
    from effc.data.statsbomb import event_content_digest

    events_dir = tmp_path / "events"
    events_dir.mkdir()
    (events_dir / "10.json").write_text('[{"index": 1}]')
    (events_dir / "20.json").write_text('[{"index": 2}]')
    d1 = event_content_digest([10, 20], data_dir=str(tmp_path))
    assert d1 is not None
    assert event_content_digest([20, 10], data_dir=str(tmp_path)) == d1
    (events_dir / "20.json").write_text('[{"index": 3}]')
    assert event_content_digest([10, 20], data_dir=str(tmp_path)) != d1
    assert event_content_digest([99], data_dir=str(tmp_path)) is None


def test_stage3_efficient_frontier_fc_01_base_rate_definition_recorded():
    import importlib

    manifest = load_manifest()
    base_rate = manifest["base_rate"]
    module_name, func_name = base_rate["function"].rsplit(".", 1)
    module = importlib.import_module(module_name)
    assert hasattr(module, func_name)
    assert "training-frame" in base_rate["definition"]
    assert "training frames only" in base_rate["definition"]
