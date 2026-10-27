"""Immutable FC6 manifest: load, validate (fail closed), audit report.

The manifest (protocol/fc6_manifest.json) freezes the FC6 scoreboard:
the Bundesliga track from the v1.5.0 beat-baseline run (train 130,
held-out 32, transfer-train 17, transfer-scored 17, era pool 1517,
extra pool 1683 match IDs), the split config, the base-rate definition,
the feature version at run time, the trained-bundle fingerprints, the
published losses, and the hash of the frozen results file. Future
evaluation tracks (five_leagues) are recorded separately and must never
claim an FC6 artifact path.

Validation fails closed: validate_manifest raises ManifestError listing
every failed check. Disjointness is checked per arm (main arm and
transfer arm separately): cross-arm sharing (transfer_train with the
main train/held-out groups, extra_pool containing era_pool) is by
design and documented in the manifest, not leakage, because neither arm
scores the other's training matches.

Run as a script for the audit report:
    PYTHONPATH=src python -m effc.fc6_manifest
Exit 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MANIFEST_PATH = os.path.join(REPO_ROOT, "protocol", "fc6_manifest.json")
RESULTS_PATH = os.path.join(REPO_ROOT, "runs", "beat_baseline", "results.json")

FROZEN_COUNTS = {
    "train": 130,
    "held_out": 32,
    "transfer_train": 17,
    "transfer_scored": 17,
    "era_pool": 1517,
    "extra_pool": 1683,
}
FROZEN_SPLIT_CONFIG = {
    "backtest_split": 0.5,
    "calib_frac": 0.2,
    "seed": 0,
    "goal_model_config": "fc6_pool",
    "transfer_goal_model_config": "fc6_pool",
}
REQUIRED_KEYS = (
    "manifest_version",
    "protocol",
    "frozen",
    "derivation",
    "split_config",
    "competitions",
    "groups",
    "disjointness_checked_pairs",
    "base_rate",
    "feature_version_at_run",
    "bundle_fingerprints",
    "published_losses",
    "frozen_artifacts",
    "future_tracks",
)


class ManifestError(ValueError):
    """Raised when the FC6 manifest fails any certification check."""


def load_manifest(path: str | None = None) -> dict:
    """Load the manifest JSON (default: the committed protocol file)."""
    with open(path or MANIFEST_PATH) as f:
        return json.load(f)


def _sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _check_schema(m: dict) -> tuple[bool, str]:
    missing = [k for k in REQUIRED_KEYS if k not in m]
    if missing:
        return False, f"missing keys: {missing}"
    if m["manifest_version"] != 1:
        return False, f"manifest_version {m['manifest_version']!r} != 1"
    if m["frozen"] is not True:
        return False, "frozen flag is not true"
    return True, "schema present, version 1, frozen"


def _check_counts(m: dict) -> tuple[bool, str]:
    for name, group in m["groups"].items():
        if group["n"] != len(group["match_ids"]):
            return False, f"{name}: n {group['n']} != {len(group['match_ids'])} IDs"
        if name in FROZEN_COUNTS and group["n"] != FROZEN_COUNTS[name]:
            return False, f"{name}: n {group['n']} != frozen {FROZEN_COUNTS[name]}"
    return True, "group counts match frozen FC6 counts"


def _check_ids_valid(m: dict) -> tuple[bool, str]:
    for name, group in m["groups"].items():
        ids = group["match_ids"]
        if any(not isinstance(i, int) or isinstance(i, bool) for i in ids):
            return False, f"{name}: non-integer match ID present"
        if len(set(ids)) != len(ids):
            return False, f"{name}: duplicate match IDs present"
    return True, "all match IDs are integers, no duplicates within any group"


def _check_disjoint(m: dict) -> tuple[bool, str]:
    groups = m["groups"]
    for a, b in m["disjointness_checked_pairs"]:
        overlap = sorted(set(groups[a]["match_ids"]) & set(groups[b]["match_ids"]))
        if overlap:
            return False, (
                f"{a} and {b} share {len(overlap)} IDs "
                f"(e.g. {overlap[:5]}); held-out evaluation requires disjoint IDs"
            )
    return True, "all checked pairs disjoint"


def _check_split_config(m: dict) -> tuple[bool, str]:
    if m["split_config"] != FROZEN_SPLIT_CONFIG:
        return False, f"split config {m['split_config']} != frozen {FROZEN_SPLIT_CONFIG}"
    return True, "split config matches frozen FC6 config"


def _check_base_rate(m: dict) -> tuple[bool, str]:
    func_path = m["base_rate"]["function"]
    definition = m["base_rate"]["definition"]
    if not definition:
        return False, "base-rate definition is empty"
    module_name, func_name = func_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        return False, f"base-rate function module {module_name} not importable: {exc}"
    if not hasattr(module, func_name):
        return False, f"base-rate function {func_path} does not exist"
    return True, f"base-rate function {func_path} exists"


def _check_fingerprints(m: dict) -> tuple[bool, str]:
    from effc.vaep.improved import improved_cache_key

    groups = m["groups"]
    for arm, fp in m["bundle_fingerprints"].items():
        recomputed = improved_cache_key(
            fp["config"],
            fp["seed"],
            groups[fp["group"]]["match_ids"],
            groups[fp["pool"]]["match_ids"],
            feature_version=fp["feature_version"],
        )
        if recomputed != fp["improved_cache_key"]:
            return False, (
                f"{arm} bundle fingerprint {fp['improved_cache_key']} != "
                f"recomputed {recomputed}"
            )
    return True, "bundle fingerprints recompute from manifest IDs"


def _check_published_losses(m: dict, results_path: str) -> tuple[bool, str]:
    with open(results_path) as f:
        results = json.load(f)
    for track in ("held_out", "season_transfer"):
        if m["published_losses"][track] != results[track]:
            return False, f"published {track} losses differ from {results_path}"
    return True, "published losses match frozen results file"


def _check_frozen_artifacts(m: dict, repo_root: str) -> tuple[bool, str]:
    for artifact in m["frozen_artifacts"]:
        path = artifact["path"]
        if not os.path.isabs(path):
            path = os.path.join(repo_root, path)
        if not os.path.exists(path):
            return False, f"frozen artifact {artifact['path']} is missing"
        if _sha256_file(path) != artifact["sha256"]:
            return False, f"frozen artifact {artifact['path']} hash mismatch (mutated?)"
    return True, "frozen artifacts unmutated"


def _check_future_tracks(m: dict) -> tuple[bool, str]:
    frozen = {os.path.normpath(a["path"]) for a in m["frozen_artifacts"]}
    for track in m["future_tracks"]:
        if os.path.normpath(track["results_path"]) in frozen:
            return False, (
                f"future track {track['name']} claims FC6 path "
                f"{track['results_path']}; future outputs must stay separate"
            )
    return True, "future tracks separate from FC6 artifacts"


_CHECKS = (
    ("schema", lambda m, ctx: _check_schema(m)),
    ("counts", lambda m, ctx: _check_counts(m)),
    ("ids_valid", lambda m, ctx: _check_ids_valid(m)),
    ("disjoint", lambda m, ctx: _check_disjoint(m)),
    ("split_config", lambda m, ctx: _check_split_config(m)),
    ("base_rate", lambda m, ctx: _check_base_rate(m)),
    ("fingerprints", lambda m, ctx: _check_fingerprints(m)),
    ("published_losses", lambda m, ctx: _check_published_losses(m, ctx["results_path"])),
    ("frozen_artifacts", lambda m, ctx: _check_frozen_artifacts(m, ctx["repo_root"])),
    ("future_tracks", lambda m, ctx: _check_future_tracks(m)),
)


def audit(
    manifest: dict,
    results_path: str | None = None,
    repo_root: str | None = None,
) -> list[dict]:
    """Run every certification check; never raises, reports per check."""
    ctx = {
        "results_path": results_path or RESULTS_PATH,
        "repo_root": repo_root or REPO_ROOT,
    }
    checks = []
    for name, fn in _CHECKS:
        try:
            passed, detail = fn(manifest, ctx)
        except Exception as exc:  # fail closed: a broken check is a failed check
            passed, detail = False, f"check raised {type(exc).__name__}: {exc}"
        checks.append({"name": name, "passed": passed, "detail": detail})
    return checks


def validate_manifest(
    manifest: dict,
    results_path: str | None = None,
    repo_root: str | None = None,
) -> list[dict]:
    """Fail closed: raise ManifestError listing every failed check."""
    checks = audit(manifest, results_path=results_path, repo_root=repo_root)
    failures = [c for c in checks if not c["passed"]]
    if failures:
        raise ManifestError(
            "FC6 manifest FAILED: "
            + "; ".join(f"{c['name']}: {c['detail']}" for c in failures)
        )
    return checks


def audit_report(
    manifest: dict,
    results_path: str | None = None,
    repo_root: str | None = None,
) -> str:
    """Clear pass/fail report, one line per check plus a summary."""
    checks = audit(manifest, results_path=results_path, repo_root=repo_root)
    lines = []
    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        lines.append(f"{status}: {c['name']} - {c['detail']}")
    n_pass = sum(c["passed"] for c in checks)
    verdict = "PASS" if n_pass == len(checks) else "FAIL"
    lines.append(f"RESULT: {verdict} ({n_pass}/{len(checks)} checks)")
    return "\n".join(lines)


def main() -> int:
    manifest = load_manifest()
    print(audit_report(manifest))
    try:
        validate_manifest(manifest)
    except ManifestError:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
