"""FC1: VAEP caches are keyed by model, split, features and config.

The old compute_target_vaep loaded a single fixed target_vaep.pkl no
matter which model was supplied, so changing the seed, features or
training matches silently reused another model's values.
"""

import os
import pickle

import pandas as pd
import pytest

from effc import pipeline as P
from effc.pipeline import vaep_cache_key


def _dummy_models(tag):
    # Intentionally different dummy models: picklable, distinct bytes.
    return {"tag": tag, "columns": ["x"]}


def _stub_pipeline(monkeypatch, tmp_path, tag_fn=None):
    """Point the cache at tmp_path and stub the expensive data steps."""
    monkeypatch.setattr(P, "CACHE", str(tmp_path))
    matches = pd.DataFrame({"match_id": [1, 2]})
    monkeypatch.setattr(P, "list_matches", lambda target: matches)
    monkeypatch.setattr(P, "load_events", lambda mid: pd.DataFrame())

    def fake_featurize(events, match_id):
        return pd.DataFrame({"match_id": [match_id]})

    def fake_add_vaep(df, models):
        tag = tag_fn(models) if tag_fn else 1.0
        return df.assign(vaep=tag)

    monkeypatch.setattr(P, "featurize_match", fake_featurize)
    monkeypatch.setattr(P, "add_vaep", fake_add_vaep)
    monkeypatch.setattr(
        P, "player_match_vaep", lambda df: df[["match_id", "vaep"]]
    )
    monkeypatch.setattr(
        P,
        "minutes_played",
        lambda mid: pd.DataFrame({"match_id": [mid], "minutes": [90]}),
    )


def test_different_models_get_different_keys():
    k1 = vaep_cache_key(_dummy_models(1), [10, 20], {"seed": 0})
    k2 = vaep_cache_key(_dummy_models(2), [10, 20], {"seed": 0})
    assert k1 != k2
    assert k1.startswith("target_vaep_v") and k1.endswith(".pkl")


def test_config_change_invalidates_key():
    m = _dummy_models(1)
    k1 = vaep_cache_key(m, [10, 20], {"seed": 0})
    k2 = vaep_cache_key(m, [10, 20], {"seed": 1})
    assert k1 != k2


def test_train_ids_change_invalidates_key():
    m = _dummy_models(1)
    k1 = vaep_cache_key(m, [10, 20], {"seed": 0})
    k2 = vaep_cache_key(m, [10, 30], {"seed": 0})
    assert k1 != k2


def test_same_inputs_reuse_key_exactly():
    m = _dummy_models(1)
    # unsorted IDs are sorted into the key
    k1 = vaep_cache_key(m, [20, 10], {"seed": 0})
    k2 = vaep_cache_key(m, [10, 20], {"seed": 0})
    assert k1 == k2


def test_different_models_do_not_share_cache(tmp_path, monkeypatch):
    # Behavioral: model 2 must not receive model 1's cached values.
    _stub_pipeline(monkeypatch, tmp_path, tag_fn=lambda m: float(m["tag"]))
    r1 = P.compute_target_vaep(_dummy_models(1), [10], config={})
    r2 = P.compute_target_vaep(_dummy_models(2), [10], config={})
    assert (r1[0]["vaep"] == 1.0).all()
    assert (r2[0]["vaep"] == 2.0).all()


def test_unchanged_inputs_reuse_cache(tmp_path, monkeypatch):
    _stub_pipeline(monkeypatch, tmp_path)
    m = _dummy_models(1)
    r1 = P.compute_target_vaep(m, [10, 20], config={"seed": 0})
    calls = []
    orig = P.featurize_match

    def counting(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(P, "featurize_match", counting)
    r2 = P.compute_target_vaep(m, [10, 20], config={"seed": 0})
    assert not calls, "cache hit should not recompute"
    pd.testing.assert_frame_equal(r1[0], r2[0])
    pd.testing.assert_frame_equal(r1[1], r2[1])


def test_legacy_unversioned_cache_rejected(tmp_path, monkeypatch):
    # A stale target_vaep.pkl from the old scheme must never be loaded.
    _stub_pipeline(monkeypatch, tmp_path)
    (tmp_path / "target_vaep.pkl").write_bytes(pickle.dumps(("stale",)))
    pm, mins, matches = P.compute_target_vaep(
        _dummy_models(1), [10], config={"seed": 0}
    )
    assert isinstance(pm, pd.DataFrame), "recomputed, not the legacy tuple"
    assert len(pm) == 2
    files = os.listdir(tmp_path)
    assert any(f.startswith("target_vaep_v") for f in files)


# ---------------------------------------------------------------------------
# Stage 3 C5: download integrity, schema validation, content-bound derived
# caches, and cache provenance manifests.
# ---------------------------------------------------------------------------

import hashlib
import json
import subprocess as _subprocess
from types import SimpleNamespace

import effc.data.download as download
from effc.data.download import _fetch


def _no_sleep(monkeypatch):
    monkeypatch.setattr(download.time, "sleep", lambda _s: None)


def _fake_curl(monkeypatch, body: bytes, returncode: int = 0, captured=None):
    """Stand in for curl: write `body` to the -o path, report returncode."""

    def fake_run(cmd, capture_output=True, text=True):
        if captured is not None:
            captured.append(list(cmd))
        out = cmd[cmd.index("-o") + 1]
        with open(out, "wb") as fh:
            fh.write(body)
        return SimpleNamespace(
            returncode=returncode, stdout="", stderr="curl: (22) HTTP error" if returncode else ""
        )

    monkeypatch.setattr(_subprocess, "run", fake_run)


_VALID_MATCHES = json.dumps(
    [
        {
            "match_id": 1,
            "match_date": "2020-01-01",
            "home_team": {"home_team_name": "A"},
            "away_team": {"away_team_name": "B"},
        }
    ]
).encode()


def test_stage3_efficient_frontier_fc_5_fetch_fails_on_http_error(tmp_path, monkeypatch):
    _no_sleep(monkeypatch)
    captured = []
    _fake_curl(monkeypatch, b"404: Not Found", returncode=22, captured=captured)
    dest = str(tmp_path / "matches_9_281.json")
    with pytest.raises(RuntimeError):
        _fetch("https://example.invalid/matches.json", dest, retries=2)
    assert not os.path.exists(dest), "an HTTP error response must not be cached"
    assert captured and "-f" in captured[0], "curl must be told to fail on HTTP errors"


def test_stage3_efficient_frontier_fc_5_fetch_rejects_error_document_body(tmp_path, monkeypatch):
    _no_sleep(monkeypatch)
    _fake_curl(monkeypatch, b"<html><body>500 Internal Server Error</body></html>", returncode=0)
    dest = str(tmp_path / "events" / "1.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with pytest.raises(RuntimeError):
        _fetch("https://example.invalid/events/1.json", dest, retries=1)
    assert not os.path.exists(dest)


def test_stage3_efficient_frontier_fc_5_fetch_rejects_truncated_json(tmp_path, monkeypatch):
    _no_sleep(monkeypatch)
    _fake_curl(monkeypatch, b'[{"match_id": 1, "match_da', returncode=0)
    dest = str(tmp_path / "matches_9_281.json")
    with pytest.raises(RuntimeError):
        _fetch("https://example.invalid/matches.json", dest, retries=1)
    assert not os.path.exists(dest)


def test_stage3_efficient_frontier_fc_5_fetch_rejects_schema_invalid_json(tmp_path, monkeypatch):
    _no_sleep(monkeypatch)
    # Valid JSON, wrong schema for a matches file (no match_id / match_date).
    _fake_curl(monkeypatch, b'[{"foo": 1}]', returncode=0)
    dest = str(tmp_path / "matches_9_281.json")
    with pytest.raises(RuntimeError):
        _fetch("https://example.invalid/matches.json", dest, retries=1)
    assert not os.path.exists(dest)
    # Valid JSON of the wrong top-level type for an events file.
    _fake_curl(monkeypatch, b'{"not": "a list"}', returncode=0)
    dest2 = str(tmp_path / "events" / "2.json")
    os.makedirs(os.path.dirname(dest2), exist_ok=True)
    with pytest.raises(RuntimeError):
        _fetch("https://example.invalid/events/2.json", dest2, retries=1)
    assert not os.path.exists(dest2)


def test_stage3_efficient_frontier_fc_5_fetch_accepts_valid_and_records_provenance(tmp_path, monkeypatch):
    _no_sleep(monkeypatch)
    _fake_curl(monkeypatch, _VALID_MATCHES, returncode=0)
    dest = str(tmp_path / "matches_9_281.json")
    _fetch("https://example.invalid/matches.json", dest, retries=1)
    with open(dest, "rb") as fh:
        assert fh.read() == _VALID_MATCHES
    sidecar = dest + ".provenance.json"
    assert os.path.exists(sidecar), "accepted downloads must record provenance"
    with open(sidecar) as fh:
        prov = json.load(fh)
    assert prov["sha256"] == hashlib.sha256(_VALID_MATCHES).hexdigest()
    assert prov["validated"] is True
    assert prov["url"] == "https://example.invalid/matches.json"
    assert prov["upstream"]


def test_stage3_efficient_frontier_fc_5_fetch_does_not_trust_planted_bad_cache(tmp_path, monkeypatch):
    _no_sleep(monkeypatch)
    dest = str(tmp_path / "matches_9_281.json")
    with open(dest, "wb") as fh:
        fh.write(b"<html>404 Not Found</html>")
    _fake_curl(monkeypatch, _VALID_MATCHES, returncode=0)
    _fetch("https://example.invalid/matches.json", dest, retries=1)
    with open(dest, "rb") as fh:
        assert fh.read() == _VALID_MATCHES, "planted error document must be replaced, not trusted"


def test_stage3_efficient_frontier_fc_5_interrupted_acquisition_preserves_completed_cache(tmp_path, monkeypatch):
    _no_sleep(monkeypatch)
    dest = str(tmp_path / "matches_9_281.json")
    with open(dest, "wb") as fh:
        fh.write(_VALID_MATCHES)
    # A half-written temp file from an interrupted run sits next to it.
    with open(dest + ".tmp", "wb") as fh:
        fh.write(b'[{"match_id": 1, "mat')
    _fake_curl(monkeypatch, b"should never be used", returncode=0)
    _fetch("https://example.invalid/matches.json", dest, retries=1)
    with open(dest, "rb") as fh:
        assert fh.read() == _VALID_MATCHES, "completed cache must survive interruption"


def test_stage3_efficient_frontier_fc_5_cache_put_is_atomic(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "CACHE", str(tmp_path))
    P._cache_put("entry.pkl", {"v": 1})
    # Leftover partial temp from an interrupted write must not corrupt it.
    with open(os.path.join(str(tmp_path), "entry.pkl.tmp"), "wb") as fh:
        fh.write(b"partial garbage")
    assert P._cache_get("entry.pkl") == {"v": 1}


def _stub_featurize(monkeypatch, tmp_path, digest_holder):
    monkeypatch.setattr(P, "CACHE", str(tmp_path / "cache"))
    matches = pd.DataFrame({"match_id": [1, 2]})
    monkeypatch.setattr(P, "list_matches", lambda pool=None: matches)
    monkeypatch.setattr(P, "load_events", lambda mid: [{"id": str(mid)}])
    calls = []

    def fake_featurize(events, match_id):
        calls.append(match_id)
        return pd.DataFrame({"match_id": [match_id], "x": [float(match_id)]})

    monkeypatch.setattr(P, "featurize_match", fake_featurize)
    import effc.data.statsbomb as sb

    monkeypatch.setattr(
        sb, "event_content_digest", lambda ids, data_dir=None: digest_holder["digest"]
    )
    return calls


def test_stage3_efficient_frontier_fc_5_featurize_pool_changed_bytes_recompute(tmp_path, monkeypatch):
    holder = {"digest": "aaa111"}
    calls = _stub_featurize(monkeypatch, tmp_path, holder)
    frames1 = P.featurize_pool()
    assert len(frames1) == 2 and len(calls) == 2
    calls.clear()
    frames2 = P.featurize_pool()
    assert not calls, "identical verified inputs must hit the cache"
    assert len(frames2) == 2
    # Same match IDs, changed raw bytes: the digest moves, so recompute.
    holder["digest"] = "bbb222"
    frames3 = P.featurize_pool()
    assert len(calls) == 2, "changed bytes at unchanged IDs must recompute"
    assert len(frames3) == 2
    # The reproducibility manifest explains both the hit and the miss.
    manifests = [
        f for f in os.listdir(P.CACHE) if f.endswith(".manifest.json")
    ]
    assert manifests, "cache use must write a reproducibility manifest"
    text = ""
    for name in manifests:
        with open(os.path.join(P.CACHE, name)) as fh:
            text += fh.read()
    assert "hit" in text and "miss" in text


def test_stage3_efficient_frontier_fc_5_featurize_pool_unverified_provenance_bypasses_cache(tmp_path, monkeypatch):
    holder = {"digest": None}
    calls = _stub_featurize(monkeypatch, tmp_path, holder)
    P.featurize_pool()
    P.featurize_pool()
    assert len(calls) == 4, "unverified provenance must never serve a cache hit"
    cache_files = []
    if os.path.exists(P.CACHE):
        cache_files = [f for f in os.listdir(P.CACHE) if f.endswith(".pkl")]
    assert not cache_files, "unverified inputs must not be written as a trusted cache"


def test_stage3_efficient_frontier_fc_5_featurize_matches_changed_bytes_recompute(tmp_path, monkeypatch):
    # featurize_matches featurizes in a worker pool, so recompute is
    # observed through the content-addressed cache files and manifest
    # rather than a parent-process call counter.
    holder = {"digest": "aaa111"}
    _stub_featurize(monkeypatch, tmp_path, holder)

    def cache_files():
        return sorted(f for f in os.listdir(P.CACHE) if f.endswith(".pkl"))

    def manifest_text():
        text = ""
        for name in os.listdir(P.CACHE):
            if name.endswith(".manifest.json"):
                with open(os.path.join(P.CACHE, name)) as fh:
                    text += fh.read()
        return text

    out1 = P.featurize_matches([1, 2], "pool_frames_extra.pkl")
    assert len(out1) == 2
    assert len(cache_files()) == 1
    P.featurize_matches([1, 2], "pool_frames_extra.pkl")
    assert len(cache_files()) == 1, "identical verified inputs must hit"
    assert '"hit"' in manifest_text()
    holder["digest"] = "ccc333"
    P.featurize_matches([1, 2], "pool_frames_extra.pkl")
    assert len(cache_files()) == 2, "changed bytes at unchanged IDs must recompute under a new key"


def test_stage3_efficient_frontier_fc_5_crossfit_changed_frame_content_recomputes(tmp_path, monkeypatch):
    from effc.vaep import improved as imp
    from tests.test_team_strength import league

    monkeypatch.setattr(download, "DATA_DIR", str(tmp_path))
    frames, meta = league()
    group, pool = frames[:4], frames[4:8]
    calls = []
    orig_train_bundle = imp.train_bundle

    def counting(*a, **k):
        calls.append(1)
        return orig_train_bundle(*a, **k)

    monkeypatch.setattr(imp, "train_bundle", counting)
    imp.crossfit_oof(group, pool, meta, imp.CONFIGS["fc6_pool"], seed=0)
    n_first = len(calls)
    assert n_first > 0
    imp.crossfit_oof(group, pool, meta, imp.CONFIGS["fc6_pool"], seed=0)
    assert len(calls) == n_first, "identical frames must hit the OOF cache"
    changed = [f.copy() for f in group]
    col = changed[0].columns[-1]
    changed[0].loc[changed[0].index[0], col] = float(changed[0][col].iloc[0]) + 1.0
    imp.crossfit_oof(changed, pool, meta, imp.CONFIGS["fc6_pool"], seed=0)
    assert len(calls) > n_first, "changed frame content at same IDs must recompute OOF"


def test_stage3_efficient_frontier_fc_5_improved_cached_changed_content_recomputes(tmp_path, monkeypatch):
    from effc.vaep import improved as imp
    from tests.test_team_strength import league

    monkeypatch.setattr(download, "DATA_DIR", str(tmp_path))
    frames, meta = league()
    group, pool = frames[:4], frames[4:8]
    calls = []
    orig_train = imp.train_improved_models

    def counting(*a, **k):
        calls.append(1)
        return orig_train(*a, **k)

    monkeypatch.setattr(imp, "train_improved_models", counting)
    imp.train_improved_models_cached(group, pool, meta, "fc6_pool", seed=0)
    assert len(calls) == 1
    imp.train_improved_models_cached(group, pool, meta, "fc6_pool", seed=0)
    assert len(calls) == 1, "identical frames must hit the improved-model cache"
    changed = [f.copy() for f in group]
    col = changed[0].columns[-1]
    changed[0].loc[changed[0].index[0], col] = float(changed[0][col].iloc[0]) + 1.0
    imp.train_improved_models_cached(changed, pool, meta, "fc6_pool", seed=0)
    assert len(calls) == 2, "changed frame content at same IDs must retrain"
