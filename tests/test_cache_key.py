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
