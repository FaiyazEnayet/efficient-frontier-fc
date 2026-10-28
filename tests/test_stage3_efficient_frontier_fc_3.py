"""Stage 3 efficient-frontier-fc C3: stale notebooks and version metadata.

The three legacy notebooks are archived, not active: they load retired
unversioned caches and make claims the published results contradict.
APPLY.md must mention this repo only, the placeholder package version must
be gone, and the numbers the archive note cites must match the published
five-league results.
"""

import json
import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARCHIVED = ["02_vaep.ipynb", "03_frontier.ipynb", "04_backtest.ipynb"]


def _read(path):
    with open(os.path.join(REPO_ROOT, path)) as f:
        return f.read()


def _active_notebook_texts():
    nb_dir = os.path.join(REPO_ROOT, "notebooks")
    texts = {}
    for name in os.listdir(nb_dir):
        if name.endswith(".ipynb"):
            texts[name] = _read(os.path.join("notebooks", name))
    return texts


def test_stage3_efficient_frontier_fc_3_archived_notebooks_moved():
    for name in ARCHIVED:
        assert not os.path.exists(os.path.join(REPO_ROOT, "notebooks", name)), (
            f"{name} is still active")
        assert os.path.exists(
            os.path.join(REPO_ROOT, "notebooks", "archive", name)), (
            f"{name} is missing from the archive")


def test_stage3_efficient_frontier_fc_3_archive_note_explicit():
    note = _read(os.path.join("notebooks", "archive", "README.md"))
    for marker in ["ARCHIVED", "target_vaep.pkl", "stars",
                   "Bundesliga", "300,000", "63.17%"]:
        assert marker in note, f"archive note is missing {marker!r}"


def test_stage3_efficient_frontier_fc_3_no_active_notebook_loads_cache():
    for name, text in _active_notebook_texts().items():
        assert "target_vaep.pkl" not in text, f"{name} loads target_vaep.pkl"
        assert "pool_frames.pkl" not in text, f"{name} loads pool_frames.pkl"


def test_stage3_efficient_frontier_fc_3_no_active_notebook_claims_win():
    for name, text in _active_notebook_texts().items():
        assert "at every budget" not in text, f"{name} claims every-budget win"
        assert "beats the stars" not in text, f"{name} cites stars baseline"


def test_stage3_efficient_frontier_fc_3_apply_md_no_foreign_projects():
    text = _read("APPLY.md")
    for foreign in ["SukukLab", "DeepArcade", "Julia"]:
        assert foreign not in text, f"APPLY.md still mentions {foreign}"
    assert "renderer.py" in text, "APPLY.md lost the dashboard rebuild command"


def test_stage3_efficient_frontier_fc_3_version_not_placeholder():
    text = _read("pyproject.toml")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match, "pyproject.toml has no version"
    version = match.group(1)
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), (
        f"version {version!r} is not a release version")
    assert version != "0.1.0", "placeholder version 0.1.0 is still present"


def test_stage3_efficient_frontier_fc_3_current_numbers_pinned():
    results = json.loads(_read(os.path.join("runs", "five_leagues",
                                            "results.json")))
    leagues = results["leagues"]
    assert leagues["Bundesliga"]["wage_match_rate"] == pytest.approx(
        0.6317, abs=0.001)
    losses = []
    for league, data in leagues.items():
        for row in data["backtest"]:
            if row["optimizer"] < row["value"]:
                losses.append((league, row["budget"]))
    assert losses == [("Bundesliga", 300000.0)], (
        f"published backtest losses changed: {losses}")
