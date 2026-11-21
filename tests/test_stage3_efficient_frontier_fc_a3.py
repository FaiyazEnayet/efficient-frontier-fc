"""Stage 3 efficient-frontier-fc A3: archive notebooks/01_data.ipynb.

Follow-up to C3, which archived 02_vaep, 03_frontier and 04_backtest and
left 01_data active. 01_data is now archived too: its narrative is
pre-five-league (same season 2023/24 on both sides, Bundesliga-only
ratings), superseded by the five-league run. No active notebook remains.
"""

import hashlib
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# sha256 of notebooks/01_data.ipynb on main at the v2.1.0 release merge
# (748a50d), recorded before the byte-identical move into the archive.
ARCHIVED_SHA256 = (
    "2a6e2a13f696f2e3869888599364d81142c41aa23065b86076d128bb7a98cd57"
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_stage3_efficient_frontier_fc_a3_01_data_archived_byte_identical():
    assert not os.path.exists(os.path.join(REPO_ROOT, "notebooks",
                                           "01_data.ipynb")), (
        "01_data.ipynb is still active")
    archived = os.path.join(REPO_ROOT, "notebooks", "archive",
                            "01_data.ipynb")
    assert os.path.exists(archived), "01_data.ipynb is missing from the archive"
    assert _sha256(archived) == ARCHIVED_SHA256, (
        "archived 01_data.ipynb is not byte-identical to the main version")


def test_stage3_efficient_frontier_fc_a3_archive_note_mentions_01_data():
    with open(os.path.join(REPO_ROOT, "notebooks", "archive",
                           "README.md")) as f:
        note = f.read()
    assert "01_data" in note, "archive note does not mention 01_data"
    assert "four" in note, "archive note still counts three notebooks"


def test_stage3_efficient_frontier_fc_a3_no_active_notebook_remains():
    nb_dir = os.path.join(REPO_ROOT, "notebooks")
    active = [name for name in os.listdir(nb_dir)
              if name.endswith(".ipynb")]
    assert active == [], f"active notebooks remain: {active}"
