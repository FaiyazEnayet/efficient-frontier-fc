"""Load the precomputed demo tables bundled under demo/ for the dashboard.

These tables are derived artifacts of the real pipeline
(python -m effc.pipeline configs/bundesliga_2seasons.yaml); see
scripts/make_demo.py for how they are generated. The dashboard reads
only these small CSVs, so it starts instantly with no data download.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

DEMO_DIR = Path(os.environ.get("EFFC_DEMO_DIR", Path(__file__).resolve().parents[2] / "demo"))

_TABLES = ("players", "frontier", "squads", "xi", "backtest")


def load_demo(demo_dir: Path | str = DEMO_DIR) -> dict:
    """Returns {'players': df, 'frontier': df, 'squads': df, 'xi': df,
    'backtest': df, 'meta': dict}."""
    demo_dir = Path(demo_dir)
    out = {}
    for name in _TABLES:
        path = demo_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"missing demo table {path}; regenerate with scripts/make_demo.py"
            )
        out[name] = pd.read_csv(path)
    meta_path = demo_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing {meta_path}; regenerate with scripts/make_demo.py")
    out["meta"] = json.loads(meta_path.read_text())
    return out
