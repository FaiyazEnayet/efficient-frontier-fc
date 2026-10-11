"""Tests for the Pages-ready dashboard and its animated preview."""

import os

import pytest

HTML = os.path.join("docs", "dashboard.html")
GIF = os.path.join("docs", "demo.gif")


def test_dashboard_exists_and_self_contained():
    assert os.path.exists(HTML), "run scripts/make_dashboard.py"
    size = os.path.getsize(HTML)
    assert size > 1_000_000, "plotly.js bundle should be inlined"
    with open(HTML) as f:
        html = f.read()
    # self-contained: the plotly.js bundle is inlined; no external scripts
    assert 'script src="http' not in html
    assert "script src='http" not in html
    # real content from the pipeline's demo tables
    for token in ("Wirtz", "EUR 0.70M", "optimizer", "Best XI"):
        assert token in html


def test_dashboard_has_all_sections():
    with open(HTML) as f:
        html = f.read()
    assert html.count("Plotly.newPlot") == 3  # frontier, ratings, backtest


def test_gif_is_valid():
    assert os.path.exists(GIF), "run scripts/make_gif.py"
    with open(GIF, "rb") as f:
        magic = f.read(6)
    assert magic in (b"GIF87a", b"GIF89a")
    assert os.path.getsize(GIF) > 10_000
