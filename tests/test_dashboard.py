"""Tests for the Pages-ready dashboard and its animated preview."""

import os
import json

import pytest

HTML = os.path.join("docs", "dashboard.html")
GIF = os.path.join("docs", "demo.gif")


def test_dashboard_exists_and_self_contained():
    assert os.path.exists(HTML), "run scripts/make_dashboard.py"
    size = os.path.getsize(HTML)
    assert size > 1_000_000, "plotly.js bundle should be inlined"
    with open(HTML, encoding="utf-8") as f:
        html = f.read()
    # self-contained: the plotly.js bundle is inlined; no external scripts
    assert 'script src="http' not in html
    assert "script src='http" not in html
    # Current release content must be the frozen five-league FV4 payload.
    marker = "const EVIDENCE="
    assert marker in html
    evidence, _ = json.JSONDecoder().raw_decode(html.split(marker, 1)[1])
    assert evidence.get("release") == "v2.0.0"
    assert "five_leagues" in evidence
    results = evidence["five_leagues"]
    assert results["feature_version"] == 4
    assert list(results["leagues"]) == [
        "English Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"
    ]
    budgets = [300000.0, 500000.0, 700000.0]
    assert results["budgets_eur"] == budgets
    for league in results["leagues"].values():
        assert [point["budget"] for point in league["frontier"]] == budgets
        assert [point["budget"] for point in league["backtest"]] == budgets
    bundesliga = results["leagues"]["Bundesliga"]
    assert bundesliga["audit"]["matches"] == 68
    exception = bundesliga["backtest"][0]
    assert round(exception["optimizer"], 2) == 40.83
    assert round(exception["value"], 2) == 41.53
    losses = [
        (name, point["budget"])
        for name, league in results["leagues"].items()
        for point in league["backtest"]
        if point["optimizer"] < point["value"]
    ]
    assert losses == [("Bundesliga", 300000.0)]


def test_dashboard_has_all_sections():
    with open(HTML, encoding="utf-8") as f:
        html = f.read()
    # The shared renderer creates charts from three embedded figure specs.
    # Counting call sites cannot count charts created by a rendering function.
    marker = "const EVIDENCE="
    assert marker in html
    evidence, _ = json.JSONDecoder().raw_decode(html.split(marker, 1)[1])
    assert evidence["repo"] == "efficient-frontier-fc"
    assert len(evidence["figures"]) == 3  # frontier, ratings, backtest
    assert all(figure["data"] for figure in evidence["figures"])
    assert "Plotly.newPlot" in html
    for section in ("frontier", "Best XI", "backtest"):
        assert section in html


def test_gif_is_valid():
    assert os.path.exists(GIF), "run scripts/make_gif.py"
    with open(GIF, "rb") as f:
        magic = f.read(6)
    assert magic in (b"GIF87a", b"GIF89a")
    assert os.path.getsize(GIF) > 10_000
