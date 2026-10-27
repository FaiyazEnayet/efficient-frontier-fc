"""Squad-name persistence for the five-league run (FV4 follow-up).

The FV4 five-league results carry a squads section: per league per
budget, the selected 23-man squad rows plus the best XI names, all
drawn from the run's own rated/selected players. This test guards that
section in the committed results file; it needs no data directory.
"""

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "runs" / "five_leagues" / "results.json"
LEAGUES = ["English Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"]
BUDGETS = ["300000", "500000", "700000"]


@pytest.fixture(scope="module")
def payload():
    assert RESULTS.exists(), "run scripts/run_five_leagues.py"
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def test_squads_present_for_all_leagues_and_budgets(payload):
    assert payload["feature_version"] == 4
    for league in LEAGUES:
        squads = payload["leagues"][league]["squads"]
        for b in BUDGETS:
            cell = squads[b]
            assert "squad" in cell and "best_xi" in cell, (league, b)
            assert cell["best_xi"] is not None, (league, b)


def test_squad_sizes_and_xi_subset(payload):
    for league in LEAGUES:
        entry = payload["leagues"][league]
        for b in BUDGETS:
            cell = entry["squads"][b]
            squad = cell["squad"]
            xi = cell["best_xi"]
            assert len(squad) == 23, (league, b)
            assert len(xi) == 11, (league, b)
            names = [row["player"] for row in squad]
            assert len(set(names)) == 23, (league, b)
            assert set(xi) <= set(names), (league, b)
            assert entry["squad_checks"][b]["n_players"] == 23, (league, b)


def test_squad_row_columns(payload):
    for league in LEAGUES:
        for b in BUDGETS:
            for row in payload["leagues"][league]["squads"][b]["squad"]:
                assert set(row) == {
                    "player",
                    "team",
                    "pos_group",
                    "vaep_per90",
                    "wage_eur",
                }
                assert isinstance(row["player"], str) and row["player"]
                assert isinstance(row["vaep_per90"], (int, float))
                assert isinstance(row["wage_eur"], (int, float))
