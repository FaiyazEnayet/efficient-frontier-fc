"""Stage 3 FC7: licenses and provenance for the demo and wage pipeline.

Cache-free small fixture build: synthetic StatsBomb-like players and a
synthetic EA FC 24-style wage table live only in memory, so no file under
data/ is read. A brute-force oracle (not the MILP) picks the expected
squad, and the provenance record must disclose wage units, proxy status,
source URLs, licensing notes, the matching audit and the config hash.
"""

import hashlib
import itertools
import json

import pandas as pd
import pytest

from effc.data.provenance import (
    SOURCES,
    WAGE_UNIT,
    WEEKS_PER_YEAR,
    annual_wage_eur,
    build_provenance,
    config_hash,
    matching_audit,
)
from effc.data.wages import (
    WAGE_UNIT as WAGES_WAGE_UNIT,
)
from effc.data.wages import (
    annual_wage_eur as wages_annual_wage_eur,
)
from effc.data.wages import (
    assign_proxy_wages,
    match_players,
    norm,
    position_group,
)
from effc.portfolio.portfolio import select_squad

# 14 fixture players: 2 GK, 4 DEF, 4 MID, 4 FWD. Wages are EUR per week,
# the unit EA FC 24 reports and the unit the README budgets use.
_FIXTURE_PLAYERS = [
    ("Ana Keeper", "Test FC", "GK", 0.10, 20000.0),
    ("Bo Keeper", "Test FC", "GK", 0.08, 15000.0),
    ("Cy Defender", "Test FC", "DEF", 0.20, 30000.0),
    ("Di Defender", "Test FC", "DEF", 0.18, 25000.0),
    ("Ed Defender", "Test FC", "DEF", 0.15, 12000.0),
    ("Fi Defender", "Test FC", "DEF", 0.12, 10000.0),
    ("Gi Midfielder", "Test FC", "MID", 0.30, 40000.0),
    ("Ho Midfielder", "Test FC", "MID", 0.28, 35000.0),
    ("Id Midfielder", "Test FC", "MID", 0.22, 18000.0),
    ("Jo Midfielder", "Test FC", "MID", 0.19, 14000.0),
    ("Ki Forward", "Test FC", "FWD", 0.40, 50000.0),
    ("Lo Forward", "Test FC", "FWD", 0.35, 45000.0),
    ("Mo Forward", "Test FC", "FWD", 0.26, 20000.0),
    ("No Forward", "Test FC", "FWD", 0.21, 16000.0),
]

_FIXTURE_QUOTA = {"GK": 1, "DEF": 2, "MID": 2, "FWD": 1}
_FIXTURE_SQUAD_SIZE = 8
# The unconstrained max-value 8 under these quotas costs well above the
# fixture budget, so the budget binds and the pick is not the greedy top 8.
_FIXTURE_BUDGET = 200_000.0


def _wage_table():
    # Mirrors load_wages output: match_players consumes the keyed frame
    # load_wages produces, so the fixture carries the same columns.
    rows = []
    for i, (name, team, group, vaep, wage) in enumerate(_FIXTURE_PLAYERS):
        pos = {"GK": "GK", "DEF": "CB", "MID": "CM", "FWD": "ST"}[group]
        rows.append(
            {
                "long_name": name,
                "short_name": name.split()[0],
                "club_name": team,
                "positions": pos,
                "age": 25,
                "overall": 80,
                "value_eur": wage * 100.0,
                "wage_eur": wage,
                "key": norm(name),
                "club_key": norm(team),
                "pos_group": position_group(pos),
            }
        )
    return pd.DataFrame(rows)


def _sb_players():
    return pd.DataFrame(
        [
            {"player": name, "team": team}
            for name, team, _, _, _ in _FIXTURE_PLAYERS
        ]
    )


def _ratings(matched):
    rows = []
    for i, (name, team, group, vaep, _) in enumerate(_FIXTURE_PLAYERS):
        rows.append(
            {
                "player_id": 1000 + i,
                "player": name,
                "team": team,
                "pos_group": group,
                "vaep_per90": vaep,
                "wage_eur": float(
                    matched.loc[matched["player"] == name, "wage_eur"].iloc[0]
                ),
                "minutes": 900.0,
            }
        )
    return pd.DataFrame(rows)


def _brute_force_oracle(ratings, budget, squad_size, quota):
    """Independent oracle: enumerate every feasible squad, keep the best.

    Returns (best_ids, best_value, n_optimal). The fixture is built so
    the optimum is unique; a tie would make the MILP comparison
    meaningless, so the caller asserts n_optimal == 1.
    """
    best_ids, best_value, n_optimal = None, -1.0, 0
    idx = list(ratings.index)
    for combo in itertools.combinations(idx, squad_size):
        sub = ratings.loc[list(combo)]
        if float(sub["wage_eur"].sum()) > budget:
            continue
        counts = sub["pos_group"].value_counts().to_dict()
        if any(counts.get(g, 0) < q for g, q in quota.items()):
            continue
        value = float(sub["vaep_per90"].sum())
        if value > best_value + 1e-12:
            best_value = value
            best_ids = frozenset(sub["player_id"].tolist())
            n_optimal = 1
        elif abs(value - best_value) <= 1e-12:
            n_optimal += 1
    assert best_ids is not None, "fixture has no feasible squad"
    return best_ids, best_value, n_optimal


def test_stage3_efficient_frontier_fc_7_wage_units_disclosed():
    assert WAGE_UNIT == "EUR/week"
    assert WAGES_WAGE_UNIT == "EUR/week"
    assert WEEKS_PER_YEAR == 52
    assert annual_wage_eur(1000.0) == pytest.approx(52000.0)
    assert wages_annual_wage_eur(1000.0) == pytest.approx(52000.0)


def test_stage3_efficient_frontier_fc_7_sources_and_licenses():
    names = {s["name"] for s in SOURCES}
    assert "statsbomb_open_data" in names
    assert "ea_fc_24_wages" in names
    for source in SOURCES:
        assert source["url"].startswith("https://")
        assert source["license_note"], f"no license note for {source['name']}"
        # Retrieval dates must be stated honestly: a recorded date, or an
        # explicit statement that none was recorded. Never a blank.
        assert source["retrieval_date"], f"no retrieval date for {source['name']}"
    statsbomb = next(s for s in SOURCES if s["name"] == "statsbomb_open_data")
    assert "non-commercial" in statsbomb["license_note"].lower()
    assert "StatsBomb" in statsbomb["license_note"]
    wages = next(s for s in SOURCES if s["name"] == "ea_fc_24_wages")
    assert "404" in wages["license_note"] or "no license" in wages["license_note"].lower()


def test_stage3_efficient_frontier_fc_7_config_hash(tmp_path):
    cfg = tmp_path / "fixture.yaml"
    cfg.write_text("run_name: fc7_fixture\nseed: 0\n")
    digest = config_hash(str(cfg))
    assert digest == hashlib.sha256(cfg.read_bytes()).hexdigest()
    assert len(digest) == 64


def test_stage3_efficient_frontier_fc_7_matching_audit_and_proxy_status():
    wages = _wage_table()
    # Drop the last two forwards from the wage table: they stay unmatched
    # and must come back as flagged proxies, never as silent matches.
    wages = wages[~wages["long_name"].isin(["Mo Forward", "No Forward"])]
    matched = match_players(_sb_players(), wages)
    matched = assign_proxy_wages(matched)
    audit = matching_audit(matched)
    assert audit["exact"] == 12
    assert audit["fuzzy"] == 0
    assert audit["proxy"] == 2
    assert audit["unmatched"] == 0
    assert audit["n_players"] == 14
    assert audit["match_rate"] == pytest.approx(12 / 14)
    proxies = matched[matched["match_type"] == "proxy"]
    assert set(proxies["player"]) == {"Mo Forward", "No Forward"}
    assert not proxies["matched"].any()
    assert proxies["wage_eur"].notna().all()


def test_stage3_efficient_frontier_fc_7_fixture_reproduces_selection():
    wages = _wage_table()
    wages = wages[~wages["long_name"].isin(["Mo Forward", "No Forward"])]
    matched = assign_proxy_wages(match_players(_sb_players(), wages))
    ratings = _ratings(matched)
    expected_ids, expected_value, n_optimal = _brute_force_oracle(
        ratings, _FIXTURE_BUDGET, _FIXTURE_SQUAD_SIZE, _FIXTURE_QUOTA
    )
    assert n_optimal == 1, "fixture optimum must be unique for the MILP check"
    squad = select_squad(
        ratings,
        _FIXTURE_BUDGET,
        squad_size=_FIXTURE_SQUAD_SIZE,
        quota=_FIXTURE_QUOTA,
    )
    assert len(squad) == _FIXTURE_SQUAD_SIZE
    assert frozenset(squad["player_id"].tolist()) == expected_ids
    assert float(squad["vaep_per90"].sum()) == pytest.approx(expected_value)
    assert float(squad["wage_eur"].sum()) <= _FIXTURE_BUDGET
    assert float(squad["wage_eur"].sum()) * WEEKS_PER_YEAR == pytest.approx(
        float(squad["wage_eur"].sum()) * 52
    )
    counts = squad["pos_group"].value_counts().to_dict()
    for group, minimum in _FIXTURE_QUOTA.items():
        assert counts.get(group, 0) >= minimum
    # Deterministic: a second run reproduces the same selected IDs.
    again = select_squad(
        ratings,
        _FIXTURE_BUDGET,
        squad_size=_FIXTURE_SQUAD_SIZE,
        quota=_FIXTURE_QUOTA,
    )
    assert frozenset(again["player_id"].tolist()) == expected_ids


def test_stage3_efficient_frontier_fc_7_provenance_record(tmp_path):
    cfg = tmp_path / "fixture.yaml"
    cfg.write_text("run_name: fc7_fixture\nseed: 0\n")
    wages = _wage_table()
    wages = wages[~wages["long_name"].isin(["Mo Forward", "No Forward"])]
    matched = assign_proxy_wages(match_players(_sb_players(), wages))
    record = build_provenance(
        config_path=str(cfg),
        matched=matched,
        feature_version=4,
        seed=0,
    )
    assert record["config_hash"] == hashlib.sha256(cfg.read_bytes()).hexdigest()
    assert record["wage_units"]["weekly"] == "EUR/week"
    assert record["wage_units"]["annual"] == "EUR/year (weekly x 52)"
    assert record["matching_audit"]["proxy"] == 2
    assert record["feature_version"] == 4
    assert record["seed"] == 0
    assert {s["name"] for s in record["sources"]} == {
        "statsbomb_open_data",
        "ea_fc_24_wages",
    }
    assert json.dumps(record)


def test_stage3_efficient_frontier_fc_7_demo_meta_has_provenance():
    with open("demo/meta.json") as f:
        meta = json.load(f)
    prov = meta["provenance"]
    assert prov["config_hash"]
    assert prov["wage_units"]["weekly"] == "EUR/week"
    assert prov["wage_units"]["annual"] == "EUR/year (weekly x 52)"
    assert prov["matching_audit"]["match_rate"] == pytest.approx(
        meta["wage_match_rate"]
    )
    # The audit covers every distinct player in the target VAEP; the
    # rated subset is smaller (matched players with enough minutes).
    assert prov["matching_audit"]["n_players"] > meta["n_rated"]
    assert {s["name"] for s in prov["sources"]} == {
        "statsbomb_open_data",
        "ea_fc_24_wages",
    }
