# Reproducing efficient-frontier-fc

This file is the raw-event, wage and model provenance record for the
demo and the pipeline (Stage 3 FC7). It states only verified facts;
anything not recorded anywhere is stated as not recorded, never
invented.

## Sources

### StatsBomb open data (raw events, matches, lineups)

- Repository: https://github.com/statsbomb/open-data
- Data files: https://raw.githubusercontent.com/statsbomb/open-data/master/data
- License file: https://raw.githubusercontent.com/statsbomb/open-data/master/LICENSE.pdf
- License: StatsBomb Public Data User Agreement (last updated
  8 September 2023). Not a Creative Commons license. Free for
  non-commercial research and personal use; the data and any derived
  analysis may not be commercially exploited; any publication of
  analysis must credit StatsBomb with its brand logo.
- Retrieval date: not recorded in the repo (verified 2026-11-05).
- Competitions used: 1. Bundesliga 2023/2024 and 2015/2016 as the
  target (34 matches each, all StatsBomb publishes); the goal models
  also train on pooled competitions (FIFA World Cup 2018 and 2022,
  UEFA Euro 2020, Indian Super League 2021/22, and the 2015/2016
  La Liga, Serie A, Ligue 1 and Premier League seasons). See
  `src/effc/data/download.py` (`MODEL_POOL`, `EXTRA_POOL`,
  `ERA_2015_POOL`) for the exact competition and season IDs.

### EA FC 24 wages (wage proxy)

- Source file recorded in `src/effc/data/download.py`:
  https://raw.githubusercontent.com/mzafram2001/ea-fc/main/data/dataset_ea_fc_24.csv
- License: no license statement found for this file. The source
  repository page and the raw file URL both return 404 when checked
  on 2026-11-05, so the file is used as a wage proxy only and is not
  redistributed. A fresh clone cannot re-fetch it until a live
  source is found and recorded here and in
  `src/effc/data/provenance.py`.
- Retrieval date: not recorded in the repo (verified 2026-11-05).
- The local copy under `data/` is gitignored. The README states the
  standing caveat: EA FC 24 wages are not verified historical or
  current contracts.

## Wage units

`wage_eur` is EUR per week, the unit EA FC 24 reports and the unit
the README budgets use (weekly wage caps of 300,000, 500,000 and
700,000 EUR). Annual figures are weekly x 52; see
`effc.data.wages.annual_wage_eur` and the `wage_units` block in
`demo/meta.json`.

## Matching audit

`demo/meta.json` carries a `provenance` block with the full record:
source URLs, licensing notes, retrieval dates, the config hash, the
feature version, the seed, the wage units, and the matching audit.
For the committed demo the audit is: 435 exact matches, 11 fuzzy
matches (same club, last name plus first initial), 0 proxies and 260
unmatched out of 706 distinct players in the target VAEP, a match
rate of 0.6317. The demo build applies no proxy wages: unmatched
players are excluded from the ratings by `build_ratings`, so the
rated set is 268 players. Proxy wages exist only through
`assign_proxy_wages`, which marks them `match_type` "proxy" and
leaves `matched` False; a proxy is a labeled stand-in, not a match.

## Config hash

The demo config is `configs/bundesliga_2seasons.yaml`. Its SHA-256
is recorded as `config_hash` in `demo/meta.json` (value
`86e40fda714d21a78027747473d8de4a6dc2c9f561a372ea1c8123ece54435c7`
for the committed demo). Recompute it with:

```sh
PYTHONPATH=src python -c "from effc.data.provenance import config_hash; print(config_hash('configs/bundesliga_2seasons.yaml'))"
```

## Reproducing the demo

From the repo root, after the pipeline has been run at least once
and the cached data under `data/` is present:

```sh
python scripts/make_demo.py
```

This rebuilds the ratings, the frontier, the selected squads, the
best XI and the backtest with the same functions and seed as the
pipeline, then writes `demo/*.csv` plus `demo/meta.json` with the
provenance block. The goal models are retrained exactly as the
pipeline trains them, so the versioned VAEP cache hits and no event
re-processing is needed.

Note: a fresh clone without the cached data cannot run this yet.
`make_demo.py` needs the pooled training competitions (1,683
matches) that `python -m effc.data.download` and `download_extra`
fetch, and the EA FC 24 wage file is currently unreachable at its
recorded URL (see above). Until a live wage source is recorded, a
fresh clone must reuse an existing `data/` copy.

## Cache-free small fixture build

The regression tests in `tests/test_stage3_efficient_frontier_fc_7.py`
rebuild a small competition from synthetic in-memory frames only, so
no file under `data/` is read. A brute-force oracle (not the MILP)
picks the expected squad, and the test asserts the selector returns
exactly those player IDs under the budget and positional quotas,
with wage units and proxy status disclosed. Run them with:

```sh
python -m pytest -q -k stage3_efficient_frontier_fc_7
```
