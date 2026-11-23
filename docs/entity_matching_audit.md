# Real-player entity matching and selection coverage audit (Stage 3 U05)

## What this is, and what it is not

This audit covers the player side of the pipeline: which real players get
a wage, which do not, and what the exclusions cost the optimizer. It is
the part of the entity-matching story the C6 rebuild did not cover. C6
rebuilt the matcher itself (club crosswalk, ambiguity quarantine,
aliases with evidence, deterministic crosswalk IDs, proxy wages, the
join audit and the gold-set scorer, all in `effc.data.wages`). U06
audited action values against a reference VAEP replication
(`effc.analysis.action_audit`, `docs/action_value_audit.md`): which
events get valued, and whether the values agree. Neither of them built a
player ID crosswalk, recorded a human decision on any uncertain match,
measured coverage by position from the event side, or measured what the
excluded players cost the selected squads. That is this unit.

The file overlap with U06 is only apparent. U06 touched the VAEP
formula, the StatsBomb loader, `tests/test_effc.py` and the new action
audit module. This unit touches none of those files: it adds
`src/effc/analysis/entity_audit.py`, `data/mapping_registry.csv`, this
document, one new match type in `effc.data.wages`, one new test file,
one updated assertion in the C6 test file, a restated audit block in
`demo/meta.json`, a one-line `.gitignore` exception for the registry,
and a CHANGELOG entry. The two audits measure different coverage
(actions vs players) and share no code.

## Method

The pool is the Bundesliga target pool: competitions (9, 281) and
(9, 27), 68 matches, 706 distinct (player, team) rows, 672 distinct
StatsBomb player IDs. The pool was rebuilt from the cached target VAEP
frame (`data/cache/target_vaep_v2_8276fc0c41a04201.pkl`), which is the
pipeline's own output, and matched with the repo's own `match_players`
against the local EA FC 24 wage file (18,508 rows read, 18,412 kept, 95
invalid wages quarantined at load, 82 homonym names; file SHA-256
`1ddeb423c9271c50a301fde6fccca8aa19d1c11d8a92f3a25f5a5cec93f0e65b`).
Config `configs/bundesliga_2seasons.yaml`
(SHA-256 `86e40fda714d21a78027747473d8de4a6dc2c9f561a372ea1c8123ece54435c7`),
FEATURE_VERSION 4, seed 0. Positions come from the event side: each
player's modal StatsBomb position across his minutes rows, mapped to
GK/DEF/MID/FWD by `statsbomb_position_group`. Selections use the exact
MILP selector (`effc.portfolio.portfolio.select_squad`); nothing here
retrains a model, re-runs the five-league backtest, or changes the demo
rated output, and the frozen FC6 results stay historical.

## Coverage: the current matcher on the true pool

Match counts on the 706-player pool, before and after the reviewed
registry is applied:

| match type | before registry | after registry |
|---|---:|---:|
| exact | 314 | 314 |
| exact_other_club | 120 | 120 |
| alias | 0 | 0 |
| fuzzy | 27 | 0 |
| reviewed | 0 | 43 |
| ambiguous | 1 | 0 |
| proxy | 0 | 0 |
| unmatched | 244 | 229 |
| **matched** | **461 (65.30%)** | **477 (67.56%)** |

The registry moves 16 players from unmatched or quarantined to matched
and re-labels the 27 fuzzy matches as reviewed; the match rate rises
from 65.30% to 67.56%.

### The published audit was stale

`demo/meta.json` still reported the pre-rebuild matcher: 435 exact, 11
fuzzy, 260 unmatched of 706 players, 63.17%. That audit was computed
before C6, and it was wrong in both directions: the old matcher matched
by name alone, so it counted 435 "exact" rows that the current matcher
splits into 314 exact and 120 exact-other-club (genuine transfers and
loans, flagged as such), and it missed 15 players the current matcher
and registry now match (fuzzy variants the old club-string comparison
blocked, plus the reviewed rows). The restatement below replaces that
block with the current matcher's numbers on the same 706-player pool.
The demo rated output (`demo/players.csv`, 268 players) is untouched:
it is the frozen output of the demo run, and the one row in it whose
match was in doubt (Max Meyer, below) turns out to carry the right wage.

### Per-position coverage, measured from the event side

The wage-side audit cannot place an unmatched player in a position
group, because the group comes from the wage row. These numbers use
each player's modal StatsBomb position instead, so the unmatched
players are counted where they actually play:

| position group | players | matched | match rate |
|---|---:|---:|---:|
| GK | 41 | 34 | 82.9% |
| DEF | 241 | 162 | 67.2% |
| MID | 268 | 176 | 65.7% |
| FWD | 156 | 105 | 67.3% |

Coverage is flat across the outfield groups (66-67%) and higher for
goalkeepers (82.9%). The gap is not positional: it is the era gap.

### Per-season coverage: the gap is the 2015/16 season

| season | players | matched | match rate |
|---|---|---:|---:|
| 2015/16 | 347 | 134 | 38.6% |
| 2023/24 | 375 | 359 | 95.7% |

Sixteen players appear in both seasons' pools, so the rows sum to 722.
The 2023/24 pool matches at 95.7% (16 unmatched of 375). The 2015/16
pool matches at 38.6% (213 unmatched of 347): most of those players had
left top football before the EA FC 24 roster was cut. The same pattern
shows by club. The lowest-coverage clubs are all 2015/16-only sides:
Hamburger SV 3 of 18 (16.7%), Hertha Berlin 5 of 18 (27.8%), Hannover
96 7 of 21 (33.3%), Ingolstadt 7 of 18 (38.9%). The highest are
Freiburg 20 of 20, FC Heidenheim 19 of 19, Union Berlin 23 of 24,
Bochum 21 of 22.

## Confidence categories

Every match type now carries a confidence category
(`effc.analysis.entity_audit.match_confidence`), grounded in the wage
uncertainty intervals:

| match type | confidence | wage interval half width |
|---|---|---:|
| exact | high | 5% |
| exact_other_club | medium | 15% |
| reviewed | medium | 15% |
| alias | low | 25% |
| fuzzy | low | 25% |
| ambiguous | uncertain | no wage |
| proxy | proxy | 50% |
| unmatched | none | no wage |

A reviewed match stays medium, not high: the identity is verified by a
human, but the wage still comes from another club and season. On the
reviewed pool the split is high 314, medium 163 (120 exact-other-club
plus 43 reviewed), none 229.

## Player ID crosswalk

`build_player_crosswalk` ties StatsBomb player IDs to wage rows: 444
player IDs have a wage row, 229 pool rows are excluded because they
have no wage row, and 31 player IDs sit at more than one club (a
transfer across the pooled seasons; the person key is the same for
both, because a transfer does not change the person). The crosswalk
fails closed when one player ID points at two different people (two
person keys, or two normalized names) or when a matched row carries no
wage row ID. On this pool there are zero such conflicts: no player ID
maps to more than one name, and no player ID maps to more than one
person key.

## Reviewed mapping registry

`data/mapping_registry.csv` holds 43 reviewed rows. Each row records
the player, team, player ID, the match type before review, the decision
(accept), the wage row ID, the wage, the evidence, the reviewer and the
review date. `apply_mapping_registry` applies accepts as overrides
(match type becomes "reviewed", wage and source come from the reviewed
wage row) and rejects as quarantine (the row becomes ambiguous and any
wage is cleared, even if the matcher had matched it). An accept whose
wage disagrees with the wage file by more than 1 EUR/week fails, as
does a registry row for a (player, team) pair the matched frame never
saw. Every row carries `prior_match_type`, so the registry's effect
stays auditable.

### The 27 fuzzy matches: all accepted

Each fuzzy match is the unique (club, last name, first initial)
candidate, and in every case the wage-file name is the StatsBomb name
with middle names dropped (Timmy Chandler, Joe Scally, Dayot
Upamecano, Danny da Costa) or with the maternal surname dropped
(Victor Boniface, Exequiel Palacios, Edmond Tapsoba). All 27 were
accepted. The position group agrees between the wage file and the
event-side modal position in 21 of 27; the 6 that differ are players
listed at both roles across the two sources (Timothy Chandler RM/RWB
vs right back, Yussuf Yurary Poulsen ST vs left midfield, Junior Adamu
ST vs left center midfield, Justin Njinmah ST vs right center
midfield, Elias Baum RB vs left defensive midfield, Raphael Guerreiro
LB vs right back). Those are position-label differences, not identity
differences, and each registry row says so.

### Max Meyer: the one quarantined player, resolved

Max Meyer at Schalke 04 was the pool's only ambiguous row: the name
appears at FC Luzern and SSV Jahn Regensburg, and Schalke agrees with
neither. The review accepts the FC Luzern row (11,000 EUR/week).
Maximilian Meyer, born 1995, played at Schalke 2013-18, then Crystal
Palace, 1. FC Koln, Fenerbahce and Midtjylland, then FC Luzern from
2022; the wage file's CAM/CM positions and age 27 fit. The Regensburg
row (age 16, overall 53, 500 EUR/week) is a different youth player and
is rejected by the decision. The old matcher had assigned Meyer the
Luzern wage by taking the first row, so the demo rated output already
carries 11,000 for him: the wage was right, reached for the wrong
reason, and the registry now records the right reason.

### Dante: two pool rows, one wage row

Dante Bonfim da Costa Santos appears twice in the pool (Bayern Munich
and Wolfsburg, the same StatsBomb player ID) and was unmatched in both
rows. The review accepts the OGC Nice row (29,000 EUR/week) for both:
Bayern 2012-15, Wolfsburg 2015-16, Nice 2016-26.

### Thirteen same-club name variants

Thirteen more unmatched rows (twelve players; Choupo-Moting has two
pool rows) resolve to a wage row at the same club,
where the wage-file name is the StatsBomb name with middle names
dropped and the candidate is the only token-subset candidate at that
club: Lois Openda (RB Leipzig, 77,000), Castello Lukeba (RB Leipzig,
36,000), Gustavo Puerta (Bayer Leverkusen, 6,000), Nathan Ngoumou
(Borussia Monchengladbach, 17,000), Eric Maxim Choupo-Moting (Bayern
Munich, 57,000; accepted for both his pool rows, Bayern and Schalke
04), Piero Hincapie (Bayer Leverkusen, 36,000), Junior Dina Ebimbe
(Eintracht Frankfurt, 30,000), Willian Pacho (Eintracht Frankfurt,
25,000), Cristian Gamboa (Bochum, 14,000), Bernardo Fernandes (Bochum,
19,000), Jordan Siebatcheu (Borussia Monchengladbach, 31,000) and
Jean-Matteo Bahoya (Eintracht Frankfurt, 3,000).

## The 229 who stay unmatched, by cause

Of the 229 players still unmatched after the registry:

- **202 have no wage row under any name variant.** No exact normalized
  name, and no token-subset candidate (middle names dropped, double
  surnames shortened) anywhere in the file. These are players the EA FC
  24 roster does not hold at all, and no matcher change reaches them.
  The largest minutes among them are full-season regulars: Karim
  Bellarabi (2,690.7 minutes), Javier Hernandez Balcazar (2,252.7),
  Stefan Kiessling (2,007.6), Odilon Kossonou (1,905.9) and Admir
  Mehmedi (1,435.1).
- **42 had a token-subset candidate; 15 were accepted** (the thirteen
  same-club variant rows above, plus Dante twice). **27 remain**, each
  needing its own career-path check before any alias is asserted: the
  clearest are Fabian Schar (Newcastle United row), Arturo Vidal
  (Colo-Colo), David Alaba (Real Madrid), John Brooks (Hoffenheim),
  Emil Forsberg (New York Red Bulls) and Joel Matip (Liverpool). They
  are recorded here, not in the registry, because the registry holds
  decisions, not guesses.

159 of the 229 still-unmatched players have at least 90 minutes, and 13
have at least 270 minutes, so the exclusions bite on players who would
otherwise be rated.

## Selection sensitivity: what the exclusions cost

### Wage-interval sensitivity

Re-solving the 268 rated players at the low and high wage of every
player's match-type interval, at the five headline budgets (summed
VAEP/90 objective; base objectives 8.88, 9.66, 10.26, 10.63, 10.74):

| budget (EUR/week) | low-wage Jaccard | low objective change | high-wage Jaccard | high objective change |
|---:|---:|---:|---:|---:|
| 300,000 | 0.840 | +0.10 | 0.769 | -0.09 |
| 500,000 | 0.769 | +0.15 | 0.643 | -0.12 |
| 700,000 | 0.840 | +0.12 | 0.917 | -0.14 |
| 943,000 | 0.917 | +0.06 | 0.840 | -0.06 |
| 1,500,000 | 1.000 | +0.00 | 1.000 | +0.00 |

Wage uncertainty moves the squad (Jaccard 0.64 to 1.00) but moves the
objective by at most 0.15 VAEP/90, and not at all at 1,500,000, where
the budget stops binding. Match uncertainty is a squad-composition
question, not an objective question, at these budgets.

### Exclusion sensitivity: the what-if rated set

Admitting the unmatched players at position-group proxy wages, with
their real VAEP, rates 447 players instead of 268 (179 newly rated
player IDs) and re-solves at the same five budgets:

| budget (EUR/week) | Jaccard vs base | objective | change |
|---:|---:|---:|---:|
| 300,000 | 0.394 | 11.53 vs 8.88 | +2.65 |
| 500,000 | 0.314 | 12.54 vs 9.66 | +2.88 |
| 700,000 | 0.314 | 13.13 vs 10.26 | +2.87 |
| 943,000 | 0.314 | 13.34 vs 10.63 | +2.71 |
| 1,500,000 | 0.353 | 13.34 vs 10.74 | +2.60 |

About two thirds of each squad turns over (10 to 12 players enter and
the same number leave at every budget), and the entering players are
the excluded stars at proxy wages: Javier Hernandez Balcazar, Eric
Maxim Choupo-Moting, Karim Bellarabi, Andre Hahn, Claudio Pizarro,
Dominik Maroh, John Anthony Brooks, Lois Openda and Nathan Tella enter
at every budget.

Read this as the scale of the exclusion effect, not as a corrected
optimum. The proxy wages are position-group medians (FWD 23,000,
MID 25,000, DEF 22,000 EUR/week), far below what those players were
actually paid, so the what-if systematically over-selects the excluded
stars. The honest statement is that the rated set is not
representative: the players the matcher cannot price include some of
the best VAEP/90 in the pool, and any squad built from the rated set
alone is chosen from a pool that is missing them. Fixing that needs
real wages for the 2015/16 era, not a better proxy.

## Match quality, measured separately

The independently checked real-player gold set from C6 (16 players,
wages read directly from the wage file) still scores precision 1.0 and
recall 1.0 under the new matcher, including the reviewed Max Meyer row
at 11,000 EUR/week. The scorer is exercised in the U05 test file with a
reviewed row included, and corrupting one expected wage still drops
precision, so the scorer measures rather than rubber-stamps.

## What was restated

`demo/meta.json`: the provenance matching audit now reads exact 314,
exact_other_club 120, alias 0, fuzzy 0, reviewed 43, ambiguous 0,
proxy 0, unmatched 229, 706 players, match rate 0.6756 (was exact 435,
fuzzy 11, proxy 0, unmatched 260, 63.17%), and the top-level
`wage_match_rate` moves with it to 0.6756. Nothing else in the file
changed. The five-league results (`runs/five_leagues/results.json`)
still carry the old-matcher per-league rates (Bundesliga 63.17%, the
other leagues 26.8% to 37.7%): those are the frozen FV4 run, and they
change only when the five-league run is recomputed, which is not this
unit.

## Limits

- One pool, two Bundesliga seasons, 68 matches. The registry resolves
  the uncertain matches in this pool; other leagues and seasons have
  their own uncertain matches, unreviewed.
- The registry accepts are only as good as the evidence behind them:
  27 fuzzy accepts rest on uniqueness at the club, 13 variant accepts
  on same-club token subsets, and 3 on verified career paths (Meyer,
  Dante twice). The 27 remaining variant candidates are deliberately
  not accepted.
- The exclusion what-if uses proxy wages, so its objective gains are
  an upper bound on the exclusion effect, not an estimate of a better
  squad.
- The wage file itself is not redistributed and is not in this repo;
  the audit reproduces from the local copy whose SHA-256 is recorded
  above.
