# Action-value audit: repo VAEP against a reference SPADL / Atomic-SPADL replication

Stage 3 U06. The repo values on-ball actions with its own VAEP-style
formula (`src/effc/vaep/values.py`) on its own featurization
(`src/effc/vaep/features.py`). This report checks that pipeline against
an independent replication built on socceraction 1.5.3, the reference
implementation behind the VAEP paper (Decroos et al. 2019). The audit
code lives in `src/effc/analysis/action_audit.py` and the tests in
`tests/test_effc.py` (names starting `test_stage3_efficient_frontier_fc_06_`).

## Spot-check match

- Bayer Leverkusen 2-2 VfB Stuttgart, 2024-04-27, Bundesliga 2023/24.
- StatsBomb open data, match 3895320, 3,771 events.
- Source: `https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/3895320.json`
- Event file SHA-256: `87b2347238c7946d625b1ba2a20042aef2f906dafbd7a65c7624871b13ae2bce`
- Committed test fixture: `tests/fixtures/statsbomb/events_3895320_excerpt.json`
  (141 events, indices 1990-2130, one Stuttgart goal; SHA-256
  `9c72c600d6b6cafda2e6f58b6869e19958c3b513f61c9cd705da44ea375b862b`).

## Method note: the shared probability source

The value comparison shares one probability source on purpose: both
pipelines use the repo goal-model probabilities, carried over by matched
event id. The comparison therefore isolates the action transformation
(action set, coordinates, sequence membership), not the goal model. The
probabilities for this one match come from an in-sample fit of the repo
goal models on this match alone (seed 0, `train_goal_models`). That fit
is a comparison device only: it is not the shipped model, and the
absolute values below (especially the goal rows, where the in-sample fit
predicts the goal before it happens) are not model behavior. The
shipped models train on pooled matches with proper splits. What the
numbers below establish is how much the two transformations disagree
when the model is held fixed.

## Headline findings

1. **The repo coordinate frame is right and the reference direction fix
   is wrong for this data.** StatsBomb records every event in the acting
   team's attacking frame. Across 195 possession changes, the next
   action starts a median 1.53 m from the mirror of the previous
   action's end and 88.17 m from the shared-frame point; the mirror is
   closer in 99.5% of pairs. Shots by both teams sit at raw x 89-114 and
   goalkeepers at raw x 1.5-3.0 in both halves. The reference conversion
   flips every away-team action to the opposite end: away shots land at
   SPADL x 5.2-25.2 (mean 13.1) instead of 91-114, and only 0.6% of
   matched away actions agree with the repo coordinates within 0.5 m.
   A VAEP built naively on the reference conversion would value
   away-team actions at mirrored locations. The repo scale-only
   conversion is the correct one here.
2. **The reference also mirrors y; the repo does not.** For home-team
   actions x agrees (median |dx| 0.04 m) and y is mirrored (median |dy|
   38.93 m, matching 68 - y). For away-team actions the two flips cancel
   on y (median |dy| 0.04 m) but x is mirrored (median |dx| 41.78 m).
   The y difference is benign for the repo model (distances and angles
   use absolute y displacement), but it is a real frame difference.
3. **The two pipelines value different action sets.** The repo values
   2,300 actions, the reference 2,157, with 2,095 matched. The repo
   alone values Ball Recovery (99), Block (39), Dispossessed (24),
   non-tackle Duels (19), goalkeeper events the reference drops (20) and
   4 passes with Injury Clearance or Unknown outcomes. The reference
   alone values fouls (29) and 29 synthesized dribbles that are not
   real events, plus 4 extra pass rows from splitting interception
   passes into an interception and a pass.
4. **Matched values agree closely, and the disagreements have a clear
   cause.** Mean absolute difference 0.00049, median 0, 90th percentile
   0; 1.24% of matched actions differ by more than 0.01 and a single
   action differs by more than 0.10 (max 0.121). Turnover rows disagree
   about twice as much as same-team rows (mean 0.0010 vs 0.0004): the
   dropped actions shift the before-state and the lookahead window of
   the actions that remain. The largest per-player effect is Serhou
   Guirassy, +0.149 (repo -0.002 over 83 actions vs reference -0.152
   over 79): four dropped actions carry that much of his total.
5. **The goal-conceded debit works as designed.** All six formula
   invariants pass on the full match. Each of the four goals debits
   exactly one later row, and the pre-debit value of those rows is
   +0.94 to +0.96 defensive value: without the debit, the keeper's
   post-goal event would earn almost a full goal of defensive credit
   for a goal that already happened. Three of the four debits land on
   Goal Keeper rows; the fourth lands on a Block (Serhou Guirassy,
   after the late Andrich goal), so the debit rule reaches the first
   conceding-team action whatever its type.
6. **Atomic-SPADL credits receivers; the repo does not.** Atomic-SPADL
   produces 3,253 rows from the same match: 895 receival rows and 159
   interception rows that have no repo counterpart. After every
   pass-like action it inserts one row at the pass end: a receival when
   the next action is by the same team, an interception credited to the
   next actor when the next action is by the other team. The repo gives
   the full pass value to the passer and nothing to the receiver. That
   is a design choice, not an error, but it is now a measured gap:
   receiving credit is 895 actions the repo never sees.

## Action-coverage table (full match)

| Event type | Raw | Repo | SPADL | Matched | Repo only | SPADL only |
|---|---:|---:|---:|---:|---:|---:|
| Pass | 1076 | 1076 | 1076 | 1072 | 4 | 4 |
| Ball Receipt* | 1027 | 0 | 0 | 0 | 0 | 0 |
| Carry | 872 | 872 | 872 | 872 | 0 | 0 |
| Pressure | 343 | 0 | 0 | 0 | 0 | 0 |
| Ball Recovery | 99 | 99 | 0 | 0 | 99 | 0 |
| Duel | 51 | 51 | 32 | 32 | 19 | 0 |
| Block | 39 | 39 | 0 | 0 | 39 | 0 |
| Goal Keeper | 34 | 34 | 14 | 14 | 20 | 0 |
| Foul Committed | 29 | 0 | 29 | 0 | 0 | 29 |
| Foul Won | 29 | 0 | 0 | 0 | 0 | 0 |
| Miscontrol | 28 | 28 | 28 | 28 | 0 | 0 |
| Shot | 25 | 25 | 25 | 25 | 0 | 0 |
| Dispossessed | 24 | 24 | 0 | 0 | 24 | 0 |
| Dribble | 21 | 21 | 21 | 21 | 0 | 0 |
| Clearance | 16 | 16 | 16 | 16 | 0 | 0 |
| Interception | 15 | 15 | 15 | 15 | 0 | 0 |
| Dribbled Past | 13 | 0 | 0 | 0 | 0 | 0 |
| Substitution | 10 | 0 | 0 | 0 | 0 | 0 |
| Half Start | 4 | 0 | 0 | 0 | 0 | 0 |
| Half End | 4 | 0 | 0 | 0 | 0 | 0 |
| Tactical Shift | 4 | 0 | 0 | 0 | 0 | 0 |
| Starting XI | 2 | 0 | 0 | 0 | 0 | 0 |
| Error | 2 | 0 | 0 | 0 | 0 | 0 |
| Bad Behaviour | 2 | 0 | 0 | 0 | 0 | 0 |
| Shield | 1 | 0 | 0 | 0 | 0 | 0 |
| Injury Stoppage | 1 | 0 | 0 | 0 | 0 | 0 |
| synthesized | 0 | 0 | 29 | 0 | 0 | 29 |
| **Total** | **3771** | **2300** | **2157** | **2095** | **205** | **62** |

Notes: the 4 repo-only passes are Injury Clearance / Unknown outcome
passes the reference discards as non-actions. The 4 SPADL-only passes
are the extra rows from splitting interception passes. The 19 repo-only
Duels are non-tackle duels. The 20 repo-only Goal Keeper rows are
keeper types the reference maps to non-action (for example claims the
reference does not count). Neither pipeline values stoppages: Foul Won,
Substitution, Half Start / Half End, Tactical Shift, Starting XI,
Injury Stoppage, Bad Behaviour, Shield, Error, Dribbled Past, Pressure
and Ball Receipt* all score zero on both sides.

## Attributing disagreement

Matched actions, both pipelines sharing the repo probability source
(n = 2,095):

| Statistic | Value |
|---|---:|
| Mean absolute difference | 0.00049 |
| Median absolute difference | 0.00000 |
| 90th percentile | 0.00000 |
| Max absolute difference | 0.121 |
| Share differing by more than 0.01 | 1.24% |
| Share differing by more than 0.05 | 0.29% |
| Share differing by more than 0.10 | 0.05% |

By action type (mean absolute difference): Clearance 0.0060 (max
0.081), Shot 0.0021 (max 0.031), Duel 0.0012 (max 0.020), Pass 0.0005
(max 0.121), Carry 0.0004 (max 0.081), Interception 0.00003,
Dribble 0.0000, Goal Keeper 0.0000, Miscontrol 0.0000.

By possession transition: turnover rows (224) mean 0.0010, max 0.081;
same-team rows (1,871) mean 0.0004, max 0.121.

Documented causes, in order of size:

- **Dropped actions shift the sequence.** The 205 repo-only actions
  are invisible to the reference, so the reference before-state and
  lookahead window for the actions around them differ. This is the
  whole disagreement: every matched action whose repo neighbours
  include a dropped action can differ, and Clearance, Shot and Duel
  sit closest to those gaps.
- **Split interception passes.** Four passes become two reference
  actions each; the extra row shifts the reference sequence by one.
- **Own-goal debit is absent from the reference sequence.** Own goals
  have no action rows, so the reference cannot apply the debit. This
  match has no own goals, so it contributes zero here; the own-goal
  debit is covered by synthetic fixtures instead.
- **Coordinates never enter this comparison.** The reference
  coordinates are wrong for away-team actions (see finding 1), but the
  comparison carries repo probabilities, so coordinate error cannot
  move these values. A reference built on its own coordinates would
  disagree far more; that is a separate, coordinate-level finding.

Per-player aggregates: repo total 5.539 over 2,300 actions vs reference
total 5.210 over the 2,095 matched actions, a gap of 0.329 carried by
the 205 repo-only actions plus sequence effects. Largest absolute
per-player differences: Serhou Guirassy +0.149, Odilon Kossounou
+0.071, Chris Fuhrich +0.058, Leonidas Stergiou +0.055, Jonas Hofmann
+0.049, Anthony Rouault -0.043, Edmond Faycal Tapsoba +0.041,
Woo-Yeong Jeong -0.041, Hiroki Ito +0.040, Robert Andrich -0.039.

Repo-only action totals: Ball Recovery +0.139 over 99 actions, Duel
+0.112 over 51, Goal Keeper +0.175 over 34, Block -0.042 over 39,
Dispossessed -0.008 over 24. The repo-only actions are small individually
and sum to +0.377; the rest of the 0.329 gap is sequence effects on
matched actions.

## Invariants (full match, all pass)

- Values add up: max |vaep - (vaep_off + vaep_def)| = 5.55e-17.
- Probabilities in range [0.0000, 0.9783].
- Goal realization: max error 0.00 over 4 goals.
- Debit exactly once per goal: debited rows match the rule's rows
  exactly, each by exactly -1.0.
- No NaN values.
- Debit total -4.0 over 4 goals with a following action.

Debit rows (match index, player, action, pre-debit defensive value,
net): 2003, Lukas Hradecky, Goal Keeper, +0.961, -0.039; 2461, Lukas
Hradecky, Goal Keeper, +0.951, -0.049; 2569, Alexander Nubel, Goal
Keeper, +0.939, -0.061; 3757, Serhou Guirassy, Block, +0.937, -0.063.

## Chance creation and goalkeeper inspection

The highest-valued repo actions are buildup actions just before goals,
not the shots themselves: Hincapie Carry 0.943 and Ito Carry 0.943,
Hincapie Pass 0.919, Anton Pass 0.896, Ito Pass 0.894, Tapsoba Pass
0.893. The four goal shots score 0.085 (Fuhrich), 0.062 (Undav),
0.088 (Adli) and 0.135 (Andrich) under the in-sample probability
source, which already expects the goal from the buildup; under the
shipped models these numbers would look different, and the audit makes
no claim about them beyond the transformation comparison.

Goalkeeper rows total +0.175 over 34 actions. The largest single
keeper value is Hradecky at match index 2501 (+0.119), a save sequence.
Nubel's Goal Keeper row at 2569 (-0.061) is the post-Adli-goal debit
row, not a save.

## What is not claimed

- **No numeric VAEP equivalence is claimed.** The repo models are its
  own, and the reference values here share the repo probabilities, so
  nothing here demonstrates agreement with published VAEP numbers. A
  true numeric replication would need a reference model trained on the
  same data, which is out of scope for this audit.
- **One match only.** The spot check is a single Bundesliga match with
  four goals and no own goals. Own goals, red cards and penalty shots
  are covered by synthetic fixtures in the test suite, not by real
  data here.
- **Atomic-VAEP values are not computed.** Atomic-SPADL actions are
  compared as an action set (counts, receival and interception credit),
  but valuing them would need a goal model trained on atomic actions,
  which does not exist in this repo.

## Reproducing the audit

```python
from effc.analysis.action_audit import audit_match
from effc.vaep.features import featurize_match
from effc.vaep.models import train_goal_models
from effc.vaep.values import add_vaep

events = ...  # raw StatsBomb events, sorted by index
frame = featurize_match(events, match_id=3895320)
models = train_goal_models([frame], seed=0)  # comparison source only
valued = add_vaep(frame, models)
result = audit_match(events, match_id=3895320, home_team_id=904,
                     repo_valued=valued)
result["coverage"]          # per-type coverage table
result["disagreement"]      # disagreement distribution
result["player_aggregates"] # repo vs reference per-player totals
result["invariants"]        # six invariant checks
```

The committed excerpt fixture reproduces the coverage, orientation and
disagreement checks in the test suite without any download.
