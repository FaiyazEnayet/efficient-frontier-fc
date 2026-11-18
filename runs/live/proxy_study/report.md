# Event-summary proxy vs fitted VAEP

League: English Premier League, season 2015/2016 (380 matches, StatsBomb open data).
Baseline: StatsBomb open data, English Premier League 2015/2016 (380 matches), EA FC 24 wages; ratings as published in runs/five_leagues/results.json (n_rated 179)

## What was measured

The event-summary proxy is the live refresh's proxy: goals 0.9 (penalties 0.7), assists 0.4, yellow cards -0.05, red cards -0.30, per 90 minutes, with minutes from lineups and substitution spans. It was computed from the same season's match events and lineups, and each player was matched to the committed baseline ratings with the same team-constrained matcher the refresh uses. Only players with at least 90 proxy minutes are compared.

## Result

- Players compared: 179 of 179 rated baseline players (660 players had any proxy minutes).
- Mean proxy per 90: 0.116; mean fitted VAEP per 90: 0.396.
- Pearson r: 0.694; Spearman rho: 0.607.
- Calibration of VAEP on the proxy: slope 0.440, intercept 0.345.
- Mean absolute error of the best linear rescaling: 0.046 VAEP per 90.

Verdict: the proxy is not calibrated: the calibration slope is 0.44, not near 1; the intercept is 0.345, not near 0; the Pearson correlation is 0.69, below 0.7.

## What this means for the live refresh

The blended rating the refresh produces is a refresh mechanism, not a measured improvement, until this study says otherwise. The proxy and VAEP move together only as far as the numbers above show; the blend weights and the shrinkage were not tuned to this study, and the study was not used to tune them.

## Highest proxy per 90 (compared players)

- Jamie Vardy (Leicester City): proxy 0.599, VAEP 0.688, 3315 minutes
- Marcus Rashford (Manchester United): proxy 0.584, VAEP 0.483, 878 minutes
- Harry Kane (Tottenham Hotspur): proxy 0.556, VAEP 0.676, 3511 minutes
- Riyad Mahrez (Leicester City): proxy 0.533, VAEP 0.727, 3117 minutes
- Roberto Firmino Barbosa de Oliveira (Liverpool): proxy 0.517, VAEP 0.632, 2054 minutes
- Callum Wilson (AFC Bournemouth): proxy 0.477, VAEP 0.580, 811 minutes
- Adnan Januzaj (Manchester United): proxy 0.477, VAEP 0.494, 160 minutes
- Andy Carroll (West Ham United): proxy 0.477, VAEP 0.596, 1548 minutes
- Kevin De Bruyne (Manchester City): proxy 0.420, VAEP 0.602, 2108 minutes
- Dušan Tadić (Southampton): proxy 0.397, VAEP 0.612, 2359 minutes

Full per-player table: results.json (matched_players).
