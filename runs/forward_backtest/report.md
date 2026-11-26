# Forward-only backtest experiment (U04 part 2)

Premier League 2015/16, three expanding decisions. Goal models retrain per cutoff on pre-cutoff frames only from contemporaneous 2015/16 competitions; UEFA Euro 2020 and Indian Super League 2021/22 never enter any training pool.

## Decisions

- fwd-01-2015-11-01: cutoff 2015-11-01, 432 input matches, model 6d1ccb5ac8555a49, retrain=True
- fwd-02-2016-01-01: cutoff 2016-01-01, 734 input matches, model b928a75641328392, retrain=True
- fwd-03-2016-03-01: cutoff 2016-03-01, 1097 input matches, model 3f919c26974dc998, retrain=True

## Policy comparison (penalized realized VAEP)

- fwd-01-2015-11-01 budget 300000: optimizer-random +24.51, optimizer-value +12.85, value-random +11.66
- fwd-01-2015-11-01 budget 500000: optimizer-random +29.53, optimizer-value +22.68, value-random +6.85
- fwd-01-2015-11-01 budget 700000: optimizer-random +36.17, optimizer-value +31.47, value-random +4.70
- fwd-02-2016-01-01 budget 300000: optimizer-random +17.40, optimizer-value +12.71, value-random +4.69
- fwd-02-2016-01-01 budget 500000: optimizer-random +16.98, optimizer-value +17.92, value-random -0.94
- fwd-02-2016-01-01 budget 700000: optimizer-random +26.65, optimizer-value +31.09, value-random -4.44
- fwd-03-2016-03-01 budget 300000: optimizer-random +34.12, optimizer-value +28.66, value-random +5.46
- fwd-03-2016-03-01 budget 500000: optimizer-random +50.92, optimizer-value +48.97, value-random +1.95
- fwd-03-2016-03-01 budget 700000: optimizer-random +60.34, optimizer-value +60.13, value-random +0.22

Gaps are observed differences on this run, not wins; a null (the optimizer not ahead) is reported as the observed gap.

## Limitations

- Wages are the repo's FC24 snapshot, not a 2015-dated wage history.
- Three cutoffs on one competition (Premier League 2015/16).
- Goal models retrain per cutoff on pre-cutoff frames only; the shipped FC6 models are untouched.
- No published number is replaced by this experiment.
