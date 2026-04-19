# Statistical decision rules (public reproducibility)

This document summarizes the key, predeclared decision rules used by the analysis scripts in this package.

## Forecast horizons and holdout

- Lead times evaluated: 1–4 weeks.
- Holdout: the final 52 available weeks per series (time-ordered split).

## Primary accuracy metric

- Root mean squared error (RMSE) over the holdout period.
- Relative RMSE is reported as `RMSE_fusion / RMSE_baseline`.
- Skill score is reported as `1 − (RMSE_fusion / RMSE_baseline)` (higher is better).

## Paired predictive-accuracy checks

- A paired Diebold–Mariano-style test is computed per state–pathogen series at one-week lead time using holdout-week squared-error differentials.
- Multiple testing: Benjamini–Hochberg false discovery rate correction is applied across the 150 paired tests (two-sided).

## Turning-point and alarm sensitivity checks

In addition to average-error metrics, the package includes sensitivity checks oriented to decision utility:

- Direction-of-change accuracy and peak-timing error (one-week-ahead).
- Surge alarm thresholds defined as 75th and 90th percentiles computed from pre-holdout history; summary includes detection rates and false discovery rates.

## Stability-window definitions (revision-history replay)

- Published-value stability: days from first release to last observed change for that value.
- Forecast stability: earliest time after first release such that all subsequent snapshot forecasts remain within a tolerance of the final forecast.
- Tolerance used by default: `max(0.05, 1% of final forecast value)`.

