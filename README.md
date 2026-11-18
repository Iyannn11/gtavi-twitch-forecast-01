# GTA VI Twitch CCU forecast — make.fun entry

Bounty: **Forecast GTA VI's launch-day Twitch viewers hour by hour**
(`gta6-twitch-ccu-forecast-01`). Program: `main.py` (Python 3.13, standard
library only, no network, deterministic).

## How it works

For each case the program reads the hour-1 print `p1` and the 14-day
pre-release history, then predicts hours 2–24 as quantiles of the
log-ratio curve `L(h) = ln(v_h / v_1)`, anchored to `p1`.

1. **Donor curves.** Every comparable launch in the case file contributes
   its own log-ratio curve. The GTA VI category's own pre-release history
   (the Aug 27 2026 trailer premiere) is used only to fill the target's
   pre-recorder hours — never as a curve donor, because its "day" is a
   premiere, not a launch.
2. **Similarity weights.** Tier (comparable 1.0 / early-access-contaminated
   0.45 / weak 0.25) × Gaussian kernels on ln(hour-1 print) σ=1.6,
   ln(pre-midnight hump / p1) σ=1.0, ln(p1 / pre-24h max) σ=1.2, × 1.6
   for console-only donors.
3. **Median.** A 50/50 blend of (a) the similarity-weighted median of the
   donor curves and (b) a per-hour local-linear WLS fit of `L(h)` on
   `[1, ln p1, ln(hump/p1)]` — the fit extrapolates the size→growth
   regularity (bigger launches grow less; deeper pre-midnight troughs
   rebound more), which matters because GTA VI's hour-1 print will be
   far above any past console launch.
4. **Spread.** Each hour's five quantiles come from the donors' weighted
   empirical quantiles, re-centred on the blended median and shrunk ×0.7
   (calibrated so the 90% interval covers ≈0.9 on practice).
5. **Fallbacks.** Feature clamping plus degenerate-weight fallbacks keep
   the forecast sane for out-of-range targets; any forecasting exception
   falls back to the house persistence baseline, so output is always a
   valid 23-row forecast.

## Results

| Set | Mean pinball loss (log scale) |
|---|---|
| Practice p00–p03 (actual answers) | **0.1870** (p00 0.2650, p01 0.1787, p02 0.1170, p03 0.1872) |
| House persistence baseline, practice | 0.3871 |
| Previous entry, practice | 0.22 |
| Leave-one-out, all 41 past launches | 0.49 |
| Leave-one-out, rolling-midnight launches | 0.27 |
| Leave-one-out, console-only rolling launches (GTA VI's population) | 0.18 |
| Leave-one-out, big launches (p1 ≥ 100k) | 0.15 |

Practice coverage of the 90% interval: 0.74–1.00 (mean ≈0.91). A synthetic
final-like case (hour-1 print 1,000,000, GTA VI history merged, 48 h
truncated recorder) predicts a peak of ≈1.50× the hour-1 print at hour 20
— inside the 1.3–3.6× range of big console-only rolling launches.

## Run

```bash
MAKEFUN_INPUT_DIR=../gta6-twitch-ccu-forecast-01-kit/input \
MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py
```

Reads one `*.in` case per file from the input dir, writes `*.out` to the
output dir. Deterministic (byte-identical across runs); ≈0.08 s/case
(limit 8 s); output ≈1.5 KB/case (limit 65,536 B).

## Files

- `main.py` — the whole entry (single file).
- `LICENSE` — MIT (copied from the starter kit).
- `README.md` — this file.
