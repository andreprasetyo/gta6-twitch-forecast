# gta6-twitch-forecast

Quantile forecaster for the make.fun bounty **Forecast GTA VI's launch-day Twitch viewers hour by hour** (`gta6-twitch-ccu-forecast-01`).

`main.py` reads make.fun case files (`makefun/twitch-ccu-case/v1`) and emits 23 rows of quantile forecasts (levels 0.05/0.25/0.5/0.75/0.95) for launch-day hours 2–24.

## How it works

Per-hour weighted log-log regression over ~40 comparable past game launches bundled in each case:

1. Each comparable's release-day hour-1 viewers and pre-release growth (`ln(hour1 / pre-release baseline)`) become its features; the case's own hour-1 print and pre-checkpoint growth are the query point.
2. Comparables are weighted by a Gaussian kernel on both features.
3. For each hour h = 2..24, a weighted least squares of `ln(viewers(h))` on `[1, ln(hour1), growth]` gives the median prediction; quantiles come from the weighted empirical quantiles of the residuals (shrunk ×0.5), exponentiated and rounded half-up to 2 decimals.

Python 3.13, standard library only, no network at run time, deterministic (PYTHONHASHSEED=0 safe). Local practice loss ≈ 0.181 pinball vs ~0.387 for the starter persistence baseline.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py
```

Reads every `<id>.in` in the input dir, writes `<id>.out` (`{"hours": [{"hour": h, "q": [q05..q95]}]}`).

## License

MIT — see LICENSE.
