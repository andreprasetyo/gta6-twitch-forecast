"""gta6-twitch-ccu-forecast-01 entry: per-hour weighted log-log regression over comparable game launches.

For each forecast hour h (2..24), fit a weighted least squares of ln(viewers(h)) on
[1, ln(viewers(h=1)), ln(hour1/pre_baseline)] across comparable launches, weighting each
comparable by similarity of its hour-1 print and pre-release growth to the case's.
Quantiles come from the weighted empirical quantiles of that hour's residuals, shrunk by 0.5.

Runs under `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0), standard library only:
reads every <id>.in in $MAKEFUN_INPUT_DIR (default /input) and writes <id>.out in
$MAKEFUN_OUTPUT_DIR (default /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def round2(x: float) -> float | int:
    """Round half up to 2 decimals, matching JavaScript's Math.round(x * 100) / 100."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def weighted_quantile(pairs: list[tuple[float, float]], tau: float) -> float:
    """Weighted empirical quantile: smallest value whose cumulative weight reaches tau * total."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    acc = 0.0
    target = tau * total
    for value, w in pairs:
        acc += w
        if acc >= target:
            return value
    return pairs[-1][0]


def solve3(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve a 3x3 linear system by Gaussian elimination with partial pivoting."""
    n = 3
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            m[piv][col] = 1e-12
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))
        x[r] = s / m[r][r]
    return x


def build_comparable(comp: dict) -> dict | None:
    """Release-day hour -> viewers, plus pre-release baseline, for one comparable."""
    day_start = parse_ts(comp["day_start"])
    hours = {}
    pre = []
    for pt in comp.get("hourly") or []:
        t = parse_ts(pt["from"])
        delta = (t - day_start).total_seconds() / 3600.0
        if delta < 0 and delta >= -24:
            if pt.get("viewers") is not None:
                pre.append(pt["viewers"])
        elif delta >= 0:
            h = int(delta) + 1
            if 1 <= h <= 24 and pt.get("viewers") is not None:
                hours[h] = pt["viewers"]
    if 1 not in hours or hours[1] <= 0:
        return None
    baseline = (sum(pre) / len(pre)) if pre else None
    g = math.log(hours[1] / baseline) if baseline and baseline > 0 else 0.0
    return {"hours": hours, "ly1": math.log(hours[1]), "g": g}


def forecast_case(case: dict, name: str) -> dict:
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    y1 = max(1, first["viewers"])
    ly1 = math.log(y1)

    checkpoint = parse_ts(case["checkpoint"])
    lo = checkpoint - timedelta(hours=25)
    hi = checkpoint - timedelta(hours=1)
    pre_vals = [p["viewers"] for p in (case.get("history") or [])
                if p.get("viewers") is not None and lo <= parse_ts(p["at"]) < hi]
    g_case = math.log(y1 / (sum(pre_vals) / len(pre_vals))) if pre_vals and sum(pre_vals) > 0 else 0.0

    comps = []
    for c in case.get("comparables") or []:
        b = build_comparable(c)
        if b is None:
            continue
        w = math.exp(-((b["ly1"] - ly1) / 3.0) ** 2) * math.exp(-((b["g"] - g_case) / 1.0) ** 2) + 1e-3
        b["w"] = w
        comps.append(b)

    out = []
    for row in case["forecast"]:
        h = row["hour"]
        rows = [(c["ly1"], c["g"], math.log(c["hours"][h]), c["w"])
                for c in comps if h in c["hours"] and c["hours"][h] > 0]
        if not rows:
            q = [round2(y1)] * 5
            out.append({"hour": h, "q": q})
            continue
        # Weighted least squares: (X'WX) beta = X'Wy, X columns [1, ly1, g].
        xtwx = [[0.0] * 3 for _ in range(3)]
        xtwy = [0.0] * 3
        for a, b_, y, w in rows:
            x = (1.0, a, b_)
            for i in range(3):
                xtwy[i] += w * x[i] * y
                for j in range(3):
                    xtwx[i][j] += w * x[i] * x[j]
        beta = solve3(xtwx, xtwy)
        pred = beta[0] + beta[1] * ly1 + beta[2] * g_case
        resid = [(y - (beta[0] + beta[1] * a + beta[2] * b_), w) for a, b_, y, w in rows]
        q = []
        for tau in LEVELS:
            r = weighted_quantile(resid, tau) * 0.5
            val = math.exp(pred + r)
            if not math.isfinite(val) or val <= 0:
                val = y1
            val = min(val, 1e9)
            q.append(round2(val))
        # Enforce non-decreasing.
        for i in range(1, 5):
            if q[i] < q[i - 1]:
                q[i] = q[i - 1]
        out.append({"hour": h, "q": q})
    return {"hours": out}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
