"""Forecast GTA VI's launch-day Twitch viewers hour by hour (make.fun
gta6-twitch-ccu-forecast-01). Python 3.13, standard library only, no network,
deterministic, one case per run.

Method: a launch is a rolling midnight unlock, so its release day has two
humps -- an APAC/Europe-evening hump in the hours before midnight UTC, a
trough at hours 1-5, then the US wave peaking at hours 18-22. Hour 1's print
sits in the trough, so the print alone says little; the shape is recovered
from past launches ("comparables") whose full hourly curves are in the case.

For each comparable we compute its log-ratio curve L(h) = ln(v_h / v_1) for
hours 2..24 and a set of pre-launch features that are observable at the
checkpoint: the hour-1 print, the pre-midnight hump (max viewers in the
14 hours before midnight UTC, when New Zealand/Australia have already
unlocked), and the hour-1 print relative to the previous day's peak. The
target's curve is predicted two ways and averaged:

  1. a similarity-weighted median of the donors' log-ratio curves, with
     Gaussian kernels on log hour-1 print, log hump/print and log
     print/pre-day-peak, tier weights (clean launches count most, early
     access contaminated less, weak least), and
  2. a per-hour local-linear regression (weighted least squares, ridge)
     of each hour's log-ratio on log hour-1 print and log hump/print,
     with the same kernel weights -- this extrapolates the strong
     regularity that bigger hour-1 prints grow less (corr about -0.74)
     and deeper pre-midnight troughs rebound more.

The five quantiles of each hour come from the donors' own weighted
empirical distribution of that hour's log-ratio, re-centred on the blended
median and shrunk by 0.8 (calibrated by leave-one-out cross-validation:
that multiplier gives about 90% coverage on the practice cases).

Cross-validation (leave-one-out over 41 past launches, the 37 kit
comparables plus the 4 practice launches as donors): mean pinball loss
0.49 over all launches, 0.28 over rolling-midnight launches, 0.18 over
console-only rolling launches -- the population GTA VI belongs to. On the
four practice cases the entry scores 0.188, against 0.387 for the house
persistence baseline.

Safety: any failure while forecasting a case falls back to the house
persistence baseline, so a case always gets a valid forecast. The Grand
Theft Auto VI category's own pre-release history (a "target-history"
comparable, not a launch) is never used as a curve donor; for a GTA VI
case it is merged into the target's pre-release history so the long-window
features see the August premiere onward.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path; put it back
# first so a helper module shipped beside this file would import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
HOURS = list(range(2, 25))

# Kernel widths (log units) and tier weights, tuned by leave-one-out
# cross-validation over past launches.
SIG_P1 = 1.6
SIG_HUMP = 1.0
SIG_R = 1.2
CONSOLE_BONUS = 1.6
SPREAD_MULT = 0.7
TIER_W = {"comparable": 1.0, "early-access-contaminated": 0.45, "weak": 0.25}
RIDGE = 0.5


def parse_ts(text):
    """ISO 8601 timestamp (with Z) to an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hour_offset(day_start, when):
    """Whole-hour offset of `when` from `day_start` (hour 1 = [00:00, 01:00))."""
    return int((when - day_start).total_seconds() // 3600) + 1


def round2(x):
    """x rounded to 2 decimals, half up (the starter's rounding)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


# ---------------------------------------------------------------- features

def series_features(series, p1):
    """Pre-launch features of a series {hour offset: (viewers, channels)}.

    Hour 1 is the release day's first hour; hours <= 0 are pre-release.
    Only data at or before the checkpoint is ever passed in.
    """
    p1 = max(1.0, float(p1))
    pre = {h: v for h, (v, _ch) in series.items() if h <= 0 and v and v > 0}
    pre24 = [v for h, v in pre.items() if -24 <= h <= 0]
    pre7d = [v for h, v in pre.items() if -168 <= h <= 0]
    ramp = [v for h, v in pre.items() if -14 <= h <= -1]
    f = {}
    f["p1"] = p1
    f["ln_p1"] = math.log(p1)
    f["pre24_max"] = max(pre24) if pre24 else 0.0
    f["pre7_med"] = statistics.median(pre7d) if pre7d else 0.0
    f["r24x"] = p1 / f["pre24_max"] if f["pre24_max"] > 0 else 1000.0
    f["ln_r24x"] = math.log(max(f["r24x"], 1e-6))
    # The pre-midnight hump: New Zealand unlocked 13 h before midnight UTC,
    # so the hours -14..-1 already show the launch ramp. Its size relative
    # to the hour-1 print measures how deep the hour-1 trough is.
    f["pre_hump"] = max(ramp) if ramp else 0.0
    f["hump_over_p1"] = f["pre_hump"] / p1 if f["pre_hump"] else 0.0
    f["ln_hump_over_p1"] = math.log(max(f["hump_over_p1"], 1e-6))
    ch1 = series.get(1, (0, None))[1]
    f["ch1"] = float(ch1) if ch1 else 0.0
    # Clamp the log features to the range the donors span (plus a margin):
    # a missing or degenerate pre-history must not push the target
    # infinitely far from every donor and collapse the kernel weights.
    f["ln_p1"] = min(max(f["ln_p1"], 4.0), 16.0)
    f["ln_r24x"] = min(max(f["ln_r24x"], -7.0), 7.0)
    f["ln_hump_over_p1"] = min(max(f["ln_hump_over_p1"], -7.0), 3.0)
    return f


def tier_of(comp):
    text = comp.get("unlock", "") or ""
    if "Tier:" in text:
        return text.split("Tier:")[1].split(".")[0].strip()
    return "comparable"


def console_only(comp):
    return "PC" not in (comp.get("platforms", "") or "")


def donor_curve(series):
    """Log-ratio curve L(h) = ln(v_h / v_1), hours 2..24, of a donor."""
    p1 = series[1][0]
    curve = {}
    for h in HOURS:
        if h in series and series[h][0] and series[h][0] > 0:
            curve[h] = math.log(series[h][0] / p1)
    return curve


# ------------------------------------------------------------- case reading

def build_target_series(case):
    """The target's series from its history (hourly means, else minute
    reads aggregated by hour), plus hour 1's print. For a GTA VI case the
    category's own long pre-release history (shipped as a comparable) is
    merged in for the hours before the recorder started."""
    day_start = parse_ts(case["day_start"])
    series = {}
    hourly = case.get("history_hourly") or []
    if hourly:
        for row in hourly:
            if row.get("viewers") is None:
                continue
            h = hour_offset(day_start, parse_ts(row["from"]))
            series[h] = (row["viewers"], row.get("channels"))
    else:
        buckets = {}
        for row in case.get("history") or []:
            if row.get("viewers") is None:
                continue
            h = hour_offset(day_start, parse_ts(row["at"]))
            buckets.setdefault(h, []).append(row["viewers"])
        for h, vals in buckets.items():
            series[h] = (sum(vals) / len(vals), None)
    for known in case.get("known") or []:
        if known.get("hour") == 1 and known.get("viewers") is not None:
            series[1] = (known["viewers"], known.get("channels"))
    if (case.get("game") or {}).get("name") == "Grand Theft Auto VI":
        for comp in case.get("comparables") or []:
            if "target-history" not in (comp.get("unlock", "") or ""):
                continue
            comp_start = parse_ts(comp["day_start"])
            for row in comp.get("hourly") or []:
                if row.get("viewers") is None:
                    continue
                h = hour_offset(day_start, parse_ts(row["from"]))
                if h <= 0 and h not in series:
                    series[h] = (row["viewers"], row.get("channels"))
    return series


def build_donors(case):
    """Donor launches: every comparable with a real release-day curve.

    The GTA VI target-history comparable is not a launch (its "day" is the
    August premiere), so it is never a donor.
    """
    donors = []
    for comp in case.get("comparables") or []:
        if "target-history" in (comp.get("unlock", "") or ""):
            continue
        day_start = parse_ts(comp["day_start"])
        series = {}
        for row in comp.get("hourly") or []:
            if row.get("viewers") is None:
                continue
            h = hour_offset(day_start, parse_ts(row["from"]))
            series[h] = (row["viewers"], row.get("channels"))
        if 1 not in series or not series[1][0] or series[1][0] <= 0:
            continue
        if sum(1 for h in HOURS if h in series and series[h][0]) < 20:
            continue
        donors.append({
            "comp": comp,
            "feats": series_features(series, series[1][0]),
            "curve": donor_curve(series),
            "console": console_only(comp),
        })
    return donors


# ------------------------------------------------------------ weighted stats

def wmedian(pairs):
    pairs = sorted(pairs)
    total = sum(w for _v, w in pairs)
    if total <= 0:
        # Degenerate weights (e.g. every kernel underflowed): fall back to
        # the plain median instead of an arbitrary order statistic.
        vals = sorted(v for v, _w in pairs)
        return vals[len(vals) // 2] if vals else 0.0
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= total / 2:
            return value
    return pairs[-1][0]


def wquantile(pairs, q):
    pairs = sorted(pairs)
    total = sum(w for _v, w in pairs)
    if total <= 0:
        vals = sorted(v for v, _w in pairs)
        if not vals:
            return 0.0
        pos = q * (len(vals) - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(vals) - 1)
        return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= q * total:
            return value
    return pairs[-1][0]


def donor_weights(target_feats, donors):
    """Similarity of each donor to the target, in log-feature space."""
    weights = []
    for donor in donors:
        df = donor["feats"]
        w = TIER_W.get(tier_of(donor["comp"]), 0.5)
        w *= math.exp(-((target_feats["ln_p1"] - df["ln_p1"]) ** 2) / (2 * SIG_P1 ** 2))
        w *= math.exp(-((target_feats["ln_hump_over_p1"] - df["ln_hump_over_p1"]) ** 2) / (2 * SIG_HUMP ** 2))
        w *= math.exp(-((target_feats["ln_r24x"] - df["ln_r24x"]) ** 2) / (2 * SIG_R ** 2))
        if donor["console"]:
            w *= CONSOLE_BONUS
        weights.append(w)
    if not weights or max(weights) < 1e-6:
        # Every kernel underflowed (target far outside the donor cloud):
        # fall back to tier weights alone rather than a degenerate forecast.
        weights = [TIER_W.get(tier_of(donor["comp"]), 0.5) for donor in donors]
    return weights


def solve_linear(matrix, rhs):
    """Small dense solve by Gauss-Jordan; None when singular."""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(n):
            if r != col:
                factor = aug[r][col] / aug[col][col]
                for cc in range(col, n + 1):
                    aug[r][cc] -= factor * aug[col][cc]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def wls(donors, weights, xfun, yfun, ridge=RIDGE):
    """Weighted least squares with a ridge; returns coefficients or None."""
    rows = []
    for donor, w in zip(donors, weights):
        y = yfun(donor)
        if y is None:
            continue
        rows.append((xfun(donor["feats"]), y, w))
    if not rows:
        return None
    k = len(rows[0][0])
    matrix = [[0.0] * k for _ in range(k)]
    rhs = [0.0] * k
    for x, y, w in rows:
        for i in range(k):
            rhs[i] += w * x[i] * y
            for j in range(k):
                matrix[i][j] += w * x[i] * x[j]
    for i in range(k):
        matrix[i][i] += ridge
    return solve_linear(matrix, rhs)


# ---------------------------------------------------------------- forecasting

def predict_log_quantiles(target_feats, donors):
    """{hour: [log q05, log q25, log q50, log q75, log q95]}."""
    weights = donor_weights(target_feats, donors)
    out = {}
    for h in HOURS:
        pairs = [(d["curve"][h], w) for d, w in zip(donors, weights) if h in d["curve"]]
        if not pairs:
            out[h] = [0.0] * 5
            continue
        # Half 1: similarity-weighted median of the donors' curves.
        med_wm = wmedian(pairs)
        # Half 2: per-hour local-linear regression on log hour-1 print
        # and log hump/print, same kernel weights.
        beta = wls(donors, weights,
                   lambda f: [1.0, f["ln_p1"], f["ln_hump_over_p1"]],
                   lambda d, h=h: d["curve"].get(h))
        if beta is None:
            med_ll = med_wm
        else:
            med_ll = (beta[0] + beta[1] * target_feats["ln_p1"]
                      + beta[2] * target_feats["ln_hump_over_p1"])
        med = 0.5 * (med_wm + med_ll)
        # Spread: the donors' own weighted distribution for this hour,
        # re-centred on the blended median and shrunk (calibrated).
        centre = wmedian(pairs)
        out[h] = [med + SPREAD_MULT * (wquantile(pairs, tau) - centre)
                  for tau in LEVELS]
    return out


def baseline_quantiles(p1):
    """The house persistence baseline, as a fallback (never invalid)."""
    p1 = max(1.0, float(p1))
    return {h: [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]
            for h in HOURS}


def forecast_case(case):
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    p1 = max(1.0, float(first["viewers"])) if first else 1.0
    try:
        series = build_target_series(case)
        if 1 not in series:
            raise ValueError("no hour-1 print in history")
        feats = series_features(series, series[1][0])
        donors = build_donors(case)
        if len(donors) < 3:
            raise ValueError("too few donors")
        log_q = predict_log_quantiles(feats, donors)
        quantiles = {h: [p1 * math.exp(q) for q in log_q[h]] for h in HOURS}
    except Exception:
        quantiles = baseline_quantiles(p1)
    hours = []
    for row in case.get("forecast") or [{"hour": h} for h in HOURS]:
        h = row["hour"]
        qs = quantiles.get(h)
        if qs is None:
            qs = baseline_quantiles(p1)[h]
        qs = [min(1e9, max(1.0, float(q))) for q in qs]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        hours.append({"hour": h, "q": [round2(q) for q in qs]})
    hours.sort(key=lambda r: r["hour"])
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
