#!/usr/bin/env python3
"""Forecast launch-day Twitch viewers, hours 2-24, for make.fun
competition gta6-twitch-ccu-forecast-01.

Reads every *.in case file in $MAKEFUN_INPUT_DIR (else /input/) and writes
one /output/<id>.out per case to $MAKEFUN_OUTPUT_DIR (else /output/).

Method (fitted at runtime from the case's own comparables; nothing is
hardcoded from practice data).  v3 replaces v2's one-size-fits-all fit
with a regime-matched fit plus a pre-release signal:

  1. REGIME FILTER.  Past launches fall into three shapes: rolling
     midnight unlocks (regions unlock through the day, viewers climb
     into the US evening), simultaneous/global unlocks (tiny hour-1,
     then a 100-2000x explosion), and early-access-contaminated flat
     curves (peak at hour 1-2, then decay).  GTA VI is a rolling
     midnight launch, so the model trains ONLY on rolling-midnight
     comparables (falling back to rolling+mixed, then all, if too
     few).  Training on all regimes at once averages the shapes away:
     on the four practice launches the rolling-only fit alone beats
     the all-regimes fit by ~0.02 pinball loss.

  2. PRE-RELEASE SURGE FEATURE.  The hour-1 print alone does not fix
     a launch's shape amplitude: two launches with the same hour-1
     can peak 2x or 9x.  What separates them is how much audience was
     already watching just before UTC midnight, when New Zealand /
     Australia and the UK / Europe had already unlocked but the US
     had not: log(viewers at day_start - 1h) minus log(hour-1).  A
     high pre-midnight surge relative to hour-1 predicts a bigger
     peak and a shallower midday dip (correlation ~+0.6 with the
     peak ratio among rolling launches).  A second feature, log
     hour-1 channels, carries the streamer breadth.  Both enter the
     ridge as extra columns, centred on the training mean, with a
     much smaller penalty (0.1x) than the base features so they can
     carry real weight.  Missing values fall back to the training
     mean, so a sparse case degrades to the base fit, never to an
     error.

  3. ENSEMBLE.  Two per-hour models of log(viewers), combined as a
     geometric-mean ensemble (50/50):
       A: ridge regression of log viewers on
          [1, log(h1), (log(h1)-11)^2, rolling-only dummy,
           pre-midnight-surge ratio, log hour-1 channels],
          fitted only on comparables with hour-1 >= 10,000, with
          quantiles from the kernel-weighted empirical distribution
          of that hour's residuals, scaled by 0.7.
       B: local-linear kernel regression in log(h1) (Gaussian
          bandwidth 1.5) on the same rolling subset, with normal
          quantiles from the weighted residual spread, also scaled
          by 0.7.  This adapts to curvature locally and falls back
          to the nearest comparables outside the training range.
  The effective log(h1) is clamped to the training range +- 1.
  When the target's log(h1) lies outside the training range, the
  forecast spread widens in proportion to the distance (the median
  is untouched).  Any failure falls back to the house-style forecast
  (hour-1 print with a spread widening over the day), so output is
  always valid.

Standard library only, no network, no clock, no randomness: deterministic.
"""

import json
import math
import os
import re
from datetime import datetime, timedelta

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
HOURS = list(range(2, 25))

BW = 1.5            # kernel bandwidth in log(h1) for component B
RIDGE = 1.0         # ridge penalty for the base features of component A
EXTRA_RIDGE = 0.1   # ridge penalty multiplier for the two new features
W_A = 0.5           # ensemble weight of component A (log space)
W_B = 0.5           # ensemble weight of component B (log space)
A_MIN_H1 = 10000.0  # component A trains only on comparables this big
QSCALE = 0.7        # spread calibration for component A's quantiles
B_SIGMA_MULT = 0.7  # spread calibration for component B's sigma
X_CENTER = 11.0     # centre for the quadratic log(h1) term
X_PAD = 1.0         # allowed extrapolation beyond the training log(h1) range
SIGMA_FLOOR = 0.15  # minimum log-spread for component B
EXTRA_SPREAD = 0.25  # extra log-spread per log-unit the target sits outside
                     # the training log(h1) range (extrapolation honesty)
LOG_Q_MIN = -10.0   # exp() of this is ~4.5e-5, safely above 0
LOG_Q_MAX = 20.72326583694641  # ln(1e9)
MIN_TRAIN = 5       # minimum comparables needed for the regime fit


# ---------------------------------------------------------------- utilities

def parse_ts(text):
    """Parse an ISO-8601 timestamp; return a datetime or None."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def safe_log(value):
    return math.log(max(float(value), 1.0))


def solve_linear(matrix, rhs):
    """Solve a small dense linear system by Gauss-Jordan with pivoting."""
    n = len(matrix)
    aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            aug[piv][col] = 1e-12
        aug[col], aug[piv] = aug[piv], aug[col]
        for row in range(n):
            if row != col:
                factor = aug[row][col] / aug[col][col]
                if factor != 0.0:
                    for k in range(col, n + 1):
                        aug[row][k] -= factor * aug[col][k]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def ridge_fit_pen(feats, ys, pen):
    """Ridge regression with a per-feature penalty vector.

    pen[i] is the penalty on column i (pen[0] applies to the intercept,
    normally 0.0).  A smaller penalty lets a feature carry more weight.
    """
    p = len(feats[0])
    ata = [[0.0] * p for _ in range(p)]
    aty = [0.0] * p
    for row, y in zip(feats, ys):
        for i in range(p):
            aty[i] += row[i] * y
            for j in range(p):
                ata[i][j] += row[i] * row[j]
    for i in range(p):
        ata[i][i] += pen[i]
    return solve_linear(ata, aty)


def weighted_linear(rows, target, weights, penalty):
    """Weighted ridge regression (penalty on non-intercept columns)."""
    p = len(rows[0])
    ata = [[0.0] * p for _ in range(p)]
    atb = [0.0] * p
    for row, y, w in zip(rows, target, weights):
        for i in range(p):
            atb[i] += w * row[i] * y
            for j in range(p):
                ata[i][j] += w * row[i] * row[j]
    for i in range(1, p):
        ata[i][i] += penalty
    return solve_linear(ata, atb)


def smooth_seq(values):
    """One [1, 2, 1] / 4 pass; edges replicate their neighbour."""
    out = []
    n = len(values)
    for i in range(n):
        left = values[i - 1] if i > 0 else values[i]
        right = values[i + 1] if i < n - 1 else values[i]
        out.append((left + 2.0 * values[i] + right) / 4.0)
    return out


def weighted_offsets(values, weights):
    """Interpolated weighted quantiles of values at LEVELS.

    Sorts by value and walks the cumulative weight; interpolates
    between neighbouring values when the target cumulative weight
    falls between them.
    """
    pairs = sorted(zip(values, weights))
    total = sum(weights)
    offsets = []
    for tau in LEVELS:
        target = tau * total
        acc = 0.0
        prev = None
        found = pairs[-1][0]
        for value, w in pairs:
            if acc + w >= target:
                if prev is None:
                    found = value
                else:
                    frac = (target - acc) / w if w > 0.0 else 0.0
                    found = prev * (1.0 - frac) + value * frac
                break
            acc += w
            prev = value
        offsets.append(found)
    return offsets


# ------------------------------------------------------- unlock classification

_NEG_WORDS = ("no", "not", "without", "never", "n't")


def _mentioned(text, pattern):
    """True if `pattern` occurs in text without a negation just before it."""
    for match in re.finditer(re.escape(pattern), text):
        before = text[max(0, match.start() - 16):match.start()].rstrip()
        if any(before.endswith(w) for w in _NEG_WORDS):
            continue
        return True
    return False


def classify_unlock(text):
    """Classify an unlock description.

    Returns 'rollonly' (rolling midnight local, no simultaneous/global
    unlock), 'global' (a simultaneous or single global unlock, not
    rolling), 'mixed' (both), 'other' (neither mentioned), or None when
    there is no usable text at all.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    low = text.lower()
    rolling = ("rolling" in low) or ("midnight local" in low)
    glob = (_mentioned(low, "simultaneous")
            or _mentioned(low, "global unlock")
            or _mentioned(low, "single global")
            or _mentioned(low, "global launch"))
    if rolling and not glob:
        return "rollonly"
    if glob and not rolling:
        return "global"
    if rolling and glob:
        return "mixed"
    return "other"


# ------------------------------------------------------------ data extraction

def extract_curves(case):
    """Turn comparables into training rows with complete release days.

    Each row: {'x': log(hour-1 viewers), 'v1': hour-1 viewers,
    'roll': 0/1 rolling-only dummy, 'y': [log viewers hours 1..24],
    'cls': unlock class, 'pre23_ratio': log(pre-midnight viewers) -
    log(hour-1) or None, 'ch1': log hour-1 channels or None}.
    A comparable whose own release day is not fully present (e.g. the
    target's own pre-release history) is skipped.
    """
    rows = []
    for comp in case.get("comparables") or []:
        if not isinstance(comp, dict):
            continue
        day_start = parse_ts(comp.get("day_start"))
        hourly = comp.get("hourly")
        if day_start is None or not isinstance(hourly, list):
            continue
        by_hour = {}
        pre23 = None
        ch1 = None
        for point in hourly:
            if not isinstance(point, dict):
                continue
            when = parse_ts(point.get("from"))
            if when is None:
                continue
            viewers = point.get("viewers")
            if not isinstance(viewers, (int, float)) or viewers <= 0:
                continue
            delta = (when - day_start).total_seconds()
            if when == day_start - timedelta(hours=1):
                pre23 = float(viewers)
            if 0 <= delta < 24 * 3600:
                hour = int(delta // 3600) + 1
                by_hour[hour] = float(viewers)
                if hour == 1:
                    channels = point.get("channels")
                    if isinstance(channels, (int, float)) and channels > 0:
                        ch1 = float(channels)
        if len(by_hour) != 24 or 1 not in by_hour:
            continue
        cls = classify_unlock(comp.get("unlock"))
        x = safe_log(by_hour[1])
        rows.append({
            "x": x,
            "v1": by_hour[1],
            "roll": 1.0 if cls == "rollonly" else 0.0,
            "y": [safe_log(by_hour[h]) for h in range(1, 25)],
            "cls": cls,
            "pre23_ratio": (safe_log(pre23) - x) if pre23 is not None else None,
            "ch1": safe_log(ch1) if ch1 is not None else None,
        })
    return rows


def select_train(rows):
    """Regime-matched training set: rolling-midnight-only launches first.

    GTA VI is a rolling-midnight launch; the other regimes (global
    simultaneous explosions, early-access flat curves) distort the
    fit when mixed in.  Fall back to rolling+mixed, then to all
    comparables, if the strict set is too small.
    """
    rollonly = [r for r in rows if r.get("cls") == "rollonly"]
    if len(rollonly) >= MIN_TRAIN:
        return rollonly
    roll_mixed = [r for r in rows if r.get("cls") in ("rollonly", "mixed")]
    if len(roll_mixed) >= MIN_TRAIN:
        return roll_mixed
    return rows


def target_roll(case, rows):
    """Rolling-only dummy for the target game: 1.0 / 0.0 / training mean."""
    mean_roll = sum(r["roll"] for r in rows) / len(rows) if rows else 0.0
    game = case.get("game") or {}
    name = game.get("name") if isinstance(game, dict) else None
    text = None
    if isinstance(name, str) and name:
        lowered = name.lower()
        for comp in case.get("comparables") or []:
            if not isinstance(comp, dict):
                continue
            comp_name = comp.get("name")
            if isinstance(comp_name, str) and lowered in comp_name.lower():
                text = comp.get("unlock")
                break
    if text is None:
        text = case.get("notes")
    cls = classify_unlock(text)
    if cls == "rollonly":
        return 1.0
    if cls in ("global", "mixed", "other"):
        return 0.0
    return mean_roll


def hour_one_print(case):
    """Hour-1 viewers from `known`, else the last history point, else 1."""
    known = case.get("known")
    rows = known if isinstance(known, list) else [known]
    for row in rows:
        if isinstance(row, dict):
            viewers = row.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                return float(viewers)
    history = case.get("history_hourly") or case.get("history") or []
    if isinstance(history, list):
        for row in reversed(history):
            if isinstance(row, dict):
                viewers = row.get("viewers")
                if isinstance(viewers, (int, float)) and viewers > 0:
                    return float(viewers)
    return 1.0


def target_extra_features(case, h1):
    """The two new features for the target game.

    Returns (pre23_ratio, ch1): log(viewers at day_start - 1h) minus
    log(hour-1), and log hour-1 channels.  Either may be None when the
    case does not carry the datum; the caller then falls back to the
    training mean for that feature.
    """
    day_start = parse_ts(case.get("day_start"))
    pre23 = None
    if day_start is not None:
        history = case.get("history_hourly") or case.get("history") or []
        if isinstance(history, list):
            for point in history:
                if not isinstance(point, dict):
                    continue
                when = parse_ts(point.get("from"))
                if when is None:
                    continue
                if when == day_start - timedelta(hours=1):
                    viewers = point.get("viewers")
                    if isinstance(viewers, (int, float)) and viewers > 0:
                        pre23 = float(viewers)
                    break
    pre23_ratio = (safe_log(pre23) - safe_log(h1)) if pre23 is not None else None
    ch1 = None
    known = case.get("known")
    rows = known if isinstance(known, list) else [known]
    for row in rows:
        if isinstance(row, dict):
            channels = row.get("channels")
            if isinstance(channels, (int, float)) and channels > 0:
                ch1 = safe_log(float(channels))
                break
    return pre23_ratio, ch1


# ------------------------------------------------------------------ components

def component_a(rows, tx, troll, t_pre23, t_ch1):
    """Per-hour ridge on [1, x, (x-11)^2, roll, pre23_ratio, log ch1].

    The two new features are centred on the training mean (missing
    values take the mean) and carry a much smaller ridge penalty than
    the base features, so the pre-release surge can move the forecast.
    Fitted only on comparables with hour-1 >= A_MIN_H1 (the tiny
    hour-1 exploders distort the fit and the spread for big launches);
    falls back to all rows if that leaves too few.  Residual quantiles
    are weighted by Gaussian kernel distance in log(h1) to the target
    and scaled by QSCALE.  Medians and offsets are smoothed across
    neighbouring hours.

    Returns {hour: [5 log-quantiles]}.
    """
    big = [r for r in rows if r.get("v1", 0.0) >= A_MIN_H1]
    use = big if len(big) >= MIN_TRAIN else rows
    weights = [math.exp(-0.5 * ((r["x"] - tx) / BW) ** 2) for r in use]
    means = []
    for key in ("pre23_ratio", "ch1"):
        vals = [r.get(key) for r in use if r.get(key) is not None]
        means.append(sum(vals) / len(vals) if vals else 0.0)
    feats = []
    for r in use:
        f = [1.0, r["x"], (r["x"] - X_CENTER) ** 2, r["roll"]]
        for key, mean in zip(("pre23_ratio", "ch1"), means):
            v = r.get(key)
            f.append((v - mean) if v is not None else 0.0)
        feats.append(f)
    target_feat = [1.0, tx, (tx - X_CENTER) ** 2, troll]
    for tval, mean in zip((t_pre23, t_ch1), means):
        target_feat.append((tval - mean) if tval is not None else 0.0)
    pen = [0.0, RIDGE, RIDGE, RIDGE,
           RIDGE * EXTRA_RIDGE, RIDGE * EXTRA_RIDGE]
    medians = []
    offsets = [[] for _ in LEVELS]
    for hour in HOURS:
        ys = [r["y"][hour - 1] for r in use]
        beta = ridge_fit_pen(feats, ys, pen)
        mu = sum(b * f for b, f in zip(beta, target_feat))
        residuals = [y - sum(b * f for b, f in zip(beta, feat))
                     for y, feat in zip(ys, feats)]
        medians.append(mu)
        for k, off in enumerate(weighted_offsets(residuals, weights)):
            offsets[k].append(off * QSCALE)
    medians = smooth_seq(medians)
    offsets = [smooth_seq(seq) for seq in offsets]
    return {h: [medians[i] + offsets[k][i] for k in range(len(LEVELS))]
            for i, h in enumerate(HOURS)}


def component_b(rows, tx):
    """Per-hour local-linear kernel regression in log(h1), normal quantiles.

    Returns {hour: [5 log-quantiles]}.
    """
    weights = [math.exp(-0.5 * ((r["x"] - tx) / BW) ** 2) for r in rows]
    out = {}
    for hour in HOURS:
        ys = [r["y"][hour - 1] for r in rows]
        design = [[1.0, r["x"] - tx] for r in rows]
        beta = weighted_linear(design, ys, weights, 1e-6)
        mu = beta[0]
        residuals = [y - (beta[0] + beta[1] * (r["x"] - tx))
                     for y, r in zip(ys, rows)]
        total_w = sum(weights)
        mean_res = sum(w * e for w, e in zip(weights, residuals)) / total_w
        var = sum(w * (e - mean_res) ** 2
                  for w, e in zip(weights, residuals)) / total_w
        sigma = max(math.sqrt(max(var, 0.0)) * B_SIGMA_MULT, SIGMA_FLOOR)
        out[hour] = [mu + z * sigma for z in Z]
    return out


def house_forecast(print_viewers):
    """House-style fallback: hour-1 print, spread widening over the day."""
    p = max(float(print_viewers), 1.0)
    log_p = math.log(p)
    return {h: [log_p + z * 0.15 * math.sqrt(h - 1) for z in Z]
            for h in HOURS}


def forecast_case(case):
    """Return the output dict for one case; never raises."""
    try:
        p = hour_one_print(case)
        rows = extract_curves(case)
        if len(rows) < MIN_TRAIN:
            log_q = house_forecast(p)
        else:
            train = select_train(rows)
            xs = [r["x"] for r in train]
            lo, hi = min(xs) - X_PAD, max(xs) + X_PAD
            raw_x = safe_log(p)
            tx = min(max(raw_x, lo), hi)
            troll = target_roll(case, train)
            t_pre23, t_ch1 = target_extra_features(case, p)
            qa = component_a(train, tx, troll, t_pre23, t_ch1)
            qb = component_b(train, tx)
            # Widen the spread when the target sits outside the training
            # range: above the largest comparable hour-1 (a GTA-scale
            # launch) the shape is an extrapolation judgement, not
            # evidence, so the quantiles honestly widen with the
            # distance.  The median (Z = 0) is untouched.
            extrap = (max(0.0, raw_x - max(xs))
                      + max(0.0, min(xs) - raw_x))
            log_q = {}
            for h in HOURS:
                log_q[h] = [W_A * qa[h][k] + W_B * qb[h][k]
                            + Z[k] * EXTRA_SPREAD * extrap
                            for k in range(len(LEVELS))]
    except Exception:
        try:
            p = hour_one_print(case)
        except Exception:
            p = 1.0
        log_q = house_forecast(p)
    hours = []
    for h in HOURS:
        quantiles = []
        running_max = 0.0
        for value in log_q[h]:
            if not math.isfinite(value):
                value = 0.0
            value = min(max(value, LOG_Q_MIN), LOG_Q_MAX)
            value = max(value, running_max)
            running_max = value
            quantiles.append(math.exp(value))
        hours.append({"hour": h, "q": quantiles})
    return {"hours": hours}


# ------------------------------------------------------------------------ main

def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(out_dir, exist_ok=True)
    try:
        names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    except OSError:
        names = []
    for name in names:
        path = os.path.join(in_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                case = json.load(handle)
        except Exception:
            continue
        if not isinstance(case, dict):
            continue
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            case_id = name[:-3]
        out_path = os.path.join(out_dir, case_id + ".out")
        result = forecast_case(case)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, separators=(",", ":"))


if __name__ == "__main__":
    main()
