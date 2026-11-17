#!/usr/bin/env python3
"""Forecast launch-day Twitch viewers, hours 2-24, for make.fun
competition gta6-twitch-ccu-forecast-01.

Reads every *.in case file in $MAKEFUN_INPUT_DIR (else /input/) and writes
one /output/<id>.out per case to $MAKEFUN_OUTPUT_DIR (else /output/).

Method (fitted at runtime from the case's own comparables; nothing is
hardcoded from practice data):
  The release-day curve of a past launch is driven mostly by how much of
  the world was already unlocked in hour 1, which the hour-1 print encodes:
  huge hour-1 counts stay flat, tiny hour-1 counts explode as regions
  unlock.  Two per-hour models of log(viewers) are fitted on the
  comparables and combined as a geometric-mean ensemble:
    A (40%): ridge regression (lambda=1) of log viewers on
       [1, log(h1), (log(h1)-11)^2, rolling-only dummy], with quantiles
       from the empirical distribution of that hour's residuals,
       smoothed across neighbouring hours.  The quadratic term keeps
       extrapolation sane for hour-1 counts above the training range
       (a plain linear log-regression collapses there); the dummy adds
       ~+0.3 log units at later hours for rolling-midnight-only launches
       (regions keep unlocking through the day).
    B (60%): local-linear kernel regression in log(h1) (Gaussian
       bandwidth 1.5), with normal quantiles from the weighted residual
       spread.  This adapts to the curvature locally and falls back to
       the nearest comparables outside the training range.
  The target's rolling-only flag is read from the target-named
  comparable's unlock text, else from the case notes, else the training
  mean.  The effective log(h1) is clamped to the training range +- 1.
  Any failure falls back to the house-style forecast (hour-1 print with
  a spread widening over the day), so output is always valid.

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
RIDGE = 1.0         # ridge penalty for component A (not on the intercept)
W_A = 0.4           # ensemble weight of component A (log space)
W_B = 0.6           # ensemble weight of component B (log space)
X_CENTER = 11.0     # centre for the quadratic log(h1) term
X_PAD = 1.0         # allowed extrapolation beyond the training log(h1) range
SIGMA_FLOOR = 0.15  # minimum log-spread for component B
LOG_Q_MIN = -10.0   # exp() of this is ~4.5e-5, safely above 0
LOG_Q_MAX = 20.72326583694641  # ln(1e9)


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


def ridge_fit(rows, target, penalty):
    """Ridge regression with an unpenalised intercept column (rows[0])."""
    p = len(rows[0])
    ata = [[0.0] * p for _ in range(p)]
    atb = [0.0] * p
    for row, y in zip(rows, target):
        for i in range(p):
            atb[i] += row[i] * y
            for j in range(p):
                ata[i][j] += row[i] * row[j]
    for i in range(1, p):
        ata[i][i] += penalty
    return solve_linear(ata, atb)


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


def empirical_offsets(residuals):
    """Interpolated empirical quantiles of residuals at LEVELS."""
    ordered = sorted(residuals)
    n = len(ordered)
    offsets = []
    for tau in LEVELS:
        pos = tau * (n - 1)
        lo = int(math.floor(pos))
        frac = pos - lo
        if lo + 1 >= n:
            offsets.append(ordered[lo])
        else:
            offsets.append(ordered[lo] * (1.0 - frac) + ordered[lo + 1] * frac)
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

    Each row: {'x': log(hour-1 viewers), 'roll': 0/1 rolling-only dummy,
    'y': [log viewers for hours 1..24]}.  A comparable whose own release
    day is not fully present (e.g. the target's own pre-release history)
    is skipped.
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
        for point in hourly:
            if not isinstance(point, dict):
                continue
            when = parse_ts(point.get("from"))
            if when is None:
                continue
            delta = (when - day_start).total_seconds()
            if 0 <= delta < 24 * 3600:
                hour = int(delta // 3600) + 1
                viewers = point.get("viewers")
                if isinstance(viewers, (int, float)) and viewers > 0:
                    by_hour[hour] = float(viewers)
        if len(by_hour) != 24 or 1 not in by_hour:
            continue
        cls = classify_unlock(comp.get("unlock"))
        rows.append({
            "x": safe_log(by_hour[1]),
            "roll": 1.0 if cls == "rollonly" else 0.0,
            "y": [safe_log(by_hour[h]) for h in range(1, 25)],
        })
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


# ------------------------------------------------------------------ components

def component_a(rows, tx, troll):
    """Per-hour ridge on [1, x, (x-11)^2, roll]; empirical residual quantiles.

    Returns {hour: [5 log-quantiles]} with medians and offsets smoothed
    across neighbouring hours.
    """
    feats = [[1.0, r["x"], (r["x"] - X_CENTER) ** 2, r["roll"]] for r in rows]
    target_feat = [1.0, tx, (tx - X_CENTER) ** 2, troll]
    medians = []
    offsets = [[] for _ in LEVELS]
    for hour in HOURS:
        ys = [r["y"][hour - 1] for r in rows]
        beta = ridge_fit(feats, ys, RIDGE)
        mu = sum(b * f for b, f in zip(beta, target_feat))
        residuals = [y - sum(b * f for b, f in zip(beta, feat))
                     for y, feat in zip(ys, feats)]
        medians.append(mu)
        for k, off in enumerate(empirical_offsets(residuals)):
            offsets[k].append(off)
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
        sigma = max(math.sqrt(max(var, 0.0)), SIGMA_FLOOR)
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
        if len(rows) < 5:
            log_q = house_forecast(p)
        else:
            xs = [r["x"] for r in rows]
            lo, hi = min(xs) - X_PAD, max(xs) + X_PAD
            tx = min(max(safe_log(p), lo), hi)
            troll = target_roll(case, rows)
            qa = component_a(rows, tx, troll)
            qb = component_b(rows, tx)
            log_q = {}
            for h in HOURS:
                log_q[h] = [W_A * qa[h][k] + W_B * qb[h][k]
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
