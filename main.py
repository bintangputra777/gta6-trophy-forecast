"""gta6-trophy-rarity-forecast-01 entry: earners/players decomposition forecaster (Python 3.13, stdlib only).

Model
-----
Key identity: rate_i(t) = earners_i(t) / players_tracked(t). A case gives rates AND players_tracked at
every read, so earners are reconstructed exactly: E_i(t) = rate_i(t)/100 * players_tracked(t). The two
series are forecast separately and divided at the target time.

Players P(t) and, per trophy, earners E_i(t) each use the same three candidate families:

  (a) persistence          -- last observed value;
  (b) saturating exp       -- Cap - exp(a + b*t), Cap = m * last value for m in
                              {1.02, 1.05, 1.1, 1.25, 1.5, 2, 3, 5}; a, b by linear least squares on
                              ln(Cap - y) vs t (grid points with any Cap - y <= 0 are skipped); the grid
                              point kept is the one with the smallest training SSE on the evaluation
                              scale below;
  (c) log-time linear      -- evaluation-scale value linear in ln(1 + t).

Evaluation scale: ln P for players, ln(max(E, 0.25)) for earners. Family selection is by backtest: fit on
the first 70% of points, squared error on the evaluation scale over the last 30%. Ties go to the earlier
family in (a), (b), (c) order, so a flat history selects persistence. Players use persistence outright
when there are fewer than 6 reads.

Guardrail 1 (median-of-3 prediction): the backtest winner is NOT used for the target point prediction.
For players and for each trophy's earners, the point prediction is the median of the three families'
all-data refit predictions (the median of the fittable ones when a family cannot be fitted). The backtest
still runs unchanged: the winning family's train fit is what the residual spread s_i is computed from.
This stops a single family's refit -- e.g. a saturating exp that legitimately wins a short backtest and
then runs to its asymptote cap over a long horizon -- from dominating the forecast on its own.

Earners edge cases: reads where the trophy's rate is null (or players_tracked is not a positive number)
are skipped for that trophy. A trophy with no usable reads, or with earners 0 in every read, gets median
rate 0.05 with the default spread. A trophy with fewer than 3 usable reads uses its last rate as the
median rate directly. Otherwise the refit prediction is floored at 0.98 * E_last (holders do not
evaporate) and capped at 0.999 * P_hat(target).

Median rate m_i = 100 * E_hat_i / P_hat, clamped to 0.01..99.99.

Guardrail 2 (logit growth guard): on the main path above (not the no-reads / all-zero / fewer-than-3
fallbacks, which keep their behaviour), logit(m_i) is clamped into
[logit(r_last) - 2.0, logit(r_last) + 1.5], where r_last is the trophy's rate at its last usable read,
clamped to 0.01..99.99 first -- the median rate cannot sprint away from the last observed rate on the
logit scale, however the earners/players forecasts divide out. Spread on the logit scale:
sigma_i = sqrt((0.45 * sqrt(horizon_days))^2 + s_i^2), floored at 0.35 and capped at 1.6, where
horizon_days is (target - last read) in days and s_i is the std of the trophy's backtest residuals on the
logit-of-rate scale (rate implied by the winning family's train fit, using the actual players_tracked of
each test read, vs the actual rate; |residual| when there is a single test point; 0.3 when unavailable).
Quantiles: q_tau = 100 / (1 + exp(-(logit(m_i) + z_tau * sigma_i))), z as in the starter, rounded to 4
decimals (half up, the starter's round4), then made non-decreasing.

Times are absolute epoch hours parsed from the ISO timestamps; t is hours since the case's first read.
Deterministic: no randomness, no wall clock, no network. Runs in O(reads) per fit/grid point.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first (kept from the starter; this entry is one file).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Saturation-cap grid, as multiples of the series' last value.
MULTS = [1.02, 1.05, 1.1, 1.25, 1.5, 2, 3, 5]


def clamp_rate(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct)
    return math.log(p / (100 - p))


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_hours(ts: str) -> float:
    """ISO timestamp -> epoch hours (float)."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() / 3600.0


def exp_safe(x: float) -> float:
    return math.exp(min(x, 700.0))


def linregress(xs: list, ys: list) -> tuple:
    """Least squares y ~ a + b*x; b = 0 when x has no spread."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def tr_players(v: float) -> float:
    return math.log(max(v, 1e-9))


def tr_earners(v: float) -> float:
    return math.log(max(v, 0.25))


def fit_persistence(ts: list, ys: list, transform) -> object:
    last = ys[-1]
    return lambda t: last


def fit_satexp(ts: list, ys: list, transform) -> object:
    """Cap - exp(a + b*t); grid over Cap = mult * last value, LS on ln(Cap - y) vs t,
    grid point chosen by training SSE on the evaluation (transform) scale."""
    last = ys[-1]
    best = None
    for mult in MULTS:
        cap = mult * last
        if any(cap - y <= 0 for y in ys):
            continue
        a, b = linregress(ts, [math.log(cap - y) for y in ys])
        sse = 0.0
        for t, y in zip(ts, ys):
            d = transform(cap - exp_safe(a + b * t)) - transform(y)
            sse += d * d
        if best is None or sse < best[0]:
            best = (sse, a, b, cap)
    if best is None:
        return None
    _, a, b, cap = best
    return lambda t: cap - exp_safe(a + b * t)


def fit_logtime(ts: list, ys: list, transform) -> object:
    """transform(y) linear in ln(1 + t)."""
    a, b = linregress([math.log1p(t) for t in ts], [transform(y) for y in ys])
    return lambda t: exp_safe(a + b * math.log1p(t))


FAMILIES = (
    ("persistence", fit_persistence),
    ("satexp", fit_satexp),
    ("logtime", fit_logtime),
)
FITTERS = dict(FAMILIES)


def median_of(values: list) -> float:
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def backtest_select(ts: list, ys: list, transform) -> tuple:
    """Fit each family on the first 70% of points, score on the last 30% (SSE on the
    transform scale). The winner's train fit is returned for the residual spread.

    Guardrail 1: the final predictor is not the winner's refit -- it returns, at any t,
    the median of every fittable family's all-data refit prediction. Ties in the
    backtest keep the earlier family."""
    n = len(ts)
    n_train = max(2, min(n - 1, int(0.7 * n)))
    tr_t, tr_y = ts[:n_train], ys[:n_train]
    te_t, te_y = ts[n_train:], ys[n_train:]
    best = None
    for name, fitter in FAMILIES:
        pred = fitter(tr_t, tr_y, transform)
        if pred is None:
            continue
        sse = sum((transform(pred(t)) - transform(y)) ** 2 for t, y in zip(te_t, te_y))
        if best is None or sse < best[0]:
            best = (sse, name, pred)
    _, name, train_pred = best
    finals = []
    for _name, fitter in FAMILIES:
        pred = fitter(ts, ys, transform)
        if pred is not None:
            finals.append(pred)
    if finals:
        final_pred = lambda t: median_of([p(t) for p in finals])
    else:
        final_pred = train_pred
    return train_pred, final_pred, n_train


def forecast_players(ts: list, ps: list, t_target: float) -> float:
    if len(ts) < 6:
        return ps[-1]
    _, pred, _ = backtest_select(ts, ps, tr_players)
    return max(pred(t_target), 1.0)


def sigma_from(s_i: float, horizon_days: float) -> float:
    sig = math.sqrt((0.45 * math.sqrt(max(horizon_days, 0.0))) ** 2 + s_i * s_i)
    return min(1.6, max(0.35, sig))


def forecast_trophy(pts: list, p_hat: float, t_target: float, horizon_days: float) -> tuple:
    """pts: the trophy's usable reads as (t_hours, earners, players, rate), oldest first.
    Returns (median_rate_percent, sigma)."""
    default_sigma = sigma_from(0.3, horizon_days)
    if not pts:
        return 0.05, default_sigma
    if all(e == 0 for (_, e, _, _) in pts):
        return 0.05, default_sigma
    if len(pts) < 3:
        return clamp_rate(pts[-1][3]), default_sigma
    ts = [p[0] for p in pts]
    es = [p[1] for p in pts]
    train_pred, final_pred, n_train = backtest_select(ts, es, tr_earners)
    # s_i: std of backtest residuals on the logit-of-rate scale. The implied rate uses the
    # winning family's train fit for earners and the actual players_tracked of each test read.
    residuals = []
    for (t, _e, players, rate) in pts[n_train:]:
        rate_hat = 100.0 * train_pred(t) / players if players > 0 else 0.0
        residuals.append(logit(rate_hat) - logit(rate))
    if len(residuals) >= 2:
        mean = sum(residuals) / len(residuals)
        s_i = math.sqrt(sum((r - mean) ** 2 for r in residuals) / len(residuals))
    elif len(residuals) == 1:
        s_i = abs(residuals[0])
    else:
        s_i = 0.3
    e_hat = final_pred(t_target)
    e_hat = max(e_hat, 0.98 * es[-1])
    e_hat = min(e_hat, 0.999 * p_hat)
    median = clamp_rate(100.0 * e_hat / p_hat) if p_hat > 0 else 0.05
    # Guardrail 2 (logit growth guard): keep the median within a logit window of the
    # last observed rate -- at most +1.5 above it, at most -2.0 below it.
    r_last = clamp_rate(pts[-1][3])
    lg = logit(median)
    lg = min(max(lg, logit(r_last) - 2.0), logit(r_last) + 1.5)
    median = clamp_rate(100.0 / (1.0 + math.exp(-lg)))
    return median, sigma_from(s_i, horizon_days)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]

    read_hours = [parse_hours(r.get("at") or r["slot"]) for r in history]
    t0 = read_hours[0]
    ts_all = [h - t0 for h in read_hours]
    target_hours = parse_hours(case["target"])
    t_target = target_hours - t0
    horizon_days = max(0.0, (target_hours - read_hours[-1]) / 24.0)

    # Players series: reads with a positive players_tracked.
    p_ts, p_ps = [], []
    for t, r in zip(ts_all, history):
        p = r.get("players_tracked")
        if isinstance(p, (int, float)) and p > 0:
            p_ts.append(t)
            p_ps.append(float(p))
    p_hat = forecast_players(p_ts, p_ps, t_target) if p_ps else 1.0

    rows = []
    for i, trophy in enumerate(trophies):
        pts = []
        for t, r in zip(ts_all, history):
            rates = r.get("rates") or []
            rate = rates[i] if i < len(rates) else None
            p = r.get("players_tracked")
            if rate is None or not (isinstance(p, (int, float)) and p > 0):
                continue
            pts.append((t, rate / 100.0 * float(p), float(p), float(rate)))
        median, sigma = forecast_trophy(pts, p_hat, t_target, horizon_days)
        lg = logit(median)
        qs = [round4(100 / (1 + math.exp(-(lg + z * sigma)))) for z in Z]
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
