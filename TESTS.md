# Test log — gta6-trophy-rarity-forecast-01 entry (main.py)

Run: `MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py` (local Python 3.12.3;
the grader uses 3.13 — the code uses no version-specific features beyond the starter's own).

## Practice cases (kit input: p00.in, p01.in — flat launch-night page, 7 and 8 identical reads)

- Both outputs VALID: exactly `{"trophies":[{"id":…,"q":[5 numbers]}…]}`, 55 rows, ids match the case
  in order, every q finite, within 0..100, non-decreasing; 2,683 bytes each (limit 65,536).
- Deterministic: two full runs byte-identical. sha256 of both p00.out and p01.out (both runs):
  `e28824af562c24642c87177671dd72d6be6c600f7a57d89f1e864ab3ad6809c9`
  (p00 and p01 outputs are identical because the recorded page is identical; only the checkpoint differs.)
- Timing: 0.12 s and 0.18 s for the two-case directory (limit is 8 s per case).

## Flat-history sanity (medians vs last rates, p00)

| trophy | last rate | forecast median | q |
|---|---|---|---|
| 1 | 0.63 | 0.63 | [0.3552, 0.4983, 0.63, 0.7963, 1.115] |
| 2 | 100 | 99.9 | [99.8223, 99.8734, 99.9, 99.921, 99.9437] |
| 3 | 1.25 | 1.25 | [0.7067, 0.9899, 1.25, 1.5773, 2.2017] |
| 20 | 2.04 | 2.04 | [1.1574, 1.6183, 2.04, 2.5688, 3.5714] |
| 38 | 1.57 | 1.57 | [0.8889, 1.2442, 1.57, 1.9794, 2.7585] |
| 55 | 0.78 | 0.78 | [0.4401, 0.6171, 0.78, 0.9855, 1.3788] |

Max |median − last rate| over all 55 trophies = 0.0000 (the 100% trophy lands on 99.9 via the
0.999 × P̂ earners cap). On a flat history the model stays at persistence, exactly as designed:
all families tie at ~0 backtest error and the tie-break picks persistence; s_i = 0, so sigma sits
at its 0.35 floor for the 90-minute practice horizon.

## Synthetic stress case (not in the kit; built for this test)

350 reads over 60 h, players saturating 638 → 50,792, target +4.52 days (final-case shape);
includes a never-earned trophy, an always-null trophy, a 100% trophy, a flat-earners trophy whose
rate falls as players grow, sparse-null trophies, and 50 logistic-growth trophies.

- Runtime 0.59 s for the whole case; output valid (55 rows, 2,978 bytes) and byte-identical across runs.
- Never-earned → median 0.05 ✓ · always-null → median 0.05 ✓ · 100% → 99.9 ✓
- Flat earners (16,000) with growing players: last rate 31.5 → median 30.0 (true value at target
  ≈ 30.8 if earners stay flat) ✓ direction and size right.
- ⚠ Two still-rising trophies (last rates 48.06 and 57.95, true ceilings 60% and 73%) got median
  99.9: the saturating-exp family legitimately wins their backtests (SSE 1.30 vs persistence 1.54;
  0.64 vs 2.19), but its refit (cap = 5 × E_last under both grid-selection criteria tested) reaches
  Ê = 60,976 / 73,156 at the +108.5 h target — above the 0.999 × P̂ clamp (53,270), so the median
  pins at 99.9. Log-time loses those backtests by a wide margin (SSE 36.5 / 50.4). This is the
  spec'd design's known weak spot — a ≤18 h backtest window certifying a curve that then runs to
  its cap over a 108.5 h horizon — not an implementation bug; the spec's own cap is what catches it.

## Guardrails added (post-stress fix) + full re-test

Two changes to main.py, nothing else (same I/O contract, stdlib only, deterministic):

1. **Median-of-3 prediction** — for players and for each trophy's earners, the point prediction is
   now the median of the three families' all-data refit predictions (median of the fittable ones
   when a family cannot be fitted). The backtest is unchanged and still supplies the winning train
   fit used for the residual spread s_i.
2. **Logit growth guard** — after m_i = 100·Ê_i/P̂, logit(m_i) is clamped into
   [logit(r_last) − 2.0, logit(r_last) + 1.5], r_last = the trophy's rate at its last usable read
   (clamped to 0.01..99.99 first). The no-reads / all-zero / <3-reads fallbacks are not guarded.

### Practice cases (same kit input)

- Both outputs VALID: exact shape, 55 rows, ids in order, q finite, 0..100, non-decreasing;
  2,687 bytes each (limit 65,536).
- Deterministic: two full runs byte-identical. sha256 of both p00.out and p01.out (both runs):
  `0f1684bc8867514c900410518d3fb2867083aa614894ee4b1784126d5d71a6eb`
  (changed from the pre-guardrail `e28824af…` — see the 100% trophy note below).
- Timing: 0.21 s and 0.13 s for the two-case directory.

### Flat-history sanity (medians vs last rates, p00)

- Max |median − last rate| = **0.0000 over 54 of 55 trophies** — median-of-3 on a flat history is
  still exactly the last value for every family, so persistence behaviour is preserved.
- The one exception is the 100% trophy: median **99.9262** (was 99.9 pre-guardrail). Its pre-guard
  median 99.9 (from the 0.999 × P̂ earners cap) has logit 6.907, below the guard's lower bound
  logit(99.99) − 2.0 = 7.210, so the guard pulls it up to 99.9262 — |median − last| = 0.0738,
  closer to the last rate than the pre-guardrail 0.1 gap.

### Synthetic stress case (rebuilt — the original generator was not saved)

Rebuilt to the description above: players P(t) = 52,000 − 51,362·e^(−t/16) (638 → 50,792 over
60 h, ≈51,999 at target); growers are logistic in rate — A: ceiling 60, midpoint 32.156 h, slope
20 h (last rate 48.06, true at target 59.81); B: ceiling 73, midpoint 26.3 h, slope 25 h (last rate
57.95, true at target 72.75); flat earners 16,000 (last rate 31.5, true at target 30.77); plus
never-earned, always-null, 100%, a sparse-constant trophy, two sparse-null logistics and 48 more
logistics (seeded). Checkpoints match the original log: players 638 → 50,792, last rates
48.06 / 57.95 / 31.5, and the ORIGINAL code on this rebuild reproduces the flaw exactly —
growers pinned at **99.9 / 99.9**, flat earners **30.0051** (original log: 99.9 / 99.9, 30.0).

With the guardrails (runtime 0.60 s; output valid, 55 rows, 3,025 bytes, byte-identical runs):

| trophy | last rate | true at target | median before | median after |
|---|---|---|---|---|
| grower A (ceiling 60) | 48.06 | 59.81 | 99.9 | **80.5708** |
| grower B (ceiling 73) | 57.95 | 72.75 | 99.9 | **86.0653** |
| flat earners | 31.5 | 30.77 | 30.0051 | **30.005** |

- Fallbacks unchanged: never-earned → 0.05 ✓ · always-null → 0.05 ✓ · sparse-constant → 12.4937 ·
  100% → 99.9262 (same guard lower-bound effect as p00).
- ⚠ The growers improve from 99.9 but land **on the guard's upper cap**, not near their ceilings:
  80.5708 = logit⁻¹(logit(48.06) + 1.5) and 86.0653 = logit⁻¹(logit(57.95) + 1.5), exactly.
  Mechanism, from the family predictions at target on this rebuild — players: persistence 50,792 /
  satexp 53,324 / logtime 132,576 → P̂ = 53,324 (satexp, the median). Grower A earners: persistence
  24,411 / satexp 58,508 / logtime 95,060 → median is satexp again, because logtime overshoots even
  harder (earners grow ~240× in-window — players alone grow 80× — so a ln(1+t) extrapolation is
  explosive). Satexp's 58,508 still exceeds the 0.999 × P̂ cap, so the pre-guard median is 99.9 and
  the +1.5 logit cap is what binds. Grower B earners: 29,434 / 60,927 / 100,619 — same story.
  So on this rebuilt data the binding guardrail is the logit cap; median-of-3 alone would only
  reach the ceilings if logtime's refit landed between persistence and satexp for the growers.
