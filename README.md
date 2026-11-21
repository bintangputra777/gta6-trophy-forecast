# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bintangputra777/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bintangputra777/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for GTA VI's PS5 trophy earn rates on Exophase: for each of the game's 55 trophies it predicts the percent of tracked players who will have earned it at the case's target, as five quantiles (5%, 25%, 50%, 75%, 95%).

## How it works

An earn rate is a ratio — earners ÷ players tracked — so the program rebuilds both series from the case's reads (`earners = rate/100 × players_tracked`) and forecasts them separately:

- Three model families per series (persistence, a saturating-exponential fit, a log-time linear fit), each refit on all reads; the point forecast is the **median of the three families**, which keeps any single runaway fit from dominating.
- The median rate is guarded on the logit scale: it may move at most +1.5 / −2.0 logit units from the trophy's last observed rate.
- The spread comes from a 70/30 backtest's residuals per trophy, combined with a horizon term, on the logit scale — so rare trophies and common ones are scored fairly.

Standard library only, no network, fully deterministic: two runs on the same case write the same forecast. See `TESTS.md` for the validation runs, including a synthetic stress test shaped like the final case.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13. Reads every `<id>.in` case file in the input directory and writes `<id>.out` forecasts to the output directory.

---

Entered on [make.fun](https://make.fun/?ref=gr3weejb&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards
