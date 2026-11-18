# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/nerdwallet6/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/nerdwallet6/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts Grand Theft Auto VI's launch-day Twitch viewers hour by hour: for each of hours 2–24 of the release day (UTC), five quantiles (5%, 25%, 50%, 75%, 95%) of that hour's viewer count in the "Grand Theft Auto VI" Twitch category, predicted from the first hour's count and the release-day curves of past big game launches. Standard library only, no network, deterministic.

## Method (v3)

Past launches fall into three shapes — rolling-midnight unlocks, simultaneous/global unlocks, and early-access-contaminated flat curves — and GTA VI is a rolling-midnight launch, so the model trains only on rolling-midnight comparables. Two extra signals sharpen the shape: the pre-midnight surge (viewers at day_start − 1h relative to hour 1, when NZ/Australia and UK/Europe had already unlocked but the US had not) and the hour-1 channel count, both entering a per-hour ridge regression with a small penalty. A local-linear kernel regression on the same rolling subset is ensembled 50/50, with calibrated quantile spreads. Practice loss (four past launches): 0.1406, vs 0.1684 for the v2 all-regimes fit.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=aleca&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
