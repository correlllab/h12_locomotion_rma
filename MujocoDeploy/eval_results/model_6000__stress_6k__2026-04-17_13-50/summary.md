# RMA Comprehensive Evaluation — Summary

- Paired trials: **708** (each run twice: RMA + baseline)

- Overall survival: **RMA** 531/708 (75.0%) vs **baseline** 429/708 (60.6%)


## Paired survival breakdown

- Both survived: **428** (60.5%)
- RMA only: **103** (14.5%)   ← RMA lift
- Baseline only: **1** (0.1%)   ← RMA regression
- Both fell: **176** (24.9%)


## Per-body × per-magnitude survival rate (paired)

| Body | Magnitude | RMA % | Baseline % | Pairs | RMA-only | Base-only |
|------|-----------|-------|------------|-------|----------|-----------|
| combined | 30/20/20N | 100.0 | 100.0 | 12 | 0 | 0 |
| combined | 50/30/30N | 91.7 | 83.3 | 12 | 1 | 0 |
| combined | 75/50/50N | 91.7 | 50.0 | 12 | 5 | 0 |
| combined | 100/75/75N | 50.0 | 16.7 | 12 | 4 | 0 |
| left_wrist | 0N | 100.0 | 100.0 | 4 | 0 | 0 |
| left_wrist | 30N | 100.0 | 100.0 | 36 | 0 | 0 |
| left_wrist | 50N | 100.0 | 94.4 | 36 | 2 | 0 |
| left_wrist | 75N | 94.4 | 63.9 | 36 | 11 | 0 |
| left_wrist | 100N | 66.7 | 30.6 | 36 | 13 | 0 |
| left_wrist | 150N | 22.2 | 22.2 | 36 | 0 | 0 |
| left_wrist | 200N | 22.2 | 5.6 | 36 | 6 | 0 |
| right_wrist | 0N | 100.0 | 100.0 | 4 | 0 | 0 |
| right_wrist | 30N | 100.0 | 100.0 | 36 | 0 | 0 |
| right_wrist | 50N | 100.0 | 97.2 | 36 | 1 | 0 |
| right_wrist | 75N | 100.0 | 97.2 | 36 | 1 | 0 |
| right_wrist | 100N | 91.7 | 44.4 | 36 | 18 | 1 |
| right_wrist | 150N | 33.3 | 13.9 | 36 | 7 | 0 |
| right_wrist | 200N | 33.3 | 16.7 | 36 | 6 | 0 |
| torso | 0N | 100.0 | 100.0 | 4 | 0 | 0 |
| torso | 30N | 100.0 | 100.0 | 36 | 0 | 0 |
| torso | 50N | 100.0 | 88.9 | 36 | 4 | 0 |
| torso | 75N | 97.2 | 86.1 | 36 | 4 | 0 |
| torso | 100N | 97.2 | 63.9 | 36 | 12 | 0 |
| torso | 150N | 50.0 | 27.8 | 36 | 8 | 0 |
| torso | 200N | 22.2 | 22.2 | 36 | 0 | 0 |

## Per-command survival rate (paired)

| Command | RMA % | Baseline % | Pairs |
|---------|-------|------------|-------|
| side_left | 73.7 | 58.8 | 354 |
| walk | 76.3 | 62.4 | 354 |

## Tracking RMSE on common survivors (mean ± std, m/s)

| Body | Magnitude | RMA | Baseline | Pairs |
|------|-----------|-----|----------|-------|
| combined | 30/20/20N | 0.1387±0.0158 | 0.1794±0.0621 | 12 |
| combined | 50/30/30N | 0.1838±0.0294 | 0.4147±0.1913 | 10 |
| combined | 75/50/50N | 0.1598±0.0470 | 0.2516±0.1823 | 6 |
| combined | 100/75/75N | 0.3577±0.0756 | 1.4171±0.0522 | 2 |
| left_wrist | 0N | 0.1194±0.0075 | 0.1194±0.0075 | 4 |
| left_wrist | 30N | 0.1417±0.0185 | 0.1674±0.0339 | 36 |
| left_wrist | 50N | 0.1588±0.0296 | 0.2386±0.0973 | 34 |
| left_wrist | 75N | 0.1902±0.0839 | 0.3921±0.1869 | 23 |
| left_wrist | 100N | 0.1598±0.0520 | 0.2857±0.2041 | 11 |
| left_wrist | 150N | 0.1424±0.0168 | 0.2293±0.1199 | 8 |
| left_wrist | 200N | 0.1760±0.0048 | 0.3486±0.1218 | 2 |
| right_wrist | 0N | 0.1194±0.0075 | 0.1194±0.0075 | 4 |
| right_wrist | 30N | 0.1436±0.0183 | 0.1655±0.0376 | 36 |
| right_wrist | 50N | 0.1766±0.0353 | 0.2678±0.1041 | 35 |
| right_wrist | 75N | 0.1972±0.0688 | 0.4058±0.2254 | 35 |
| right_wrist | 100N | 0.3366±0.2848 | 0.4748±0.4347 | 15 |
| right_wrist | 150N | 0.1950±0.0382 | 0.3441±0.3297 | 5 |
| right_wrist | 200N | 0.1896±0.0253 | 0.2207±0.0460 | 6 |
| torso | 0N | 0.1194±0.0075 | 0.1194±0.0075 | 4 |
| torso | 30N | 0.1427±0.0165 | 0.1720±0.0352 | 36 |
| torso | 50N | 0.1599±0.0317 | 0.2686±0.1000 | 32 |
| torso | 75N | 0.1831±0.0675 | 0.4954±0.2492 | 31 |
| torso | 100N | 0.2064±0.0807 | 0.8148±0.5104 | 23 |
| torso | 150N | 0.1816±0.1068 | 0.3071±0.3022 | 10 |
| torso | 200N | 0.1379±0.0235 | 0.1786±0.0470 | 8 |