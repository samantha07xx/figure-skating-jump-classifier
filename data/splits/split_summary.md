# FS-Jump3D Grouped Split Summary

## Strategy

- Split level: physical-attempt `group_id`, never individual videos.
- Split ratios: `70/15/15` for train/validation/test.
- Seed: `42`.
- Scope: six target classes only; `Comb` rows are excluded from the ML split.
- Decision note: the blueprint allowed `70/15/15` or `80/10/10`; `70/15/15` was selected because 40 groups per class divides cleanly into 28/6/6 groups and gives larger validation/test sets.

## Counts

| Split | Physical-attempt groups | Video samples |
| --- | ---: | ---: |
| train | 168 | 2016 |
| val | 36 | 432 |
| test | 36 | 432 |

## Class x Split Distribution (Groups)

| Item | Train | Val | Test |
| --- | ---: | ---: | ---: |
| Axel | 28 | 6 | 6 |
| Flip | 28 | 6 | 6 |
| Loop | 28 | 6 | 6 |
| Lutz | 28 | 6 | 6 |
| Salchow | 28 | 6 | 6 |
| Toeloop | 28 | 6 | 6 |

## Class x Split Distribution (Videos)

| Item | Train | Val | Test |
| --- | ---: | ---: | ---: |
| Axel | 336 | 72 | 72 |
| Flip | 336 | 72 | 72 |
| Loop | 336 | 72 | 72 |
| Lutz | 336 | 72 | 72 |
| Salchow | 336 | 72 | 72 |
| Toeloop | 336 | 72 | 72 |

## Skater x Split Distribution (Groups)

| Item | Train | Val | Test |
| --- | ---: | ---: | ---: |
| A | 42 | 9 | 9 |
| B | 42 | 9 | 9 |
| C | 42 | 9 | 9 |
| D | 42 | 9 | 9 |

## Skater x Split Distribution (Videos)

| Item | Train | Val | Test |
| --- | ---: | ---: | ---: |
| A | 504 | 108 | 108 |
| B | 504 | 108 | 108 |
| C | 504 | 108 | 108 |
| D | 504 | 108 | 108 |

## Leakage And Integrity Checks

- Zero group leakage: True
- All groups have 12 views: True
- Duplicate filepaths: None
- All usable samples assigned exactly once: True
- Combination jumps excluded: True
- Only target labels appear: True
- Assigned target rows: 2880
- Expected target rows: 2880
- Assigned groups: 240
- Expected groups: 240

## Notes

- This is the primary MVP group-aware split, not a skater-independent or leave-one-skater-out evaluation.
- Skater distribution is balanced exactly across splits for this audited dataset, but the split still includes all four skaters in each split.
