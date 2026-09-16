# FS-Jump3D Dataset Audit

## Headline Counts

- Total real MP4 files: 3036
- Ignored metadata MP4 sidecars: 3036
- Physical jump attempts, all labels: 253
- Physical jump attempts, six-class MVP only: 240
- Usable six-class MP4 files: 2880
- Excluded combination-jump videos: 156
- Structure matches expected skater/camera layout: True

## Counts By Jump Class

| Jump class | Videos |
| --- | ---: |
| Axel | 480 |
| Flip | 480 |
| Loop | 480 |
| Lutz | 480 |
| Salchow | 480 |
| Toeloop | 480 |

## Counts By Skater

| Skater | Videos |
| --- | ---: |
| A | 756 |
| B | 756 |
| C | 768 |
| D | 756 |

## Counts By Camera

| Camera | Videos |
| --- | ---: |
| 1 | 253 |
| 10 | 253 |
| 11 | 253 |
| 12 | 253 |
| 2 | 253 |
| 3 | 253 |
| 4 | 253 |
| 5 | 253 |
| 6 | 253 |
| 7 | 253 |
| 8 | 253 |
| 9 | 253 |

## Counts By Skater And Class

| Skater | Axel | Flip | Loop | Lutz | Salchow | Toeloop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 120 | 120 | 120 | 120 | 120 | 120 |
| B | 120 | 120 | 120 | 120 | 120 | 120 |
| C | 120 | 120 | 120 | 120 | 120 | 120 |
| D | 120 | 120 | 120 | 120 | 120 | 120 |

## Physical Attempt Camera Views

- Minimum views per group: 12
- Maximum views per group: 12
- View-count distribution: {12: 253}

## Missing Camera Views

None

## Data Quality Checks

- Duplicate paths: None
- Malformed/unexpected filenames: None
- MP4 header issues: None
- Unexpected jump labels: None
- Group ID verification: {'mismatches': [], 'groups_with_multiple_cameras': 253, 'all_groups_camera_free': True}

## Notes

- macOS metadata paths under `__MACOSX`, `.DS_Store`, and `._*` are ignored.
- `Comb` videos are indexed but excluded from the six-class MVP dataset.
- The `split` column is intentionally empty in Milestone 1.
