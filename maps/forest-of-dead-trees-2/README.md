# Forest of Dead Trees 2

This workspace is the first labeling pilot for the recording at
`C:/projects/input-flag-inspector/saves_m/ForestOfDeadTreesSample2_Lich`.

Geometry was extracted in `C:/projects/map-info-extractor` from direct yellow
minimap-marker detections only. No missing positions were interpolated. The
geometry manifest and direct positions CSV are referenced by `map.json` so a
later HUMAN routine can consume reviewed evidence without coupling it to this
dataset repository.

The current `pilot` split contains exactly five deterministic samples:

| ID | Source frame | PTS seconds |
|---|---:|---:|
| `pilot-0000` | 1,051 | 39.499867 |
| `pilot-0001` | 68,057 | 2,569.665233 |
| `pilot-0002` | 70,576 | 2,665.021967 |
| `pilot-0003` | 102,543 | 3,872.216167 |
| `pilot-0004` | 105,024 | 3,965.572533 |

Selection uses seed `0`, at least 30 seconds between samples, direct-marker
eligibility, and no UI/menu filtering. `dataset/pilot_manifest.json` records
the source and evidence hashes plus the exact marker observation for each
frame. All five annotations begin pending with zero boxes.

Run `run.bat` to label them. The two neutral presets in
`dataset/review_settings.json` are placeholders for this pilot; adjust their
names and dimensions after inspecting the five frames. Pilot annotations
cannot be exported to YOLO. A later, separately approved step must create
larger train/validation/test splits.
