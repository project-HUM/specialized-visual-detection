# Forest of Dead Trees 2

This workspace is the first labeling pilot for the recording at
`C:/projects/input-flag-inspector/saves_m/ForestOfDeadTreesSample2_Lich`.

Geometry was extracted in `C:/projects/map-info-extractor` from direct yellow
minimap-marker detections only. No missing positions were interpolated. The
geometry manifest and direct positions CSV are referenced by `map.json` so a
later HUMAN routine can consume reviewed evidence without coupling it to this
dataset repository.

The current `pilot` split contains five deterministic random samples plus two
explicitly requested Lich samples:

| ID | Source frame | PTS seconds |
|---|---:|---:|
| `pilot-0000` | 1,051 | 39.499867 |
| `pilot-0001` | 68,057 | 2,569.665233 |
| `pilot-0002` | 70,576 | 2,665.021967 |
| `pilot-0003` | 102,543 | 3,872.216167 |
| `pilot-0004` | 105,024 | 3,965.572533 |
| `pilot-0005` | 267 | 9.960067 |
| `pilot-0006` | 122,443 | 4,622.296900 |

Selection uses seed `0`, at least 30 seconds between samples, direct-marker
eligibility, and no UI/menu filtering. `dataset/pilot_manifest.json` records
the source and evidence hashes plus the exact marker observation for each
frame. The two requested Lich frames are also pending so they appear in the
same review queue without changing previously saved annotations.

Run `run.bat` to label them. The three presets in
`dataset/review_settings.json` are Zombie, Hero, and Lich (special monster).
Edit each preset's source-pixel `width`, `height`, and display `hue`, then close
and reopen the reviewer. Use `Ctrl+1`, `Ctrl+2`, or `Ctrl+3`, or drag a preset
card onto the image. The right-side hotkey/help panel scrolls when the mouse
wheel is over it. For remote-desktop-friendly sizing, select a preset or one
of its boxes, then use `Shift+Left`/`Shift+Right` to shrink/expand width and
`Shift+Down`/`Shift+Up` to shrink/expand height one source pixel at a time. The
letter forms `Shift+L`/`Shift+R` and `Shift+D`/`Shift+U` work too. The
bottom-left corner remains fixed and the new preset size is saved immediately
to `dataset/review_settings.json`. Select any box and use
`Ctrl+Left`/`Ctrl+Right`/`Ctrl+Up`/`Ctrl+Down` to move it one source pixel at a
time; boxes that are already fully inside remain inside. `Ctrl+L`/`Ctrl+R`/
`Ctrl+U`/`Ctrl+D` are equivalent. Pilot annotations
cannot be exported to YOLO. A later, separately approved step must create
larger train/validation/test splits.

Two requested Lich frames are stored under
`dataset/reference_samples/lich/`: one at 00:10 from the requested 00:00-00:19
period and one at 01:17:03 from the requested 01:16:57-01:17:45 period. Their
manifest records provenance, hashes, and exact source-frame anchors. They were
explicitly added as pending `pilot-0005` and `pilot-0006` annotations so they
are visible when `run.bat` is reopened.
