# OWLv2 UI-masked residual prompt sweep

This is a prompt-selection probe over four representative registered residual
cutouts. It is not a detection metric: only the five highest-scoring boxes per
prompt were retained, and these frames do not have exhaustive monster labels.

All inputs had the capture-local fixed UI rectangles removed before OWLv2. The
model was `google/owlv2-base-patch16-ensemble`, text threshold was deliberately
lowered to `0.05` to inspect score separation, and raw prompt identity was kept.

The four old residual masks contained 41,161, 22,955, 48,242, and 40,356
surviving pixels inside fixed-UI regions. Their regenerated counterparts each
contain exactly zero, while retaining 65,409, 82,964, 72,409, and 55,565
gameplay-region residual pixels respectively.

| timestamp | prompt | retained score range |
|---:|---|---:|
| 30.859 | pirate monster | 0.402-0.469 |
| 30.859 | pirate enemy | 0.355-0.389 |
| 30.859 | pirate mushroom | 0.256-0.321 |
| 30.859 | cartoon pirate mushroom | 0.265-0.317 |
| 30.859 | mushroom monster | 0.256-0.299 |
| 112.985 | pirate monster | 0.321-0.492 |
| 112.985 | pirate enemy | 0.328-0.447 |
| 112.985 | pirate mushroom | 0.167-0.233 |
| 116.263 | pirate monster | 0.297-0.360 |
| 116.263 | pirate enemy | 0.252-0.337 |
| 116.263 | pirate mushroom | 0.222-0.298 |
| 1747.064 | pirate monster | 0.419-0.507 |
| 1747.064 | pirate enemy | 0.417-0.461 |
| 1747.064 | pirate mushroom | 0.181-0.191 |

`pirate monster` was the strongest prompt in every tested frame. The live HUD
setting was therefore retained but corrected from the malformed
`pirate monster::0.32` to `pirate monster:0.29`. A single colon is the syntax
accepted by the detector's per-prompt threshold parser.

Visual review found an important persistent false positive: the player's own
pirate outfit can match `pirate monster`. The tracker accepts an exact
`exclusion_boxes` list for this reason. Pass the independently recognized
character box and reject only substantial box overlap (default IoU 0.35); do
not mask a broad area around the character where real monsters may be present.

Representative command, run from `C:\projects\owlv2-visual-detector`:

```powershell
.\.venv\Scripts\python.exe -m owlv2_detector.cli image `
  --input <ui-masked-residual.png> `
  --target "pirate monster" --target "pirate mushroom" `
  --target "pirate enemy" --strategy text `
  --text-threshold 0.05 --raw-prompt-results --save-image
```

For a video stream, pass each accepted detection to
`perception.tracking.MonsterTracker.update`. Supply the registered map-space
origin of scene pixel `(0, 0)` as `camera_origin`; count confirmed tracks only.
The default two-hit confirmation suppresses isolated boxes, while a maximum
0.55-second temporal hold bridges brief OWLv2 misses and reports
`monster_temporal_hold` in `quality_flags`.
