# Forest of Dead Trees 2

This workspace is the first labeling pilot for the recording at
`C:/projects/input-flag-inspector/saves_m/ForestOfDeadTreesSample2_Lich`.

Geometry was extracted in `C:/projects/map-info-extractor` from direct yellow
minimap-marker detections only. No missing positions were interpolated. The
geometry manifest and direct positions CSV are referenced by `map.json` so a
later HUMAN routine can consume reviewed evidence without coupling it to this
dataset repository.

The dataset began with five deterministic random samples plus two explicitly
requested Lich samples:

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
frame. These seven frames are now reviewed members of the combined
train/validation dataset described below.

Run `review_codex_labels_0.bat` to inspect the seven original reference labels across their
current train/validation assignments. The three presets in
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
`Ctrl+U`/`Ctrl+D` are equivalent. Partially visible boxes can be dragged beyond
the image boundary while retaining at least one sixteenth of their area in the
frame. `Esc` only cancels the current interaction and returns to default mode;
it never closes the reviewer. Use `Q` to save and quit. Closing with the window
X also saves automatically.

Only one monster-review window may hold the canonical `annotations.jsonl` at a
time. A second launcher exits with an error instead of loading an independent
snapshot that could overwrite work saved by the first window.

Two requested Lich frames are stored under
`dataset/reference_samples/lich/`: one at 00:10 from the requested 00:00-00:19
period and one at 01:17:03 from the requested 01:16:57-01:17:45 period. Their
manifest records provenance, hashes, and exact source-frame anchors. They were
explicitly added as `pilot-0005` and `pilot-0006`; both are now reviewed.

## Tiny three-class learning experiment

`tools/prepare_forest_tiny_experiment.py` preserves this canonical pilot and
derives an ignored experiment snapshot under
`dataset/runs/tiny-3class-v1`. Presets map to three detector classes: Zombie,
Hero, and Lich. The fixed split is four regular frames plus `pilot-0005` for
training, and one regular frame plus `pilot-0006` for validation. Its sealed
visual test set contains 16 deterministic general frames and 4 deterministic
frames from 01:16:57-01:17:45. Those 20 frames remain unlabeled, so their
rendered predictions support visual inspection only, not accuracy metrics.

When this model is used through `perception_cli.py`, detections use YOLO's
native boxes by default. Pass `--monster-fixed-boxes` to use the three
registered source-pixel sizes in `dataset/review_settings.json`: Zombie
83x131, Hero 123x144, and Lich 123x202. In that optional mode, YOLO supplies
the class, confidence, and bottom-center position; post-processing supplies
the exact class box size and scales it for the actual input-frame resolution.

## Codex manual-label review batch

`dataset/codex_manual_batch_v1_manifest.json` records ten deterministic random
frames plus two frames from 00:00-00:19 and two from 01:16:57-01:17:45. Codex
placed all proposed Zombie, Hero, and Lich boxes by visual inspection of the
user's seven reference frames; neither OWLv2 nor another detector was used.
The user completed all 14 frames in the interactive reviewer. The random
town/dialog frame is preserved in canonical history but excluded from
development because it fails the minimap reference check.

A second manual edge pass adds 15 partial-Zombie proposals across five of those
frames. These include top-edge feet/lower bodies, a right-edge partial sprite,
and lower-playfield heads obscured by the frame/UI boundary. Each proposal keeps
the full 83x131 Zombie preset around the inferred sprite position, even when
part of that box lies outside the image; none of the boxes is tightened to only
the visible fragment.

`dataset/codex_manual_batch_v2_manifest.json` records a separate ten-frame
random batch from `ForestOfDeadTreeSample4_LottaLich`. Every selected frame
passes the `0.90` native-minimap correlation gate and is at least eight seconds
from the other samples. The detector supplied starting proposals, then Codex
visually checked all ten source frames and corrected classes, duplicates,
missed Liches, and partial edge/UI sprites. The resulting 55 Zombie, 9 Hero,
and 8 Lich boxes remain pending and review-required. Run
`review_codex_labels_2.bat` to confirm only this Sample4 batch against its own
source video. `review_codex_labels_1.bat` opens only the 14-frame Sample2 batch;
the legacy `review_codex_labels.bat` name delegates to that batch-1 launcher.

`dataset/codex_manual_batch_v3_manifest.json` records a third, separate
15-frame Sample4 batch selected from a 2.5-second survey of all valid-minimap
candidates. It is stratified around Lich observation conditions rather than
being purely random: 12 frames contain a visible Lich across clear, crowded,
effect-obscured, overlapping, high/low platform, and edge/UI-cutoff views; 3
frames are intentional Lich-absent comparisons. Codex visually audited the
full-resolution proposals after the detector-assisted pass, including
overlapping sprites and inferred full preset boxes at image boundaries. The
resulting 95 Zombie, 15 Hero, and 12 Lich boxes remain pending and
review-required. Run `review_codex_labels_3.bat` to confirm only this batch.

`dataset/codex_manual_batch_v4_manifest.json` records a blank, from-scratch
teaching batch from Sample4: five known false-positive/low-confidence Lich
diagnostics plus ten random valid-minimap frames. The random subset uses the
recorded seed `20260924`, a 0.5-second candidate grid, separation from existing
canonical and combined-v2 test frames, and at least three seconds between new
random selections. All 15 entries intentionally begin with zero boxes and
remain pending, so they cannot be exported for training until a reviewer labels
and confirms them. Run `review_codex_labels_4.bat` to open only this batch.
After the explanatory pass, the user's manual boxes on the opening frames were
preserved. The remaining blank frames received model-assisted Codex proposals
that were visually checked at full resolution to remove duplicates and obvious
snow/tree/UI false positives and to add missed partial/effect-obscured sprites.
All proposals remain review-required and the launcher resumes at frame `0002`
for human correction and confirmation.

## Combined confirmed experiment

`tools/prepare_forest_combined_v1.py` promotes the explicitly confirmed labels,
combines them with the original seven references, and produces a group-aware
15-train/5-validation split with 99 and 34 instances respectively. The early
Lich sequence is held together in validation; late-Lich examples remain in
training. Every development frame must have an exact direct marker observation
and at least `0.90` correlation with `minimap_native.png`. This excludes
`codex-random-0009`, whose town/dialog view scores `0.004`.

The same gate selects a separate unlabeled visual test of 20 new frames: 16
general, 2 early-Lich, and 2 late-Lich. Test minimap correlations range from
`0.981` to `0.996`. The YOLO11n 640px, 200-epoch CPU run reached validation
precision `0.927`, recall `0.815`, mAP50 `0.847`, and mAP50-95 `0.589` on only
five validation images. Lich recall was `0.667` across three validation boxes.
Because the 20 test frames are unlabeled, their prediction sheets are visual
evidence, not accuracy measurements. At confidence `0.25`, Lich was detected
in one late-window frame; at diagnostic confidence `0.05`, it appeared in both
early-window frames and one late-window frame, with extra low-confidence
proposals. Full provenance, weights hash, class metrics, and contact-sheet
paths are in `dataset/runs/combined-v1/experiment-report.json`.

## Full-collection experiment

`tools/prepare_forest_combined_v2.py` snapshots the current 45 valid-minimap
labeled frames without changing their canonical review statuses. The
source-aware, temporal-group split contains 36 training frames with 262 boxes
and 9 validation frames with 67 boxes. No train/validation pair from the same
recording is within 10 seconds. The town/dialog frame `codex-random-0009`
remains excluded because its minimap correlation is only `0.004`.

The YOLO11n 640px, 200-epoch CPU run selected epoch 157 and reached validation
precision `0.959`, recall `0.957`, mAP50 `0.980`, and mAP50-95 `0.746` on the
nine held-out images. Lich precision was `0.980` and recall was `1.000` across
six validation boxes. This is still a small validation set, so those figures
should not be treated as a broad production estimate.

A separate deterministic test uses 30 random frames from
`ForestOfDeadTreeSample4_LottaLich`; all pass the minimap gate, with correlation
from `0.980` to `0.996`, and remain unlabeled. Visual inspection at confidence
`0.25` found Lich detections in eight frames, each corresponding to a visible
full, obscured, or edge-cropped Lich. The original post-processing left
duplicate boxes on two partial Liches and on the Hero during effects in four
frames. Class-aware NMS now uses IoU `0.80` for Zombie and `0.50` for Hero/Lich;
rerendering removes those normal-confidence Hero/Lich duplicates.
Confidence `0.05` recovers another plausible extreme-edge Lich but increases
total proposals from 181 to 244, so `0.25` is the preferred display threshold. Full provenance,
weights hash, per-class metrics, test counts, audit notes, and contact-sheet
paths are in `dataset/runs/combined-v2/experiment-report.json`.

## Expanded full-collection experiment

`tools/prepare_forest_combined_v3.py` snapshots all 60 valid-minimap labeled
frames currently in the collection. The user's explicit retraining request
authorizes their use only in this experiment snapshot; canonical review and
review-required flags remain unchanged. The source-aware temporal split has 48
training frames with 340 boxes and 12 validation frames with 60 boxes. The
invalid-minimap town/dialog frame `codex-random-0009` remains excluded.

The fresh YOLO11n 640px, 200-epoch CPU run selected its best checkpoint at
epoch 164 and reached validation precision `0.970`, recall `0.957`, mAP50
`0.985`, and mAP50-95 `0.742`. Lich precision was `0.980` and recall was
`1.000` across eight validation boxes. These figures still come from a small
held-out set and should not be treated as a broad production estimate.

The separate visual test contains 15 deterministic seeded random frames from
`ForestOfDeadTreesSample4_LottaLich` and 15 from
`ForestOfDeadTreesSample6`. Seeds are `20260923` and `20260924`; all 30 frames
pass the `0.90` minimap-correlation gate. At confidence `0.25`, all five Lich
detections in Sample4 are visually valid full, obscured, or cropped Liches;
Sample6 contains no Lich detection at that threshold. Confidence `0.05`
recovers three additional visually real difficult Liches, including one in
Sample6, but raises total proposals from 187 to 281 and adds substantial
clutter. A few UI lookalikes remain, including two inventory-item Hero false
positives in Sample6. Full provenance, metrics, model hash, prediction counts,
and contact-sheet paths are in
`dataset/runs/combined-v3/experiment-report.json`.
