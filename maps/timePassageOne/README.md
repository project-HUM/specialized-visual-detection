# timePassageOne review preparation

## CPU training experiment

All 50 canonical frames are reviewed (441 mob boxes, 49 hero boxes, no remaining
review-required flags). The frozen `pretrained-v1-20260930` experiment preserves
the canonical annotations, presets and order. Event frames 27, 42 and 43 are
excluded from training and validation but retained in the portable snapshot.

The user-selected split samples both videos with seed 20260930: 38 training
images (370 mob, 37 hero) and 9 validation images (71 mob, 9 hero). Validation
positions are 8, 13, 15, 21, 24, 36, 45, 49 and 50. These share video sources with
training and provide development metrics, not independent-source accuracy.

Training starts from generic pretrained YOLO11n, using CPU with six threads,
640px, batch 16, workers 0 and APO's AdamW configuration. Minimum 200 epochs,
patience 100, maximum 1000. Only strictly higher validation mAP50-95 resets
patience; the counter is not reset at epoch 200. Cosine learning-rate decay is
scheduled over 1000 epochs; mosaic closes for the final 10 if the cap is reached.

The standalone runner, frozen inputs, logs and checkpoints are in
`dataset/runs/pretrained-v1-20260930`. Lightweight provenance and recovery
instructions are in `experiments/pretrained-v1-20260930`. Training completed at
epoch **276**, with the best checkpoint at **176**, stopping after 100 epochs
without improvement. The CPU run took 65.1 minutes. Validation mAP50 is **92.8%**
and mAP50-95 is **71.1%**. At confidence 0.25, mob counts are 67 matched / 9 extra /
4 missed; hero counts are 7 matched / 0 extra / 2 missed. The hero misses are
heavily obscured frames 13 and 45. All nine comparison images were inspected.

See [the results report](experiments/pretrained-v1-20260930/RESULTS.md) for metrics,
limitations and artifacts. Best weights are at
`dataset/runs/pretrained-v1-20260930/runs/pretrained-640/weights/best.pt`.
The archive receipt records the recovery ZIP and verified extracted copy under
`C:/Users/MAD/Documents/Project HUM/Training Backups/timePassageOne`.
Eight focused tests and the one-epoch CPU smoke test passed. Canonical annotations,
presets and order remain byte-identical. `resume_cpu_training.bat` is only for an
interrupted unfinished run; this completed run must not be resumed.
No checkpoint has been deployed.

## Labeling history

The final pre-labeling batch added 155 mob and 20 hero proposals to positions
31-50, subsequently reviewed by the user. All 50 samples have been inspected.
Saved presets remain 104x116 for mob and 108x148 for hero. Frames 42-43 are off-map
event scenes with hero-only proposals, like frame 27. Review the heavily obscured
hero in frame 45, translucent sprites, overlaps, minimap/chat occlusion, and
top/bottom edge fragments carefully. Notes record the uncertain cases.
Original images, settings, order, and the first 30 annotation lines are unchanged.
Batch provenance is in `dataset/codex_manual_batch_v3_manifest.json`; overlays and
inspection sheets are in `dataset/prelabel_contacts/batch3`. The before-state
backup is in `dataset/runs/before-codex-labels-3`. Canonical frames retain their
pilot split; the separate experiment snapshot has validated training splits.

Historical batch 2 added 35 mob and 3 hero proposals to positions 26-30 while
preserving the user's frame-26 hero exactly. The user accepted its box positions
unchanged. Frame 30's hero is fully hidden by inventory and has no box.
Its artifacts remain in `dataset/codex_manual_batch_v2_manifest.json`,
`dataset/prelabel_contacts/batch2` and `dataset/runs/before-codex-labels-2`.

Historical batch 1 added 176 mob and 17 hero proposals to positions 9-25.
Its manifest records four small mob-position corrections from the prior review;
artifacts remain in `dataset/codex_manual_batch_v1_manifest.json`,
`dataset/prelabel_contacts/batch1` and `dataset/runs/before-codex-labels-1`.
Batch 0 retains the earlier positions 5-8 proposals and original backup.

Open `review_codex_labels_0.bat` to label all 50 saved images.
Ctrl+1 = mob; Ctrl+2 = hero. Current saved presets are 104x116 and 108x148 pixels.
Resize a preset box using its corner handles or Shift+arrows; future placements use the saved size.
Click the Frame number and press Enter to jump; S saves and Q saves and quits.

Positions 1-25 come from timePassageOneSample0/screen.mp4; positions 26-50
come from timePassageOneSample1/screen.mp4, under C:/projects/input-flag-inspector/saves_m/.
Each source is chronological, and the canonical JSONL plus review_order.json preserve this order.
Divide each recording's first-to-last presentation-timestamp span into 25 equal intervals;
choose the nearest source frame to each midpoint, breaking ties toward the earlier frame.
No random sampling, shuffling, or content filtering is applied.

Images are full-resolution 1920x1080 PNGs with verified lossless decoded-pixel roundtrips.
Initial preparation generated no boxes: all 50 frames started pending in the pilot split. Empty annotations
are unlabeled, not confirmed negatives (the generic report counts empty frames as negative_frames).
Training readiness remains false. Geometry, minimap, and UI masks are uncalibrated.
Use images-only review because this collection spans two source videos.

dataset/pilot_manifest.json records both source hashes, original frame indices and timestamps,
target midpoint times, image hashes, class mapping, and display positions.
dataset/contacts contains refreshed overview sheets. artifact_hashes.json describes initial preparation.
Reproduce into a fresh map directory with `python tools/prepare_time_passage_one.py`
from the repository root; the script refuses to overwrite any existing map/review work.
