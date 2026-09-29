# APO detector preparation

Initial dataset preparation follows the Forest of Dead Trees 2 map workspace,
canonical JSONL, pending review, contact sheet, and provenance workflow.

- Source: `C:/projects/input-flag-inspector/saves_m/APO_sample2/screen.mp4`.
- 50 distinct randomly selected source frames, sampled without replacement
  over the entire video with Python `random.Random(20260929)`.
- Full-resolution 1920x1080 PNGs in `dataset/images`, without crops or overlays.
- Exact zero-based source frame indices, original presentation timestamps,
  source hash, and image hashes are in `dataset/pilot_manifest.json`.
- All 80 canonical frames are reviewed, with 700 boxes and no remaining
  review-required flags. They remain in the `pilot` split for the reviewer;
  training uses a separate frozen snapshot.
- Positions 51-65 append 15 random frames from `APO_sample1`; positions 66-80
  append 15 from `APO_sample3_woSpecial`. Sampling and saved shuffling are independent
  per source; the original 50 positions are unchanged. Source paths, hashes,
  exact frame indices, timestamps, and seeds are in
  `dataset/codex_manual_batch_v3_manifest.json`.
- Geometry, minimap crop, UI masks, and trained weights are not available yet.
  No frames were filtered by minimap or assumed map membership.
- The first frame stays first. The other 49 are shuffled once with seed
  `20260929`; canonical JSONL order and `dataset/review_order.json` preserve
  the same display order across launches and saves.

Open `review_codex_labels_0.bat` to review all 80 still images.
Ctrl+1 = `mob`, Ctrl+2 = `hero`, Ctrl+3 = `special`; A enables freehand boxes.
Preset sizes follow the user's first-image work, including the hero size
recovered from its manually resized annotation. Corner dragging previews the
border immediately and saves the new preset dimensions on release. Existing
other boxes keep their dimensions; future placements use the updated preset.
Click the Frame number, type a display position (1-80), and press Enter to
save current edits and jump. Esc cancels number editing.
The launcher includes all frames so it can revisit reviewed images.
Contact sheets are in `dataset/contacts`; full-resolution proposal overlays
are in `dataset/prelabel_contacts/final30` for positions 21-50 and
`dataset/prelabel_contacts/additional-sources` for positions 51-80. Earlier
overlays are historical proposals and precede the user's corrections;
the canonical annotations and current contact sheets contain those corrections.
Use only `review_codex_labels_0.bat` for this combined 80-frame list. The
numbered preparation scripts and overlay folders record labeling passes,
not separate review datasets. Use the editable Frame number to jump anywhere.

## Label policy

The ghost-like character on the right with a **book icon in a thought bubble
is an NPC**, not a mob. Leave that individual unlabeled. Do not exclude a
fixed right-side region: actual mobs can overlap the NPC and still need their
own boxes. Keep pets, dropped items, UI icons and fixed wall monitors unlabeled.
`special` refers to the moving monitor-and-camera robot.
Position 60 has dialogue occlusion: the fully hidden special is not inferred.
Position 66 is a random town frame retained from sample3, tagged
`outside_APO_town`; only the user's hero is labeled. Review its inclusion when
building the training split. Use images-only review for this multiple-video list.

From the repository root:

```powershell
python perception_cli.py --map APO monster-review-report
python perception_cli.py --map APO monster-contact-sheet --split pilot
```

`tools/prepare_apo_random_batch.py` reproduces the extraction into a fresh APO
dataset and refuses to overwrite an existing dataset or review work.
The source recording is read-only and remains outside this workspace.
`tools/prepare_apo_codex_labels_0.py` documents the one-time shuffle and visual
proposal coordinates, refuses repeated application, and keeps the original
annotations/settings in `dataset/runs/before-codex-labels-0`.
`tools/prepare_apo_codex_labels_1.py` records the next 15 visual proposals and
the NPC exclusion policy. Its pre-edit backup is in
`dataset/runs/before-codex-labels-1`; existing reviewed frames and presets are
not overwritten.
`tools/prepare_apo_codex_labels_final30.py` records positions 21-50, with the
pre-edit backup in `dataset/runs/before-codex-labels-final30`.
`tools/prepare_apo_additional_sources.py` extracts the two additional groups;
`tools/label_apo_additional_sources.py` records the visual proposals and appends
them. The original 50 annotation lines, settings, and order are backed up in
`dataset/runs/before-additional-sources`. Existing annotation bytes and preset
settings are preserved.

## Pretrained detector experiment

`pretrained-v1-20260929` freezes the reviewed collection without changing the
canonical annotation bytes, presets, or review order. It uses 64 training images
(555 boxes) from sample2/sample3 and 15 validation images (144 boxes) from sample1.
The source videos are disjoint between splits. Town frame 66 is retained in the
snapshot's excluded folder but is not used in either split.

Classes are `0: mob`, `1: hero`, and `2: special`. The run starts from generic
pretrained YOLO11n, following Forest's pretrained configuration: 640px, CPU,
batch 16, AdamW lr=0.001, maximum 200 epochs, patience 50, seed 20260929.
The experiment is under `dataset/runs/pretrained-v1-20260929`; portable recovery
instructions and lightweight provenance are under
`experiments/pretrained-v1-20260929`. Training and evaluation results are recorded
there when complete. No checkpoint is automatically deployed.

The canonical pilot report is not training-ready because its split is deliberately
unchanged; the experiment snapshot has validated train/validation splits.
Geometry can be added independently when evidence is available.
