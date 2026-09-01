# Gameplay perception investigation

This repository contains the specialized model/dataset work for the exact
pirate-ship recording. The source capture remains at the path in
`capture_manifest.json` (or set `PERCEPTION_CAPTURE_DIR`). It does **not** claim cross-map or
unseen-outfit generalization.

## What is implemented

- `build_session_profile(video_or_frames, events=None, background_assets=None)`
- `analyze_frame(image, profile, timestamp=None, key_state=None)`
- `analyze_video(video, profile, events=None)`
- A stateful `FrameAnalyzer` for streaming integration without rereading video.
- Presentation-timestamp handling using the synchronized per-frame cache, with
  an `ffprobe` VFR fallback.
- SIFT translation registration against the reconstruction reference, stable
  background residuals, fixed-UI masking, and registration abstention.
- Detector-ready unstable residual cutouts remove the same fixed HUD regions
  used by registration before preserving foreground RGB pixels.
- `MonsterTracker` camera-compensates detector boxes in map coordinates,
  requires two observations by default, and separates unsupported temporal
  holds from visually supported occlusion. Persistent `OcclusionGroup` objects
  retain member counts without claiming fresh individual observations.
- Capture-local character templates with independent left/right banks. The
  current bootstrap has 4 left and 3 right anchors; the profile truthfully sets
  `guided_calibration_required=true` because neither side has 30 reviewed
  examples. Changed outfits require a new profile/hints.
- `FrameAnalyzer(profile, monster_detector=...)` accepts the template baseline
  or learned `YoloMonsterDetector`; backend objects do not leak through the
  normalized interface and provenance remains explicit.
- A backend-independent `MonsterDetection`/`SpecializedMonsterDetector`
  contract, conservative duplicate suppression, explicit tentative/visible/
  occluded/temporal-hold track states, many-track/one-detection occlusion
  groups, and separate `visual_detection_count` versus `estimated_count`.
- `monster_dataset/` contains the canonical JSONL annotation schema (including
  ground positions, occlusion, visibility, and review status), deterministic
  VFR sampling from the benchmark, contact sheets, dataset reports, and
  reviewed-only YOLO export. Pending proposals are intentionally not labels.
- Black/red A-wave segmentation, relative-position evidence, short-term
  centroid tracking, and camera-local optical flow. Symmetric onset and visual
  versus key disagreement are explicit quality flags.
- Visual-only results stored separately from fused/key-assisted results.
- JSONL, per-frame CSV, annotated images/video, deterministic benchmark
  manifests, contact sheets, runtime measurement, and reviewed-only evaluation.

## Reproduce

Run from this directory with Python 3.14 and the versions in
`requirements-lock.txt`:

```powershell
python perception_cli.py build-profile --video "$env:PERCEPTION_CAPTURE_DIR\screen.mp4" --events "$env:PERCEPTION_CAPTURE_DIR\events.csv" --background "$env:PERCEPTION_CAPTURE_DIR\background_reconstruction"
python perception_cli.py benchmark-manifest
python perception_cli.py extract-benchmark
python -m pytest -q tests
python perception_cli.py benchmark-runtime --frames 60
python perception_cli.py discover-video
python perception_cli.py monster-sample
python -m pip install -r requirements-prelabel.txt
python perception_cli.py monster-prelabel --split train
python perception_cli.py monster-review-report
python perception_cli.py monster-review --split train --queue pending
python perception_cli.py monster-contact-sheet --split train
python perception_cli.py monster-temporal-context --split train
python perception_cli.py monster-yolo-export --split train
python perception_cli.py monster-yolo-export --split validation
```

Regenerate the fixed-UI-masked residual proof inputs:

```powershell
python render_background_subtraction.py `
  --output proof\ui_masked_background_subtraction.png `
  --cutout-output proof\ui_masked_unstable_residual_cutouts.png `
  --cutout-dir proof\ui_masked_unstable_residual_cases
```

The OWLv2 text-prompt experiment and annotated outputs are documented in
`proof/owlv2_prompt_sweep/README.md`. It found `pirate monster` stronger than
the more literal mushroom prompts on the four reviewed proof inputs. This is a
prompt-selection probe, not an accuracy score.

Analyze one frame from the source video:

```powershell
python perception_cli.py image "$env:PERCEPTION_CAPTURE_DIR\screen.mp4" --timestamp 30.447689 `
  --events "$env:PERCEPTION_CAPTURE_DIR\events.csv" --json proof\right_cast.json `
  --annotated proof\right_cast.png
```

Analyze/render a retained video interval:

```powershell
python perception_cli.py video "$env:PERCEPTION_CAPTURE_DIR\screen.mp4" --start 29.8 --end 31.4 `
  --sample-fps 5 --jsonl proof\right_clip.jsonl `
  --csv proof\right_clip.csv --annotated proof\right_clip.mp4
```

Use learned inference after training without changing `FrameAnalyzer` or the
tracker:

```powershell
python perception_cli.py video "$env:PERCEPTION_CAPTURE_DIR\screen.mp4" `
  --monster-backend yolo --monster-weights monster_dataset\runs\v0-768\weights\best.pt `
  --monster-confidence 0.21 --monster-nms-iou 0.90 --monster-imgsz 768 `
  --jsonl evaluation\validation-observations.jsonl
```

Template and YOLO detections are not silently fused; select one backend per
run. YOLO's permissive NMS (`0.90` by default) only prevents pathological
proposal explosion. The tracker-side deduplicator is authoritative and only
suppresses boxes when high IoU, near-identical centers, and similar dimensions
all agree; strongly overlapping plausible monsters remain separate proposals.

### Monster output semantics

The three count concepts intentionally answer different questions:

- `visual_detection_count`: retained detector observations in the current
  frame, after player exclusion and conservative duplicate suppression.
- `estimated_count`: live confirmed persistent monster identities, including
  identities supported inside an unresolved overlap.
- `count_min` / `count_max`: the range justified by current independent visual
  evidence plus temporal identity evidence. One merged observation supporting
  two identities yields `[1, 2]`; adding one independently visible monster
  yields `[2, 3]`; unsupported holds contribute only to the upper bound.

Track states are exact: `tentative` lacks enough observations for a persistent
claim; `visible` has its own current detector observation; `occluded` lacks an
individual box but belongs to a currently supported overlap group; and
`temporal_hold` has neither fresh nor current supporting visual evidence.
Accordingly, `observed=true` only for `visible`, while `supported=true` only
for `visible` and `occluded`. Occluded tracks expose `group_member_count` and
`monster_overlap`; ordinary holds expose `monster_temporal_hold`.

Occlusion decisions are local to candidate members. For an existing or new
group, one shared geometrically relevant observation supports the group, while
two distinct individually explanatory observations allow ordinary association
to resolve it. Unrelated detector boxes elsewhere in the frame never enable or
disable group support merely by changing the global detection count.

`detector_debug` records raw confidence/box data, retention or suppression and
its reason, association IDs/costs, track ages/hits/fresh/support ages, group
membership, and expired IDs. It is diagnostic metadata rather than another
source of detection truth.

The full overview intentionally samples at approximately 0.5 fps while spanning
the full retained 00:00-36:03 timeline. Exact source presentation timestamps
remain in JSONL; `annotated_full_overview_timeline.mp4` has rescaled media PTS
so playback spans that timeline. It is a review artifact, not a claim that every
source frame was inferred:

```powershell
python perception_cli.py video "$env:PERCEPTION_CAPTURE_DIR\screen.mp4" --start 0 --end 2163.8 `
  --sample-fps 0.5 --jsonl proof\full_overview_current.jsonl `
  --csv proof\full_overview_current.csv `
  --annotated proof\annotated_full_overview_current_timeline.mp4
```

## Benchmark protocol

`benchmark/annotations.csv` has 360 deterministic candidates. Train,
validation, and sealed test are complete non-overlapping time ranges with
216/72/72 frames; each split is balanced across the three mining categories.
The `weak_*` columns come only from synchronized keys and are never consumed by
the evaluator. Contact sheets and individual review images are under
`benchmark/frames`.

Review canonical monster boxes and ground positions in
`monster_dataset/annotations.jsonl`. Keep the sealed test untouched until
weights and every threshold are frozen. During development run:

```powershell
python perception_cli.py monster-evaluate `
  --annotations monster_dataset\annotations.jsonl `
  --observations <matching-observations.jsonl> --split validation
```

This reports visual detections and persistent estimates separately, including
center precision/recall, count MAE, exact-count accuracy, condition breakdowns,
duplicate rate, player false positives, fragmentation proxy, and p50/p95
latency. It rejects pending labels and the sealed test split.

## Current proof boundary

Registration and both A-wave directions are demonstrated in the proof clips,
including a flagged transient disagreement. Monster templates return plausible
proposals in exact/near-template poses but visibly undercount crowded and
occluded groups. Character matching recognizes reviewed bootstrap poses but
abstains across many other animation phases. Therefore the target accuracy
metrics are deliberately `not_evaluated`; no ONNX fallback was trained because
reviewed detector ground truth does not yet exist. The next minimum intervention
is visual review of the 216 train and 72 validation frames. The 72 sealed frames
remain unreviewed until model weights and configuration are frozen.
