# Specialized visual detection

Shared capture-local perception and one-class monster-detection tooling. Map
identity, capture locations, prompts, geometry references, and generated
artifacts live under `maps/<map-id>/`; shared Python implementation lives in
`perception/` and `monster_dataset/`.

There is deliberately no root `run.bat`. Launch the map you intend to review:

```text
maps/rednose3/run.bat
maps/forest-of-dead-trees-2/run.bat
```

The same boundary applies to the CLI:

```powershell
python perception_cli.py --map rednose3 monster-review-report
python perception_cli.py --map forest-of-dead-trees-2 monster-review --split pilot --queue pending
```

`--map` defaults to `rednose3` for command-line compatibility and can also be
set with `SVD_MAP`. `PERCEPTION_CAPTURE_DIR` overrides only the selected map's
capture directory.

## Map layout

Each `map.json` uses schema `specialized-visual-detection.map.v1` and owns:

- capture video/event locations and expected source resolution;
- minimap and fixed-UI geometry;
- reviewed pre-label prompts, if any;
- profile, benchmark, proof, evaluation, dataset, and training paths;
- optional external direct-geometry evidence.

`MapWorkspace` validates map identity and refuses configured artifact paths
that escape the selected map directory. Cross-map artifacts are never selected
implicitly.

## Forest of Dead Trees 2 dataset

The Forest workspace began as a five-frame deterministic labeling pilot and
two explicitly requested Lich frames. Those seven references and a later
fourteen-frame manual batch have now been combined into the canonical dataset.
One town/dialog frame failed the minimap validity gate and remains excluded;
the development split contains 15 training and 5 validation frames:

```powershell
python tools/prepare_forest_combined_v1.py
maps\forest-of-dead-trees-2\run.bat
```

A second, still-pending ten-frame review batch comes from
`ForestOfDeadTreeSample4_LottaLich`. It is isolated behind
`maps\forest-of-dead-trees-2\review_codex_labels_2.bat`, uses only frames that
pass the minimap-reference gate, and must be confirmed before any retraining.

The combined experiment uses a direct minimap observation plus a `0.90`
correlation check against the native minimap reference. Its separate 20-frame
visual test contains 16 general, 2 early-Lich, and 2 late-Lich frames. See the
map README and `dataset/runs/combined-v1/experiment-report.json` for provenance
and results. Run outputs remain ignored derived artifacts.

The reproducible tiny three-class experiment derives a separate reviewed
5-train/2-validation snapshot and a deterministic unlabeled visual test set of
16 general frames plus 4 frames from the Lich interval:

```powershell
python tools/prepare_forest_tiny_experiment.py
python train_specialized_detector.py --map forest-of-dead-trees-2 `
  --annotations maps/forest-of-dead-trees-2/dataset/runs/tiny-3class-v1/annotations.jsonl `
  --data maps/forest-of-dead-trees-2/dataset/runs/tiny-3class-v1/yolo_dataset/dataset.yaml `
  --output maps/forest-of-dead-trees-2/dataset/runs/tiny-3class-v1/training-200 `
  --model maps/forest-of-dead-trees-2/dataset/runs/tiny-3class-v1/pretrained-yolo11n.pt `
  --imgsz 640 --epochs 200 --device cpu
```

## Train, validation, and test

The canonical JSONL is map-local. YOLO files are reviewed-only derived output:

```powershell
python perception_cli.py --map rednose3 monster-prelabel --split train
python perception_cli.py --map rednose3 monster-review --split train --queue pending
python perception_cli.py --map rednose3 monster-review --split validation --queue pending
python perception_cli.py --map rednose3 monster-yolo-export --split train
python perception_cli.py --map rednose3 monster-yolo-export --split validation
python train_specialized_detector.py --map rednose3 --imgsz 640 768 960 --epochs 40
```

For explicit classes encoded by box presets, repeat `--preset-class` in class
ID order, for example `--preset-class 1=zombie --preset-class 2=hero
--preset-class 3=lich`. Multi-class export rejects missing or unknown presets
instead of silently collapsing them into the default single `monster` class.

Training fits weights on `train`; validation selects confidence/resolution and
other configuration; the sealed `test` split is opened only after those choices
are frozen. Pending or `review_required` frames fail export. Automatic
pre-labeling is restricted to train; validation remains manually labeled.

Overlapping monster boxes are represented as independent annotation rows and
are valid YOLO input. YOLO learns one target per box even when boxes overlap.
At inference, NMS can suppress highly overlapping same-class predictions. The
default class-aware thresholds are `0.80` for Zombie and `0.50` for Hero/Lich;
they can be overridden with `--monster-nms-iou`, `--monster-hero-nms-iou`, and
`--monster-lich-nms-iou`. JSONL retains ground position, visibility,
occlusion, and review metadata that YOLO text cannot.

YOLO inference uses the model's native variable-size boxes by default. Pass
`--monster-fixed-boxes` to retain each prediction's bottom-center ground
position while replacing its extent with the class-ordered dimensions in the
map's `dataset/review_settings.json`, scaled from the map source resolution to
the input frame. In that mode, predictions for classes without a registered
preset are discarded. Use `--monster-box-presets PATH` to select another
preset file.

## Shared perception contract

The shared implementation provides `build_session_profile`, `analyze_frame`,
`analyze_video`, and stateful `FrameAnalyzer`. It includes capture-local map
registration, fixed-UI masking, character/template anchors, pluggable template
or YOLO monster proposals, camera-compensated tracking, explicit occlusion
groups, and normalized JSONL/CSV/rendered outputs.

Count fields have distinct meanings:

- `visual_detection_count`: current retained detector observations;
- `estimated_count`: confirmed persistent identities, including supported
  overlap groups;
- `count_min` / `count_max`: the evidence-supported current range.

Template and YOLO proposals are never silently fused. Select one backend per
run. Offline replay and dataset checks do not establish live-game acceptance.

## Verification

Use Python 3.14 and the pinned runtime dependencies:

```powershell
python -m pip install -r requirements-lock.txt
python -m pytest -q tests
```

Optional pre-label and training dependencies are in
`requirements-prelabel.txt` and `requirements-training.txt`.
