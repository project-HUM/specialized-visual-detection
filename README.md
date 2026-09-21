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

## Forest of Dead Trees 2 pilot

The current Forest workspace is intentionally a five-frame labeling pilot, not
a training dataset. It is selected deterministically from direct minimap-marker
observations with seed `0`, a minimum 30-second gap, and no UI/menu filter:

```powershell
python perception_cli.py --map forest-of-dead-trees-2 monster-pilot-sample
python perception_cli.py --map forest-of-dead-trees-2 monster-contact-sheet --split pilot
maps\forest-of-dead-trees-2\run.bat
```

The generated annotations use split `pilot`. Review and contact-sheet tools
accept that split, while YOLO export accepts only `train` and `validation`.
This keeps labeling experiments out of future training by construction. See
`maps/forest-of-dead-trees-2/README.md` for provenance and selected frames.

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

Training fits weights on `train`; validation selects confidence/resolution and
other configuration; the sealed `test` split is opened only after those choices
are frozen. Pending or `review_required` frames fail export. Automatic
pre-labeling is restricted to train; validation remains manually labeled.

Overlapping monster boxes are represented as independent annotation rows and
are valid YOLO input. YOLO learns one target per box even when boxes overlap.
At inference, NMS can suppress highly overlapping same-class predictions, so
this project keeps detector NMS permissive (`0.90` by default) and leaves the
conservative multi-signal deduplicator authoritative. JSONL retains ground
position, visibility, occlusion, and review metadata that YOLO text cannot.

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
