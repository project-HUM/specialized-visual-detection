# Specialized monster perception implementation note

## Reused

- `perception.core.SessionProfile`, VFR presentation timestamps, map registration,
  stable-background residuals, fixed-UI masking, character localization, and the
  stateful `FrameAnalyzer` API.
- `benchmark/annotations.csv` and its 216/72/72 compatibility manifest. The
  sealed split remains read-only for tuning.
- `perception.render` JSONL/CSV/annotated-output conventions and the existing
  pytest suite.

## Extended

- `MonsterTracker` now has explicit track states, conservative box
  deduplication, many-tracks/one-detection occlusion groups, uncertainty
  bounds, and detector-cadence-safe aging.
- `FrameAnalyzer` reports fresh visual detections separately from the estimated
  count supplied by persistent tracks.
- The capture CLI gains deterministic specialized-dataset sampling/reporting
  commands.

## Newly introduced

- `monster_dataset/`: canonical reviewed annotation schema, VFR sampler,
  annotation I/O, contact sheets, validation/reporting, and YOLO export.
- `perception/specialized_detector.py`: backend-independent one-class detector
  types and optional training/runtime adapters. OWLv2 remains a proposal or
  disagreement source; it is not fine-tuned here.

All new artifacts are capture-local and preserve explicit pending/reviewed
provenance; pending proposals are never treated as ground truth.
