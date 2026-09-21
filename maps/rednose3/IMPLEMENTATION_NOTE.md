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

## Learned-detector and occlusion extension

- `FrameAnalyzer` now receives a `SpecializedMonsterDetector` by dependency
  injection; template and YOLO modes remain separate and keep explicit
  provenance.
- Train/validation review is supported by an interactive VFR-aware three-frame
  UI. Strict export and evaluation reject pending/malformed labels and reject
  the sealed test split during development.
- Tracks now distinguish fresh visual timestamps from support timestamps.
  Persistent occlusion groups preserve confirmed counts through repeated merged
  detections and recover members when detections separate.
- Specialized evaluation reports raw visual and persistent-count results
  independently and writes a failure queue for the later active-learning loop.

## Pre-training semantic hardening

- YOLO NMS defaults to a permissive `0.90`; custom multi-signal deduplication
  owns duplicate decisions after backend NMS.
- Tracks are expired before association, preventing an already-invalid identity
  from acquiring an enlarged motion gate and attaching to a new monster.
- New occlusion groups use intersecting predicted geometry and a plausible
  horizontally contiguous member span. Nearby tracks admitted only by the
  broad motion gate are not automatically absorbed.
- Group formation and continued support are decided from observations locally
  relevant to the candidate members. Global detection/track totals do not
  suppress a valid many-to-one group when an unrelated false positive appears
  elsewhere.
- Count bounds use independent current evidence units: a supported group adds
  one lower-bound unit while each live confirmed member remains represented in
  the upper bound. Unsupported temporal holds add no lower-bound evidence.
- Debug observations retain preprocessing, exclusion, deduplication,
  association, lifetime, support, and group provenance for validation failure
  analysis.

These changes freeze tracker/detector semantics before labeling. They do not
measure learned-detector accuracy; train and validation annotations are still
required, and the sealed test remains untouched.
