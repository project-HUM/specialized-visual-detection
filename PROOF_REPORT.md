# Proof report

## Outcome

A functioning capture-local pipeline now exists for the exact pirate-ship
recording. It registers frames to the reconstructed map, emits conservative
monster proposals, recognizes reviewed character poses, detects both visual
A-wave directions in short sequential clips, and explicitly abstains on failed
registration or missing/ambiguous character evidence.

This is an investigation baseline, not a passing detector. The reviewed-ground-
truth stage has not been performed, so none of the requested accuracy targets
are reported as passed.

The specialized-monster slice is now wired into the same pipeline. Detector
boxes are conservatively deduplicated, fed to persistent camera-compensated
tracks, and exposed as separate fresh-visual and estimated counts. Established
tracks survive a many-monsters/one-detection overlap as an explicit occlusion
group. A backend-independent one-class detector contract and optional YOLO
training entry point are present, but training is correctly blocked while the
canonical labels remain pending.

## Direct observations

- The recording contains 58,743 VFR frames; the retained same-map interval ends
  at 2163.8 s. The synchronized event log contains 158 A keydowns.
- Exclusive held direction at A keydown is scarce on the left: 14 left versus
  103 right; 41 casts have no exclusive held direction at the instant.
- `right_clip.mp4` visually detects an active right wave at 30.443 s and again
  after onset at 30.859-31.290 s.
- `left_clip.mp4` visually detects an active left wave at 116.048, 116.263,
  116.680, 116.888, and 117.103 s. At 116.471 s visual motion votes right while
  the held key is left, and the record retains the visual result with a
  disagreement flag instead of silently changing it.
- The black synthetic 1920x1080 input produces `registration_failed`, unknown
  character, and unknown A-wave rather than fabricated coordinates.
- The current repeated-sprite templates visibly undercount crowded scenes. For
  example, the crowded proof frame reports two proposals while several pirate
  sprites are visible. No accuracy claim is made from these proposals.

## Measured implementation diagnostics

The 1080p runtime benchmark used 60 consecutive frames on this machine:

| Diagnostic | Result |
|---|---:|
| Mean latency | 180.11 ms |
| p50 latency | 177.27 ms |
| p95 latency | 199.84 ms |
| Throughput | 5.55 frames/s |
| Peak process RSS | 111,345,664 bytes |

The full retained overview contains 1,074 sampled observations spanning exact
source PTS 0.000-2163.789 s:

| Diagnostic (not accuracy) | Result |
|---|---:|
| Registration success | 100% |
| Visual character-facing coverage | 1.96% |
| Frames with visual active-wave output | 1.02% |
| Mean monster proposals | 0.957/frame |
| p95 monster proposals | 3/frame |

The low character coverage is expected from only seven reviewed bootstrap
poses and is the clearest evidence that guided calibration/automatic mining
must be completed before this is usable as an authoritative recognizer.

## Benchmark integrity

- `benchmark/annotations.csv` contains 360 deterministic candidates.
- Time ranges are complete and non-overlapping: 216 train, 72 validation, and
  72 sealed test.
- Every split is internally balanced across clean-candidate, A-skill, and
  heavy-effect candidate groups.
- All 360 rows remain `pending`. The synchronized `weak_*` columns are mining
  hints only. The evaluator rejects them until `review_status=reviewed`.
- The contact-sheet review showed that some no-recent-key "clean" candidates
  still contain lingering effects or open UI. This is precisely why candidate
  mining is not treated as truth.

## Specialized dataset status

`monster_dataset/annotations.jsonl` contains all 360 VFR-sampled benchmark
frames, with source-coordinate annotation fields and zero guessed instances.
All 360 are `pending`, so the report is intentionally 360 negatives only as a
dataset scaffold, not as ground truth. Eighteen contact sheets and a reviewed-
only YOLO exporter are available. `monster_dataset/dataset_report.json` records
the review and count distribution, and `monster_dataset/TRAINING.md` gives the
reproducible 640/768/960 resolution sweep command once labels are reviewed.

## Target status

| Target | Status |
|---|---|
| Monster F1 >= 0.85 at 64 px | Not evaluated; zero reviewed sealed frames |
| Monster count MAE <= 0.5 | Not evaluated; zero reviewed sealed frames |
| Facing accuracy >= 95%, coverage >= 85% | Not evaluated; observed visual coverage is only 1.96% |
| A-wave event F1 >= 0.90 | Not evaluated; short clips prove both directions only |
| A-wave direction accuracy >= 95% per side | Not evaluated; left class is scarce |

The learned ONNX monster fallback was not trained: detector training and model
selection before reviewed labels would turn weak keys/template outputs into
self-confirming ground truth. After train/validation labels are reviewed, run
the baseline evaluation first; only then is the F1/MAE gate meaningful.

## Minimum next input

1. Review the 72 sealed frames for a first honest test score, without tuning on
   them.
2. Review train/validation monster centers before deciding whether the ONNX
   fallback is required.
3. Supply or approve at least 30 clean crops for each visible facing, preferably
   using the planned two-second left and two-second right calibration after any
   outfit change.

Until then, the current outputs are useful for proposal inspection and A-wave
method proof, but not for authoritative gameplay automation.
