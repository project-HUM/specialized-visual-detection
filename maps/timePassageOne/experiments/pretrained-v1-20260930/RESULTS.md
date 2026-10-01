# timePassageOne CPU training results

Completed on 2026-09-30 from generic pretrained YOLO11n, following the APO workflow.
Training stopped at epoch **276**, after **100 epochs without improvement**.
The best checkpoint is from epoch **176**. The minimum 200 epochs was honored;
the configured maximum was 1000. Training took 65.1 minutes.

## Dataset and configuration

All 50 canonical frames remain reviewed and unchanged. Event frames 27, 42 and 43
were excluded from this experiment. The seeded split has 38 training frames
(370 mob, 37 hero) and 9 validation frames (71 mob, 9 hero), sampled from both videos.
YOLO class IDs are 0=mob and 1=hero. CPU: six threads, 640px, batch 16, workers 0;
AdamW, initial learning rate 0.001, cosine schedule over the 1000-epoch horizon.
Full settings and the strict-improvement stopping policy are in the adjacent JSON files.

## Best checkpoint evaluation

| Class | mAP50 | mAP50-95 |
| --- | ---: | ---: |
| Overall | 0.9278 | 0.7115 |
| mob | 0.9707 | 0.7218 |
| hero | 0.8850 | 0.7011 |

At confidence 0.25, class-aware matching IoU 0.50 and NMS IoU 0.70:

| Class | Matched | Extra detections | Missed |
| --- | ---: | ---: | ---: |
| mob | 67 | 9 | 4 |
| hero | 7 | 0 | 2 |

All nine validation comparison images were visually inspected. Hero misses are
frames 13 and 45, under effects or heavy UI occlusion. Mob misses involve overlap
and tiny edge/UI fragments; extras cluster around inventory/minimap regions.
Frames 24, 49 and 50 have no unmatched detections or missed labeled objects.
See visual-review.json for frame-level findings and metrics.json for exact metrics.

These are development results on a small validation set sharing video sources
with training, not independent-source or production accuracy.

## Artifacts and recovery

Snapshot location relative to the map: dataset/runs/pretrained-v1-20260930.
Best model within that snapshot: runs/pretrained-640/weights/best.pt.
Best model SHA-256: a33343f6204f8b8650938b3defe8bc2b8246d983cd2165393b973aa89dba1dea.
Prediction comparisons: evaluation/pretrained-640/overlays.
No checkpoint has been deployed to Human.

The recovery bundle includes frozen data, source weights, environment requirements,
runner, checkpoints, epoch stopping states, logs, metrics and visual comparisons.
See archive-receipt.json for the archive path and independent extraction/hash verification.
See RECOVERY-START-HERE.md for the portable runner instructions. The completed run
must not be resumed; the resume launcher is only for an interrupted unfinished run.

Eight focused training tests passed, the CPU smoke run passed, all 276 stopping
states were checked, and canonical annotations, order and presets remain byte-identical.
