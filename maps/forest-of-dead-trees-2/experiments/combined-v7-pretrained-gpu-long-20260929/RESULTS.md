# Forest batches 0-6: extended pretrained GPU training

Completed all 500 epochs in 840.926 seconds (14m 01s) on RTX 5060 Ti.
Patience was 300 epochs; the best checkpoint was epoch 364. The last checkpoint
had lower validation mAP50-95 (0.73434), so use best.pt for this comparison.
Initialization was original COCO-pretrained yolo11n.pt, not Forest weights.

The 111 reviewed Forest images / 757 boxes are frozen at 89 train / 22 validation.
One town/dialog frame was excluded. Images, corrected labels, train/validation
assignments, initializer and seed match the shorter GPU experiment exactly.
Only the maximum epochs and patience changed (200/50 to 500/300); changing the
maximum also stretches the cosine learning-rate schedule. This is not a resume.
Older temporal groups and batch-6 source-run splits remain isolated.

| Same 22 validation images | Existing combined-v3 | Short GPU | Long GPU |
|---|---:|---:|---:|
| Overall mAP50-95 | 0.7504 | 0.7364 | 0.7585 |
| Lich mAP50-95 | 0.6904 | 0.7162 | 0.7375 |
| Lich true positives at conf 0.25 | 14 | 12 | 13 |
| Lich false positives at conf 0.25 | 2 | 0 | 1 |
| Lich misses at conf 0.25 | 0 | 2 | 1 |
| Batch-6 held-out Lich TP / FP / FN | 4 / 0 / 0 | 3 / 0 / 1 | 4 / 1 / 0 |

Fixed-threshold matches require IoU >= 0.5 and use NMS IoU 0.7. All models use
identical settings within each comparison. Standard validation mAP and its
reported precision/recall summarize different operating points from the
single-image conf-0.25 audit; do not substitute those values for these counts.
Standalone overall mAP50-95 is 0.758470; in-training best was 0.75835.

Visual inspection covered all 22 candidate prediction overlays and full-size
error/recovery examples. The long run recovers the spell-obscured Lich in
codex-human-batch6-evidence-0058 at confidence 0.87. It still misses the real
Lich clipped at the right edge in codex-human-batch5-evidence-0049. It adds one
0.280 false positive over a narrow portal/background strip at the left border
in codex-human-batch6-evidence-0076. Both existing-model false positives are absent.

Longer training improved aggregate metrics and difficult-Lich recall over the
short run, with one additional false positive. This remains a tradeoff against
the existing model, which missed none at confidence 0.25. These selected images
are a development validation set repeatedly used for model selection, not an
independent final test or live HUMAN acceptance. No detector was deployed.

Checkpoints, curves, exact inputs, hashes, package pins, console logs, prediction
overlays and comparison reports are retained. The complete recovery ZIP is
extracted and checked file-by-file; archive-receipt.json identifies it. The
backup resides on the same volume, not off-device storage.

The dedicated Python 3.12 CUDA environment passed tensor/backpropagation,
torchvision NMS, a smoke epoch and three training tests. All 51 focused checks
passed in the original environment. One unrelated reviewer/prelabel test fails
in the isolated CUDA environment because OpenCV 5 changes HoughLinesP shape;
use the unchanged original environment for label review.
