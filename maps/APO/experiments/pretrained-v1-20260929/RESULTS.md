# APO pretrained-v1 results

Completed 200 epochs in approximately 27.5 minutes after restarting with verified
six-thread CPU execution. The earlier eight-thread attempt is retained under
runs/interrupted-thread-override; history.jsonl records the reason and runner correction.

The best training-epoch mAP50-95 was 0.78658 at epoch 173. The saved best.pt was
independently evaluated at mAP50 0.96464 and mAP50-95 0.78480. Use the latter
saved-checkpoint evaluation for reporting. The complete per-class results are in
evaluation/pretrained-640/metrics.json.

At confidence 0.25 and matching IoU 0.50:

| Class | Correct | Extra | Missed |
| --- | ---: | ---: | ---: |
| mob | 112 | 7 | 3 |
| hero | 15 | 1 | 0 |
| special | 10 | 0 | 4 |

The 15 validation images all come from sample1, held out as a complete source.
All four special misses are neighboring attack-heavy frames near the end of
sample1. No clear NPC-centered detection was observed in the visual review at
0.25; this small set does not establish general NPC discrimination.
Clean overlays and truth/prediction pairs are in evaluation/pretrained-640/overlays.

mAP50-95 averages per-class AP at IoU thresholds 0.50, 0.55, ..., 0.95: thirty AP
values across our three classes. AP integrates an interpolated precision-recall
envelope over 101 recall points. It is not the percentage of objects detected.
The fixed-confidence counts above are a separate operating-point evaluation
using class-aware greedy confidence-order IoU matching, not the validator's
interpolated max-F1 precision/recall values.

Best weights: runs/pretrained-640/weights/best.pt.
The original 80 reviewed annotation lines, presets, and display order are
unchanged. The town frame remains archived under excluded/. This model has not
been deployed into HUMAN. See RECOVERY-START-HERE.md for portable rerun commands.
