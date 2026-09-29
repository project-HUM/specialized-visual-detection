# Forest batches 0-6: pretrained GPU experiment

Completed 110 epochs in 188 seconds on RTX 5060 Ti 16 GB; early stopping
selected epoch 60. Initialization was original COCO-pretrained yolo11n.pt.
The dataset contains 111 reviewed Forest images, 757 boxes, split 89/22.
The one town/dialog frame was excluded. Current reviewed labels were used;
legacy experiment split assignments and batch-6 source-run splits were retained.

| Same 22 validation images | Existing combined-v3 | New candidate |
|---|---:|---:|
| Overall mAP50-95 | 0.7504 | 0.7364 |
| Lich mAP50-95 | 0.6904 | 0.7162 |
| Lich true positives at conf 0.25 | 14 | 12 |
| Lich false positives at conf 0.25 | 2 | 0 |
| Lich misses at conf 0.25 | 0 | 2 |
| Batch-6 held-out Lich TP / FP / FN | 4 / 0 / 0 | 3 / 0 / 1 |

Fixed-threshold matches require IoU >= 0.5. Validator precision/recall uses
its own operating point and differs from these fixed-threshold counts.
The two misses are a right-edge clipped Lich and a spell-obscured Lich.
This is a precision/recall tradeoff, not a clear replacement win. No HUMAN
model, detector threshold, or map configuration was changed.

The standalone post-training comparison is authoritative for the table.
In-training best mAP50-95 was 0.7372; automatic final validation and standalone
validation differ slightly due to evaluation batching/numerical execution.
Both comparison models were evaluated with the same settings and dataset.
These selected validation images are a development set, not live acceptance.

CUDA tensor/backpropagation and torchvision NMS passed, as did a one-epoch
smoke train. Fifty-one focused tests passed in the original environment;
the dedicated CUDA environment passes the three training tests but one unrelated
reviewer test exposes an OpenCV 5 HoughLinesP shape incompatibility. Use the
original environment for review; its installation was unchanged.

Recovery inputs, exact images/labels, initialization, environment pins, logs,
curves, periodic checkpoints, best/last checkpoints, and prediction overlays
are preserved in the snapshot and the verified complete archive.
