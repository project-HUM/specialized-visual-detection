# timePassageOne CPU training recovery

This portable bundle includes frozen images, clipped YOLO labels, all reviewed
source annotations (including excluded event frames), pretrained weights and runner.
Original videos and the repository are not needed. Use the Python version in
environment.json. Install the recorded CPU PyTorch wheel from its CPU index,
then requirements-training.txt; requirements-frozen.txt records all dependencies.

    python runner.py verify
    python runner.py smoke --name smoke-640
    python runner.py train --name pretrained-640
    python runner.py train --name pretrained-640 --resume
    python runner.py evaluate --name pretrained-640

Use a new name for repeats. Resume requires an unfinished last.pt and its matching
runs/<name>/stopping/epoch-NNNNNN.json; copy the entire bundle. Stopping state
ahead of the checkpoint is ignored. Minimum 200 epochs, patience 100, cap 1000;
strict improvement in validation mAP50-95. Ties and zero plateaus do not reset
patience. The 200-epoch boundary does not reset the accumulated plateau.
CPU, six threads, workers 0, batch 16, 640px, AdamW lr=0.001, seed 20260930.
Schedule: cosine decay over the 1000-epoch cap; mosaic closes at epoch 991 if reached.

38 training frames and 9 validation frames share video sources. Event frames
27, 42 and 43 are excluded. Classes: 0 mob, 1 hero. Canonical data stays unchanged.
Evaluation reports mAP, per-class metrics and confidence .25 / matching IoU .50
counts with NMS IoU .70 and full-frame overlays. Inspect all nine comparisons and
write evaluation/<name>/visual-review.json, then run:

    python runner.py archive --name pretrained-640

The manifest records the local backup directory. Archive extraction is verified
by hash. This is a same-volume backup, not off-device protection. No automatic deployment.
