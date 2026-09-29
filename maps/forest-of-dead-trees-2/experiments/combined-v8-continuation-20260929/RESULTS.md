# Forest continuation until 300 epochs without improvement

Training stopped at epoch 664: exactly 300 consecutive epochs without a
strict improvement after the historical best at epoch 364 (mAP50-95 0.75835).
The continuation computed 173 epochs (492-664), taking about 273 seconds.

The completed 500-epoch run had stripped optimizer state from last.pt. Resume
used epoch490.pt, which retains optimizer/scaler/EMA through completed epoch
491. Epochs 492-500 were replayed. The RNG state is not serialized, so this is
an optimizer-state continuation with a short replay, not a bit-identical
extension from epoch 500. The original experiment remains intact.

Historical best score and epoch were restored before training, so patience
was not reset to zero on resume. Equal scores do not reset the counter.
The original 500-epoch cosine learning-rate schedule was retained, then held
at its final learning rate of 0.00001. Mosaic augmentation remained closed.
Images, labels, train/validation split, batch size and GPU remain unchanged.
A high implementation ceiling of 100000 allowed patience to determine the end;
that ceiling was not reached. The final terminal-resumable.pt retains optimizer,
scaler and EMA before Ultralytics strips last.pt.

All 499 tensors of the retained best model exactly match the prior best.
The checkpoint file hash differs because final metadata includes the new history.
Same 22-image standalone validation: overall mAP50-95 0.7585, Lich mAP50-95
0.7375. At confidence 0.25 and match IoU 0.5: 13 Lich true positives,
1 false positive and 1 miss (batch-6 subset 4/1/0), unchanged from the 500 run.
The prior visual audit remains applicable to the identical best model; no new
candidate was selected. This remains a development validation set, not live
acceptance or an independent final test. No model was deployed.

results.csv retains epochs 1-491 from the source and appends the continuation;
its time column restarts for epoch 492. Original epochs 492-500 remain in the
original archived experiment. The original best, resume input, terminal full
checkpoint, outputs, logs, exact labels/images and environment pins are preserved.
See continuation-verification.json and archive-receipt.json. The recovery ZIP
is extracted and checked file-by-file on the same local volume.

Two focused continuation tests passed: the cosine learning rate stays at its
floor after epoch 500, and equal scores retain the original patience anchor.
Runtime assertions confirmed the restored patience and the exact stopping rule.
