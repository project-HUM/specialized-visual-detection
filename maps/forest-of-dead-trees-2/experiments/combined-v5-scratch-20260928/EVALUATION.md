# Repeat the comparison after recovery

From the recovered folder, after installing requirements-training.txt:

```powershell
python compare.py --name scratch-640
python compare_lich.py --name scratch-640
```

compare.py evaluates original, fine-tuned and scratch checkpoints on the same
16 validation images. It refuses to replace an existing comparison JSON; retain
existing evidence and use a separate recovered copy for repeat evaluation.
compare_lich.py reports Lich matches at confidence 0.25 and 0.40, separately
for legacy validation, new validation and new training-fit frames. The latter
is not generalization evidence. The scripts use raw YOLO boxes with IoU 0.5
matching, not HUMAN temporal gates. The same validation set has been reused
across experiments, so this is a development comparison, not a sealed test.
