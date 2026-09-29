# Recover the random-initialization comparison

All 82 images and labels are included with the same 66 train / 16 validation
split as combined-v4. Original and fine-tuned comparison checkpoints are also
bundled. Training scratch mode loads the YAML architecture and pretrained=False.
Neither comparison checkpoint nor the original pretrained weights initialize it.

Use the recorded Python version in a fresh environment, then from this folder:

```powershell
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-training.txt
python run.py verify
python run.py train --mode scratch --name scratch-640
python run.py evaluate --name scratch-640
```

Use a new run name to repeat. training-configs.json records the frozen settings:
YOLO11n, 640px, CPU, batch 16, AdamW lr=0.001, seed 20260928, at most 300 epochs,
patience 75. The runner records full effective options and versions in history.jsonl.
Full per-epoch history, plots and checkpoints live under runs/scratch-640.
The runner resolves relocated dataset paths without needing original videos/logs.
Requirements-frozen.txt is the full environment audit; install only the training
requirements above. Wheels are not bundled, so package index access is needed.

Provenance includes the previous experiment and original dataset source records.
Validation remains the same small selected set; it is not a new independent test.
No model is installed into HUMAN automatically.
