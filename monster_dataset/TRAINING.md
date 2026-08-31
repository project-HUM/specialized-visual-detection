# Specialized detector training

The canonical source is `annotations.jsonl`; YOLO text files are a derived
export. Training is intentionally blocked until every selected frame is
visually reviewed:

```powershell
python perception_cli.py monster-review-report
python perception_cli.py monster-yolo-export --split train
python perception_cli.py monster-yolo-export --split validation
python -m pip install -r requirements-training.txt
python train_specialized_detector.py --imgsz 640 768 960 --epochs 40
```

Training device selection is automatic unless `--device cpu` or `--device 0`
is supplied. Each resolution gets a separate run directory and records backend
metrics and timing. The sealed test is rejected by development export and
evaluation.
