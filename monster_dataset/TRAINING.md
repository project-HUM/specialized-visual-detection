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

For the first 640/768/960 comparison, keep the tracker fixed (`min_hits=2`,
`max_gap_s=0.55`, supported occlusion `2.5 s`) and use permissive YOLO NMS
around `0.90`. Sweep detector confidence on validation in the planned
`0.15-0.25` range, but do not retune tracker behavior independently per input
resolution. This isolates detector improvements from tracking changes.
