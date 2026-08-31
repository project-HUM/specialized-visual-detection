# Specialized detector training

The canonical source is `annotations.jsonl`; YOLO text files are a derived
export. Training is intentionally blocked until every selected frame is
visually reviewed:

```powershell
python perception_cli.py monster-yolo-export --annotations monster_dataset\annotations.jsonl --output monster_dataset\yolo_labels
python train_specialized_detector.py --annotations monster_dataset\annotations.jsonl --data monster_dataset\dataset.yaml --imgsz 640 --epochs 40
```

The first resolution sweep should repeat the command at 640, 768, and 960,
recording precision/recall, overlap and occlusion recall, latency, and RSS.
The sealed test is not used by this training entry point.
