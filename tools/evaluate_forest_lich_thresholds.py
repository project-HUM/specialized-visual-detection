"""Compare Lich counts at fixed confidence on disjoint train/validation cohorts."""
import argparse
import json
from pathlib import Path
import sys

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.forest_training_snapshot_runner import sha, event


def iou(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "maps/forest-of-dead-trees-2/dataset/runs/combined-v4-20260928")
    parser.add_argument("--name", default="finetune-640")
    args = parser.parse_args()
    run = args.snapshot.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    frames = [f for f in manifest["frames"] if f["split"] == "validation" or f["label_origin"] == "batch5_human_confirmed"]
    report = {"method": "YOLO raw detections at imgsz=640; greedy confidence-order IoU>=0.5 matches against clipped YOLO labels",
              "warning": "Training cohort results measure fit, not generalization. Does not reproduce HUMAN temporal gating or preset-box postprocessing.",
              "models": {}}
    models = [("parent", run / "models/parent-best.pt")]
    if (run / "models/finetuned-best.pt").is_file():
        models.append(("finetuned", run / "models/finetuned-best.pt"))
    models.append(("candidate", run / "runs" / args.name / "weights/best.pt"))
    for name, weights in models:
        results = YOLO(str(weights)).predict(source=[str(run / f["image_path"]) for f in frames],
                                            imgsz=640, conf=.05, device="cpu", batch=16, verbose=False)
        model_report = {"weights_sha256": sha(weights), "thresholds": {}}
        for threshold in (.25, .4):
            cohorts = {}
            for frame, result in zip(frames, results):
                cohort = "new_training_fit" if frame["split"] == "train" else (
                    "new_validation" if frame["label_origin"] == "batch5_human_confirmed" else "legacy_validation")
                counts = cohorts.setdefault(cohort, {"frames": 0, "tp": 0, "fp": 0, "fn": 0, "frames_with_false_lich": 0})
                truth = []
                for line in (run / frame["label_path"]).read_text().splitlines():
                    c, x, y, w, h = map(float, line.split())
                    if c == 2:
                        truth.append([x-w/2, y-h/2, x+w/2, y+h/2])
                predictions = [(float(conf), box.tolist()) for c, conf, box in zip(
                    result.boxes.cls, result.boxes.conf, result.boxes.xyxyn) if int(c) == 2 and float(conf) >= threshold]
                matched = set()
                fp = 0
                for confidence, box in sorted(predictions, reverse=True):
                    candidates = [(iou(box, gt), i) for i, gt in enumerate(truth) if i not in matched]
                    score, index = max(candidates, default=(0., -1))
                    if score >= .5:
                        matched.add(index)
                    else:
                        fp += 1
                counts["frames"] += 1
                counts["tp"] += len(matched)
                counts["fp"] += fp
                counts["fn"] += len(truth)-len(matched)
                counts["frames_with_false_lich"] += int(fp > 0)
            model_report["thresholds"][str(threshold)] = cohorts
        report["models"][name] = model_report
    path = run / "evaluation/lich-fixed-thresholds.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    event(run, "fixed_threshold_evaluation_completed", report=str(path.relative_to(run)), sha256=sha(path))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
