"""Evaluate original, fine-tuned and scratch checkpoints on one frozen dataset."""
import argparse
import json
from pathlib import Path
import sys

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.forest_training_snapshot_runner import dataset_yaml, event, metrics, sha, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "maps/forest-of-dead-trees-2/dataset/runs/combined-v5-scratch-20260928")
    parser.add_argument("--name", default="scratch-640")
    args = parser.parse_args()
    root = args.snapshot.resolve()
    verify(root)
    output = root / "evaluation" / f"{args.name}-comparison.json"
    if output.exists():
        raise FileExistsError(output)
    report = {"evaluation_set": "Same 16 held-out frames: 12 legacy and 4 new run-isolated frames",
              "caution": "Small selected validation set reused across experiments; not an independent production test.",
              "candidate_initialization": "random weights; pretrained=False",
              "models": {}}
    for name, weights in [("parent", root / "models/parent-best.pt"),
                          ("finetuned", root / "models/finetuned-best.pt"),
                          ("candidate", root / "runs" / args.name / "weights/best.pt")]:
        result = YOLO(str(weights)).val(data=str(dataset_yaml(root)), imgsz=640, batch=16,
            device="cpu", workers=0, plots=True, project=str(root / "evaluation"),
            name=f"{args.name}-{name}", exist_ok=False, verbose=False)
        report["models"][name] = {"weights": str(weights.relative_to(root)),
                                 "sha256": sha(weights), **metrics(result)}
    output.write_text(json.dumps(report, indent=2)+"\n")
    event(root, "evaluation_completed", report=str(output.relative_to(root)), sha256=sha(output))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
