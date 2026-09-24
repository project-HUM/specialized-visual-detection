"""Validate the combined Forest checkpoint and write its evidence report."""
from __future__ import annotations

import argparse
import json
import numbers
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perception.core import sha256_file


def _numeric(values) -> list[float]:
    return [float(value) for value in values]


def main() -> int:
    default_run = ROOT / "maps/forest-of-dead-trees-2/dataset/runs/combined-v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=default_run)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "maps/forest-of-dead-trees-2/dataset/yolo_dataset/dataset.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from ultralytics import YOLO

    run = args.run.resolve()
    weights = run / "training-640/weights/best.pt"
    manifest_path = run / "manifest.json"
    validation_dir = run / "validation-best"
    result = YOLO(str(weights)).val(
        data=str(args.data.resolve()), imgsz=640, device=args.device,
        workers=0, plots=False, project=str(run), name=validation_dir.name,
        exist_ok=True, verbose=False,
    )
    names = result.names
    class_ids = sorted(int(class_id) for class_id in names)
    per_class = {
        str(names[class_id]): {
            "precision": _numeric(result.box.p)[class_id],
            "recall": _numeric(result.box.r)[class_id],
            "map50": _numeric(result.box.ap50)[class_id],
            "map50_95": _numeric(result.box.maps)[class_id],
        }
        for class_id in class_ids
    }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction_runs = {}
    for name in ("predictions-conf025", "predictions-conf005"):
        path = run / "test" / name / "predictions.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        prediction_runs[name] = {
            "confidence": payload["confidence"],
            "counts_by_kind": payload["counts_by_kind"],
            "lich_detections_by_frame": {
                row["id"]: sum(detection["class_name"] == "lich" for detection in row["detections"])
                for row in payload["frames"]
                if row["kind"].startswith("lich")
            },
            "contact_sheet": payload["contact_sheet"],
        }

    minimap_scores = [float(row["minimap_correlation"]) for row in manifest["test_frames"]]
    lich_interval_visual_audit = {
        "combined-test-lich_early-00": "no_visible_lich",
        "combined-test-lich_early-01": "visible_lich",
        "combined-test-lich_late-00": "visible_lich",
        "combined-test-lich_late-01": "no_visible_lich",
    }
    report = {
        "schema": "specialized-visual-detection.combined-learning-result.v1",
        "warning": (
            "Validation contains only five images and 34 boxes. The 20 test frames are unlabeled, "
            "so test predictions are visual evidence rather than accuracy measurements."
        ),
        "dataset": {
            "train_frames": manifest["exports"]["train"]["frames"],
            "train_instances": manifest["exports"]["train"]["instances"],
            "validation_frames": manifest["exports"]["validation"]["frames"],
            "validation_instances": manifest["exports"]["validation"]["instances"],
            "excluded_labeled_frames": manifest["eligibility"]["excluded_labeled_frames"],
        },
        "minimap_gate": {
            "threshold": manifest["eligibility"]["minimum_reference_correlation"],
            "test_frames": len(minimap_scores),
            "test_minimum": min(minimap_scores),
            "test_maximum": max(minimap_scores),
        },
        "training": {
            "model": "YOLO11n",
            "epochs": 200,
            "imgsz": 640,
            "device": args.device,
            "weights": str(weights),
            "weights_sha256": sha256_file(weights),
            "overall": {
                str(key): float(value)
                for key, value in result.results_dict.items()
                if isinstance(value, numbers.Real)
            },
            "per_class": per_class,
        },
        "heldout_visual_results": {
            "manual_lich_presence_audit": lich_interval_visual_audit,
            "interpretation": (
                "At confidence 0.25 the clear late Lich is detected and the clear early Lich is missed. "
                "At confidence 0.05 both visible Liches are detected, but the early frame without a "
                "visible Lich receives a false-positive Lich and the late Lich receives duplicate boxes."
            ),
            "prediction_runs": prediction_runs,
        },
    }
    destination = run / "experiment-report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
