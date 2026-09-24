"""Validate the combined-v3 Forest checkpoint and write its evidence report."""
from __future__ import annotations

import argparse
import csv
import json
import numbers
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perception.core import sha256_file


def _numeric(values) -> list[float]:
    return [float(value) for value in values]


def _training_summary(run: Path) -> dict[str, object]:
    results_path = run / "training-640/results.csv"
    rows = list(csv.DictReader(results_path.open(encoding="utf-8")))
    fitness_key = "metrics/mAP50-95(B)"
    best = max(rows, key=lambda row: float(row[fitness_key]))
    return {
        "epochs_completed": len(rows),
        "best_epoch": int(best["epoch"]),
        "best_training_row": {
            key.strip(): float(value)
            for key, value in best.items()
            if key.strip() != "epoch" and value.strip()
        },
    }


def main() -> int:
    default_run = ROOT / "maps/forest-of-dead-trees-2/dataset/runs/combined-v3"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=default_run)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from ultralytics import YOLO

    run = args.run.resolve()
    data = run / "yolo_dataset/dataset.yaml"
    weights = run / "training-640/weights/best.pt"
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    model = YOLO(str(weights))
    result = model.val(
        data=str(data), imgsz=640, device=args.device, workers=0, plots=False,
        project=str(run), name="validation-best", exist_ok=True, verbose=False,
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

    prediction_runs = {}
    for name in ("predictions-conf025", "predictions-conf005"):
        payload = json.loads((run / "test" / name / "predictions.json").read_text(encoding="utf-8"))
        per_frame = {
            row["id"]: {
                class_name: sum(detection["class_name"] == class_name for detection in row["detections"])
                for class_name in ("zombie", "hero", "lich")
            }
            for row in payload["frames"]
        }
        prediction_runs[name] = {
            "confidence": payload["confidence"],
            "nms_iou_by_class": payload["nms_iou_by_class"],
            "counts_by_kind": payload["counts_by_kind"],
            "counts_by_frame": per_frame,
            "frames_with_lich_detections": [
                frame_id for frame_id, counts in per_frame.items() if counts["lich"]
            ],
            "contact_sheet": payload["contact_sheet"],
        }

    minimap_scores = [float(row["minimap_correlation"]) for row in manifest["test_frames"]]
    torch_model = model.model
    report = {
        "schema": "specialized-visual-detection.combined-learning-result.v3",
        "warning": (
            "The validation set has 12 labeled images. The 15 Sample4 and 15 Sample6 random "
            "test frames are unlabeled, so their predictions are visual evidence rather than "
            "accuracy measurements."
        ),
        "dataset": {
            "train": manifest["exports"]["train"],
            "validation": manifest["exports"]["validation"],
            "split_strategy": manifest["split_strategy"],
            "excluded_labeled_frames": manifest["eligibility"]["excluded_labeled_frames"],
            "snapshot_authorization": manifest["confirmation"],
        },
        "minimap_gate": {
            "threshold": manifest["eligibility"]["minimum_minimap_reference_correlation"],
            "invalid_minimap_frames_allowed": False,
            "test_frames": len(minimap_scores),
            "test_minimum": min(minimap_scores),
            "test_maximum": max(minimap_scores),
        },
        "training": {
            "model": "YOLO11n",
            "parameters": sum(parameter.numel() for parameter in torch_model.parameters()),
            "imgsz": 640,
            "device": args.device,
            "weights": str(weights),
            "weights_sha256": sha256_file(weights),
            **_training_summary(run),
            "fresh_validation_overall": {
                str(key): float(value)
                for key, value in result.results_dict.items()
                if isinstance(value, numbers.Real)
            },
            "fresh_validation_per_class": per_class,
            "fresh_validation_speed_ms_per_image": {
                str(key): float(value) for key, value in result.speed.items()
            },
        },
        "random_visual_test": {
            "selection": manifest["test_selection"],
            "prediction_runs": prediction_runs,
            "visual_audit": {
                "scope": (
                    "Both 30-frame contact sheets were inspected. Every normal-confidence Lich "
                    "frame, every additional low-confidence Lich frame, and all 15 Sample6 frames "
                    "were then inspected at full resolution."
                ),
                "observations": [
                    "At confidence 0.25, all five Sample4 Lich detections correspond to visible full, obscured, or edge/UI-cropped Liches; Sample6 has no Lich detection at that threshold.",
                    "At confidence 0.05, Sample4 random-07 and random-14 and Sample6 random-12 add visually real but heavily obscured or edge-cropped Liches.",
                    "Normal-confidence Zombie boxes are generally aligned, including partial sprites at frame boundaries, but a few non-monster UI lookalikes remain on the minimap/action bar.",
                    "Sample6 random-12 and random-13 each contain an extra Hero false positive on an inventory item icon while the real Hero is also detected.",
                    "Lowering confidence from 0.25 to 0.05 raises total proposals from 187 to 281 and introduces substantial extra low-confidence clutter, especially in Sample6.",
                    "Class-aware NMS uses IoU 0.80 for Zombie and 0.50 for Hero/Lich; no obvious normal-confidence Hero/Lich duplicate survived the visual audit."
                ],
                "nms_iou_by_class": {
                    "zombie": 0.80,
                    "hero": 0.50,
                    "lich": 0.50
                },
                "preferred_display_threshold": 0.25,
                "diagnostic_threshold": 0.05,
                "limitation": (
                    "This is a visual audit of unlabeled frames. It does not establish precision or "
                    "recall for either random test source."
                ),
            },
        },
    }
    destination = run / "experiment-report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
