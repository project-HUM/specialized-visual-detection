"""Render class-aware YOLO predictions for a tiny experiment's sealed test frames."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perception.specialized_detector import YoloMonsterDetector


COLORS = {0: (70, 210, 255), 1: (255, 170, 70), 2: (255, 70, 210)}
TITLE_BAR_HEIGHT = 34
NMS_IOU_BY_CLASS = {"zombie": .80, "hero": .50, "lich": .50}


def _class_box_sizes(path: Path) -> dict[int, tuple[float, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        class_id: (float(preset["width"]), float(preset["height"]))
        for class_id, preset in enumerate(payload["box_presets"])
    }


def _draw_detection(image: np.ndarray, detection) -> None:
    class_id = int(detection.metadata["class_id"])
    class_name = str(detection.metadata["class_name"])
    color = COLORS.get(class_id, (255, 255, 255))
    x1, y1, x2, y2 = (round(value) for value in detection.bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, f"{class_name} {detection.detector_confidence:.2f}",
                (x1, max(48, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5,
                color, 1, cv2.LINE_AA)


def _prepend_caption(image: np.ndarray, caption: str) -> np.ndarray:
    """Return the complete image with a separate caption bar above it."""
    title_bar = np.zeros((TITLE_BAR_HEIGHT, image.shape[1], image.shape[2]), dtype=image.dtype)
    cv2.putText(title_bar, caption, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                .58, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.vconcat((title_bar, image))


def render(weights: Path, manifest_path: Path, output: Path, *, confidence: float,
           imgsz: int, device: str, box_presets: Path | None = None,
           source_size: tuple[int, int] = (1920, 1080)) -> dict[str, object]:
    repo_root = ROOT
    weights = weights.resolve()
    manifest_path = manifest_path.resolve()
    output = output.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    class_box_sizes = _class_box_sizes(box_presets) if box_presets else None
    detector = YoloMonsterDetector(
        weights, confidence=confidence, nms_iou=NMS_IOU_BY_CLASS["zombie"],
        class_nms_iou={1: NMS_IOU_BY_CLASS["hero"], 2: NMS_IOU_BY_CLASS["lich"]},
        input_resolution=imgsz,
        device=device, max_detections=100, class_box_sizes=class_box_sizes,
        box_size_reference=source_size if class_box_sizes else None,
    )
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    counts_by_kind: dict[str, Counter[str]] = {}
    panels = []
    for entry in manifest["test_frames"]:
        image_path = repo_root / entry["image"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read {image_path}")
        found = detector.detect(image, float(entry["time_s"]))
        detections = []
        kind_counts = counts_by_kind.setdefault(entry["kind"], Counter())
        annotated = image.copy()
        for detection in found:
            class_name = str(detection.metadata["class_name"])
            kind_counts[class_name] += 1
            _draw_detection(annotated, detection)
            detections.append({
                "class_id": int(detection.metadata["class_id"]),
                "class_name": class_name,
                "confidence": detection.detector_confidence,
                "bbox_xyxy": list(detection.bbox),
                "raw_yolo_bbox_xyxy": detection.metadata.get("raw_yolo_box"),
            })
        caption = f"{entry['id']}  t={entry['time_s']:.3f}s  detections={len(detections)}"
        annotated = _prepend_caption(annotated, caption)
        destination = output / f"{entry['id']}.jpg"
        if not cv2.imwrite(str(destination), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write {destination}")
        panel_height = round(384 * annotated.shape[0] / annotated.shape[1])
        panel = cv2.resize(annotated, (384, panel_height), interpolation=cv2.INTER_AREA)
        panels.append(panel)
        rows.append({**entry, "detections": detections,
                     "annotated_image": destination.relative_to(repo_root).as_posix()})

    columns = 5
    blank = np.zeros_like(panels[0])
    while len(panels) % columns:
        panels.append(blank.copy())
    sheet = cv2.vconcat([
        cv2.hconcat(panels[index:index + columns])
        for index in range(0, len(panels), columns)
    ])
    contact_sheet = output / "contact-sheet.jpg"
    if not cv2.imwrite(str(contact_sheet), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not write {contact_sheet}")

    summary = {
        "schema": "specialized-visual-detection.yolo-visual-results.v1",
        "weights": str(weights.resolve()), "confidence": confidence,
        "imgsz": imgsz, "device": device, "frames": rows,
        "nms_iou_by_class": NMS_IOU_BY_CLASS,
        "box_mode": "registered" if class_box_sizes else "native_yolo",
        "box_presets": str(box_presets.resolve()) if box_presets else None,
        "box_size_reference": list(source_size) if class_box_sizes else None,
        "counts_by_kind": {
            kind: dict(sorted(counts.items())) for kind, counts in sorted(counts_by_kind.items())
        },
        "contact_sheet": contact_sheet.relative_to(repo_root).as_posix(),
    }
    (output / "predictions.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=.05)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--box-presets", type=Path,
                        help="class-ordered registered sizes; omit for native YOLO boxes")
    parser.add_argument("--source-size", type=int, nargs=2, default=(1920, 1080),
                        metavar=("WIDTH", "HEIGHT"))
    args = parser.parse_args()
    summary = render(args.weights, args.manifest, args.output,
                     confidence=args.confidence, imgsz=args.imgsz, device=args.device,
                     box_presets=args.box_presets, source_size=tuple(args.source_size))
    print(json.dumps({
        "frames": len(summary["frames"]),
        "counts_by_kind": summary["counts_by_kind"],
        "box_mode": summary["box_mode"],
        "contact_sheet": summary["contact_sheet"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
