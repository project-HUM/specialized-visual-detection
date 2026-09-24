"""Backend-independent monster detector implementations."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
import cv2
import numpy as np

@dataclass(frozen=True)
class MonsterDetection:
    bbox: tuple[float, float, float, float]
    detector_confidence: float
    ground_position: tuple[float, float]
    backend: str
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"box": list(self.bbox), "score": self.detector_confidence,
                "ground_position": list(self.ground_position), "backend": self.backend,
                "provenance": f"visual_{self.backend}", "metadata": self.metadata or {}}

class SpecializedMonsterDetector(Protocol):
    backend: str
    def detect(self, image: np.ndarray, timestamp: float | None = None) -> list[MonsterDetection]: ...

class TemplateMonsterDetector:
    backend = "template"
    def __init__(self, templates: list[np.ndarray], threshold: float = .42):
        self.templates, self.threshold = templates, threshold

    def detect(self, image: np.ndarray, timestamp: float | None = None) -> list[MonsterDetection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        edge = cv2.Canny(gray, 55, 140); output: list[MonsterDetection] = []
        for template in self.templates:
            if template.shape[0] > edge.shape[0] or template.shape[1] > edge.shape[1]: continue
            scores = cv2.matchTemplate(edge, template, cv2.TM_CCOEFF_NORMED)
            maxima = scores == cv2.dilate(scores, np.ones((9, 9), np.uint8))
            ys, xs = np.where(maxima & (scores >= self.threshold))
            for y, x in zip(ys, xs):
                box = (float(x), float(y), float(x + template.shape[1]), float(y + template.shape[0]))
                output.append(MonsterDetection(box, float(scores[y, x]), ((box[0]+box[2])/2, box[3]),
                                                self.backend, {"timestamp": timestamp}))
        return output

class YoloMonsterDetector:
    """Lazy Ultralytics adapter with class-aware backend NMS.

    Ultralytics first retains candidates at the least restrictive configured
    IoU. A second pass applies the requested per-class thresholds.
    """
    backend = "yolo"
    def __init__(self, weights: Path | str, *, confidence: float = .21, nms_iou: float = .80,
                 input_resolution: int = 768, device: str | int | None = None, max_detections: int = 100,
                 class_nms_iou: dict[int, float] | None = None,
                 class_box_sizes: dict[int, tuple[float, float]] | None = None,
                 box_size_reference: tuple[int, int] | None = None):
        if not 0 <= confidence <= 1 or not 0 <= nms_iou <= 1: raise ValueError("confidence and nms_iou must be within [0, 1]")
        if class_nms_iou is None:
            class_nms_iou = {1: .50, 2: .50}
        if any(class_id < 0 or not 0 <= threshold <= 1
               for class_id, threshold in class_nms_iou.items()):
            raise ValueError("class_nms_iou requires non-negative class IDs and thresholds within [0, 1]")
        if input_resolution <= 0 or max_detections <= 0: raise ValueError("input_resolution and max_detections must be positive")
        if class_box_sizes is not None:
            if not class_box_sizes or any(class_id < 0 or width <= 0 or height <= 0
                                          for class_id, (width, height) in class_box_sizes.items()):
                raise ValueError("class_box_sizes requires non-negative class IDs and positive sizes")
            if box_size_reference is None or any(value <= 0 for value in box_size_reference):
                raise ValueError("box_size_reference requires positive width and height")
        try: from ultralytics import YOLO
        except ImportError as exc: raise RuntimeError("YoloMonsterDetector requires the optional ultralytics package") from exc
        self.model = YOLO(str(weights)); self.weights = str(weights); self.confidence = confidence
        self.nms_iou = nms_iou; self.class_nms_iou = dict(class_nms_iou)
        self.input_resolution = input_resolution; self.device = device; self.max_detections = max_detections
        self.class_box_sizes = class_box_sizes
        self.box_size_reference = box_size_reference

    def _registered_box(self, box: np.ndarray, class_id: int,
                        image_shape: tuple[int, ...]) -> tuple[float, float, float, float] | None:
        if self.class_box_sizes is None:
            return tuple(float(value) for value in box)
        source_size = self.class_box_sizes.get(class_id)
        if source_size is None:
            return None
        assert self.box_size_reference is not None
        image_height, image_width = image_shape[:2]
        width = source_size[0] * image_width / self.box_size_reference[0]
        height = source_size[1] * image_height / self.box_size_reference[1]
        ground_x, ground_y = (float(box[0]) + float(box[2])) / 2.0, float(box[3])
        return (ground_x - width / 2.0, ground_y - height,
                ground_x + width / 2.0, ground_y)

    def _class_aware_nms(self, xyxy: np.ndarray, confidence: np.ndarray,
                         classes: np.ndarray) -> np.ndarray:
        kept: list[int] = []
        thresholds = getattr(self, "class_nms_iou", {})
        for class_id in np.unique(classes):
            candidates = np.flatnonzero(classes == class_id)
            candidates = candidates[np.argsort(-confidence[candidates], kind="stable")]
            threshold = thresholds.get(int(class_id), self.nms_iou)
            while len(candidates):
                current = int(candidates[0])
                kept.append(current)
                candidates = candidates[1:]
                if not len(candidates):
                    break
                box = xyxy[current]
                others = xyxy[candidates]
                left_top = np.maximum(box[:2], others[:, :2])
                right_bottom = np.minimum(box[2:], others[:, 2:])
                intersection = np.prod(np.maximum(0.0, right_bottom - left_top), axis=1)
                box_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
                other_area = np.maximum(0.0, others[:, 2] - others[:, 0]) * np.maximum(
                    0.0, others[:, 3] - others[:, 1]
                )
                union = box_area + other_area - intersection
                iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
                candidates = candidates[iou <= threshold]
        return np.asarray(sorted(kept, key=lambda index: -float(confidence[index])), dtype=int)

    def detect(self, image: np.ndarray, timestamp: float | None = None) -> list[MonsterDetection]:
        thresholds = getattr(self, "class_nms_iou", {})
        backend_nms_iou = max([self.nms_iou, *thresholds.values()])
        result = self.model.predict(source=image, conf=self.confidence, iou=backend_nms_iou,
                                    imgsz=self.input_resolution, device=self.device,
                                    max_det=self.max_detections, agnostic_nms=False, verbose=False)[0]
        boxes = result.boxes
        if boxes is None: return []
        xyxy = boxes.xyxy.detach().cpu().numpy(); confidence = boxes.conf.detach().cpu().numpy()
        classes = (boxes.cls.detach().cpu().numpy().astype(int) if hasattr(boxes, "cls")
                   else np.zeros(len(xyxy), dtype=int))
        keep = self._class_aware_nms(xyxy, confidence, classes)
        xyxy, confidence, classes = xyxy[keep], confidence[keep], classes[keep]
        names = getattr(result, "names", getattr(self.model, "names", {}))
        output = []
        for raw_box, score, class_id_value in zip(xyxy, confidence, classes):
            class_id = int(class_id_value)
            box = self._registered_box(raw_box, class_id, image.shape)
            if box is None:
                continue
            metadata = {"timestamp": timestamp, "imgsz": self.input_resolution,
                        "nms_iou": self.nms_iou, "class_id": class_id,
                        "class_nms_iou": thresholds.get(class_id, self.nms_iou),
                        "class_name": names.get(class_id, str(class_id))}
            if self.class_box_sizes is not None:
                metadata.update({
                    "raw_yolo_box": [float(value) for value in raw_box],
                    "registered_box_size": list(self.class_box_sizes[class_id]),
                    "box_size_reference": list(self.box_size_reference or ()),
                })
            output.append(MonsterDetection(
                box, float(score), ((box[0] + box[2]) / 2.0, box[3]), self.backend, metadata,
            ))
        return output

def train_yolo(dataset_yaml: Path, output_dir: Path, *, model: str = "yolo11n.pt", imgsz: int = 768,
               epochs: int = 40, device: str | int | None = None) -> dict[str, Any]:
    try: from ultralytics import YOLO
    except ImportError as exc: raise RuntimeError("Optional specialized training requires ultralytics") from exc
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    run = YOLO(model).train(data=str(dataset_yaml), imgsz=imgsz, epochs=epochs,
                            project=str(output_dir.parent), name=output_dir.name, workers=0, device=device)
    metrics = {str(key): float(value) for key, value in getattr(run, "results_dict", {}).items()
               if isinstance(value, (int, float, np.number))}
    speed = {str(key): float(value) for key, value in getattr(run, "speed", {}).items()
             if isinstance(value, (int, float, np.number))}
    return {"backend": "ultralytics", "model": model, "imgsz": imgsz, "epochs": epochs,
            "device": "auto" if device is None else device, "run": str(getattr(run, "save_dir", output_dir)),
            "metrics": metrics, "speed_ms": speed}
