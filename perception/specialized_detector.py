"""Backend-independent one-class monster detector implementations."""
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
    """Lazy Ultralytics adapter with intentionally permissive backend NMS.

    YOLO NMS limits pathological proposal explosion.  The tracker-side
    multi-signal deduplicator remains authoritative for deciding whether two
    surviving observations are duplicates.
    """
    backend = "yolo"
    def __init__(self, weights: Path | str, *, confidence: float = .21, nms_iou: float = .90,
                 input_resolution: int = 768, device: str | int | None = None, max_detections: int = 100):
        if not 0 <= confidence <= 1 or not 0 <= nms_iou <= 1: raise ValueError("confidence and nms_iou must be within [0, 1]")
        if input_resolution <= 0 or max_detections <= 0: raise ValueError("input_resolution and max_detections must be positive")
        try: from ultralytics import YOLO
        except ImportError as exc: raise RuntimeError("YoloMonsterDetector requires the optional ultralytics package") from exc
        self.model = YOLO(str(weights)); self.weights = str(weights); self.confidence = confidence
        self.nms_iou = nms_iou; self.input_resolution = input_resolution; self.device = device; self.max_detections = max_detections

    def detect(self, image: np.ndarray, timestamp: float | None = None) -> list[MonsterDetection]:
        result = self.model.predict(source=image, conf=self.confidence, iou=self.nms_iou,
                                    imgsz=self.input_resolution, device=self.device, classes=[0],
                                    max_det=self.max_detections, agnostic_nms=False, verbose=False)[0]
        boxes = result.boxes
        if boxes is None: return []
        xyxy = boxes.xyxy.detach().cpu().numpy(); confidence = boxes.conf.detach().cpu().numpy()
        return [MonsterDetection(tuple(float(v) for v in box), float(score),
                                 ((float(box[0])+float(box[2]))/2, float(box[3])), self.backend,
                                 {"timestamp": timestamp, "imgsz": self.input_resolution,
                                  "nms_iou": self.nms_iou}) for box, score in zip(xyxy, confidence)]

def train_yolo(dataset_yaml: Path, output_dir: Path, *, model: str = "yolo11n.pt", imgsz: int = 768,
               epochs: int = 40, device: str | int | None = None) -> dict[str, Any]:
    try: from ultralytics import YOLO
    except ImportError as exc: raise RuntimeError("Optional specialized training requires ultralytics") from exc
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
