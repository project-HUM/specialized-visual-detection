from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json

@dataclass
class MonsterAnnotation:
    bbox_xyxy: list[float]
    ground_position: list[float]
    occluded: bool = False
    visibility: float = 1.0
    annotation_confidence: float = 1.0
    review_required: bool = False
    source: str = "visual_review"
    def __post_init__(self):
        if len(self.bbox_xyxy) != 4 or len(self.ground_position) != 2:
            raise ValueError("monster bbox must have four values and ground_position two values")
        if self.bbox_xyxy[2] <= self.bbox_xyxy[0] or self.bbox_xyxy[3] <= self.bbox_xyxy[1]:
            raise ValueError("monster bbox must have positive extent")
        self.visibility = max(0.0, min(1.0, float(self.visibility)))

@dataclass
class FrameAnnotation:
    frame_id: str
    timestamp: float
    image_path: str
    monsters: list[MonsterAnnotation] = field(default_factory=list)
    review_status: str = "pending"
    category: str = "unspecified"
    notes: str = ""
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FrameAnnotation":
        data = dict(payload)
        data["monsters"] = [MonsterAnnotation(**item) for item in data.get("monsters", [])]
        return cls(**data)

def write_jsonl(items: list[FrameAnnotation], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item.to_dict(), separators=(",", ":")) + "\n" for item in items), encoding="utf-8")

def read_jsonl(path: Path) -> list[FrameAnnotation]:
    return [FrameAnnotation.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
