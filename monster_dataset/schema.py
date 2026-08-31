"""Canonical, backend-independent monster annotation schema."""
from __future__ import annotations
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

VALID_SPLITS = {"train", "validation", "test"}
VALID_REVIEW_STATUSES = {"pending", "reviewed", "needs_review"}

@dataclass
class MonsterAnnotation:
    bbox_xyxy: list[float]
    ground_position: list[float]
    visibility: float = 1.0
    occluded: bool = False
    annotation_confidence: float = 1.0
    review_required: bool = False
    source: str = "visual_review"

    def validate(self, width: int = 1920, height: int = 1080) -> list[str]:
        errors: list[str] = []
        if len(self.bbox_xyxy) != 4 or not all(math.isfinite(float(v)) for v in self.bbox_xyxy):
            return ["bbox_xyxy must contain four finite values"]
        if len(self.ground_position) != 2 or not all(math.isfinite(float(v)) for v in self.ground_position):
            errors.append("ground_position must contain two finite values")
        x1, y1, x2, y2 = (float(v) for v in self.bbox_xyxy)
        if x2 <= x1 or y2 <= y1:
            errors.append("bbox_xyxy must have positive extent")
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            errors.append(f"bbox_xyxy must be inside {width}x{height}")
        if not 0.0 <= float(self.visibility) <= 1.0:
            errors.append("visibility must be within [0, 1]")
        if not 0.0 <= float(self.annotation_confidence) <= 1.0:
            errors.append("annotation_confidence must be within [0, 1]")
        if len(self.ground_position) == 2:
            gx, gy = (float(v) for v in self.ground_position)
            if not (x1 <= gx <= x2 and y1 <= gy <= y2 + 2.0):
                errors.append("ground_position must lie within the monster box")
        if not self.source:
            errors.append("source must be non-empty")
        return errors

@dataclass
class FrameAnnotation:
    frame_id: str
    timestamp: float
    image_path: str
    monsters: list[MonsterAnnotation] = field(default_factory=list)
    split: str = "train"
    review_status: str = "pending"
    category: str = "unspecified"
    conditions: list[str] = field(default_factory=list)
    player_bbox_xyxy: list[float] | None = None
    notes: str = ""

    def validate(self, width: int = 1920, height: int = 1080) -> list[str]:
        errors: list[str] = []
        if not self.frame_id:
            errors.append("frame_id must be non-empty")
        if not math.isfinite(float(self.timestamp)) or float(self.timestamp) < 0:
            errors.append("timestamp must be a finite non-negative number")
        if self.split not in VALID_SPLITS:
            errors.append(f"split must be one of {sorted(VALID_SPLITS)}")
        if self.review_status not in VALID_REVIEW_STATUSES:
            errors.append(f"review_status must be one of {sorted(VALID_REVIEW_STATUSES)}")
        if self.player_bbox_xyxy is not None:
            if len(self.player_bbox_xyxy) != 4:
                errors.append("player_bbox_xyxy must have four values")
            else:
                x1, y1, x2, y2 = self.player_bbox_xyxy
                if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                    errors.append("player_bbox_xyxy must be a valid in-frame box")
        for index, monster in enumerate(self.monsters):
            errors.extend(f"monster[{index}]: {error}" for error in monster.validate(width, height))
        return errors

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
