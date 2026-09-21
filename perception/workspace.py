"""Validated map-scoped paths and capture settings."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAP_SCHEMA = "specialized-visual-detection.map.v1"
MAP_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class MapWorkspace:
    repo_root: Path
    map_dir: Path
    config: dict[str, Any]

    @classmethod
    def load(cls, repo_root: Path | str, map_id: str) -> "MapWorkspace":
        root = Path(repo_root).resolve()
        if not MAP_ID.fullmatch(map_id):
            raise ValueError(f"Invalid map id: {map_id!r}")
        map_dir = (root / "maps" / map_id).resolve()
        maps_root = (root / "maps").resolve()
        if map_dir.parent != maps_root:
            raise ValueError(f"Map id escapes maps directory: {map_id!r}")
        config_path = map_dir / "map.json"
        if not config_path.is_file():
            available = sorted(path.parent.name for path in maps_root.glob("*/map.json"))
            raise ValueError(f"Unknown map {map_id!r}; available maps: {available}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("schema") != MAP_SCHEMA:
            raise ValueError(f"{config_path}: unsupported schema {config.get('schema')!r}")
        if config.get("map_id") != map_id:
            raise ValueError(f"{config_path}: map_id must be {map_id!r}")
        width, height = config.get("source_size", [0, 0])
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValueError(f"{config_path}: source_size must contain two positive integers")
        crop = config.get("minimap", {}).get("crop_xywh", [])
        if len(crop) != 4 or any(not isinstance(value, int) or value < 0 for value in crop) or crop[2] <= 0 or crop[3] <= 0:
            raise ValueError(f"{config_path}: minimap.crop_xywh must be x,y,width,height")
        return cls(root, map_dir, config)

    @property
    def map_id(self) -> str:
        return str(self.config["map_id"])

    @property
    def display_name(self) -> str:
        return str(self.config["display_name"])

    @property
    def source_size(self) -> tuple[int, int]:
        width, height = self.config["source_size"]
        return int(width), int(height)

    @property
    def minimap_crop(self) -> tuple[int, int, int, int]:
        return tuple(int(value) for value in self.config["minimap"]["crop_xywh"])

    @property
    def fixed_ui_rects(self) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(tuple(int(value) for value in rect) for rect in self.config.get("fixed_ui_rects", []))

    @property
    def prompts(self) -> list[str]:
        return [str(value) for value in self.config.get("prelabel", {}).get("prompts", [])]

    @property
    def capture_dir(self) -> Path:
        override = os.environ.get("PERCEPTION_CAPTURE_DIR")
        return Path(override or self.config["capture"]["directory"]).resolve()

    @property
    def video(self) -> Path:
        return self.capture_dir / str(self.config["capture"].get("video", "screen.mp4"))

    @property
    def events(self) -> Path:
        return self.capture_dir / str(self.config["capture"].get("events", "events.csv"))

    @property
    def video_end_s(self) -> float | None:
        value = self.config["capture"].get("end_s")
        return None if value is None else float(value)

    def path(self, name: str) -> Path:
        relative = self.config.get("paths", {}).get(name)
        if not relative:
            raise ValueError(f"Map {self.map_id!r} has no configured path {name!r}")
        resolved = (self.map_dir / str(relative)).resolve()
        if resolved != self.map_dir and self.map_dir not in resolved.parents:
            raise ValueError(f"Map path {name!r} escapes {self.map_dir}")
        return resolved

    def external_path(self, section: str, name: str) -> Path:
        value = self.config.get(section, {}).get(name)
        if not value:
            raise ValueError(f"Map {self.map_id!r} has no configured {section}.{name}")
        return Path(str(value)).resolve()
