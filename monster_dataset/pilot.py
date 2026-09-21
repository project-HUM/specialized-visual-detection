"""Deterministic pilot sampling from direct minimap-marker observations."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import cv2

from perception.core import sha256_file

from .schema import FrameAnnotation, write_jsonl


def _direct_positions(path: Path) -> list[dict[str, float | int]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [
            {
                "frame": int(row["frame"]),
                "time_s": float(row["time_s"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "area": int(row["area"]),
            }
            for row in csv.DictReader(stream)
        ]
    if not rows:
        raise ValueError(f"No direct minimap positions in {path}")
    if any(float(right["time_s"]) < float(left["time_s"]) for left, right in zip(rows, rows[1:])):
        raise ValueError(f"Direct minimap positions are not monotonic: {path}")
    return rows


def choose_pilot_positions(positions: list[dict[str, float | int]], *, count: int,
                           seed: int, minimum_separation_s: float) -> list[dict[str, float | int]]:
    if count <= 0 or minimum_separation_s < 0:
        raise ValueError("count must be positive and minimum separation non-negative")
    shuffled = list(positions)
    random.Random(seed).shuffle(shuffled)
    selected: list[dict[str, float | int]] = []
    for candidate in shuffled:
        timestamp = float(candidate["time_s"])
        if all(abs(timestamp - float(existing["time_s"])) >= minimum_separation_s for existing in selected):
            selected.append(candidate)
            if len(selected) == count:
                return sorted(selected, key=lambda row: float(row["time_s"]))
    raise ValueError(
        f"Could select only {len(selected)} of {count} direct positions with "
        f"minimum separation {minimum_separation_s:g}s"
    )


def sample_pilot(video: Path, positions_csv: Path, images_dir: Path, annotations_path: Path,
                 manifest_path: Path, *, repo_root: Path, map_id: str,
                 minimap_crop: tuple[int, int, int, int], count: int = 5,
                 source_size: tuple[int, int] = (1920, 1080),
                 seed: int = 0, minimum_separation_s: float = 30.0,
                 replace: bool = False) -> dict[str, object]:
    outputs = (annotations_path, manifest_path)
    if not replace and any(path.exists() for path in outputs):
        raise FileExistsError("Pilot annotations already exist; pass --replace to regenerate them")
    selected = choose_pilot_positions(_direct_positions(positions_csv), count=count, seed=seed,
                                      minimum_separation_s=minimum_separation_s)
    images_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video}")
    annotations: list[FrameAnnotation] = []
    manifest_rows: list[dict[str, object]] = []
    for index, position in enumerate(selected):
        frame_index = int(position["frame"])
        timestamp = float(position["time_s"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Could not decode pilot frame {frame_index}")
        frame_id = f"pilot-{index:04d}"
        image_path = images_dir / f"{frame_id}_{timestamp:010.3f}.jpg"
        preview = cv2.resize(
            frame, (max(1, source_size[0] // 2), max(1, source_size[1] // 2)),
            interpolation=cv2.INTER_AREA,
        )
        if not cv2.imwrite(str(image_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            capture.release()
            raise RuntimeError(f"Could not write {image_path}")
        try:
            stored_path = image_path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            stored_path = str(image_path.resolve())
        annotations.append(FrameAnnotation(frame_id, timestamp, stored_path, split="pilot",
                                           review_status="pending", category="pilot_random"))
        manifest_rows.append({
            "frame_id": frame_id,
            "frame": frame_index,
            "time_s": timestamp,
            "image": stored_path,
            "minimap_position": [float(position["x"]), float(position["y"])],
            "marker_area": int(position["area"]),
        })
    capture.release()
    write_jsonl(annotations, annotations_path)
    manifest = {
        "schema": "specialized-visual-detection.pilot.v1",
        "map_id": map_id,
        "selection": {
            "count": count,
            "seed": seed,
            "minimum_separation_s": minimum_separation_s,
            "eligibility": "direct_minimap_marker",
            "ui_filter": "none",
        },
        "source": {
            "video": str(video.resolve()),
            "video_sha256": sha256_file(video),
            "positions": str(positions_csv.resolve()),
            "positions_sha256": sha256_file(positions_csv),
            "minimap_crop_xywh": list(minimap_crop),
        },
        "frames": manifest_rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
