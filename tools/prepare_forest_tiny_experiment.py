"""Prepare the isolated Forest 5-train/2-validation/20-test experiment."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monster_dataset.annotation_io import export_yolo
from monster_dataset.pilot import _direct_positions, choose_pilot_positions
from monster_dataset.schema import read_jsonl, write_jsonl
from perception.core import sha256_file
from perception.workspace import MapWorkspace


TRAIN_IDS = ("pilot-0000", "pilot-0001", "pilot-0003", "pilot-0004", "pilot-0005")
VALIDATION_IDS = ("pilot-0002", "pilot-0006")
PRESET_CLASSES = {"1": "zombie", "2": "hero", "3": "lich"}
LICH_INTERVAL_S = (4617.0, 4665.0)
SOURCE_IMAGE_SIZE = (960, 540)


def _write_normalized_image(source: Path, destination: Path) -> None:
    image = cv2.imread(str(source))
    if image is None:
        raise RuntimeError(f"Could not read {source}")
    if (image.shape[1], image.shape[0]) != SOURCE_IMAGE_SIZE:
        image = cv2.resize(image, SOURCE_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not write {destination}")


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def prepare(output: Path, *, seed: int) -> dict[str, object]:
    workspace = MapWorkspace.load(ROOT, "forest-of-dead-trees-2")
    if output.exists():
        raise FileExistsError(f"Experiment output already exists: {output}")
    output.mkdir(parents=True)

    canonical = read_jsonl(workspace.path("annotations"))
    by_id = {item.frame_id: item for item in canonical}
    expected = set(TRAIN_IDS + VALIDATION_IDS)
    if set(by_id) != expected:
        raise ValueError(f"Expected exactly {sorted(expected)}, got {sorted(by_id)}")

    experiment_items = []
    normalized_dir = output / "source_images"
    for frame_id in TRAIN_IDS + VALIDATION_IDS:
        item = copy.deepcopy(by_id[frame_id])
        item.split = "train" if frame_id in TRAIN_IDS else "validation"
        item.review_status = "reviewed"
        item.notes = (item.notes + " " if item.notes else "") + (
            "Derived for tiny-3class-v1 after explicit human labeling confirmation."
        )
        source = workspace.repo_root / item.image_path
        normalized = normalized_dir / f"{frame_id}.jpg"
        _write_normalized_image(source, normalized)
        item.image_path = _relative(normalized, workspace.repo_root)
        experiment_items.append(item)

    annotations = output / "annotations.jsonl"
    write_jsonl(experiment_items, annotations)
    yolo_dir = output / "yolo_dataset"
    train_export = export_yolo(
        experiment_items, yolo_dir, split="train", source_size=workspace.source_size,
        image_root=workspace.repo_root, preset_classes=PRESET_CLASSES,
    )
    validation_export = export_yolo(
        experiment_items, yolo_dir, split="validation", source_size=workspace.source_size,
        image_root=workspace.repo_root, preset_classes=PRESET_CLASSES,
    )

    positions_path = workspace.external_path("geometry", "positions")
    positions = _direct_positions(positions_path)
    labeled_times = [float(item.timestamp) for item in experiment_items]
    general_candidates = [
        row for row in positions
        if not (LICH_INTERVAL_S[0] <= float(row["time_s"]) <= LICH_INTERVAL_S[1])
        and all(abs(float(row["time_s"]) - timestamp) >= 5.0 for timestamp in labeled_times)
    ]
    lich_candidates = [
        row for row in positions
        if LICH_INTERVAL_S[0] <= float(row["time_s"]) <= LICH_INTERVAL_S[1]
        and all(abs(float(row["time_s"]) - timestamp) >= 2.0 for timestamp in labeled_times)
    ]
    general = choose_pilot_positions(general_candidates, count=16, seed=seed,
                                     minimum_separation_s=60.0)
    lich = choose_pilot_positions(lich_candidates, count=4, seed=seed + 1,
                                  minimum_separation_s=6.0)

    capture = cv2.VideoCapture(str(workspace.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {workspace.video}")
    test_rows = []
    test_images = output / "test" / "images"
    try:
        for kind, rows in (("general", general), ("lich_interval", lich)):
            for index, row in enumerate(rows):
                frame_index = int(row["frame"])
                timestamp = float(row["time_s"])
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not decode frame {frame_index}")
                frame = cv2.resize(frame, SOURCE_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                test_id = f"test-{kind}-{index:02d}"
                image = test_images / f"{test_id}_{timestamp:010.3f}.jpg"
                image.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(image), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"Could not write {image}")
                test_rows.append({
                    "id": test_id, "kind": kind, "frame": frame_index,
                    "time_s": timestamp, "image": _relative(image, workspace.repo_root),
                    "image_sha256": sha256_file(image),
                })
    finally:
        capture.release()

    manifest = {
        "schema": "specialized-visual-detection.tiny-learning-experiment.v1",
        "map_id": workspace.map_id,
        "seed": seed,
        "classes": PRESET_CLASSES,
        "train_frame_ids": list(TRAIN_IDS),
        "validation_frame_ids": list(VALIDATION_IDS),
        "test_selection": {
            "general_count": 16,
            "general_minimum_separation_s": 60.0,
            "lich_interval_s": list(LICH_INTERVAL_S),
            "lich_count": 4,
            "lich_minimum_separation_s": 6.0,
            "annotations": "sealed_unlabeled_visual_test",
        },
        "source": {
            "annotations": str(workspace.path("annotations").resolve()),
            "annotations_sha256": sha256_file(workspace.path("annotations")),
            "video": str(workspace.video.resolve()),
            "video_sha256": sha256_file(workspace.video),
            "positions": str(positions_path.resolve()),
            "positions_sha256": sha256_file(positions_path),
        },
        "exports": {"train": train_export, "validation": validation_export},
        "test_frames": test_rows,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    default = ROOT / "maps/forest-of-dead-trees-2/dataset/runs/tiny-3class-v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    manifest = prepare(args.output, seed=args.seed)
    print(json.dumps({
        "output": str(args.output),
        "train": manifest["exports"]["train"],
        "validation": manifest["exports"]["validation"],
        "test_frames": len(manifest["test_frames"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
