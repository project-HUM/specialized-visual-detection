import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from monster_dataset.annotation_io import export_yolo
from monster_dataset.pilot import choose_pilot_positions
from monster_dataset.schema import FrameAnnotation, MonsterAnnotation
from monster_dataset.validation import report
from perception.workspace import MapWorkspace


REPO = Path(__file__).resolve().parents[1]


def test_checked_in_map_workspaces_keep_artifacts_isolated():
    rednose = MapWorkspace.load(REPO, "rednose3")
    forest = MapWorkspace.load(REPO, "forest-of-dead-trees-2")

    assert rednose.map_dir != forest.map_dir
    assert rednose.path("annotations").parent == rednose.map_dir / "dataset"
    assert forest.path("annotations").parent == forest.map_dir / "dataset"
    assert forest.external_path("geometry", "positions").name == "positions.csv"
    assert rednose.source_size == forest.source_size == (1920, 1080)


def test_map_workspace_rejects_identity_mismatch_and_path_escape(tmp_path: Path):
    map_dir = tmp_path / "maps" / "wrong"
    map_dir.mkdir(parents=True)
    config = {
        "schema": "specialized-visual-detection.map.v1",
        "map_id": "another",
        "display_name": "Wrong",
        "source_size": [1920, 1080],
        "capture": {"directory": "."},
        "minimap": {"crop_xywh": [0, 0, 10, 10]},
        "paths": {"annotations": "../outside.jsonl"},
    }
    (map_dir / "map.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="map_id must be"):
        MapWorkspace.load(tmp_path, "wrong")

    config["map_id"] = "wrong"
    (map_dir / "map.json").write_text(json.dumps(config), encoding="utf-8")
    workspace = MapWorkspace.load(tmp_path, "wrong")
    with pytest.raises(ValueError, match="escapes"):
        workspace.path("annotations")


def test_pilot_selection_is_deterministic_and_time_separated():
    positions = [
        {"frame": index, "time_s": float(index * 10), "x": 1.0, "y": 2.0, "area": 10}
        for index in range(30)
    ]
    first = choose_pilot_positions(positions, count=5, seed=0, minimum_separation_s=30)
    second = choose_pilot_positions(positions, count=5, seed=0, minimum_separation_s=30)
    assert first == second
    assert all(
        right["time_s"] - left["time_s"] >= 30
        for left, right in zip(first, first[1:])
    )


def test_pilot_is_not_a_yolo_export_split(tmp_path: Path):
    image = tmp_path / "frame.jpg"
    cv2.imwrite(str(image), np.zeros((50, 100, 3), np.uint8))
    item = FrameAnnotation(
        "pilot", 0, str(image),
        [MonsterAnnotation([10, 10, 30, 30], [20, 30])],
        split="pilot", review_status="reviewed",
    )
    with pytest.raises(ValueError, match="train or validation"):
        export_yolo([item], tmp_path / "yolo", split="pilot")
    assert report([item])["training_ready"] is False


def test_yolo_export_uses_configured_source_size(tmp_path: Path):
    image = tmp_path / "frame.jpg"
    cv2.imwrite(str(image), np.zeros((50, 100, 3), np.uint8))
    item = FrameAnnotation(
        "train", 0, str(image),
        [MonsterAnnotation([10, 10, 30, 30], [20, 30])],
        split="train", review_status="reviewed",
    )
    export_yolo([item], tmp_path / "yolo", split="train", source_size=(100, 50))
    values = [float(value) for value in
              (tmp_path / "yolo/labels/train/train.txt").read_text().split()[1:]]
    assert values == pytest.approx([0.2, 0.4, 0.2, 0.4])
