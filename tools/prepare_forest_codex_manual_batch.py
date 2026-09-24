"""Add the first visually hand-labeled Codex review batch for Forest 2.

Frame selection is deterministic, but every box below was placed by visual
inspection of the extracted frame and the user's seven reference annotations.
No detector or automatic pre-labeler is called by this tool.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monster_dataset.schema import FrameAnnotation, MonsterAnnotation, read_jsonl, write_jsonl
from perception.core import sha256_file
from perception.workspace import MapWorkspace


PRESET_SIZES = {"1": (83.0, 131.0), "2": (123.0, 144.0), "3": (123.0, 202.0)}

# frame_id, source frame, PTS seconds, selection kind, (preset, ground x, ground y)
FRAMES = (
    ("codex-random-0000", 11134, 418.975000, "random",
     (("1", 245, 610), ("1", 582, 610), ("1", 660, 610),
      ("2", 810, 710), ("1", 1470, 710))),
    ("codex-random-0001", 17369, 654.432267, "random",
     (("1", 220, 770), ("2", 490, 785), ("1", 1830, 665))),
    ("codex-random-0002", 31452, 1187.628333, "random",
     (("1", 490, 710), ("2", 1040, 710), ("1", 1265, 765),
      ("1", 1350, 765), ("1", 1430, 765), ("1", 1600, 765),
      ("1", 630, 80), ("1", 760, 80), ("1", 1015, 80),
      ("1", 1125, 160), ("1", 1265, 190), ("1", 1700, 100))),
    ("codex-random-0003", 36707, 1385.822233, "random",
     (("1", 490, 720), ("1", 717, 720), ("1", 905, 720),
      ("2", 1235, 840), ("1", 1380, 840), ("1", 755, 150),
      ("1", 1250, 150))),
    ("codex-random-0004", 45784, 1728.554933, "random",
     (("1", 20, 610), ("1", 365, 710), ("2", 965, 680),
      ("1", 1690, 790), ("1", 470, 105), ("1", 710, 125),
      ("1", 770, 125), ("1", 1000, 145), ("1", 1130, 145))),
    ("codex-random-0005", 54997, 2076.649700, "random",
     (("1", 210, 690), ("1", 390, 730), ("1", 710, 730),
      ("2", 810, 710), ("1", 1245, 700), ("1", 1320, 700))),
    ("codex-random-0006", 63233, 2387.314367, "random",
     (("1", 20, 610), ("1", 265, 710), ("1", 860, 710),
      ("2", 1045, 775), ("1", 100, 610), ("1", 725, 75),
      ("1", 790, 75), ("1", 1180, 120))),
    ("codex-random-0007", 73919, 2791.293733, "random",
     (("1", 0, 600), ("1", 80, 600), ("1", 180, 600),
      ("2", 820, 720))),
    ("codex-random-0008", 97044, 3664.909633, "random",
     (("1", 1215, 180), ("1", 1360, 180), ("1", 1500, 180),
      ("1", 1585, 180), ("1", 830, 710), ("2", 930, 710))),
    ("codex-random-0009", 124221, 4690.475333, "random", ()),
    ("codex-lich-early-0000", 121, 4.521633, "lich_early",
     (("2", 810, 710), ("1", 1490, 390), ("3", 1545, 1080))),
    ("codex-lich-early-0001", 464, 17.301600, "lich_early",
     (("3", 895, 700), ("2", 1045, 700), ("1", 1185, 700),
      ("1", 1330, 700))),
    ("codex-lich-late-0000", 122572, 4627.019900, "lich_late",
     (("1", 1640, 185), ("1", 1725, 185), ("1", 1805, 185),
      ("1", 1900, 185), ("2", 885, 610), ("3", 1280, 935))),
    ("codex-lich-late-0001", 123494, 4661.782867, "lich_late",
     (("1", 560, 800), ("2", 800, 800), ("3", 1115, 700))),
)

# Additional partial sprites found during a second manual edge pass. Ground
# points deliberately remain at their inferred full-sprite positions, so the
# fixed-size box may extend beyond the captured image or through the lower UI
# occlusion instead of being shrunk to the visible fragment.
CUTOFF_LABELS = {
    "codex-random-0000": (
        ("1", 1015, 80), ("1", 1155, 80),
        ("1", 1650, 80), ("1", 1715, 80),
        ("1", 1935, 790),
    ),
    "codex-random-0007": (
        ("1", 985, 35), ("1", 1080, 35), ("1", 1230, 35),
    ),
    "codex-random-0008": (
        ("1", 805, 35), ("1", 850, 35),
    ),
    "codex-lich-early-0000": (
        ("1", 1650, 1040), ("1", 1720, 1040),
    ),
    "codex-lich-early-0001": (
        ("1", 1245, 40), ("1", 1535, 40), ("1", 120, 880),
    ),
}


def _monster(preset: str, ground_x: float, ground_y: float) -> MonsterAnnotation:
    width, height = PRESET_SIZES[preset]
    return MonsterAnnotation(
        bbox_xyxy=[ground_x - width / 2.0, ground_y - height,
                   ground_x + width / 2.0, ground_y],
        ground_position=[float(ground_x), float(ground_y)],
        visibility=1.0,
        occluded=False,
        annotation_confidence=.75,
        review_required=True,
        source="codex_prelabel",
        box_preset=preset,
    )


def _labels_with_cutoffs(frame_id: str, labels: tuple[tuple[str, int, int], ...]) -> tuple[tuple[str, int, int], ...]:
    return labels + CUTOFF_LABELS.get(frame_id, ())


def prepare() -> dict[str, object]:
    workspace = MapWorkspace.load(ROOT, "forest-of-dead-trees-2")
    annotations_path = workspace.path("annotations")
    existing = read_jsonl(annotations_path)
    reference_annotations_sha256 = sha256_file(annotations_path)
    existing_ids = {item.frame_id for item in existing}
    duplicate_ids = sorted(existing_ids & {row[0] for row in FRAMES})
    if duplicate_ids:
        raise ValueError(f"Batch already present in canonical annotations: {duplicate_ids}")

    images_dir = workspace.path("dataset_images")
    images_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(workspace.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {workspace.video}")

    additions = []
    manifest_frames = []
    try:
        for frame_id, frame_index, timestamp, kind, labels in FRAMES:
            labels = _labels_with_cutoffs(frame_id, labels)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode source frame {frame_index}")
            image_path = images_dir / f"{frame_id}_{timestamp:010.3f}.jpg"
            if image_path.exists():
                raise FileExistsError(f"Refusing to overwrite {image_path}")
            if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Could not write {image_path}")

            conditions = []
            if kind == "lich_early":
                conditions = ["lich_window_00m00s_00m19s", "lich_visible"]
            elif kind == "lich_late":
                conditions = ["lich_window_01h16m57s_01h17m45s", "lich_visible"]
            elif frame_id == "codex-random-0009":
                conditions = ["ui_dialog", "negative_frame"]
            monsters = [_monster(*label) for label in labels]
            additions.append(FrameAnnotation(
                frame_id=frame_id,
                timestamp=timestamp,
                image_path=image_path.relative_to(ROOT).as_posix(),
                monsters=monsters,
                split="train",
                review_status="pending",
                category="codex_manual_lich_window" if kind != "random" else "codex_manual_random",
                conditions=conditions,
                notes="Codex manual visual proposal from user reference convention; no detector or OWLv2 used. Human review required.",
            ))
            manifest_frames.append({
                "frame_id": frame_id,
                "source_frame": frame_index,
                "timestamp": timestamp,
                "selection_kind": kind,
                "image_path": image_path.relative_to(ROOT).as_posix(),
                "image_sha256": sha256_file(image_path),
                "proposal_count": len(monsters),
                "class_counts": {
                    "zombie": sum(label[0] == "1" for label in labels),
                    "hero": sum(label[0] == "2" for label in labels),
                    "lich": sum(label[0] == "3" for label in labels),
                },
            })
    finally:
        capture.release()

    write_jsonl(existing + additions, annotations_path)
    manifest = {
        "schema": "specialized-visual-detection.codex-manual-review-batch.v1",
        "map_id": workspace.map_id,
        "labeling_method": "manual_visual_inspection_no_detector",
        "reference_frame_ids": [f"pilot-{index:04d}" for index in range(7)],
        "selection": {
            "general": {"count": 10, "seed": 20260922, "minimum_separation_s": 180},
            "lich_early": {"count": 2, "seed": 20260923, "interval_s": [0, 19], "minimum_separation_s": 4},
            "lich_late": {"count": 2, "seed": 20260924, "interval_s": [4617, 4665], "minimum_separation_s": 10},
        },
        "review": {
            "split": "train",
            "status": "pending",
            "command": "maps\\forest-of-dead-trees-2\\review_codex_labels.bat",
        },
        "source": {
            "video": str(workspace.video.resolve()),
            "video_sha256": sha256_file(workspace.video),
            "reference_annotations_sha256_before_append": reference_annotations_sha256,
        },
        "frames": manifest_frames,
    }
    manifest_path = workspace.map_dir / "dataset" / "codex_manual_batch_v1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "annotations": str(annotations_path),
        "manifest": str(manifest_path),
        "frames_added": len(additions),
        "proposals_added": sum(len(item.monsters) for item in additions),
    }


def augment_cutoffs() -> dict[str, object]:
    """Append the second-pass cutoff proposals to an existing first batch."""
    workspace = MapWorkspace.load(ROOT, "forest-of-dead-trees-2")
    annotations_path = workspace.path("annotations")
    items = read_jsonl(annotations_path)
    by_id = {item.frame_id: item for item in items}
    missing_frames = sorted(set(CUTOFF_LABELS) - set(by_id))
    if missing_frames:
        raise ValueError(f"Cutoff target frames are missing: {missing_frames}")

    added = 0
    by_frame: dict[str, int] = {}
    for frame_id, labels in CUTOFF_LABELS.items():
        item = by_id[frame_id]
        existing = {
            (
                monster.box_preset,
                round(float(monster.ground_position[0]), 6),
                round(float(monster.ground_position[1]), 6),
            )
            for monster in item.monsters
        }
        frame_added = 0
        for label in labels:
            key = (label[0], round(float(label[1]), 6), round(float(label[2]), 6))
            if key in existing:
                continue
            item.monsters.append(_monster(*label))
            existing.add(key)
            frame_added += 1
        if frame_added:
            if "partial_edge_labels" not in item.conditions:
                item.conditions.append("partial_edge_labels")
            suffix = (
                " Includes manually placed partial-sprite proposals; preset dimensions are retained "
                "through the image edge or lower UI occlusion."
            )
            if suffix.strip() not in item.notes:
                item.notes += suffix
            added += frame_added
            by_frame[frame_id] = frame_added

    errors = {
        item.frame_id: item.validate(*workspace.source_size)
        for item in items
        if item.validate(*workspace.source_size)
    }
    if errors:
        raise ValueError(f"Augmented annotations are invalid: {errors}")
    if added:
        write_jsonl(items, annotations_path)

    manifest_path = workspace.map_dir / "dataset" / "codex_manual_batch_v1_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for frame in manifest["frames"]:
        item = by_id[frame["frame_id"]]
        frame["proposal_count"] = len(item.monsters)
        frame["class_counts"] = {
            "zombie": sum(monster.box_preset == "1" for monster in item.monsters),
            "hero": sum(monster.box_preset == "2" for monster in item.monsters),
            "lich": sum(monster.box_preset == "3" for monster in item.monsters),
        }
    manifest["cutoff_augmentation"] = {
        "method": "manual_visual_edge_pass_no_detector",
        "proposal_count": sum(len(labels) for labels in CUTOFF_LABELS.values()),
        "frame_count": len(CUTOFF_LABELS),
        "convention": "retain registered preset dimensions beyond image edges or through lower UI occlusion",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "annotations": str(annotations_path),
        "manifest": str(manifest_path),
        "proposals_added": added,
        "frames_changed": by_frame,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment-cutoffs", action="store_true")
    args = parser.parse_args()
    print(json.dumps(augment_cutoffs() if args.augment_cutoffs else prepare(), indent=2))
