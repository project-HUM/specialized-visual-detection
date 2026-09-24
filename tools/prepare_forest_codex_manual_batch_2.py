"""Prepare the second pending Forest manual-review batch from Sample4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perception.core import sha256_file
from monster_dataset.annotation_io import write_jsonl_with_backup
from monster_dataset.schema import FrameAnnotation, MonsterAnnotation, read_jsonl


SOURCE_VIDEO = Path(
    r"C:\projects\input-flag-inspector\saves_m\ForestOfDeadTreeSample4_LottaLich\screen.mp4"
)
MAP_DIR = ROOT / "maps/forest-of-dead-trees-2"
IMAGES_DIR = MAP_DIR / "dataset/images"
STAGING_DIR = MAP_DIR / "dataset/runs/codex-manual-batch-2"
MINIMAP_REFERENCE = Path(
    r"C:\projects\map-info-extractor\forest-of-dead-trees-2-output\minimap_native.png"
)
MINIMAP_CROP = (14, 147, 326, 180)
MINIMAP_CORRELATION_THRESHOLD = .90
SEED = 20260923
COUNT = 10
MINIMUM_SEPARATION_S = 8.0
PRESET_SIZES = {"1": (83.0, 131.0), "2": (123.0, 144.0), "3": (123.0, 202.0)}
PRESET_NAMES = {"1": "zombie", "2": "hero", "3": "lich"}

# Visually audited fixed-size labels: (preset, ground x, ground y). Detector
# output was used only as a starting point; duplicates were removed and missed
# Liches, lower-UI cutoffs, and upper/side edge sprites were added by inspection.
LABELS = {
    "codex-sample4-random-0000": (
        ("1", 665, 45), ("1", 795, 45), ("1", 977, 130),
        ("1", 1322, 135), ("1", 1666, 136), ("1", 1896, 125),
        ("1", 56, 754), ("3", 490, 668), ("1", 688, 673),
        ("2", 977, 765), ("1", 1487, 764), ("1", 1707, 783),
        ("1", 1892, 852),
    ),
    "codex-sample4-random-0001": (
        ("1", 31, 553), ("1", 108, 556), ("1", 594, 648),
        ("2", 885, 642), ("1", 1035, 646), ("1", 1126, 646),
        ("1", 1302, 645),
    ),
    "codex-sample4-random-0002": (
        ("1", 796, 45), ("1", 1246, 133), ("1", 1345, 132),
        ("1", 1598, 130), ("1", 1684, 130), ("1", 1865, 130),
        ("3", 239, 670), ("1", 361, 671), ("1", 446, 668),
        ("2", 1101, 756), ("1", 1584, 762),
    ),
    "codex-sample4-random-0003": (
        ("1", 32, 385), ("1", 659, 470), ("1", 747, 472),
        ("2", 1011, 499), ("3", 120, 1080), ("1", 210, 1040),
    ),
    "codex-sample4-random-0004": (
        ("1", 1470, 45), ("1", 1530, 45),
        ("1", 229, 894), ("1", 327, 892), ("2", 470, 890),
        ("1", 599, 894), ("1", 640, 894), ("1", 705, 892),
        ("1", 742, 892), ("1", 891, 870), ("1", 1062, 803),
        ("3", 1125, 803),
    ),
    "codex-sample4-random-0005": (
        ("1", 1254, 348), ("1", 1378, 350), ("1", 1835, 350),
        ("1", 1406, 946), ("1", 1576, 944), ("1", 1775, 938),
    ),
    "codex-sample4-random-0006": (
        ("1", 1713, 387), ("1", 1768, 389), ("2", 968, 714),
        ("3", 1260, 1080), ("1", 1610, 1040),
    ),
    "codex-sample4-random-0007": (
        ("1", 1424, 271), ("2", 1010, 785), ("3", 900, 1080),
        ("1", 1095, 1040), ("1", 1140, 1040),
    ),
    "codex-sample4-random-0008": (
        ("1", 1428, 282), ("2", 997, 549), ("3", 1050, 1080),
    ),
    "codex-sample4-random-0009": (
        ("1", 255, 765), ("2", 622, 769), ("3", 900, 773),
        ("1", 1882, 585),
    ),
}


def _minimap_correlation(frame: np.ndarray, reference: np.ndarray) -> float:
    x, y, width, height = MINIMAP_CROP
    roi = frame[y:y + height, x:x + width]
    if roi.shape[:2] != reference.shape[:2]:
        return -1.0
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype("float32")
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype("float32")
    roi_gray -= roi_gray.mean()
    reference_gray -= reference_gray.mean()
    denominator = float(cv2.norm(roi_gray) * cv2.norm(reference_gray))
    return float((roi_gray * reference_gray).sum() / denominator) if denominator else -1.0


def stage() -> dict[str, object]:
    reference = cv2.imread(str(MINIMAP_REFERENCE))
    if reference is None:
        raise RuntimeError(f"Could not read minimap reference {MINIMAP_REFERENCE}")
    capture = cv2.VideoCapture(str(SOURCE_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {SOURCE_VIDEO}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0 or frame_count <= 0:
        raise RuntimeError("Source video has invalid FPS or frame count")

    candidates = list(range(round(4 * fps), frame_count - round(4 * fps)))
    random.Random(SEED).shuffle(candidates)
    selected: list[dict[str, object]] = []
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for frame_index in candidates:
            timestamp = frame_index / fps
            if any(abs(timestamp - float(row["timestamp"])) < MINIMUM_SEPARATION_S for row in selected):
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
            if not ok:
                continue
            correlation = _minimap_correlation(image, reference)
            if correlation < MINIMAP_CORRELATION_THRESHOLD:
                continue
            selected.append({
                "frame_index": frame_index,
                "timestamp": timestamp,
                "minimap_correlation": correlation,
                "image": image,
            })
            if len(selected) == COUNT:
                break
    finally:
        capture.release()
    if len(selected) != COUNT:
        raise RuntimeError(f"Found only {len(selected)} eligible frames")
    selected.sort(key=lambda row: float(row["timestamp"]))

    manifest_frames = []
    panels = []
    for index, row in enumerate(selected):
        frame_id = f"codex-sample4-random-{index:04d}"
        timestamp = float(row["timestamp"])
        destination = IMAGES_DIR / f"{frame_id}_{timestamp:010.3f}.jpg"
        image = row.pop("image")
        if destination.exists():
            existing = cv2.imread(str(destination))
            if existing is None or existing.shape != image.shape:
                raise FileExistsError(f"Refusing to overwrite {destination}")
        elif not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write {destination}")
        relative = destination.relative_to(ROOT).as_posix()
        manifest_frames.append({
            "id": frame_id,
            "frame_id": frame_id,
            "frame_index": int(row["frame_index"]),
            "source_frame": int(row["frame_index"]),
            "time_s": timestamp,
            "timestamp": timestamp,
            "kind": "sample4_random",
            "image": relative,
            "image_path": relative,
            "image_sha256": sha256_file(destination),
            "minimap_correlation": round(float(row["minimap_correlation"]), 6),
        })
        panel = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (0, 0), (480, 28), (0, 0, 0), -1)
        cv2.putText(panel, f"{frame_id}  t={timestamp:.3f}s", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)

    sheet = cv2.vconcat([cv2.hconcat(panels[index:index + 5]) for index in range(0, COUNT, 5)])
    sheet_path = STAGING_DIR / "raw-contact-sheet.jpg"
    if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not write {sheet_path}")
    manifest = {
        "schema": "specialized-visual-detection.codex-manual-review-batch.v1",
        "map_id": "forest-of-dead-trees-2",
        "selection": {
            "kind": "random_valid_minimap",
            "count": COUNT,
            "seed": SEED,
            "minimum_separation_s": MINIMUM_SEPARATION_S,
            "minimum_minimap_correlation": MINIMAP_CORRELATION_THRESHOLD,
        },
        "source": {"video": str(SOURCE_VIDEO), "video_sha256": sha256_file(SOURCE_VIDEO)},
        "test_frames": manifest_frames,
        "frames": manifest_frames,
        "raw_contact_sheet": sheet_path.relative_to(ROOT).as_posix(),
    }
    path = STAGING_DIR / "staging-manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"manifest": str(path), "frames": len(manifest_frames), "contact_sheet": str(sheet_path)}


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


def _draw_labels(image: np.ndarray, labels: tuple[tuple[str, int, int], ...]) -> None:
    colors = {"1": (70, 210, 255), "2": (255, 170, 70), "3": (255, 70, 210)}
    for preset, ground_x, ground_y in labels:
        width, height = PRESET_SIZES[preset]
        x1, y1 = round(ground_x - width / 2), round(ground_y - height)
        x2, y2 = round(ground_x + width / 2), round(ground_y)
        color = colors[preset]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, PRESET_NAMES[preset], (x1, max(20, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)


def finalize() -> dict[str, object]:
    staging_path = STAGING_DIR / "staging-manifest.json"
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    frames = staging["frames"]
    frame_ids = [str(row["frame_id"]) for row in frames]
    if set(frame_ids) != set(LABELS):
        raise ValueError("Staged frame IDs do not match the visually audited label table")

    annotations_path = MAP_DIR / "dataset/annotations.jsonl"
    existing = read_jsonl(annotations_path)
    duplicate_ids = sorted({item.frame_id for item in existing} & set(frame_ids))
    if duplicate_ids:
        raise ValueError(f"Batch already exists in canonical annotations: {duplicate_ids}")

    additions = []
    manifest_frames = []
    panels = []
    for row in frames:
        frame_id = str(row["frame_id"])
        labels = LABELS[frame_id]
        monsters = [_monster(*label) for label in labels]
        conditions = ["sample4_lotta_lich", "valid_minimap"]
        if any(label[0] == "3" for label in labels):
            conditions.append("lich_visible")
        additions.append(FrameAnnotation(
            frame_id=frame_id,
            timestamp=float(row["timestamp"]),
            image_path=str(row["image_path"]),
            monsters=monsters,
            split="train",
            review_status="pending",
            category="codex_manual_sample4_random",
            conditions=conditions,
            notes=(
                "Codex detector-assisted proposal, then visually audited frame-by-frame for class, "
                "duplicates, missed sprites, and edge/UI cutoffs. Human confirmation required."
            ),
        ))
        counts = {
            name: sum(label[0] == preset for label in labels)
            for preset, name in PRESET_NAMES.items()
        }
        manifest_frames.append({
            **{key: value for key, value in row.items() if key not in {"id", "time_s", "kind", "image"}},
            "proposal_count": len(labels),
            "class_counts": counts,
            "visual_audit": "completed_pending_human_confirmation",
        })
        image = cv2.imread(str(ROOT / str(row["image_path"])))
        if image is None:
            raise RuntimeError(f"Could not read {row['image_path']}")
        _draw_labels(image, labels)
        caption = f"{frame_id}  z={counts['zombie']} h={counts['hero']} l={counts['lich']}"
        cv2.rectangle(image, (0, 0), (image.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(image, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA))

    validation_errors = {
        item.frame_id: item.validate(1920, 1080)
        for item in additions if item.validate(1920, 1080)
    }
    if validation_errors:
        raise ValueError(f"Invalid audited proposals: {validation_errors}")
    write_jsonl_with_backup(existing + additions, annotations_path)

    sheet = cv2.vconcat([cv2.hconcat(panels[index:index + 5]) for index in range(0, COUNT, 5)])
    contact_path = STAGING_DIR / "audited-label-contact-sheet.jpg"
    if not cv2.imwrite(str(contact_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not write {contact_path}")

    manifest = {
        "schema": "specialized-visual-detection.codex-manual-review-batch.v1",
        "map_id": "forest-of-dead-trees-2",
        "labeling_method": "detector_assisted_then_manual_visual_audit",
        "selection": staging["selection"],
        "review": {
            "split": "train",
            "status": "pending",
            "command": "maps\\forest-of-dead-trees-2\\review_codex_labels_2.bat",
        },
        "source": staging["source"],
        "frames": manifest_frames,
        "audit": {
            "all_frames_visually_inspected": True,
            "audited_contact_sheet": contact_path.relative_to(ROOT).as_posix(),
            "checks": ["class", "duplicates", "missed_sprites", "partial_edges", "lower_ui_cutoffs"],
        },
    }
    manifest_path = MAP_DIR / "dataset/codex_manual_batch_v2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "annotations": str(annotations_path),
        "manifest": str(manifest_path),
        "frames_added": len(additions),
        "proposals_added": sum(len(item.monsters) for item in additions),
        "class_counts": {
            name: sum(label[0] == preset for labels in LABELS.values() for label in labels)
            for preset, name in PRESET_NAMES.items()
        },
        "contact_sheet": str(contact_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="store_true", help="sample and extract the deterministic frame set")
    parser.add_argument("--finalize", action="store_true", help="append the visually audited pending labels")
    args = parser.parse_args()
    if args.stage == args.finalize:
        parser.error("choose exactly one of --stage or --finalize")
    print(json.dumps(stage() if args.stage else finalize(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
