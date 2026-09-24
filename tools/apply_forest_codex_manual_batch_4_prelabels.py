"""Apply visually audited prelabels to the unfinished Forest batch-4 frames."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monster_dataset.annotation_io import write_jsonl_with_backup
from monster_dataset.schema import MonsterAnnotation, read_jsonl


MAP_DIR = ROOT / "maps/forest-of-dead-trees-2"
ANNOTATIONS = MAP_DIR / "dataset/annotations.jsonl"
MANIFEST = MAP_DIR / "dataset/codex_manual_batch_v4_manifest.json"
OUTPUT = MAP_DIR / "dataset/runs/codex-manual-batch-4/audited-prelabels"
PREFIX = "codex-sample4-hardcase-"
SIZES = {"1": (83.0, 131.0), "2": (123.0, 144.0), "3": (123.0, 202.0)}
NAMES = {"1": "Zombie", "2": "Hero", "3": "Lich"}
COLORS = {"1": (70, 210, 255), "2": (255, 170, 70), "3": (255, 70, 210)}

# Each tuple is (preset ID, ground x, ground y). These began as high-recall
# combined-v2 detections, then were audited against each full-resolution image.
# Snow/tree/UI false positives and duplicates are intentionally omitted.
AUDITED_GROUND_POINTS: dict[str, list[tuple[str, float, float]]] = {
    "codex-sample4-hardcase-0003": [
        ("2", 895.6, 671.3), ("1", 1412.1, 396.1),
    ],
    "codex-sample4-hardcase-0004": [
        ("2", 916.2, 658.2), ("1", 1425.8, 378.9), ("1", 1638.8, 375.4),
        ("1", 1487.0, 977.3), ("1", 1741.0, 988.3), ("3", 1895.4, 963.2),
    ],
    "codex-sample4-hardcase-0005": [
        ("2", 874.7, 649.5), ("1", 1809.0, 375.9), ("3", 1604.1, 999.9),
    ],
    "codex-sample4-hardcase-0006": [
        ("2", 1018.7, 775.0), ("3", 1558.5, 992.1),
    ],
    "codex-sample4-hardcase-0007": [
        ("2", 933.2, 691.0),
    ],
    "codex-sample4-hardcase-0008": [
        ("3", 207.3, 729.6), ("2", 1077.6, 801.5),
        ("1", 901.7, 82.8), ("1", 386.1, 711.0),
        ("1", 1412.1, 169.9), ("1", 1616.9, 163.9), ("1", 1857.8, 164.5),
        ("1", 1002.0, 800.0), ("1", 1157.4, 800.6), ("1", 1202.5, 800.7),
        ("1", 1825.4, 861.2),
    ],
    "codex-sample4-hardcase-0009": [
        ("2", 859.5, 677.0), ("1", 1196.3, 761.1), ("1", 1749.6, 762.9),
    ],
    "codex-sample4-hardcase-0010": [
        ("2", 893.8, 672.3), ("1", 1713.1, 363.6), ("3", 1300.8, 1064.0),
        ("1", 1825.0, 1064.0),
    ],
    "codex-sample4-hardcase-0011": [
        ("1", 389.1, 85.2), ("1", 462.7, 86.4), ("1", 579.7, 86.7),
        ("1", 1152.8, 87.6), ("1", 108.4, 624.1),
        ("2", 1277.3, 816.5), ("1", 1345.1, 808.3), ("1", 1467.0, 801.8),
    ],
    "codex-sample4-hardcase-0012": [
        ("1", 15.1, 558.0), ("1", 515.5, 560.8), ("2", 681.1, 623.5),
        ("1", 792.0, 651.1), ("1", 842.4, 652.1), ("1", 952.7, 648.9),
        ("1", 1056.9, 648.2), ("1", 1132.8, 650.5), ("1", 1386.0, 650.4),
    ],
    "codex-sample4-hardcase-0013": [
        ("2", 968.8, 809.7), ("1", 1885.1, 438.5), ("3", 1433.5, 1052.4),
    ],
    "codex-sample4-hardcase-0014": [
        ("2", 1090.1, 546.8), ("1", 1359.8, 426.7),
        ("1", 1780.0, 425.9), ("1", 1899.2, 425.9),
    ],
}


def _annotation(preset: str, ground_x: float, ground_y: float) -> MonsterAnnotation:
    width, height = SIZES[preset]
    return MonsterAnnotation(
        bbox_xyxy=[ground_x - width / 2.0, ground_y - height,
                   ground_x + width / 2.0, ground_y],
        ground_position=[ground_x, ground_y],
        annotation_confidence=.75,
        review_required=True,
        source="codex_prelabel",
        box_preset=preset,
    )


def _render(item) -> np.ndarray:
    image = cv2.imread(str(ROOT / item.image_path))
    if image is None:
        raise RuntimeError(f"Could not read {item.image_path}")
    for monster in item.monsters:
        preset = monster.box_preset or "1"
        color = COLORS[preset]
        x1, y1, x2, y2 = (round(value) for value in monster.bbox_xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, NAMES[preset], (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    title = np.zeros((38, image.shape[1], 3), dtype=image.dtype)
    cv2.putText(title, f"{item.frame_id}  audited prelabels={len(item.monsters)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.vconcat((title, image))


def main() -> int:
    items = read_jsonl(ANNOTATIONS)
    by_id = {item.frame_id: item for item in items}
    batch_ids = sorted(frame_id for frame_id in by_id if frame_id.startswith(PREFIX))
    if len(batch_ids) != 15:
        raise ValueError(f"Expected 15 batch-4 frames, found {len(batch_ids)}")

    # The user's first two frames and six existing boxes on frame 0002 are
    # preserved byte-for-field by changing only the twelve still-blank frames.
    preserved = {frame_id: [monster.__dict__.copy() for monster in by_id[frame_id].monsters]
                 for frame_id in batch_ids[:3]}
    for frame_id, points in AUDITED_GROUND_POINTS.items():
        item = by_id[frame_id]
        if item.monsters:
            raise ValueError(f"Refusing to replace existing boxes in {frame_id}")
        item.monsters = [_annotation(*point) for point in points]
        item.review_status = "pending"
        item.notes += (
            " Model-assisted Codex prelabels were checked at full resolution; duplicate and "
            "background/UI false positives were removed and obvious missed edge/effect sprites "
            "were added. Human confirmation is still required."
        )

    for frame_id, before in preserved.items():
        after = [monster.__dict__ for monster in by_id[frame_id].monsters]
        if after != before:
            raise AssertionError(f"Preserved user annotation changed: {frame_id}")

    errors = {
        frame_id: by_id[frame_id].validate(1920, 1080)
        for frame_id in batch_ids if by_id[frame_id].validate(1920, 1080)
    }
    if errors:
        raise ValueError(f"Invalid batch-4 annotations: {errors}")
    write_jsonl_with_backup(items, ANNOTATIONS)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    shutil.copy2(MANIFEST, MANIFEST.with_suffix(".json.prelabel.bak"))
    for row in manifest["frames"]:
        item = by_id[row["frame_id"]]
        row["current_box_count"] = len(item.monsters)
        if row["frame_id"] in AUDITED_GROUND_POINTS:
            row["proposal_count"] = len(item.monsters)
            row["prelabeling"] = "combined-v2_conf005_plus_full_resolution_codex_audit"
            row["visual_audit"] = "complete_pending_human_confirmation"
        elif row["frame_id"] == "codex-sample4-hardcase-0002":
            row["prelabeling"] = "existing_user_work_preserved"
    manifest["prelabeling_audit"] = {
        "model": "combined-v2 best.pt",
        "proposal_confidence": .05,
        "method": "model_assisted_then_full_resolution_codex_visual_audit",
        "preserved_user_frames": batch_ids[:3],
        "prelabelled_frames": len(AUDITED_GROUND_POINTS),
        "prelabelled_boxes": sum(len(points) for points in AUDITED_GROUND_POINTS.values()),
        "status": "pending_human_confirmation",
        "contact_sheet": "maps/forest-of-dead-trees-2/dataset/runs/"
                         "codex-manual-batch-4/audited-prelabels/contact-sheet.jpg",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    panels = []
    for frame_id in batch_ids[2:]:
        rendered = _render(by_id[frame_id])
        destination = OUTPUT / f"{frame_id}.jpg"
        if not cv2.imwrite(str(destination), rendered, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write {destination}")
        panel = cv2.resize(rendered, (640, round(640 * rendered.shape[0] / rendered.shape[1])),
                           interpolation=cv2.INTER_AREA)
        panels.append(panel)
    blank = np.zeros_like(panels[0])
    while len(panels) % 3:
        panels.append(blank.copy())
    sheet = cv2.vconcat([
        cv2.hconcat(panels[index:index + 3]) for index in range(0, len(panels), 3)
    ])
    contact = OUTPUT / "contact-sheet.jpg"
    if not cv2.imwrite(str(contact), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not write {contact}")

    print(json.dumps({
        "preserved_user_frames": batch_ids[:3],
        "prelabelled_frames": len(AUDITED_GROUND_POINTS),
        "prelabelled_boxes": sum(len(points) for points in AUDITED_GROUND_POINTS.values()),
        "contact_sheet": str(contact),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
