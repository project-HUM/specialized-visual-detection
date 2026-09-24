"""Survey and prepare a Lich-observation-diversity review batch from Sample4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
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
STAGING_DIR = MAP_DIR / "dataset/runs/codex-manual-batch-3"
MINIMAP_REFERENCE = Path(
    r"C:\projects\map-info-extractor\forest-of-dead-trees-2-output\minimap_native.png"
)
MINIMAP_CROP = (14, 147, 326, 180)
MINIMAP_CORRELATION_THRESHOLD = .90
SURVEY_INTERVAL_S = 2.5
EXISTING_SEPARATION_S = 1.5
SELECTED_CASES = (
    ("survey-000", "no_lich_wide_negative"),
    ("survey-001", "clear_lich_left_with_zombies"),
    ("survey-002", "clear_lich_far_left_dense_zombies"),
    ("survey-007", "elevated_left_lich"),
    ("survey-037", "clear_lich_center_portal_damage"),
    ("survey-015", "bottom_right_ui_cutoff"),
    ("survey-023", "aura_overlap"),
    ("survey-012", "left_lich_portal_damage"),
    ("survey-031", "no_lich_sparse_negative"),
    ("survey-036", "dense_center_lich"),
    ("survey-039", "portal_damage_overlap"),
    ("survey-022", "low_left_lich_open_scene"),
    ("survey-047", "airborne_hero_edge_case"),
    ("survey-055", "dense_late_combat"),
    ("survey-056", "lich_below_airborne_hero"),
)
PRESET_SIZES = {"1": (83.0, 131.0), "2": (123.0, 144.0), "3": (123.0, 202.0)}
PRESET_NAMES = {"1": "zombie", "2": "hero", "3": "lich"}

# Visually audited fixed-size labels: (preset, ground x, ground y). The low-
# confidence detector pass was used only as a starting point. Duplicate boxes
# were removed; missed overlapping, effect-obscured, and partial edge/UI
# sprites were added from the full-resolution frames.
LABELS = {
    "codex-sample4-lichcase-0000": (
        ("1", 558, 419), ("1", 766, 418), ("1", 1038, 420),
        ("1", 1381, 508), ("1", 1569, 508),
        ("1", 1040, 1080), ("2", 1120, 1080),
    ),
    "codex-sample4-lichcase-0001": (
        ("1", 545, 81), ("1", 693, 83), ("1", 949, 136),
        ("1", 1281, 169), ("1", 1666, 136),
        ("3", 191, 708), ("1", 588, 708), ("1", 699, 708),
        ("2", 1084, 807), ("1", 1398, 802), ("1", 1504, 802),
        ("1", 1622, 802),
    ),
    "codex-sample4-lichcase-0002": (
        ("1", 500, 60), ("1", 675, 60), ("1", 765, 60),
        ("1", 1037, 144), ("1", 1262, 147), ("1", 1790, 124),
        ("1", 1850, 124), ("3", 176, 689), ("1", 334, 686),
        ("1", 683, 683), ("2", 905, 688), ("1", 1484, 777),
        ("1", 1703, 800), ("1", 1785, 858),
    ),
    "codex-sample4-lichcase-0003": (
        ("1", 550, 37), ("1", 750, 37), ("1", 890, 37),
        ("1", 970, 37), ("1", 16, 570), ("3", 211, 573),
        ("1", 710, 670), ("2", 891, 667),
    ),
    "codex-sample4-lichcase-0004": (
        ("2", 330, 757), ("1", 603, 749), ("1", 739, 750),
        ("3", 927, 692), ("1", 1066, 663), ("1", 1904, 570),
    ),
    "codex-sample4-lichcase-0005": (
        ("2", 905, 575), ("1", 1694, 287), ("1", 1197, 946),
        ("1", 1288, 922), ("3", 1612, 922),
    ),
    "codex-sample4-lichcase-0006": (
        ("1", 1162, 130), ("1", 1486, 216), ("1", 1687, 213),
        ("1", 1765, 213), ("1", 1909, 207), ("3", 457, 786),
        ("1", 455, 787), ("2", 1053, 758), ("1", 1307, 849),
        ("1", 1356, 849), ("1", 1508, 849),
    ),
    "codex-sample4-lichcase-0007": (
        ("1", 724, 193), ("1", 1085, 281), ("1", 1469, 286),
        ("1", 52, 915), ("3", 214, 849), ("1", 519, 827),
        ("1", 587, 823), ("2", 847, 828), ("1", 1196, 914),
        ("1", 1302, 915), ("1", 1413, 917), ("1", 1594, 915),
        ("1", 1832, 934), ("1", 1899, 983),
    ),
    "codex-sample4-lichcase-0008": (
        ("2", 793, 609), ("1", 1494, 673),
    ),
    "codex-sample4-lichcase-0009": (
        ("1", 176, 769), ("1", 272, 768), ("2", 381, 766),
        ("1", 545, 768), ("1", 660, 767), ("1", 865, 754),
        ("3", 945, 680), ("1", 1066, 676),
    ),
    "codex-sample4-lichcase-0010": (
        ("1", 777, 134), ("1", 1209, 132), ("1", 1545, 219),
        ("1", 1600, 219), ("3", 243, 862), ("1", 545, 841),
        ("1", 963, 763), ("2", 1051, 762), ("1", 1179, 758),
        ("1", 1579, 853), ("1", 1802, 853), ("1", 1854, 852),
    ),
    "codex-sample4-lichcase-0011": (
        ("1", 1768, 290), ("1", 1895, 292), ("1", 247, 1080),
        ("1", 330, 1080), ("3", 691, 925), ("1", 691, 926),
        ("2", 1117, 833), ("1", 1748, 923), ("1", 1775, 923),
    ),
    "codex-sample4-lichcase-0012": (
        ("2", 790, 604), ("1", 1511, 725), ("1", 1829, 816),
    ),
    "codex-sample4-lichcase-0013": (
        ("1", 317, 788), ("1", 399, 782), ("2", 543, 785),
        ("1", 620, 788), ("1", 785, 784), ("1", 824, 782),
        ("3", 1016, 697),
    ),
    "codex-sample4-lichcase-0014": (
        ("1", 346, 816), ("1", 468, 816), ("3", 839, 810),
        ("2", 1048, 626),
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


def survey() -> dict[str, object]:
    reference = cv2.imread(str(MINIMAP_REFERENCE))
    if reference is None:
        raise RuntimeError(f"Could not read minimap reference {MINIMAP_REFERENCE}")
    prior_manifest = json.loads(
        (MAP_DIR / "dataset/codex_manual_batch_v2_manifest.json").read_text(encoding="utf-8")
    )
    prior_times = [float(row["timestamp"]) for row in prior_manifest["frames"]]
    capture = cv2.VideoCapture(str(SOURCE_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {SOURCE_VIDEO}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    rows = []
    panels = []
    try:
        timestamp = 4.0
        while timestamp <= duration - 4.0:
            if all(abs(timestamp - prior) >= EXISTING_SEPARATION_S for prior in prior_times):
                frame_index = round(timestamp * fps)
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not decode frame {frame_index}")
                correlation = _minimap_correlation(image, reference)
                if correlation >= MINIMAP_CORRELATION_THRESHOLD:
                    candidate_id = f"survey-{len(rows):03d}"
                    rows.append({
                        "candidate_id": candidate_id,
                        "source_frame": frame_index,
                        "timestamp": frame_index / fps,
                        "minimap_correlation": round(correlation, 6),
                    })
                    panel = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
                    cv2.rectangle(panel, (0, 0), (480, 30), (0, 0, 0), -1)
                    cv2.putText(
                        panel, f"{candidate_id}  t={frame_index / fps:.3f}s  map={correlation:.3f}",
                        (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA,
                    )
                    panels.append(panel)
            timestamp += SURVEY_INTERVAL_S
    finally:
        capture.release()

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    sheet_paths = []
    per_sheet = 16
    for sheet_index in range(0, len(panels), per_sheet):
        page = panels[sheet_index:sheet_index + per_sheet]
        blank = np.zeros_like(page[0])
        while len(page) < per_sheet:
            page.append(blank.copy())
        sheet = cv2.vconcat([
            cv2.hconcat(page[row:row + 4]) for row in range(0, per_sheet, 4)
        ])
        path = STAGING_DIR / f"survey-{sheet_index // per_sheet + 1:02d}.jpg"
        if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write {path}")
        sheet_paths.append(path.relative_to(ROOT).as_posix())

    manifest = {
        "schema": "specialized-visual-detection.lich-diversity-survey.v1",
        "source_video": str(SOURCE_VIDEO),
        "interval_s": SURVEY_INTERVAL_S,
        "existing_batch_separation_s": EXISTING_SEPARATION_S,
        "minimum_minimap_correlation": MINIMAP_CORRELATION_THRESHOLD,
        "candidates": rows,
        "contact_sheets": sheet_paths,
    }
    manifest_path = STAGING_DIR / "survey-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "candidates": len(rows),
        "manifest": str(manifest_path),
        "contact_sheets": [str(ROOT / path) for path in sheet_paths],
    }


def select() -> dict[str, object]:
    survey_manifest = json.loads(
        (STAGING_DIR / "survey-manifest.json").read_text(encoding="utf-8")
    )
    by_id = {row["candidate_id"]: row for row in survey_manifest["candidates"]}
    missing = [candidate_id for candidate_id, _case in SELECTED_CASES if candidate_id not in by_id]
    if missing:
        raise ValueError(f"Survey candidates are missing: {missing}")
    selected_dir = STAGING_DIR / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(SOURCE_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {SOURCE_VIDEO}")
    rows = []
    panels = []
    try:
        for index, (candidate_id, case) in enumerate(SELECTED_CASES):
            survey_row = by_id[candidate_id]
            frame_id = f"codex-sample4-lichcase-{index:04d}"
            frame_index = int(survey_row["source_frame"])
            timestamp = float(survey_row["timestamp"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {frame_index}")
            destination = selected_dir / f"{frame_id}_{timestamp:010.3f}.jpg"
            if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Could not write {destination}")
            relative = destination.relative_to(ROOT).as_posix()
            rows.append({
                "id": frame_id,
                "frame_id": frame_id,
                "candidate_id": candidate_id,
                "case": case,
                "source_frame": frame_index,
                "time_s": timestamp,
                "timestamp": timestamp,
                "kind": case,
                "image": relative,
                "image_path": relative,
                "image_sha256": sha256_file(destination),
                "minimap_correlation": survey_row["minimap_correlation"],
            })
            panel = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
            cv2.rectangle(panel, (0, 0), (480, 42), (0, 0, 0), -1)
            cv2.putText(panel, f"{frame_id}  t={timestamp:.3f}s", (7, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(panel, case, (7, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        .38, (180, 230, 255), 1, cv2.LINE_AA)
            panels.append(panel)
    finally:
        capture.release()

    blank = np.zeros_like(panels[0])
    panels.append(blank)
    sheet = cv2.vconcat([
        cv2.hconcat(panels[row:row + 4]) for row in range(0, 16, 4)
    ])
    sheet_path = STAGING_DIR / "selected-raw-contact-sheet.jpg"
    if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not write {sheet_path}")
    manifest = {
        "schema": "specialized-visual-detection.lich-diversity-selection.v1",
        "source_video": str(SOURCE_VIDEO),
        "selection_goal": "diverse Lich observation and Lich-absent comparison cases",
        "test_frames": rows,
        "frames": rows,
        "contact_sheet": sheet_path.relative_to(ROOT).as_posix(),
    }
    manifest_path = STAGING_DIR / "selection-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "frames": len(rows),
        "manifest": str(manifest_path),
        "contact_sheet": str(sheet_path),
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
    selection_path = STAGING_DIR / "selection-manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    frames = selection["frames"]
    frame_ids = [str(row["frame_id"]) for row in frames]
    if set(frame_ids) != set(LABELS):
        raise ValueError("Selected frame IDs do not match the visually audited label table")

    annotations_path = MAP_DIR / "dataset/annotations.jsonl"
    existing = read_jsonl(annotations_path)
    duplicate_ids = sorted({item.frame_id for item in existing} & set(frame_ids))
    if duplicate_ids:
        raise ValueError(f"Batch already exists in canonical annotations: {duplicate_ids}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    additions = []
    manifest_frames = []
    panels = []
    for row in frames:
        frame_id = str(row["frame_id"])
        labels = LABELS[frame_id]
        source_image = ROOT / str(row["image_path"])
        destination = IMAGES_DIR / source_image.name
        if destination.exists():
            if sha256_file(destination) != sha256_file(source_image):
                raise FileExistsError(f"Refusing to overwrite a different image: {destination}")
        else:
            shutil.copy2(source_image, destination)
        image_path = destination.relative_to(ROOT).as_posix()

        monsters = [_monster(*label) for label in labels]
        has_lich = any(label[0] == "3" for label in labels)
        conditions = ["sample4_lotta_lich", "valid_minimap", str(row["case"])]
        conditions.append("lich_visible" if has_lich else "lich_absent_comparison")
        additions.append(FrameAnnotation(
            frame_id=frame_id,
            timestamp=float(row["timestamp"]),
            image_path=image_path,
            monsters=monsters,
            split="train",
            review_status="pending",
            category="codex_manual_sample4_lich_diversity",
            conditions=conditions,
            notes=(
                "Codex detector-assisted proposal, then visually audited at full resolution for "
                "Lich observation diversity, class, duplicates, missed overlapping sprites, "
                "effects, and partial edge/UI cutoffs. Human confirmation required."
            ),
        ))

        counts = {
            name: sum(label[0] == preset for label in labels)
            for preset, name in PRESET_NAMES.items()
        }
        manifest_frames.append({
            **{key: value for key, value in row.items()
               if key not in {"id", "time_s", "kind", "image", "image_path"}},
            "image_path": image_path,
            "proposal_count": len(labels),
            "class_counts": counts,
            "lich_observation": "visible" if has_lich else "absent_comparison",
            "visual_audit": "completed_pending_human_confirmation",
        })
        image = cv2.imread(str(destination))
        if image is None:
            raise RuntimeError(f"Could not read {destination}")
        _draw_labels(image, labels)
        caption = (
            f"{frame_id} {row['case']}  "
            f"z={counts['zombie']} h={counts['hero']} l={counts['lich']}"
        )
        cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(image, caption, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    .48, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA))

    validation_errors = {
        item.frame_id: item.validate(1920, 1080)
        for item in additions if item.validate(1920, 1080)
    }
    if validation_errors:
        raise ValueError(f"Invalid audited proposals: {validation_errors}")
    write_jsonl_with_backup(existing + additions, annotations_path)

    sheet = cv2.vconcat([
        cv2.hconcat(panels[index:index + 5]) for index in range(0, len(panels), 5)
    ])
    contact_path = STAGING_DIR / "audited-label-contact-sheet.jpg"
    if not cv2.imwrite(str(contact_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not write {contact_path}")

    manifest = {
        "schema": "specialized-visual-detection.codex-manual-review-batch.v1",
        "map_id": "forest-of-dead-trees-2",
        "labeling_method": "detector_assisted_then_full_resolution_visual_audit",
        "selection": {
            "kind": "stratified_lich_observation_valid_minimap",
            "count": len(frames),
            "survey_interval_s": SURVEY_INTERVAL_S,
            "minimum_minimap_correlation": MINIMAP_CORRELATION_THRESHOLD,
            "lich_positive_frames": sum(
                any(label[0] == "3" for label in LABELS[frame_id])
                for frame_id in frame_ids
            ),
            "lich_absent_comparison_frames": sum(
                not any(label[0] == "3" for label in LABELS[frame_id])
                for frame_id in frame_ids
            ),
        },
        "review": {
            "split": "train",
            "status": "pending",
            "command": "maps\\forest-of-dead-trees-2\\review_codex_labels_3.bat",
        },
        "source": {
            "video": str(SOURCE_VIDEO),
            "video_sha256": sha256_file(SOURCE_VIDEO),
        },
        "frames": manifest_frames,
        "audit": {
            "all_frames_visually_inspected": True,
            "audited_contact_sheet": contact_path.relative_to(ROOT).as_posix(),
            "checks": [
                "valid_minimap", "lich_observation_diversity", "class", "duplicates",
                "missed_sprites", "overlap_and_effects", "partial_edges",
                "lower_ui_cutoffs",
            ],
        },
    }
    manifest_path = MAP_DIR / "dataset/codex_manual_batch_v3_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "annotations": str(annotations_path),
        "manifest": str(manifest_path),
        "frames_added": len(additions),
        "lich_positive_frames": manifest["selection"]["lich_positive_frames"],
        "lich_absent_comparison_frames": manifest["selection"]["lich_absent_comparison_frames"],
        "proposals_added": sum(len(item.monsters) for item in additions),
        "class_counts": {
            name: sum(label[0] == preset for labels in LABELS.values() for label in labels)
            for preset, name in PRESET_NAMES.items()
        },
        "minimap_correlation_range": [
            min(float(row["minimap_correlation"]) for row in frames),
            max(float(row["minimap_correlation"]) for row in frames),
        ],
        "contact_sheet": str(contact_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    selected = sum((args.survey, args.select, args.finalize))
    if selected != 1:
        parser.error("choose exactly one of --survey, --select, or --finalize")
    operation = survey if args.survey else select if args.select else finalize
    print(json.dumps(operation(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
