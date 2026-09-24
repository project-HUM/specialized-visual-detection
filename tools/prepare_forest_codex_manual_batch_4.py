"""Prepare 5 diagnostic and 10 seeded-random blank Forest labeling frames."""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monster_dataset.annotation_io import write_jsonl_with_backup
from monster_dataset.schema import FrameAnnotation, read_jsonl
from perception.core import sha256_file


SOURCE_VIDEO = Path(
    r"C:\projects\input-flag-inspector\saves_m\ForestOfDeadTreeSample4_LottaLich\screen.mp4"
)
MAP_DIR = ROOT / "maps/forest-of-dead-trees-2"
IMAGES_DIR = MAP_DIR / "dataset/images"
STAGING_DIR = MAP_DIR / "dataset/runs/codex-manual-batch-4"
MINIMAP_REFERENCE = Path(
    r"C:\projects\map-info-extractor\forest-of-dead-trees-2-output\minimap_native.png"
)
MINIMAP_CROP = (14, 147, 326, 180)
MINIMAP_CORRELATION_THRESHOLD = .90
RANDOM_SEED = 20260924
RANDOM_COUNT = 10
RANDOM_GRID_INTERVAL_S = .5
EXISTING_SEPARATION_S = 1.5
RANDOM_PAIR_SEPARATION_S = 3.0
COMBINED_V2_MANIFEST = MAP_DIR / "dataset/runs/combined-v2/manifest.json"

# Existing combined-v2 diagnostic frames, deliberately re-extracted at the
# original 1920x1080 resolution. No detector boxes are copied into batch 4.
CASES = (
    ("combined-v2-test-random-07", "low_confidence_effect_obscured_lich", 1187),
    ("combined-v2-test-random-10", "low_confidence_bottom_edge_lich", 1557),
    ("combined-v2-test-random-16", "low_confidence_duplicate_edge_lich", 2599),
    ("combined-v2-test-random-25", "snowy_platform_false_positive", 4103),
    ("combined-v2-test-random-27", "diagnostic_extreme_edge_lich", 4485),
)


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


def _select_random_cases(capture: cv2.VideoCapture, fps: float, reference: np.ndarray,
                         excluded_times: list[float]) -> list[tuple[str, str, int]]:
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    candidates: list[int] = []
    timestamp = 4.0
    while timestamp <= duration - 4.0:
        frame_index = round(timestamp * fps)
        actual_time = frame_index / fps
        if all(abs(actual_time - prior) >= EXISTING_SEPARATION_S for prior in excluded_times):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode random candidate frame {frame_index}")
            if _minimap_correlation(image, reference) >= MINIMAP_CORRELATION_THRESHOLD:
                candidates.append(frame_index)
        timestamp += RANDOM_GRID_INTERVAL_S

    random.Random(RANDOM_SEED).shuffle(candidates)
    selected: list[int] = []
    for frame_index in candidates:
        actual_time = frame_index / fps
        if all(abs(actual_time - other / fps) >= RANDOM_PAIR_SEPARATION_S for other in selected):
            selected.append(frame_index)
            if len(selected) == RANDOM_COUNT:
                break
    if len(selected) != RANDOM_COUNT:
        raise ValueError(f"Found only {len(selected)} eligible random frames")
    selected.sort()
    return [
        (f"new-random-{index:02d}", "random_valid_minimap", frame_index)
        for index, frame_index in enumerate(selected)
    ]


def main() -> int:
    annotations_path = MAP_DIR / "dataset/annotations.jsonl"
    existing = read_jsonl(annotations_path)
    reference = cv2.imread(str(MINIMAP_REFERENCE))
    if reference is None:
        raise RuntimeError(f"Could not read minimap reference {MINIMAP_REFERENCE}")
    capture = cv2.VideoCapture(str(SOURCE_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {SOURCE_VIDEO}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    prior_test_manifest = json.loads(COMBINED_V2_MANIFEST.read_text(encoding="utf-8"))
    excluded_times = [float(item.timestamp) for item in existing]
    excluded_times.extend(float(row["time_s"]) for row in prior_test_manifest["test_frames"])
    selected_cases = list(CASES)
    selected_cases.extend(_select_random_cases(capture, fps, reference, excluded_times))
    frame_ids = [f"codex-sample4-hardcase-{index:04d}" for index in range(len(selected_cases))]
    duplicate_ids = sorted({item.frame_id for item in existing} & set(frame_ids))
    if duplicate_ids:
        raise ValueError(f"Batch 4 already exists in canonical annotations: {duplicate_ids}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    additions: list[FrameAnnotation] = []
    manifest_frames = []
    panels = []
    try:
        for index, (diagnostic_id, case, source_frame) in enumerate(selected_cases):
            frame_id = frame_ids[index]
            timestamp = source_frame / fps
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {source_frame}")
            if image.shape[:2] != (1080, 1920):
                raise ValueError(f"Unexpected source size for {frame_id}: {image.shape[:2]}")
            correlation = _minimap_correlation(image, reference)
            if correlation < MINIMAP_CORRELATION_THRESHOLD:
                raise ValueError(
                    f"{frame_id} fails minimap gate: {correlation:.6f} < "
                    f"{MINIMAP_CORRELATION_THRESHOLD:.2f}"
                )

            destination = IMAGES_DIR / f"{frame_id}_{timestamp:010.3f}.jpg"
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite {destination}")
            if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Could not write {destination}")
            relative_image = destination.relative_to(ROOT).as_posix()
            additions.append(FrameAnnotation(
                frame_id=frame_id,
                timestamp=timestamp,
                image_path=relative_image,
                monsters=[],
                split="train",
                review_status="pending",
                category="codex_manual_sample4_hardcases_blank",
                conditions=["sample4_lotta_lich", "valid_minimap", "manual_from_scratch", case],
                notes=(
                    "Blank manual-label frame selected from Sample4 diagnostic/random candidates. "
                    "No detector proposals or pre-labels were copied. Label every visible "
                    "Zombie, Hero, and Lich before confirming the frame."
                ),
            ))
            manifest_frames.append({
                "frame_id": frame_id,
                "diagnostic_source_id": diagnostic_id,
                "case": case,
                "source_frame": source_frame,
                "timestamp": timestamp,
                "image_path": relative_image,
                "image_sha256": sha256_file(destination),
                "minimap_correlation": round(correlation, 6),
                "proposal_count": 0,
                "prelabeling": "none",
                "review_status": "pending",
            })

            panel = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
            title = np.zeros((38, panel.shape[1], 3), dtype=panel.dtype)
            cv2.putText(title, f"{frame_id}  t={timestamp:.3f}s", (7, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, .40, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(title, case, (7, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        .36, (180, 230, 255), 1, cv2.LINE_AA)
            panels.append(cv2.vconcat((title, panel)))
    finally:
        capture.release()

    validation_errors = {
        item.frame_id: item.validate(1920, 1080)
        for item in additions if item.validate(1920, 1080)
    }
    if validation_errors:
        raise ValueError(f"Invalid blank review frames: {validation_errors}")

    contact_sheet = cv2.vconcat([
        cv2.hconcat(panels[index:index + 5])
        for index in range(0, len(panels), 5)
    ])
    contact_path = STAGING_DIR / "blank-contact-sheet.jpg"
    if not cv2.imwrite(str(contact_path), contact_sheet, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not write {contact_path}")

    manifest = {
        "schema": "specialized-visual-detection.codex-manual-review-batch.v1",
        "map_id": "forest-of-dead-trees-2",
        "labeling_method": "manual_from_scratch_no_prelabels",
        "selection": {
            "kind": "diagnostic_hard_cases_plus_seeded_random_valid_minimap",
            "count": len(additions),
            "diagnostic_count": len(CASES),
            "new_random_count": RANDOM_COUNT,
            "random_seed": RANDOM_SEED,
            "random_grid_interval_s": RANDOM_GRID_INTERVAL_S,
            "existing_frame_separation_s": EXISTING_SEPARATION_S,
            "random_pair_separation_s": RANDOM_PAIR_SEPARATION_S,
            "minimum_minimap_correlation": MINIMAP_CORRELATION_THRESHOLD,
            "all_frames_start_with_zero_boxes": True,
        },
        "review": {
            "split": "train",
            "status": "pending",
            "command": "maps\\forest-of-dead-trees-2\\review_codex_labels_4.bat",
        },
        "source": {
            "video": str(SOURCE_VIDEO),
            "video_sha256": sha256_file(SOURCE_VIDEO),
        },
        "frames": manifest_frames,
        "contact_sheet": contact_path.relative_to(ROOT).as_posix(),
    }
    manifest_path = MAP_DIR / "dataset/codex_manual_batch_v4_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_jsonl_with_backup(existing + additions, annotations_path)
    print(json.dumps({
        "frames_added": len(additions),
        "proposals_added": 0,
        "manifest": str(manifest_path),
        "contact_sheet": str(contact_path),
        "minimap_correlation_range": [
            min(row["minimap_correlation"] for row in manifest_frames),
            max(row["minimap_correlation"] for row in manifest_frames),
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
