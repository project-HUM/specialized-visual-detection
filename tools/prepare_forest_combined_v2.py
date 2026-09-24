"""Prepare the full confirmed Forest collection and a held-out Sample4 visual test.

This creates a versioned experiment snapshot without changing canonical review
statuses or the previous combined-v1 export. Every train, validation, and test
frame must pass the native-minimap reference gate.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import shutil
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monster_dataset.annotation_io import export_yolo
from monster_dataset.pilot import _direct_positions
from monster_dataset.schema import read_jsonl, write_jsonl
from monster_dataset.validation import write_report
from perception.core import sha256_file
from perception.workspace import MapWorkspace


MAP_DIR = ROOT / "maps/forest-of-dead-trees-2"
SAMPLE4_VIDEO = Path(
    r"C:\projects\input-flag-inspector\saves_m\ForestOfDeadTreesSample4_LottaLich\screen.mp4"
)
MINIMAP_REFERENCE = Path(
    r"C:\projects\map-info-extractor\forest-of-dead-trees-2-output\minimap_native.png"
)
MINIMAP_CORRELATION_THRESHOLD = .90
PRESET_CLASSES = {"1": "zombie", "2": "hero", "3": "lich"}
TEST_COUNT = 30
TEST_GRID_INTERVAL_S = .5
TEST_LABELED_SEPARATION_S = 1.5
TEST_PAIR_SEPARATION_S = 3.0
TEST_IMAGE_SIZE = (960, 540)
GROUP_SEPARATION_S = 10.0
CONFIRMATION_NOTE = (
    "Included in the full-collection training snapshot after the user requested "
    "training from the complete current collection on 2026-09-23."
)

# Nearby frames stay together. The early Sample2 Lich cluster and two Sample4
# temporal groups provide a 20% holdout with six Lich instances, while every
# other group remains in training.
VALIDATION_IDS = {
    "pilot-0002",
    "codex-random-0004",
    "pilot-0005",
    "codex-lich-early-0000",
    "codex-lich-early-0001",
    "codex-sample4-lichcase-0011",
    "codex-sample4-lichcase-0006",
    "codex-sample4-random-0002",
    "codex-sample4-lichcase-0008",
}


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _minimap_correlation(frame: np.ndarray, reference: np.ndarray,
                         crop: tuple[int, int, int, int]) -> float:
    x, y, width, height = crop
    roi = frame[y:y + height, x:x + width]
    if roi.shape[:2] != reference.shape[:2]:
        return -1.0
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype("float32")
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype("float32")
    roi_gray -= roi_gray.mean()
    reference_gray -= reference_gray.mean()
    denominator = float(cv2.norm(roi_gray) * cv2.norm(reference_gray))
    return float((roi_gray * reference_gray).sum() / denominator) if denominator else -1.0


def _manifest_frame_indices() -> dict[str, int]:
    specs = (
        ("pilot_manifest.json", ("frame", "source_frame", "frame_index")),
        ("codex_manual_batch_v1_manifest.json", ("source_frame", "frame", "frame_index")),
        ("codex_manual_batch_v2_manifest.json", ("source_frame", "frame_index", "frame")),
        ("codex_manual_batch_v3_manifest.json", ("source_frame", "frame_index", "frame")),
        ("codex_manual_batch_v4_manifest.json", ("source_frame", "frame_index", "frame")),
    )
    result: dict[str, int] = {}
    for filename, fields in specs:
        payload = json.loads((MAP_DIR / "dataset" / filename).read_text(encoding="utf-8"))
        for row in payload["frames"]:
            field = next((candidate for candidate in fields if candidate in row), None)
            if field is None:
                raise ValueError(f"{filename} row has no source-frame field: {row}")
            frame_id = str(row["frame_id"])
            if frame_id in result:
                raise ValueError(f"Duplicate frame ID across manifests: {frame_id}")
            result[frame_id] = int(row[field])
    return result


def _source_name(frame_id: str) -> str:
    return "sample4" if frame_id.startswith("codex-sample4-") else "sample2"


def _audit_labeled_frames(items, frame_indices: dict[str, int], *, workspace: MapWorkspace,
                          reference: np.ndarray) -> tuple[dict[str, float], dict[str, str]]:
    videos = {"sample2": workspace.video, "sample4": SAMPLE4_VIDEO}
    captures = {name: cv2.VideoCapture(str(path)) for name, path in videos.items()}
    if any(not capture.isOpened() for capture in captures.values()):
        raise RuntimeError("Could not open one or more labeled source videos")
    direct_frames = {
        int(row["frame"])
        for row in _direct_positions(workspace.external_path("geometry", "positions"))
    }
    scores: dict[str, float] = {}
    exclusions: dict[str, str] = {}
    try:
        for item in items:
            if item.frame_id not in frame_indices:
                raise ValueError(f"No source-frame manifest entry for {item.frame_id}")
            source = _source_name(item.frame_id)
            frame_index = frame_indices[item.frame_id]
            if source == "sample2" and frame_index not in direct_frames:
                exclusions[item.frame_id] = "no_exact_direct_minimap_marker"
                continue
            capture = captures[source]
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode {source} frame {frame_index}")
            score = _minimap_correlation(frame, reference, workspace.minimap_crop)
            scores[item.frame_id] = score
            if score < MINIMAP_CORRELATION_THRESHOLD:
                exclusions[item.frame_id] = "minimap_reference_correlation_below_threshold"
    finally:
        for capture in captures.values():
            capture.release()
    return scores, exclusions


def _assert_group_isolation(items) -> list[dict[str, object]]:
    eligible = [item for item in items if item.split in {"train", "validation"}]
    leaks = []
    for index, left in enumerate(eligible):
        for right in eligible[index + 1:]:
            if left.split == right.split or _source_name(left.frame_id) != _source_name(right.frame_id):
                continue
            distance = abs(float(left.timestamp) - float(right.timestamp))
            if distance < GROUP_SEPARATION_S:
                leaks.append({
                    "left": left.frame_id,
                    "right": right.frame_id,
                    "distance_s": distance,
                })
    if leaks:
        raise ValueError(f"Train/validation temporal leakage: {leaks}")
    return leaks


def _sample_test(*, seed: int, labeled_times: list[float], reference: np.ndarray,
                 crop: tuple[int, int, int, int], output: Path) -> list[dict[str, object]]:
    capture = cv2.VideoCapture(str(SAMPLE4_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {SAMPLE4_VIDEO}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    candidate_times = []
    timestamp = 4.0
    while timestamp <= duration - 4.0:
        if all(abs(timestamp - labeled) >= TEST_LABELED_SEPARATION_S
               for labeled in labeled_times):
            candidate_times.append(timestamp)
        timestamp += TEST_GRID_INTERVAL_S
    random.Random(seed).shuffle(candidate_times)

    selected: list[dict[str, object]] = []
    try:
        for timestamp in candidate_times:
            if any(abs(timestamp - float(row["time_s"])) < TEST_PAIR_SEPARATION_S
                   for row in selected):
                continue
            frame_index = round(timestamp * fps)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            score = _minimap_correlation(frame, reference, crop)
            if score < MINIMAP_CORRELATION_THRESHOLD:
                continue
            selected.append({
                "frame": frame_index,
                "time_s": frame_index / fps,
                "minimap_correlation": score,
                "frame_image": frame,
            })
            if len(selected) == TEST_COUNT:
                break
    finally:
        capture.release()
    if len(selected) != TEST_COUNT:
        raise RuntimeError(f"Found only {len(selected)} eligible held-out test frames")
    selected.sort(key=lambda row: float(row["time_s"]))

    test_images = output / "test/images"
    test_images.mkdir(parents=True, exist_ok=True)
    rows = []
    panels = []
    for index, row in enumerate(selected):
        test_id = f"combined-v2-test-random-{index:02d}"
        frame = cv2.resize(row.pop("frame_image"), TEST_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        destination = test_images / f"{test_id}_{float(row['time_s']):010.3f}.jpg"
        if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write {destination}")
        result = {
            "id": test_id,
            "kind": "random_valid_minimap",
            "frame": int(row["frame"]),
            "time_s": float(row["time_s"]),
            "image": _relative(destination),
            "image_sha256": sha256_file(destination),
            "minimap_correlation": round(float(row["minimap_correlation"]), 6),
            "minimum_labeled_separation_s": min(
                abs(float(row["time_s"]) - labeled) for labeled in labeled_times
            ),
        }
        rows.append(result)
        panel = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (0, 0), (320, 25), (0, 0, 0), -1)
        cv2.putText(panel, f"{test_id} t={result['time_s']:.2f} map={result['minimap_correlation']:.3f}",
                    (5, 17), cv2.FONT_HERSHEY_SIMPLEX, .34, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    sheet = cv2.vconcat([
        cv2.hconcat(panels[index:index + 5]) for index in range(0, TEST_COUNT, 5)
    ])
    sheet_path = output / "test/contact-sheet-unlabeled.jpg"
    if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not write {sheet_path}")
    return rows


def prepare(output: Path, *, seed: int) -> dict[str, object]:
    workspace = MapWorkspace.load(ROOT, "forest-of-dead-trees-2")
    if output.exists():
        raise FileExistsError(f"Experiment output already exists: {output}")
    output.mkdir(parents=True)

    canonical_path = workspace.path("annotations")
    canonical = read_jsonl(canonical_path)
    ids = [item.frame_id for item in canonical]
    if len(ids) != len(set(ids)):
        raise ValueError("Canonical annotations contain duplicate frame IDs")
    frame_indices = _manifest_frame_indices()
    if set(ids) != set(frame_indices):
        raise ValueError(
            f"Canonical/manifest ID mismatch: missing={sorted(set(ids)-set(frame_indices))}, "
            f"extra={sorted(set(frame_indices)-set(ids))}"
        )
    reference = cv2.imread(str(MINIMAP_REFERENCE))
    if reference is None:
        raise RuntimeError(f"Could not read {MINIMAP_REFERENCE}")
    scores, exclusions = _audit_labeled_frames(
        canonical, frame_indices, workspace=workspace, reference=reference,
    )
    if set(VALIDATION_IDS) & set(exclusions):
        raise ValueError(f"Validation contains ineligible frames: {set(VALIDATION_IDS) & set(exclusions)}")

    source_copy = output / "source_annotations.current_collection.jsonl"
    shutil.copy2(canonical_path, source_copy)
    promoted = copy.deepcopy(canonical)
    for item in promoted:
        if item.frame_id in exclusions:
            item.split = "pilot"
            item.category = "excluded_invalid_minimap"
            if "invalid_minimap" not in item.conditions:
                item.conditions.append("invalid_minimap")
        else:
            item.split = "validation" if item.frame_id in VALIDATION_IDS else "train"
        item.review_status = "reviewed"
        item.notes = (item.notes.strip() + " " + CONFIRMATION_NOTE).strip()
        for monster in item.monsters:
            if monster.source == "codex_prelabel":
                monster.source = "human_confirmed_prelabel"
            monster.review_required = False
    _assert_group_isolation(promoted)

    snapshot_path = output / "training_annotations.jsonl"
    write_jsonl(promoted, snapshot_path)
    report = write_report(promoted, output / "dataset-report.json",
                          source_size=workspace.source_size)
    if not report["training_ready"]:
        raise RuntimeError("Full-collection snapshot is not training-ready")

    yolo_dir = output / "yolo_dataset"
    train_export = export_yolo(
        promoted, yolo_dir, split="train", source_size=workspace.source_size,
        image_root=workspace.repo_root, preset_classes=PRESET_CLASSES,
    )
    validation_export = export_yolo(
        promoted, yolo_dir, split="validation", source_size=workspace.source_size,
        image_root=workspace.repo_root, preset_classes=PRESET_CLASSES,
    )

    sample4_labeled_times = [
        float(item.timestamp) for item in promoted
        if _source_name(item.frame_id) == "sample4" and item.frame_id not in exclusions
    ]
    test_rows = _sample_test(
        seed=seed, labeled_times=sample4_labeled_times, reference=reference,
        crop=workspace.minimap_crop, output=output,
    )
    train_ids = [item.frame_id for item in promoted if item.split == "train"]
    validation_ids = [item.frame_id for item in promoted if item.split == "validation"]
    manifest = {
        "schema": "specialized-visual-detection.combined-learning-experiment.v2",
        "map_id": workspace.map_id,
        "seed": seed,
        "classes": PRESET_CLASSES,
        "eligibility": {
            "minimum_minimap_reference_correlation": MINIMAP_CORRELATION_THRESHOLD,
            "invalid_minimap_frames_allowed": False,
            "minimap_reference": str(MINIMAP_REFERENCE),
            "labeled_minimap_scores": {
                frame_id: round(score, 6) for frame_id, score in sorted(scores.items())
            },
            "excluded_labeled_frames": [
                {"frame_id": frame_id, "reason": reason,
                 "minimap_correlation": round(scores.get(frame_id, -1.0), 6)}
                for frame_id, reason in sorted(exclusions.items())
            ],
        },
        "confirmation": {
            "method": "explicit_user_request_to_train_from_full_current_collection",
            "date": "2026-09-23",
            "canonical_statuses_left_unchanged": True,
            "source_annotations": _relative(source_copy),
            "source_annotations_sha256": sha256_file(source_copy),
            "training_annotations": _relative(snapshot_path),
            "training_annotations_sha256": sha256_file(snapshot_path),
        },
        "split_strategy": {
            "method": "source-aware temporal-group holdout",
            "minimum_cross_split_temporal_separation_s": GROUP_SEPARATION_S,
            "train_frame_ids": train_ids,
            "validation_frame_ids": validation_ids,
            "note": (
                "Nearby timestamps from the same recording stay together. Validation contains "
                "the full early Sample2 Lich cluster and two isolated Sample4 temporal groups."
            ),
        },
        "exports": {"train": train_export, "validation": validation_export},
        "test_selection": {
            "source": str(SAMPLE4_VIDEO),
            "method": "deterministic random valid-minimap sampling",
            "count": TEST_COUNT,
            "grid_interval_s": TEST_GRID_INTERVAL_S,
            "minimum_labeled_separation_s": TEST_LABELED_SEPARATION_S,
            "minimum_pair_separation_s": TEST_PAIR_SEPARATION_S,
            "annotations": "unlabeled_visual_test",
            "contact_sheet": _relative(output / "test/contact-sheet-unlabeled.jpg"),
        },
        "sources": {
            "sample2_video": str(workspace.video.resolve()),
            "sample2_video_sha256": sha256_file(workspace.video),
            "sample4_video": str(SAMPLE4_VIDEO.resolve()),
            "sample4_video_sha256": sha256_file(SAMPLE4_VIDEO),
        },
        "test_frames": test_rows,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=MAP_DIR / "dataset/runs/combined-v2",
    )
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    manifest = prepare(args.output, seed=args.seed)
    print(json.dumps({
        "output": str(args.output),
        "train": manifest["exports"]["train"],
        "validation": manifest["exports"]["validation"],
        "excluded": manifest["eligibility"]["excluded_labeled_frames"],
        "test_frames": len(manifest["test_frames"]),
        "test_minimap_range": [
            min(row["minimap_correlation"] for row in manifest["test_frames"]),
            max(row["minimap_correlation"] for row in manifest["test_frames"]),
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
