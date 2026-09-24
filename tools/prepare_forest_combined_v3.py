"""Prepare the expanded Forest collection and balanced Sample4/Sample6 visual test."""
from __future__ import annotations

import argparse
from collections import Counter
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
from monster_dataset.schema import read_jsonl, write_jsonl
from monster_dataset.validation import write_report
from perception.core import sha256_file
from perception.workspace import MapWorkspace
from tools import prepare_forest_combined_v2 as v2


MAP_DIR = ROOT / "maps/forest-of-dead-trees-2"
SAMPLE4_VIDEO = Path(
    r"C:\projects\input-flag-inspector\saves_m\ForestOfDeadTreesSample4_LottaLich\screen.mp4"
)
SAMPLE6_VIDEO = Path(
    r"C:\projects\input-flag-inspector\saves_m\ForestOfDeadTreesSample6\screen.mp4"
)
MINIMAP_REFERENCE = v2.MINIMAP_REFERENCE
MINIMAP_CORRELATION_THRESHOLD = .90
PRESET_CLASSES = {"1": "zombie", "2": "hero", "3": "lich"}
TEST_COUNT_PER_SOURCE = 15
TEST_GRID_INTERVAL_S = .5
TEST_EXCLUSION_SEPARATION_S = 1.5
TEST_PAIR_SEPARATION_S = 3.0
TEST_IMAGE_SIZE = (960, 540)
GROUP_SEPARATION_S = 10.0

# Complete temporal groups are held out. This yields 12/60 validation frames,
# includes both source recordings, and retains 8 Lich instances in validation.
VALIDATION_IDS = {
    "codex-lich-early-0000", "pilot-0005", "codex-lich-early-0001",
    "pilot-0002",
    "codex-sample4-hardcase-0003", "codex-sample4-random-0007",
    "codex-sample4-lichcase-0013", "codex-sample4-lichcase-0014",
    "codex-sample4-random-0008", "codex-sample4-hardcase-0004",
    "codex-sample4-hardcase-0014", "codex-sample4-random-0009",
}


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _sample_source(*, source_id: str, video: Path, seed: int,
                   excluded_times: list[float], reference: np.ndarray,
                   crop: tuple[int, int, int, int], output: Path) -> list[dict[str, object]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    candidates: list[float] = []
    timestamp = 4.0
    while timestamp <= duration - 4.0:
        if all(abs(timestamp - excluded) >= TEST_EXCLUSION_SEPARATION_S
               for excluded in excluded_times):
            candidates.append(timestamp)
        timestamp += TEST_GRID_INTERVAL_S
    random.Random(seed).shuffle(candidates)

    selected: list[dict[str, object]] = []
    try:
        for timestamp in candidates:
            if any(abs(timestamp - float(row["time_s"])) < TEST_PAIR_SEPARATION_S
                   for row in selected):
                continue
            frame_index = round(timestamp * fps)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            score = v2._minimap_correlation(frame, reference, crop)
            if score < MINIMAP_CORRELATION_THRESHOLD:
                continue
            selected.append({
                "frame": frame_index,
                "time_s": frame_index / fps,
                "minimap_correlation": score,
                "frame_image": frame,
            })
            if len(selected) == TEST_COUNT_PER_SOURCE:
                break
    finally:
        capture.release()
    if len(selected) != TEST_COUNT_PER_SOURCE:
        raise RuntimeError(
            f"Found only {len(selected)} eligible held-out frames in {source_id}"
        )
    selected.sort(key=lambda row: float(row["time_s"]))

    images_dir = output / "test/images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, row in enumerate(selected):
        test_id = f"combined-v3-test-{source_id}-random-{index:02d}"
        frame = cv2.resize(row.pop("frame_image"), TEST_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        destination = images_dir / f"{test_id}_{float(row['time_s']):010.3f}.jpg"
        if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write {destination}")
        nearest_exclusion = (
            min(abs(float(row["time_s"]) - excluded) for excluded in excluded_times)
            if excluded_times else None
        )
        rows.append({
            "id": test_id,
            "kind": f"{source_id}_random_valid_minimap",
            "source": source_id,
            "frame": int(row["frame"]),
            "time_s": float(row["time_s"]),
            "image": _relative(destination),
            "image_sha256": sha256_file(destination),
            "minimap_correlation": round(float(row["minimap_correlation"]), 6),
            "minimum_excluded_time_separation_s": nearest_exclusion,
        })
    return rows


def _write_test_contact_sheet(rows: list[dict[str, object]], output: Path) -> Path:
    panels = []
    for row in rows:
        image = cv2.imread(str(ROOT / str(row["image"])))
        if image is None:
            raise RuntimeError(f"Could not read {row['image']}")
        panel = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (0, 0), (320, 25), (0, 0, 0), -1)
        caption = (
            f"{row['source']} #{str(row['id'])[-2:]} t={float(row['time_s']):.2f} "
            f"map={float(row['minimap_correlation']):.3f}"
        )
        cv2.putText(panel, caption, (5, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    .34, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    path = output / "test/contact-sheet-unlabeled.jpg"
    sheet = cv2.vconcat([
        cv2.hconcat(panels[index:index + 5]) for index in range(0, len(panels), 5)
    ])
    if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not write {path}")
    return path


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
    frame_indices = v2._manifest_frame_indices()
    if set(ids) != set(frame_indices):
        raise ValueError(
            f"Canonical/manifest ID mismatch: missing={sorted(set(ids)-set(frame_indices))}, "
            f"extra={sorted(set(frame_indices)-set(ids))}"
        )
    reference = cv2.imread(str(MINIMAP_REFERENCE))
    if reference is None:
        raise RuntimeError(f"Could not read {MINIMAP_REFERENCE}")
    scores, exclusions = v2._audit_labeled_frames(
        canonical, frame_indices, workspace=workspace, reference=reference,
    )
    if set(VALIDATION_IDS) & set(exclusions):
        raise ValueError(f"Validation contains ineligible frames: {set(VALIDATION_IDS) & set(exclusions)}")

    source_copy = output / "source_annotations.current_collection.jsonl"
    shutil.copy2(canonical_path, source_copy)
    source_statuses = Counter(item.review_status for item in canonical)
    source_review_required = sum(
        monster.review_required for item in canonical for monster in item.monsters
    )
    promoted = copy.deepcopy(canonical)
    confirmation_note = (
        "Included only in the combined-v3 experiment snapshot after the user explicitly "
        "requested retraining with the newly added collection on 2026-09-23. Canonical "
        "review metadata remains unchanged."
    )
    for item in promoted:
        if item.frame_id in exclusions:
            item.split = "pilot"
            item.category = "excluded_invalid_minimap"
            if "invalid_minimap" not in item.conditions:
                item.conditions.append("invalid_minimap")
        else:
            item.split = "validation" if item.frame_id in VALIDATION_IDS else "train"
        item.review_status = "reviewed"
        item.notes = (item.notes.strip() + " " + confirmation_note).strip()
        for monster in item.monsters:
            if monster.source == "codex_prelabel":
                monster.source = "user_authorized_training_snapshot"
            monster.review_required = False
    v2._assert_group_isolation(promoted)

    snapshot_path = output / "training_annotations.jsonl"
    write_jsonl(promoted, snapshot_path)
    report = write_report(promoted, output / "dataset-report.json",
                          source_size=workspace.source_size)
    if not report["training_ready"]:
        raise RuntimeError("Expanded snapshot is not training-ready")
    yolo_dir = output / "yolo_dataset"
    train_export = export_yolo(
        promoted, yolo_dir, split="train", source_size=workspace.source_size,
        image_root=workspace.repo_root, preset_classes=PRESET_CLASSES,
    )
    validation_export = export_yolo(
        promoted, yolo_dir, split="validation", source_size=workspace.source_size,
        image_root=workspace.repo_root, preset_classes=PRESET_CLASSES,
    )

    prior_test = json.loads(
        (MAP_DIR / "dataset/runs/combined-v2/manifest.json").read_text(encoding="utf-8")
    )
    sample4_exclusions = [
        float(item.timestamp) for item in promoted
        if v2._source_name(item.frame_id) == "sample4" and item.frame_id not in exclusions
    ]
    sample4_exclusions.extend(float(row["time_s"]) for row in prior_test["test_frames"])
    sample4_rows = _sample_source(
        source_id="sample4", video=SAMPLE4_VIDEO, seed=seed,
        excluded_times=sample4_exclusions, reference=reference,
        crop=workspace.minimap_crop, output=output,
    )
    sample6_rows = _sample_source(
        source_id="sample6", video=SAMPLE6_VIDEO, seed=seed + 1,
        excluded_times=[], reference=reference,
        crop=workspace.minimap_crop, output=output,
    )
    test_rows = sample4_rows + sample6_rows
    contact_sheet = _write_test_contact_sheet(test_rows, output)

    train_ids = [item.frame_id for item in promoted if item.split == "train"]
    validation_ids = [item.frame_id for item in promoted if item.split == "validation"]
    manifest = {
        "schema": "specialized-visual-detection.combined-learning-experiment.v3",
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
            "method": "explicit_user_request_to_retrain_with_newly_added_collection",
            "date": "2026-09-23",
            "canonical_statuses_left_unchanged": True,
            "canonical_source_statuses": dict(sorted(source_statuses.items())),
            "canonical_review_required_instances": source_review_required,
            "snapshot_promotion_scope": "combined-v3_only",
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
                "Validation holds the complete early Sample2 Lich group, one isolated "
                "Sample2 general frame, and the complete late Sample4 temporal group."
            ),
        },
        "exports": {"train": train_export, "validation": validation_export},
        "test_selection": {
            "method": "deterministic seeded random valid-minimap sampling",
            "count": len(test_rows),
            "count_per_source": TEST_COUNT_PER_SOURCE,
            "grid_interval_s": TEST_GRID_INTERVAL_S,
            "minimum_excluded_time_separation_s": TEST_EXCLUSION_SEPARATION_S,
            "minimum_pair_separation_s": TEST_PAIR_SEPARATION_S,
            "annotations": "unlabeled_visual_test",
            "contact_sheet": _relative(contact_sheet),
            "sources": {
                "sample4": {"video": str(SAMPLE4_VIDEO), "seed": seed,
                            "excluded_current_labels_and_combined_v2_test": True},
                "sample6": {"video": str(SAMPLE6_VIDEO), "seed": seed + 1,
                            "excluded_current_labels_and_combined_v2_test": False},
            },
        },
        "sources": {
            "sample2_video": str(workspace.video.resolve()),
            "sample2_video_sha256": sha256_file(workspace.video),
            "sample4_video": str(SAMPLE4_VIDEO.resolve()),
            "sample4_video_sha256": sha256_file(SAMPLE4_VIDEO),
            "sample6_video": str(SAMPLE6_VIDEO.resolve()),
            "sample6_video_sha256": sha256_file(SAMPLE6_VIDEO),
        },
        "test_frames": test_rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=MAP_DIR / "dataset/runs/combined-v3")
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    manifest = prepare(args.output, seed=args.seed)
    scores = [float(row["minimap_correlation"]) for row in manifest["test_frames"]]
    print(json.dumps({
        "output": str(args.output),
        "train": manifest["exports"]["train"],
        "validation": manifest["exports"]["validation"],
        "excluded": manifest["eligibility"]["excluded_labeled_frames"],
        "test_frames": len(manifest["test_frames"]),
        "test_minimap_range": [min(scores), max(scores)],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
