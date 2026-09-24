"""Promote confirmed Forest labels and prepare a direct-minimap-only experiment.

The canonical annotations remain the source of truth.  This tool records the
exact user-labeled input before promoting it, assigns a group-aware
train/validation split, exports YOLO data, and samples a separate unlabeled
visual test set exclusively from direct minimap-marker observations.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monster_dataset.annotation_io import export_yolo, write_jsonl_with_backup
from monster_dataset.pilot import _direct_positions, choose_pilot_positions
from monster_dataset.schema import read_jsonl
from monster_dataset.validation import write_report
from perception.core import sha256_file
from perception.workspace import MapWorkspace


EXPECTED_IDS = {f"pilot-{index:04d}" for index in range(7)} | {
    *(f"codex-random-{index:04d}" for index in range(10)),
    "codex-lich-early-0000",
    "codex-lich-early-0001",
    "codex-lich-late-0000",
    "codex-lich-late-0001",
}
VALIDATION_IDS = {
    "pilot-0002",
    "pilot-0005",
    "codex-random-0004",
    "codex-lich-early-0000",
    "codex-lich-early-0001",
}
PRESET_CLASSES = {"1": "zombie", "2": "hero", "3": "lich"}
EARLY_LICH_INTERVAL_S = (0.0, 19.0)
LATE_LICH_INTERVAL_S = (4617.0, 4665.0)
TEST_IMAGE_SIZE = (960, 540)
MINIMAP_CORRELATION_THRESHOLD = 0.90
CONFIRMATION_NOTE = "Promoted after the user explicitly confirmed labeling complete on 2026-09-22."


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _frame_indices(workspace: MapWorkspace) -> dict[str, int]:
    result: dict[str, int] = {}
    for manifest_name, field in (("pilot_manifest.json", "frame"),
                                 ("codex_manual_batch_v1_manifest.json", "source_frame")):
        path = workspace.path("annotations").parent / manifest_name
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in payload["frames"]:
            result[str(entry["frame_id"])] = int(entry[field])
    return result


def _prior_test_times(workspace: MapWorkspace) -> list[float]:
    path = workspace.path("annotations").parent / "runs/tiny-3class-v1/manifest.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [float(entry["time_s"]) for entry in payload.get("test_frames", [])]


def _outside_interval(timestamp: float, interval: tuple[float, float]) -> bool:
    return not interval[0] <= timestamp <= interval[1]


def _far_from(timestamp: float, existing: list[float], distance: float) -> bool:
    return all(abs(timestamp - value) >= distance for value in existing)


def _minimap_correlation(frame, reference, crop: tuple[int, int, int, int]) -> float:
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


def _audit_minimap_rows(video: Path, reference, crop: tuple[int, int, int, int],
                        rows: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video}")
    audited = []
    try:
        for row in rows:
            candidate = dict(row)
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(candidate["frame"]))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {candidate['frame']} for minimap audit")
            candidate["minimap_correlation"] = _minimap_correlation(frame, reference, crop)
            audited.append(candidate)
    finally:
        capture.release()
    return audited


def _sample_test_rows(positions: list[dict[str, float | int]], *, seed: int,
                      labeled_times: list[float], prior_test_times: list[float],
                      video: Path, minimap_reference, minimap_crop: tuple[int, int, int, int]) -> list[tuple[str, dict[str, float | int]]]:
    excluded = labeled_times + prior_test_times
    general_candidates = [
        row for row in positions
        if _outside_interval(float(row["time_s"]), EARLY_LICH_INTERVAL_S)
        and _outside_interval(float(row["time_s"]), LATE_LICH_INTERVAL_S)
        and _far_from(float(row["time_s"]), excluded, 10.0)
    ]
    early_candidates = [
        row for row in positions
        if EARLY_LICH_INTERVAL_S[0] <= float(row["time_s"]) <= EARLY_LICH_INTERVAL_S[1]
        and _far_from(float(row["time_s"]), excluded, 1.0)
    ]
    late_candidates = [
        row for row in positions
        if LATE_LICH_INTERVAL_S[0] <= float(row["time_s"]) <= LATE_LICH_INTERVAL_S[1]
        and _far_from(float(row["time_s"]), excluded, 2.0)
    ]
    pools = {
        "general": choose_pilot_positions(general_candidates, count=64, seed=seed,
                                          minimum_separation_s=20.0),
        "lich_early": choose_pilot_positions(early_candidates, count=8, seed=seed + 1,
                                             minimum_separation_s=1.0),
        "lich_late": choose_pilot_positions(late_candidates, count=8, seed=seed + 2,
                                            minimum_separation_s=2.0),
    }
    audited = {
        kind: [
            row for row in _audit_minimap_rows(video, minimap_reference, minimap_crop, rows)
            if float(row["minimap_correlation"]) >= MINIMAP_CORRELATION_THRESHOLD
        ]
        for kind, rows in pools.items()
    }
    general = choose_pilot_positions(audited["general"], count=16, seed=seed + 100,
                                     minimum_separation_s=90.0)
    early = choose_pilot_positions(audited["lich_early"], count=2, seed=seed + 101,
                                   minimum_separation_s=3.0)
    late = choose_pilot_positions(audited["lich_late"], count=2, seed=seed + 102,
                                  minimum_separation_s=5.0)
    return ([('general', row) for row in general] +
            [('lich_early', row) for row in early] +
            [('lich_late', row) for row in late])


def prepare(output: Path, *, seed: int) -> dict[str, object]:
    workspace = MapWorkspace.load(ROOT, "forest-of-dead-trees-2")
    if output.exists():
        raise FileExistsError(f"Experiment output already exists: {output}")
    output.mkdir(parents=True)

    annotations_path = workspace.path("annotations")
    items = read_jsonl(annotations_path)
    ids = {item.frame_id for item in items}
    if ids != EXPECTED_IDS:
        raise ValueError(f"Expected exactly {sorted(EXPECTED_IDS)}, got {sorted(ids)}")
    if len(items) != len(ids):
        raise ValueError("Canonical annotations contain duplicate frame IDs")

    frame_indices = _frame_indices(workspace)
    direct_positions = _direct_positions(workspace.external_path("geometry", "positions"))
    direct_by_frame = {int(row["frame"]): row for row in direct_positions}
    missing_direct = [item.frame_id for item in items if frame_indices.get(item.frame_id) not in direct_by_frame]
    if missing_direct:
        raise ValueError(
            "Refusing dataset preparation; frames lack a direct valid-minimap observation: "
            + ", ".join(sorted(missing_direct))
        )
    minimap_path = workspace.external_path("geometry", "positions").parent / "minimap_native.png"
    minimap_reference = cv2.imread(str(minimap_path))
    if minimap_reference is None:
        raise RuntimeError(f"Could not read minimap reference {minimap_path}")
    labeled_minimap_rows = _audit_minimap_rows(
        workspace.video, minimap_reference, workspace.minimap_crop,
        [{**direct_by_frame[frame_indices[item.frame_id]], "frame_id": item.frame_id} for item in items],
    )
    minimap_score_by_id = {
        str(row["frame_id"]): float(row["minimap_correlation"])
        for row in labeled_minimap_rows
    }
    excluded_ids = {
        frame_id for frame_id, score in minimap_score_by_id.items()
        if score < MINIMAP_CORRELATION_THRESHOLD
    }
    invalid_validation = sorted(VALIDATION_IDS & excluded_ids)
    if invalid_validation:
        raise ValueError(f"Validation assignment contains invalid-minimap frames: {invalid_validation}")

    source_copy = output / "source_annotations.user_confirmed.jsonl"
    shutil.copy2(annotations_path, source_copy)
    source_sha256 = sha256_file(source_copy)

    promoted = copy.deepcopy(items)
    for item in promoted:
        if item.frame_id in excluded_ids:
            item.split = "pilot"
            item.category = "excluded_invalid_minimap"
            if "invalid_minimap" not in item.conditions:
                item.conditions.append("invalid_minimap")
        else:
            item.split = "validation" if item.frame_id in VALIDATION_IDS else "train"
        item.review_status = "reviewed"
        previous_notes = item.notes.replace("Human review required.", "").strip()
        item.notes = (previous_notes + " " + CONFIRMATION_NOTE).strip()
        if item.frame_id in excluded_ids:
            item.notes += " Excluded from development because the minimap reference check failed."
        for monster in item.monsters:
            if monster.source == "codex_prelabel":
                monster.source = "human_confirmed_prelabel"
            monster.review_required = False

    write_jsonl_with_backup(promoted, annotations_path)
    report = write_report(promoted, workspace.path("dataset_report"),
                          source_size=workspace.source_size)
    if not report["training_ready"]:
        raise RuntimeError("Promoted canonical dataset did not become training-ready")

    yolo_dir = workspace.path("yolo_dataset")
    if yolo_dir.exists():
        raise FileExistsError(f"YOLO dataset already exists: {yolo_dir}")
    train_export = export_yolo(
        promoted, yolo_dir, split="train", source_size=workspace.source_size,
        image_root=workspace.repo_root, preset_classes=PRESET_CLASSES,
    )
    validation_export = export_yolo(
        promoted, yolo_dir, split="validation", source_size=workspace.source_size,
        image_root=workspace.repo_root, preset_classes=PRESET_CLASSES,
    )

    selected_test = _sample_test_rows(
        direct_positions, seed=seed,
        labeled_times=[float(item.timestamp) for item in promoted if item.frame_id not in excluded_ids],
        prior_test_times=_prior_test_times(workspace),
        video=workspace.video, minimap_reference=minimap_reference,
        minimap_crop=workspace.minimap_crop,
    )
    test_images = output / "test/images"
    capture = cv2.VideoCapture(str(workspace.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {workspace.video}")
    test_rows = []
    test_panels = []
    kind_indices: dict[str, int] = {"general": 0, "lich_early": 0, "lich_late": 0}
    try:
        for kind, row in selected_test:
            frame_index = int(row["frame"])
            timestamp = float(row["time_s"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode test frame {frame_index}")
            frame = cv2.resize(frame, TEST_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            kind_index = kind_indices[kind]
            kind_indices[kind] += 1
            test_id = f"combined-test-{kind}-{kind_index:02d}"
            image_path = test_images / f"{test_id}_{timestamp:010.3f}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Could not write {image_path}")
            test_rows.append({
                "id": test_id,
                "kind": kind,
                "frame": frame_index,
                "time_s": timestamp,
                "image": _relative(image_path),
                "image_sha256": sha256_file(image_path),
                "eligibility": "direct_minimap_marker",
                "minimap_position": [float(row["x"]), float(row["y"])],
                "marker_area": int(row["area"]),
                "minimap_correlation": round(float(row["minimap_correlation"]), 6),
            })
            panel = cv2.resize(frame, (384, 216), interpolation=cv2.INTER_AREA)
            cv2.rectangle(panel, (0, 0), (384, 26), (0, 0, 0), -1)
            cv2.putText(panel, f"{test_id} map={float(row['minimap_correlation']):.3f}",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .42,
                        (255, 255, 255), 1, cv2.LINE_AA)
            test_panels.append(panel)
    finally:
        capture.release()

    sheet = cv2.vconcat([
        cv2.hconcat(test_panels[index:index + 5])
        for index in range(0, len(test_panels), 5)
    ])
    test_contact_sheet = output / "test/contact-sheet-unlabeled.jpg"
    if not cv2.imwrite(str(test_contact_sheet), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not write {test_contact_sheet}")

    train_ids = [item.frame_id for item in promoted if item.split == "train"]
    validation_ids = [item.frame_id for item in promoted if item.split == "validation"]
    manifest = {
        "schema": "specialized-visual-detection.combined-learning-experiment.v1",
        "map_id": workspace.map_id,
        "seed": seed,
        "classes": PRESET_CLASSES,
        "eligibility": {
            "train_validation": "exact source frame has direct minimap-marker observation",
            "test": "sampled exclusively from direct minimap-marker observations",
            "invalid_minimap_frames_allowed": False,
            "minimap_reference": str(minimap_path.resolve()),
            "minimum_reference_correlation": MINIMAP_CORRELATION_THRESHOLD,
            "excluded_labeled_frames": [
                {"frame_id": frame_id, "minimap_correlation": minimap_score_by_id[frame_id]}
                for frame_id in sorted(excluded_ids)
            ],
        },
        "confirmation": {
            "method": "explicit_user_confirmation_after_interactive_manual_labeling",
            "date": "2026-09-22",
            "source_annotations": _relative(source_copy),
            "source_annotations_sha256": source_sha256,
        },
        "split_strategy": {
            "method": "group-aware holdout",
            "train_frame_ids": train_ids,
            "validation_frame_ids": validation_ids,
            "note": "The early-Lich sequence is held together in validation; late-Lich examples remain in training.",
        },
        "exports": {"train": train_export, "validation": validation_export},
        "test_selection": {
            "general_count": 16,
            "early_lich_count": 2,
            "late_lich_count": 2,
            "annotations": "sealed_unlabeled_visual_test",
            "excludes_prior_test_neighborhoods": True,
            "contact_sheet": _relative(test_contact_sheet),
        },
        "source": {
            "video": str(workspace.video.resolve()),
            "video_sha256": sha256_file(workspace.video),
            "positions": str(workspace.external_path("geometry", "positions").resolve()),
            "positions_sha256": sha256_file(workspace.external_path("geometry", "positions")),
        },
        "test_frames": test_rows,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    default = ROOT / "maps/forest-of-dead-trees-2/dataset/runs/combined-v1"
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
