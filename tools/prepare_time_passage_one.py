"""Prepare 25 regularly spaced, unlabeled frames from each timePassageOne video."""
from __future__ import annotations

import bisect
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monster_dataset.contact_sheet import write_contact_sheets
from monster_dataset.schema import FrameAnnotation, write_jsonl
from monster_dataset.validation import write_report
from perception.core import probe_frame_timestamps, sha256_file
from perception.workspace import MapWorkspace

MAP_ID = "timePassageOne"
MAP = ROOT / "maps" / MAP_ID
CAPTURES = Path("C:/projects/input-flag-inspector/saves_m")
SOURCES = ("timePassageOneSample0", "timePassageOneSample1")
COUNT = 25
SOURCE_SIZE = (1920, 1080)


def midpoint_samples(timestamps: list[float], count: int = COUNT) -> list[tuple[int, float]]:
    """Return distinct source indices and target times; ties favor earlier frames."""
    if count < 1 or len(timestamps) < count:
        raise ValueError("Not enough source frames for the requested sample count")
    if (not all(math.isfinite(t) and t >= 0 for t in timestamps)
            or any(b < a for a, b in zip(timestamps, timestamps[1:]))
            or timestamps[-1] <= timestamps[0]):
        raise ValueError("Expected finite, nondecreasing presentation timestamps with positive span")
    step = (timestamps[-1] - timestamps[0]) / count
    samples = []
    for slot in range(count):
        target = timestamps[0] + (slot + 0.5) * step
        insertion = bisect.bisect_left(timestamps, target)
        choices = [i for i in (insertion - 1, insertion) if 0 <= i < len(timestamps)]
        index = min(choices, key=lambda i: (abs(timestamps[i] - target), i))
        samples.append((index, target))
    if len({index for index, _ in samples}) != count:
        raise ValueError("Midpoint sampling did not produce distinct frames; refusing silent resampling")
    return samples


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def map_config() -> dict:
    return {
        "schema": "specialized-visual-detection.map.v1",
        "map_id": MAP_ID, "display_name": MAP_ID, "source_size": list(SOURCE_SIZE),
        "capture": {"directory": (CAPTURES / SOURCES[0]).as_posix(),
                    "video": "screen.mp4", "events": "events.csv", "end_s": None},
        "minimap": None, "fixed_ui_rects": [], "prelabel": {"prompts": []},
        "preparation": {"status": "unlabeled_review_ready", "geometry_status": "not_available",
                        "minimap_status": "not_calibrated", "fixed_ui_status": "not_calibrated",
                        "class_status": "defined_mob_hero", "training_ready": False},
        "paths": {
            "capture_manifest": "capture_manifest.json", "calibration_hints": "calibration_hints.json",
            "profile_dir": "session_profile", "profile": "session_profile/profile.json",
            "benchmark_annotations": "benchmark/annotations.csv", "benchmark_frames": "benchmark/frames",
            "annotations": "dataset/annotations.jsonl", "dataset_images": "dataset/images",
            "dataset_report": "dataset/dataset_report.json", "prelabel_report": "dataset/prelabel_report.json",
            "pilot_manifest": "dataset/pilot_manifest.json", "contacts": "dataset/contacts",
            "temporal_context": "dataset/temporal_context", "yolo_dataset": "dataset/yolo_dataset",
            "training_runs": "dataset/runs/v0", "evaluation": "evaluation", "proof": "proof",
        },
    }


def main() -> None:
    if MAP.exists():
        raise FileExistsError(f"Refusing to overwrite existing map or review work: {MAP}")
    # Validate both recordings before creating any review artifacts.
    prepared = []
    for source_name in SOURCES:
        video = CAPTURES / source_name / "screen.mp4"
        print(f"Probing timestamps: {source_name}", flush=True)
        digest = sha256_file(video)
        timestamps = probe_frame_timestamps(video)
        samples = midpoint_samples(timestamps)
        capture = cv2.VideoCapture(str(video))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open {video}")
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if size != SOURCE_SIZE:
                raise ValueError(f"Unexpected dimensions {size}: {video}")
        finally:
            capture.release()
        prepared.append((source_name, video, digest, timestamps, samples))
        print(f"{source_name}: {len(timestamps)} frames; interval {(timestamps[-1]-timestamps[0])/COUNT:.3f}s", flush=True)

    MAP.mkdir()
    write_json(MAP / "map.json", map_config())
    workspace = MapWorkspace.load(ROOT, MAP_ID)
    images = workspace.path("dataset_images")
    images.mkdir(parents=True)
    items, entries, sources = [], [], []
    for source_number, (source_name, video, digest, timestamps, samples) in enumerate(prepared):
        selected = dict(samples)
        capture = cv2.VideoCapture(str(video))
        decoded = extracted = 0
        first_position = len(items) + 1
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open {video}")
            while capture.grab():
                index = decoded
                decoded += 1
                if decoded % 5000 == 0:
                    print(f"{source_name}: decoded {decoded}/{len(timestamps)}; saved {extracted}/{COUNT}", flush=True)
                if index not in selected:
                    continue
                ok, frame = capture.retrieve()
                if not ok or frame is None or frame.shape != (1080, 1920, 3):
                    raise RuntimeError(f"Cannot decode full-resolution frame {index} from {video}")
                frame_id = f"tpo-sample{source_number}-{extracted:04d}"
                destination = images / f"{frame_id}_{timestamps[index]:010.3f}.png"
                if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                    raise RuntimeError(f"Cannot write {destination}")
                if not np.array_equal(cv2.imread(str(destination)), frame):
                    raise RuntimeError(f"PNG pixel roundtrip failed: {destination}")
                item = FrameAnnotation(
                    frame_id=frame_id, timestamp=timestamps[index],
                    image_path=destination.relative_to(ROOT).as_posix(), split="pilot",
                    review_status="pending", category="regular_interval_original_video",
                    conditions=[source_name, "unlabeled", "map_membership_unverified"],
                    notes="Unlabeled original-video frame. Empty boxes are not a confirmed negative. "
                          "Classes: preset 1=mob, preset 2=hero. No geometry or content filtering.",
                )
                if item.validate(*SOURCE_SIZE):
                    raise ValueError(item.validate(*SOURCE_SIZE))
                items.append(item)
                entries.append({"frame_id": frame_id, "frame": index, "timestamp": timestamps[index],
                                "target_timestamp": selected[index], "source_name": source_name,
                                "source_video": video.as_posix(), "source_video_sha256": digest,
                                "image_path": item.image_path, "image_sha256": sha256_file(destination),
                                "display_position": len(items), "review_status": "pending"})
                extracted += 1
        finally:
            capture.release()
        if decoded != len(timestamps) or extracted != COUNT:
            raise RuntimeError(f"Incomplete extraction: decoded={decoded}, PTS={len(timestamps)}, extracted={extracted}")
        if sha256_file(video) != digest:
            raise RuntimeError(f"Source video changed during extraction: {video}")
        sources.append({"name": source_name, "video": video.as_posix(), "video_sha256": digest,
                        "decoded_frames": decoded, "source_size": list(SOURCE_SIZE),
                        "first_frame_timestamp_s": timestamps[0], "last_frame_timestamp_s": timestamps[-1],
                        "interval_s": (timestamps[-1] - timestamps[0]) / COUNT,
                        "sample_count": extracted, "display_positions": [first_position, len(items)]})
        print(f"Completed {source_name}: {extracted} verified PNGs", flush=True)

    annotations = workspace.path("annotations")
    write_jsonl(items, annotations)
    write_json(annotations.parent / "review_settings.json", {"box_presets": [
        {"name": "mob", "width": 100, "height": 100, "hue": 42},
        {"name": "hero", "width": 100, "height": 100, "hue": 205},
    ]})
    write_json(annotations.parent / "review_order.json", {
        "schema": "specialized-visual-detection.review-order.v1",
        "storage": "canonical JSONL line order; sample0 then sample1, chronological within each source",
        "frame_ids": [item.frame_id for item in items],
    })
    contacts = write_contact_sheets(items, workspace.path("contacts"), split="pilot",
                                    source_size=SOURCE_SIZE, annotations_path=annotations)
    report = write_report(items, workspace.path("dataset_report"), source_size=SOURCE_SIZE)
    report["note"] = "All 50 empty annotations are unreviewed/unlabeled, not confirmed negative examples."
    write_json(workspace.path("dataset_report"), report)
    write_json(workspace.path("capture_manifest"), {
        "schema_version": 1, "map_id": MAP_ID, "sources": sources,
        "source_size": list(SOURCE_SIZE), "geometry_status": "not_available",
    })
    write_json(workspace.path("pilot_manifest"), {
        "schema": "specialized-visual-detection.pilot.v1", "map_id": MAP_ID,
        "sources": sources, "source_size": list(SOURCE_SIZE), "requested_count": 50,
        "sampling": "25 equal intervals over each first-to-last PTS span; nearest midpoint frame; ties select earlier index",
        "timestamp_basis": "ffprobe best_effort_timestamp_time; never nominal FPS",
        "image_encoding": "PNG; lossless decoded BGR pixels, verified roundtrip",
        "filter": "none; no minimap or geometry requirement", "split": "pilot",
        "training_ready": False, "class_mapping": {"1": "mob", "2": "hero"},
        "review": {"images_only": True, "queue": "all"}, "frames": entries,
        "contact_sheets": [path.relative_to(ROOT).as_posix() for path in contacts],
    })
    (MAP / "review_codex_labels_0.bat").write_text(
        '@echo off\nsetlocal\ncd /d "%~dp0\\..\\.."\n'
        'echo timePassageOne - 50 regularly spaced frames; no pre-labels.\n'
        'echo Ctrl+1 = mob   Ctrl+2 = hero\n'
        'echo 1-25: timePassageOneSample0   26-50: timePassageOneSample1\n'
        'echo Chronological within each source. Empty boxes start unreviewed.\n'
        'echo Click the Frame number, type a position, and press Enter to jump.\n'
        'python perception_cli.py --map timePassageOne monster-review --images-only --split pilot --queue all\n'
        'if errorlevel 1 (\n    echo timePassageOne review exited with an error.\n    pause\n)\nendlocal\n',
        encoding="ascii", newline="\r\n",
    )
    (MAP / "README.md").write_text(
        "# timePassageOne review preparation\n\n"
        "Open `review_codex_labels_0.bat` to label all 50 saved images.\n"
        "Ctrl+1 = mob; Ctrl+2 = hero. Both presets begin at editable 100x100 pixels.\n"
        "Resize a preset box using its corner handles or Shift+arrows; future placements use the saved size.\n"
        "Click the Frame number and press Enter to jump; S saves and Q saves and quits.\n\n"
        "Positions 1-25 come from timePassageOneSample0/screen.mp4; positions 26-50\n"
        "come from timePassageOneSample1/screen.mp4, under C:/projects/input-flag-inspector/saves_m/.\n"
        "Each source is chronological, and the canonical JSONL plus review_order.json preserve this order.\n"
        "Divide each recording's first-to-last presentation-timestamp span into 25 equal intervals;\n"
        "choose the nearest source frame to each midpoint, breaking ties toward the earlier frame.\n"
        "No random sampling, shuffling, or content filtering is applied.\n\n"
        "Images are full-resolution 1920x1080 PNGs with verified lossless decoded-pixel roundtrips.\n"
        "No boxes were generated: all 50 frames start pending in the pilot split. Empty annotations\n"
        "are unlabeled, not confirmed negatives (the generic report counts empty frames as negative_frames).\n"
        "Training readiness remains false. Geometry, minimap, and UI masks are uncalibrated.\n"
        "Use images-only review because this collection spans two source videos.\n\n"
        "dataset/pilot_manifest.json records both source hashes, original frame indices and timestamps,\n"
        "target midpoint times, image hashes, class mapping, and display positions.\n"
        "dataset/contacts contains unlabeled overview sheets. artifact_hashes.json describes initial preparation.\n"
        "Reproduce into a fresh map directory with `python tools/prepare_time_passage_one.py`\n"
        "from the repository root; the script refuses to overwrite any existing map/review work.\n",
        encoding="utf-8",
    )
    write_json(MAP / "artifact_hashes.json", {"sha256": {
        path.relative_to(MAP).as_posix(): sha256_file(path)
        for path in sorted(MAP.rglob("*")) if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps({"frames": len(items), "sources": sources, "training_ready": False,
                      "launcher": str(MAP / "review_codex_labels_0.bat")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
