"""Extract the initial APO review batch without geometry or pre-label assumptions."""
from __future__ import annotations

import json
from pathlib import Path
import random
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

COUNT = 50
SEED = 20260929


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    workspace = MapWorkspace.load(ROOT, "APO")
    images = workspace.path("dataset_images")
    annotations = workspace.path("annotations")
    manifest_path = workspace.path("pilot_manifest")
    for target in (images, annotations, manifest_path, workspace.path("capture_manifest")):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing dataset/review work: {target}")
    video = workspace.video
    source_hash = sha256_file(video)
    timestamps = probe_frame_timestamps(video)
    indices = sorted(random.Random(SEED).sample(range(len(timestamps)), COUNT))
    selected = set(indices)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    if (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) != workspace.source_size:
        capture.release()
        raise ValueError("Video dimensions differ from map source_size")
    images.mkdir(parents=True)
    items, entries = [], []
    decoded = 0
    try:
        # Sequential decoding avoids nominal-FPS seeking on this variable-rate video.
        while capture.grab():
            index = decoded
            decoded += 1
            if index not in selected:
                continue
            ok, frame = capture.retrieve()
            if not ok or frame is None:
                raise RuntimeError(f"Could not decode source frame {index}")
            if frame.shape[:2] != workspace.source_size[::-1]:
                raise ValueError(f"Unexpected dimensions at source frame {index}")
            frame_id = f"apo-random-{len(items):04d}"
            destination = images / f"{frame_id}_{timestamps[index]:010.3f}.png"
            if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise RuntimeError(f"Could not write {destination}")
            if not np.array_equal(cv2.imread(str(destination)), frame):
                raise RuntimeError(f"PNG pixel roundtrip failed for {destination}")
            item = FrameAnnotation(
                frame_id=frame_id, timestamp=timestamps[index],
                image_path=destination.relative_to(ROOT).as_posix(),
                split="pilot", review_status="pending", category="random_original_video",
                conditions=["APO_sample2", "unlabeled", "map_membership_unverified"],
                notes="Unlabeled original-video frame. Empty boxes are not a confirmed negative. "
                      "Geometry, classes, and minimap calibration are not available yet.",
            )
            if item.validate(*workspace.source_size):
                raise ValueError(item.validate(*workspace.source_size))
            items.append(item)
            entries.append({
                "frame_id": frame_id, "frame": index, "timestamp": timestamps[index],
                "image_path": item.image_path, "image_sha256": sha256_file(destination),
                "review_status": "pending",
            })
    finally:
        capture.release()
    if decoded != len(timestamps) or len(items) != COUNT:
        raise RuntimeError(f"Incomplete extraction: decoded={decoded}, PTS={len(timestamps)}, selected={len(items)}")
    if sha256_file(video) != source_hash:
        raise RuntimeError("Source video changed during extraction")
    write_jsonl(items, annotations)
    contacts = write_contact_sheets(
        items, workspace.path("contacts"), split="pilot",
        source_size=workspace.source_size, annotations_path=annotations,
    )
    result = write_report(items, workspace.path("dataset_report"), source_size=workspace.source_size)
    # The shared report calls any empty annotation a negative; document the pending meaning.
    result["note"] = "All 50 empty annotations are unreviewed/unlabeled, not confirmed negative examples."
    write_json(workspace.path("dataset_report"), result)
    write_json(workspace.path("capture_manifest"), {
        "schema_version": 1, "map_id": workspace.map_id,
        "capture_dir": video.parent.as_posix(), "video": video.as_posix(),
        "video_sha256": source_hash, "decoded_frames": decoded,
        "source_size": list(workspace.source_size),
        "first_frame_timestamp_s": timestamps[0], "last_frame_timestamp_s": timestamps[-1],
        "geometry_status": "not_available",
    })
    write_json(manifest_path, {
        "schema": "specialized-visual-detection.pilot.v1", "map_id": workspace.map_id,
        "source_video": video.as_posix(), "source_video_sha256": source_hash,
        "source_frame_count": decoded, "source_size": list(workspace.source_size),
        "seed": SEED, "requested_count": COUNT,
        "sampling": "Python random.Random(seed).sample over all zero-based source frame indices; sorted chronologically",
        "timestamp_basis": "ffprobe best_effort_timestamp_time; never nominal FPS",
        "image_encoding": "PNG; lossless decoded BGR pixels, verified roundtrip",
        "filter": "none; no minimap or geometry requirement",
        "split": "pilot", "training_ready": False, "frames": entries,
        "contact_sheets": [path.relative_to(ROOT).as_posix() for path in contacts],
    })
    hashes = {
        path.relative_to(workspace.map_dir).as_posix(): sha256_file(path)
        for path in sorted(workspace.map_dir.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(workspace.map_dir / "artifact_hashes.json", {"sha256": hashes})
    print(json.dumps({"frames": len(items), "decoded_frames": decoded,
                      "first_sample_s": items[0].timestamp, "last_sample_s": items[-1].timestamp,
                      "training_ready": result["training_ready"], "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
