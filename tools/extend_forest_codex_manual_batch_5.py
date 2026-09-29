"""Append the visually audited HUMAN evidence shortlist to pending batch 5.

Uses local review artifacts; refuses duplicate imports or replacement of review work.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monster_dataset.schema import FrameAnnotation, MonsterAnnotation, read_jsonl
from monster_dataset.review_app import ReviewInstanceLock, ReviewSession
from perception.core import sha256_file
from tools.prepare_forest_codex_manual_batch_4 import _minimap_correlation, MINIMAP_REFERENCE

MAP = ROOT / "maps/forest-of-dead-trees-2"
EVIDENCE = ROOT.parent / "human/out/lich-evidence-review-20260928/reviewed-evidence.json"
OUT = MAP / "dataset/runs/codex-manual-batch-5/extension-20260928"
PREFIX = "codex-human-batch5-evidence-"
OLD_PREFIX = "codex-human-189ec300-false-positive-"
# Indices into the saved combined-v3/conf=.05 proposals, after full-frame audit.
# Loot, UI, scenery, cross-class duplicates and duplicate overlapping boxes removed.
KEEP = {
    1: [0, 1, 2, 4],
    4: [0, 1, 2, 3, 4, 5, 6, 8, 12],
    6: [0, 1, 2, 3, 4, 5, 6, 8],
    9: [0, 1, 2, 3, 4, 5],
    16: [0, 1, 2, 3, 4, 5, 7],
    20: [0, 1, 2, 3, 4],
    21: [0, 1, 2, 3, 4, 5],
    24: [0, 1, 2, 3, 4, 5, 6, 7, 8],
    32: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16],
    33: list(range(13)),
    44: [0, 1, 2, 3, 4, 5],
    49: list(range(7)) + [8],
    50: [0, 1, 2],
    51: [0, 1, 2],
    53: [0, 1, 2, 3],
    55: [0, 1, 2, 3],
    58: [0, 1, 2, 3],
}
# Added/adjusted full preset boxes: preset, feet x/y, visible fraction estimate.
# Dying Zombies retain Zombie identity. Hidden feet remain inferred for review.
EXTRA = {
    4: [("1", 864, 40, .25), ("1", 909, 41, .25)],
    6: [("1", 1134, 766, .65), ("1", 1305, 769, .55), ("1", 1480, 766, .6)],
    9: [("1", 1652, 761, .6)],
    16: [("1", 563, 814, .6), ("1", 625, 814, .6), ("1", 869, 794, .65)],
    44: [("1", 902, 665, .35)],
    53: [("1", 1625, 394, .25)],
}
# Visibility estimates only; every proposal still requires human confirmation.
VISIBILITY = {
    1: {0: .35, 2: .85, 4: .45},
    4: {3: .7, 8: .65, 12: .65},
    6: {4: .65, 8: .2},
    9: {0: .3, 5: .55},
    16: {0: .2, 7: .55},
    20: {1: .65},
    21: {4: .75, 5: .6},
    24: {2: .25, 5: .65, 6: .65, 8: .55},
    32: {7: .6, 10: .75, 11: .8, 12: .3, 16: .45},
    33: {6: .8, 7: .8, 12: .85},
    44: {4: .6, 5: .25},
    49: {4: .55, 5: .8, 6: .7, 8: .4},
    50: {1: .55, 2: .2},
    51: {1: .2, 2: .5},
    53: {1: .2, 2: .2, 3: .45},
    55: {3: .6},
    58: {1: .7, 3: .25},
}


def main() -> None:
    predictions_path = OUT / "proposals/predictions.json"
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    proposals = {int(f["id"].split("-")[-1]): f for f in predictions["frames"]}
    evidence = {f["id"]: f for f in json.loads(EVIDENCE.read_text(encoding="utf-8")) if f["shortlisted"]}
    assert set(evidence) == set(KEEP)
    settings = json.loads((MAP / "dataset/review_settings.json").read_text(encoding="utf-8"))
    presets = {str(i + 1): p for i, p in enumerate(settings["box_presets"])}
    annotations = MAP / "dataset/annotations.jsonl"
    manifest_path = MAP / "dataset/codex_manual_batch_v5_manifest.json"
    reference = cv2.imread(str(MINIMAP_REFERENCE))
    if reference is None:
        raise ValueError("Missing minimap reference")
    additions, frames, panels, sources = [], [], [], {}
    copies = []
    with ReviewInstanceLock(annotations):
        before = annotations.read_bytes()
        manifest_before = manifest_path.read_bytes()
        manifest = json.loads(manifest_before)
        original_frames = list(manifest["frames"])
        current = read_jsonl(annotations)
        if any(f.frame_id.startswith(PREFIX) for f in current):
            raise ValueError("Evidence extension already exists; refusing to overwrite review work")
        assert len(original_frames) == 5
        assert all(f["review_status"] == "pending" for f in original_frames)
        existing_hashes = {sha256_file(ROOT / f.image_path) for f in current}
        for number in sorted(KEEP):
            event = evidence[number]
            source = Path(event["screenshot"])
            source_hash = sha256_file(source)
            assert source_hash == event["sha256"] == sha256_file(Path(event["staged_image"]))
            if source_hash in existing_hashes:
                raise ValueError(f"Duplicate image: {source}")
            existing_hashes.add(source_hash)
            image = cv2.imread(str(source))
            assert image is not None and image.shape[:2] == (1080, 1920)
            correlation = _minimap_correlation(image, reference)
            assert correlation >= .90, (number, correlation)
            run = source.parent.parent
            session = json.loads((run / "session.json").read_text(encoding="utf-8"))
            start = datetime.fromisoformat(session["startedAtUtc"].replace("Z", "+00:00"))
            captured = datetime.fromisoformat(event["checkedAt"] + "+09:00")
            sources[event["run"]] = {"run_id": session["runId"], "run_directory": str(run),
                "detection_log_sha256": sha256_file(Path(event["log"]))}
            points = []
            for index in KEEP[number]:
                detection = proposals[number]["detections"][index]
                x1, y1, x2, y2 = detection["bbox_xyxy"]
                preset = {"zombie": "1", "hero": "2", "lich": "3"}[detection["class_name"]]
                visible = VISIBILITY.get(number, {}).get(index, 1.)
                if y1 < 0:
                    visible = min(visible, y2 / presets[preset]["height"])
                points.append((preset, round((x1 + x2) / 2), round(y2), round(visible, 2)))
            points.extend(EXTRA.get(number, []))
            monsters = []
            for preset, x, y, visible in points:
                width, height = (presets[preset][k] for k in ("width", "height"))
                monsters.append(MonsterAnnotation(
                    bbox_xyxy=[x-width/2, y-height, x+width/2, y], ground_position=[x, y],
                    visibility=visible, occluded=visible < 1, annotation_confidence=.75,
                    review_required=True, source="codex_prelabel", box_preset=preset))
            frame_id = f"{PREFIX}{number:04d}"
            destination = MAP / "dataset/images" / f"{frame_id}.png"
            if destination.exists():
                raise FileExistsError(destination)
            item = FrameAnnotation(frame_id=frame_id, timestamp=(captured-start).total_seconds(),
                image_path=destination.relative_to(ROOT).as_posix(), monsters=monsters,
                split="train", review_status="pending", category="codex_human_lich_evidence",
                conditions=["valid_minimap", event["region_review"], f"human_run_{session['runId'][:8]}"],
                notes=(event.get("review_note", "") + " Full-frame model-assisted Codex proposals; "
                       "human review required. Full registered preset boxes include inferred hidden feet. "
                       "Check overlapping, dying, HUD-obscured and off-frame sprites carefully."))
            assert not item.validate(), (frame_id, item.validate())
            additions.append(item)
            copies.append((source, destination))
            frames.append({"frame_id": frame_id, "timestamp": item.timestamp, "image_path": item.image_path,
                "source_image": str(source), "image_sha256": source_hash, "run_id": session["runId"],
                "evidence_shortlist_id": number, "minimap_correlation": correlation,
                "region_review": event["region_review"], "original_detection_event": event,
                "proposal_count": len(monsters), "review_status": "pending",
                "prelabeling": "combined-v3_conf005_plus_full_resolution_codex_audit"})
            rendered = image.copy()
            colors = {"1": (70, 210, 255), "2": (255, 170, 70), "3": (255, 70, 210)}
            for monster in monsters:
                x1, y1, x2, y2 = map(round, monster.bbox_xyxy)
                cv2.rectangle(rendered, (x1, y1), (x2, y2), colors[monster.box_preset], 2)
            cv2.imwrite(str(OUT / f"{frame_id}.jpg"), rendered)
            panel = cv2.resize(rendered, (640, 360), interpolation=cv2.INTER_AREA)
            title = np.zeros((28, 640, 3), dtype=np.uint8)
            cv2.putText(title, f"Batch 5 / evidence {number:02} / pending", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
            panels.append(cv2.vconcat([title, panel]))
        contacts = []
        for page, offset in enumerate(range(0, len(panels), 6), 1):
            group = panels[offset:offset+6]
            group += [np.zeros_like(panels[0])] * (6-len(group))
            contact = OUT / f"audited-contact-{page}.jpg"
            cv2.imwrite(str(contact), cv2.vconcat([cv2.hconcat(group[:3]), cv2.hconcat(group[3:])]))
            contacts.append(contact.relative_to(ROOT).as_posix())
        manifest["frames"].extend(frames)
        manifest.setdefault("extensions", []).append({"date": "2026-09-28", "frames_added": len(frames),
            "source_evidence_index": str(EVIDENCE), "source_evidence_index_sha256": sha256_file(EVIDENCE),
            "predictions_sha256": sha256_file(predictions_path), "sources": sources,
            "contact_sheets": contacts, "frame_id_prefix": PREFIX,
            "timestamp_basis": "seconds_since_each_source_run_start", "timestamp_timezone": "Asia/Seoul"})
        assert manifest["frames"][:5] == original_frames
        review = ReviewSession(current + additions, split="all", queue="all",
                               frame_id_prefixes=(OLD_PREFIX, PREFIX))
        assert len(review.items) == 22
        assert all(f.review_status == "pending" for f in review.items)
        (OUT / "annotations-before-extension.jsonl").write_bytes(before)
        (OUT / "manifest-before-extension.json").write_bytes(manifest_before)
        for source, destination in copies:
            shutil.copy2(source, destination)
            assert sha256_file(source) == sha256_file(destination)
        appended = b"".join((json.dumps(f.to_dict(), separators=(",", ":"))+"\n").encode() for f in additions)
        temporary = annotations.with_suffix(".jsonl.tmp")
        temporary.write_bytes(before + (b"\n" if before and not before.endswith(b"\n") else b"") + appended)
        os.replace(temporary, annotations)
        manifest_temp = manifest_path.with_suffix(".json.tmp")
        manifest_temp.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        os.replace(manifest_temp, manifest_path)
        assert annotations.read_bytes().startswith(before)
        counts = Counter(m.box_preset for f in additions for m in f.monsters)
        print(json.dumps({"added": len(additions), "batch_total": len(review.items),
                          "new_boxes_by_preset": dict(counts), "contact_sheets": contacts}, indent=2))


if __name__ == "__main__":
    main()
