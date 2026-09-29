"""Import five exact HUMAN false-positive screenshots as pending review batch 5."""
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
from monster_dataset.review_app import ReviewInstanceLock
from perception.core import sha256_file
from tools.prepare_forest_codex_manual_batch_4 import _minimap_correlation, MINIMAP_REFERENCE

MAP = ROOT / "maps/forest-of-dead-trees-2"
RUN = Path(r"C:\Users\LEE\AppData\Local\Project HUM\Control Center\runs\20260928-120440-485-189ec300")
PREFIX = "codex-human-189ec300-false-positive-"
SEQUENCES = [15, 776, 1296, 1297, 1505]
# Full registered preset boxes, anchored at visually inspected sprite feet.
# Dying/effect-obscured and tooltip-obscured Zombies remain proposals for review.
# (preset, ground x, ground y, occluded, visibility)
POINTS = {
    15: [("2", 1221, 650, True, .45), ("1", 1593, 580, False, 1),
         ("1", 1743, 581, False, 1), ("1", 1905, 580, True, .68)],
    776: [("2", 1196, 671, True, .45), ("1", 1600, 582, False, 1),
          ("1", 1672, 582, False, 1), ("1", 1819, 580, False, 1),
          ("1", 1250, 641, True, .35)],
    1296: [("2", 907, 667, True, .8), ("1", 1567, 744, True, .4)],
    1297: [("2", 907, 668, True, .8), ("1", 1550, 744, True, .4)],
    1505: [("2", 1255, 633, True, .3), ("1", 301, 774, False, 1),
           ("1", 1608, 591, False, 1), ("1", 1740, 590, False, 1),
           ("1", 1348, 592, True, .6)],
}
COLORS = {"1": (70, 210, 255), "2": (255, 170, 70), "3": (255, 70, 210)}


def main() -> int:
    annotations = MAP / "dataset/annotations.jsonl"
    manifest_path = MAP / "dataset/codex_manual_batch_v5_manifest.json"
    output = MAP / "dataset/runs/codex-manual-batch-5"
    source_log = RUN / "incidents/lich-detections.jsonl"
    events = {row["entitySampleSequence"]: row for row in
              (json.loads(line) for line in source_log.read_text(encoding="utf-8").splitlines())}
    presets = json.loads((MAP / "dataset/review_settings.json").read_text(encoding="utf-8"))["box_presets"]
    reference = cv2.imread(str(MINIMAP_REFERENCE))
    if reference is None:
        raise RuntimeError(f"Missing minimap reference: {MINIMAP_REFERENCE}")
    session = json.loads((RUN / "session.json").read_text(encoding="utf-8"))
    start = datetime.fromisoformat(session["startedAtUtc"].replace("Z", "+00:00"))
    additions, frames, panels = [], [], []
    with ReviewInstanceLock(annotations):
        before = annotations.read_bytes()
        if manifest_path.exists() or any(item.frame_id.startswith(PREFIX) for item in read_jsonl(annotations)):
            raise ValueError("Batch 5 already exists; refusing to replace review work")
        output.mkdir(parents=True, exist_ok=True)
        for index, sequence in enumerate(SEQUENCES):
            event = events[sequence]
            source = RUN / "incidents" / f"lich-low-confidence-{sequence}.png"
            image = cv2.imread(str(source))
            if image is None or image.shape[:2] != (1080, 1920):
                raise ValueError(f"Invalid source screenshot: {source}")
            correlation = _minimap_correlation(image, reference)
            if correlation < .90:
                raise ValueError(f"Invalid minimap: {source}: {correlation}")
            frame_id = f"{PREFIX}{index:04d}"
            destination = MAP / "dataset/images" / f"{frame_id}.png"
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copy2(source, destination)
            assert sha256_file(source) == sha256_file(destination)
            monsters = []
            for preset, x, y, occluded, visibility in POINTS[sequence]:
                width, height = (float(presets[int(preset)-1][key]) for key in ("width", "height"))
                monsters.append(MonsterAnnotation(
                    bbox_xyxy=[x-width/2, y-height, x+width/2, y], ground_position=[x, y],
                    visibility=visibility, occluded=occluded, annotation_confidence=.75,
                    review_required=True, source="codex_prelabel", box_preset=preset))
            cause = "inventory_tooltip_icon" if sequence in (1296, 1297) else "left_edge_warning_sign"
            captured = datetime.fromisoformat(event["checkedAt"] + "+09:00")
            item = FrameAnnotation(
                frame_id=frame_id, timestamp=(captured-start).total_seconds(),
                image_path=destination.relative_to(ROOT).as_posix(), monsters=monsters,
                split="train", review_status="pending", category="codex_human_lich_false_positive",
                conditions=["human_run_189ec300", "valid_minimap", "lich_absent", cause],
                notes=("Exact HUMAN incident screenshot. The reported Lich box is a false positive "
                       f"on {cause}; no Lich proposal was retained. Model-assisted, visually checked "
                       "Hero/Zombie preset boxes require human review, especially obscured/dying sprites."))
            if item.validate():
                raise ValueError(item.validate())
            additions.append(item)
            frames.append({
                "frame_id": frame_id, "timestamp": item.timestamp, "image_path": item.image_path,
                "source_image": str(source), "image_sha256": sha256_file(destination),
                "minimap_correlation": correlation, "entity_sample_sequence": sequence,
                "false_positive_cause": cause, "original_detection_event": event,
                "proposal_count": len(monsters), "review_status": "pending",
                "prelabeling": "combined-v3_conf005_plus_full_resolution_codex_audit"})
            rendered = image.copy()
            for monster in monsters:
                x1, y1, x2, y2 = map(round, monster.bbox_xyxy)
                cv2.rectangle(rendered, (x1,y1), (x2,y2), COLORS[monster.box_preset], 2)
                cv2.putText(rendered, presets[int(monster.box_preset)-1]["name"],
                            (max(0,x1), max(20,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .6,
                            COLORS[monster.box_preset], 2, cv2.LINE_AA)
            cv2.imwrite(str(output / f"{frame_id}.jpg"), rendered)
            panel = cv2.resize(rendered, (640,360), interpolation=cv2.INTER_AREA)
            title = np.zeros((30,640,3), dtype=np.uint8)
            cv2.putText(title, f"Batch 5 / {index:04d} / incident {sequence}", (8,21),
                        cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
            panels.append(cv2.vconcat([title,panel]))
        panels.append(np.zeros_like(panels[0]))
        contact = output / "prelabels-contact-sheet.jpg"
        cv2.imwrite(str(contact), cv2.vconcat([cv2.hconcat(panels[:3]),cv2.hconcat(panels[3:])]))
        shutil.copy2(source_log, output / "source-lich-detections.jsonl")
        manifest = {
            "schema": "specialized-visual-detection.codex-manual-review-batch.v1",
            "map_id": "forest-of-dead-trees-2", "labeling_method": "model_assisted_visual_audit_pending_review",
            "source": {"kind": "still_images", "run_directory": str(RUN), "run_id": session["runId"],
                       "detection_log_sha256": sha256_file(source_log), "timestamp_timezone": "Asia/Seoul",
                       "timestamp_basis": "seconds_since_run_start"},
            "review": {"split": "train", "status": "pending", "images_only": True,
                       "command": "maps\\forest-of-dead-trees-2\\review_codex_labels_5.bat"},
            "prelabel_model_sha256": sha256_file(MAP / "dataset/runs/combined-v3/training-640/weights/best.pt"),
            "frames": frames, "contact_sheet": contact.relative_to(ROOT).as_posix()}
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        # Preserve every pre-existing JSONL byte, with the usual atomic write and backup.
        appended = b"".join((json.dumps(item.to_dict(), separators=(",", ":"))+"\n").encode() for item in additions)
        temporary = annotations.with_suffix(".jsonl.tmp")
        temporary.write_bytes(before + (b"\n" if before and not before.endswith(b"\n") else b"") + appended)
        shutil.copy2(annotations, annotations.with_suffix(".jsonl.bak"))
        os.replace(temporary, annotations)
        assert annotations.read_bytes().startswith(before)
    print(json.dumps({"frames_added":len(additions), "boxes":dict(Counter(m.box_preset for i in additions for m in i.monsters)),
                      "manifest":str(manifest_path), "contact_sheet":str(contact)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
