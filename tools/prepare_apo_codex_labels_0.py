"""Freeze APO review order and add four visually inspected pending pre-labels."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monster_dataset.contact_sheet import write_contact_sheets
from monster_dataset.review_app import ReviewInstanceLock
from monster_dataset.schema import FrameAnnotation, MonsterAnnotation
from monster_dataset.validation import write_report
from perception.core import sha256_file

MAP = ROOT / "maps/APO"
SEED = 20260929
# (preset ID, ground X, ground Y, visible fraction). Full preset extents include
# occluded parts, matching the user's manually labeled first-image convention.
# Pets, loot, damage digits, UI and static wall monitors are excluded.
POINTS = {
    "apo-random-0004": [
        ("1", 161, 734, .4), ("1", 584, 733, .85), ("1", 661, 733, .8),
        ("1", 1080, 731, .75), ("1", 1336, 734, .35),
        ("1", 1539, 734, .8),
        ("1", 1726, 734, .5), ("1", 1763, 734, .7), ("1", 1800, 734, .65),
        ("2", 849, 738, .35), ("3", 422, 737, 1.0),
    ],
    "apo-random-0040": [
        ("1", 484, 732, 1.0), ("1", 603, 732, .9), ("1", 1338, 734, 1.0),
        ("1", 1639, 732, .85),
        ("2", 850, 739, .2), ("3", 1708, 746, .65),
    ],
    "apo-random-0013": [
        ("1", 526, 731, .85), ("1", 742, 734, .85), ("1", 878, 734, .75),
        ("1", 1491, 734, .55), ("1", 1546, 734, .85),
        ("2", 1150, 739, .85), ("3", 642, 739, 1.0),
    ],
    "apo-random-0035": [
        ("1", 417, 731, 1.0), ("1", 597, 731, 1.0), ("1", 734, 734, .85),
        ("1", 954, 734, .8), ("1", 1152, 734, .9), ("1", 1220, 734, .8),
        ("1", 1270, 734, .6), ("1", 1433, 732, .35),
        ("2", 873, 739, .8), ("3", 1395, 740, .65),
    ],
}


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    annotations = MAP / "dataset/annotations.jsonl"
    settings_path = MAP / "dataset/review_settings.json"
    order_path = MAP / "dataset/review_order.json"
    manifest_path = MAP / "dataset/codex_manual_batch_v0_manifest.json"
    proof = MAP / "dataset/prelabel_contacts/batch0"
    backup = MAP / "dataset/runs/before-codex-labels-0"
    with ReviewInstanceLock(annotations):
        if order_path.exists() or manifest_path.exists() or backup.exists():
            raise FileExistsError("Batch/order already exists; refusing to overwrite review work")
        before = annotations.read_bytes()
        lines = before.splitlines(keepends=True)
        original = [FrameAnnotation.from_dict(json.loads(line)) for line in lines]
        assert len(original) == 50 and original[0].frame_id == "apo-random-0000"
        tail = list(range(1, len(lines)))
        random.Random(SEED).shuffle(tail)
        order = [0] + tail
        selected = [original[i].frame_id for i in order[1:5]]
        assert selected == list(POINTS)
        for i in order[1:5]:
            assert not original[i].monsters and original[i].review_status == "pending"
        settings_before = settings_path.read_bytes()
        settings = json.loads(settings_before)
        for preset, name in zip(settings["box_presets"], ("mob", "hero", "special")):
            preset["name"] = name
        # Recover the user's measured hero dimensions that the old corner drag
        # saved only on the annotation, leaving the preset at its 100x100 default.
        hero = next(m for m in original[0].monsters if m.box_preset == "2")
        settings["box_presets"][1]["width"] = round(hero.bbox_xyxy[2] - hero.bbox_xyxy[0], 6)
        settings["box_presets"][1]["height"] = round(hero.bbox_xyxy[3] - hero.bbox_xyxy[1], 6)
        backup.mkdir(parents=True)
        (backup / "annotations.jsonl").write_bytes(before)
        (backup / "review_settings.json").write_bytes(settings_before)
        proof.mkdir(parents=True, exist_ok=True)
        frames = []
        pilot = json.loads((MAP / "dataset/pilot_manifest.json").read_text())
        anchors = {f["frame_id"]: f for f in pilot["frames"]}
        for position, index in enumerate(order[1:5], 2):
            item = original[index]
            image = cv2.imread(str(ROOT / item.image_path))
            assert image is not None
            for preset_id, x, y, visibility in POINTS[item.frame_id]:
                preset = settings["box_presets"][int(preset_id) - 1]
                width, height = preset["width"], preset["height"]
                item.monsters.append(MonsterAnnotation(
                    [x - width / 2, y - height, x + width / 2, y], [x, y],
                    visibility=visibility, occluded=visibility < 1,
                    annotation_confidence=.6 if visibility < .5 else .8,
                    review_required=True, source="codex_prelabel", box_preset=preset_id))
            item.conditions = [c for c in item.conditions if c != "unlabeled"] + ["codex_prelabel_pending"]
            item.notes = "Manual visual proposals using first-image class references. Review all boxes, especially effects, overlapping ghosts and ladder occlusion."
            assert not item.validate()
            lines[index] = (json.dumps(item.to_dict(), separators=(",", ":")) + "\n").encode()
            for monster in item.monsters:
                preset_id = monster.box_preset
                color = {"1": (50, 210, 255), "2": (255, 170, 70), "3": (255, 70, 210)}[preset_id]
                x1, y1, x2, y2 = map(round, monster.bbox_xyxy)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, settings["box_presets"][int(preset_id)-1]["name"],
                            (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
            overlay = proof / f"position-{position:02d}-{item.frame_id}.jpg"
            assert cv2.imwrite(str(overlay), image)
            frames.append({"frame_id": item.frame_id, "display_position": position,
                           "source_frame": anchors[item.frame_id]["frame"],
                           "image_sha256": anchors[item.frame_id]["image_sha256"],
                           "proposal_count": len(item.monsters), "review_status": "pending",
                           "overlay": overlay.relative_to(ROOT).as_posix()})
        temporary = annotations.with_suffix(".jsonl.tmp")
        temporary.write_bytes(b"".join(lines[i] for i in order))
        temporary.replace(annotations)
        assert annotations.read_bytes().splitlines(keepends=True)[0] == before.splitlines(keepends=True)[0]
        write_json(settings_path, settings)
        write_json(order_path, {"schema": "specialized-visual-detection.review-order.v1",
                               "seed": SEED, "fixed_first": original[0].frame_id,
                               "storage": "canonical JSONL line order; no per-launch shuffle",
                               "frame_ids": [original[i].frame_id for i in order]})
        write_json(manifest_path, {
            "schema": "specialized-visual-detection.codex-manual-review-batch.v1",
            "map_id": "APO", "labeling_method": "manual_visual_proposals_pending_review",
            "class_mapping": {"1": "mob", "2": "hero", "3": "special"},
            "reference_frame_id": original[0].frame_id,
            "reference_line_sha256": hashlib.sha256(lines[0]).hexdigest(),
            "source_video_sha256": pilot["source_video_sha256"],
            "review": {"split": "pilot", "status": "pending", "images_only": True},
            "frames": frames})
        ordered_items = [original[i] for i in order]
        write_contact_sheets(ordered_items, MAP / "dataset/contacts", split="pilot", annotations_path=annotations)
        write_report(ordered_items, MAP / "dataset/dataset_report.json")
        print(json.dumps({"first_five": [i.frame_id for i in ordered_items[:5]],
                          "prelabel_counts": dict(Counter(m.box_preset for i in order[1:5] for m in original[i].monsters)),
                          "first_image_preserved": True}, indent=2))


if __name__ == "__main__":
    main()
