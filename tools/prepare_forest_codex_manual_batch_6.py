"""Import the visually audited 2026-09-29 HUMAN shortlist, without overwriting review work.

Run from the repository root after producing inventory.json and predictions.json
in dataset/runs/codex-manual-batch-6. The durable manifest preserves both inputs.
All labels remain pending; source-run grouping determines the train/validation split.
"""
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monster_dataset.schema import FrameAnnotation, MonsterAnnotation, read_jsonl
from monster_dataset.review_app import ReviewInstanceLock, ReviewSession
from perception.core import sha256_file
from tools.prepare_forest_codex_manual_batch_4 import MINIMAP_REFERENCE

MAP = ROOT / "maps/forest-of-dead-trees-2"
OUT = MAP / "dataset/runs/codex-manual-batch-6"
PREFIX = "codex-human-batch6-evidence-"
VALIDATION_RUN = "5be330d2"
# Indices in combined-v3, conf=.05 proposals. Loot, scenery, duplicate boxes,
# cross-class confusions and UI pictures are explicitly excluded after visual audit.
KEEP = {
    0: [0, 1, 2, 3, 4], 1: [0, 1, 2], 4: [0, 1],
    5: list(range(8)), 7: [0, 1, 2], 15: list(range(8)),
    17: list(range(8)), 19: list(range(6)) + [7],
    23: list(range(8)), 28: [0, 1],
    29: [0, 1, 2, 3, 4, 7, 9], 31: list(range(6)),
    33: list(range(7)), 38: list(range(5)) + [6, 7, 8, 9],
    40: list(range(9)) + [14], 48: list(range(12)),
    54: [0, 1, 2, 3, 5, 10, 11, 12],
    56: list(range(11)), 58: list(range(6)), 61: [0, 1, 3],
    70: list(range(7)), 73: list(range(14)) + [15, 16, 17],
    76: [1, 2, 3, 4, 5, 13, 15],
    78: [0, 1, 2, 3, 4, 5, 6, 7, 9],
    87: list(range(7)), 92: list(range(10)),
    94: [0, 1, 3, 6], 98: list(range(8)), 100: [0, 1],
}
# Full registered preset boxes, grounded by visible feet or explicitly inferred
# hidden feet. These estimates are proposals, never human-confirmed ground truth.
EXTRA = {
    0: [("3", 1915, 594, .45)],
    1: [("1", 1128, 600, .6)],
    5: [("1", 1733, 761, .55)],
    15: [("1", 1662, 784, .5), ("1", 1144, 784, .6), ("1", -20, 608, .2)],
    17: [("1", 993, 644, .5), ("1", 1283, 644, .55)],
    19: [("3", 786, 1031, .55), ("1", 213, 1023, .3),
         ("1", 542, 1023, .3), ("1", 670, 1023, .3), ("1", 843, 1023, .4)],
    23: [("1", 493, 1031, .3), ("1", 257, 1031, .3)],
    28: [("1", 1109, 658, .35)],
    29: [("1", 1220, 845, .65), ("1", 1472, 845, .65),
         ("1", 770, 768, .4), ("1", 1909, 860, .3)],
    31: [("1", 963, 709, .65)],
    33: [("1", 665, 634, .35)],
    38: [("1", 1539, 665, .65)],
    54: [("1", 128, 827, .4)],
    56: [("1", 548, 748, .6)],
    61: [("3", 739, 749, .65)],
    70: [("1", 327, 806, .5)],
    76: [("2", 1078, 646, .65), ("1", 1117, 646, .5), ("1", -12, 820, .15)],
    78: [("1", -22, 799, .18), ("1", 733, 719, .3)],
    87: [("1", 1210, 655, .45)],
    94: [("3", 418, 1063, .4), ("2", 846, 974, .6),
         ("1", 799, 974, .25)],
    100: [("1", 167, 370, .2)],
}
VISIBILITY = {
    0: {4: .25}, 1: {0: .3, 2: .9}, 4: {0: .25}, 5: {0: .25, 7: .5},
    7: {0: .3}, 15: {5: .3, 4: .6, 7: .55}, 17: {7: .3, 4: .65},
    19: {5: .65, 7: .45}, 23: {0: .55, 1: .3, 4: .3},
    28: {0: .7, 1: .25}, 29: {1: .2, 2: .55, 7: .55, 9: .25},
    31: {0: .2, 5: .35}, 33: {4: .2, 6: .85},
    38: {3: .65, 7: .7}, 40: {1: .6, 3: .6, 6: .5, 14: .5},
    54: {1: .8}, 56: {0: .3, 6: .65, 8: .5, 9: .9},
    58: {4: .2, 5: .7}, 61: {0: .75, 3: .2},
    70: {4: .15, 5: .8, 6: .18},
    73: {3: .7, 5: .6, 9: .5, 11: .5, 12: .6, 13: .6, 15: .45, 16: .45, 17: .45},
    76: {4: .65}, 78: {2: .3, 3: .75, 6: .6, 9: .7},
    87: {0: .3, 5: .65, 6: .2},
    92: {3: .5, 5: .7, 6: .45, 8: .7, 7: .6, 9: .55},
    94: {6: .4}, 98: {0: .7, 6: .35, 7: .4}, 100: {0: .7},
}
NOTES = {
    0: "Real Lich clipped by right edge and overlapping Zombie; hidden center/feet estimated.",
    1: "Real Lich under damage numbers; dying Zombie nearby is not Lich.",
    4: "False Lich on left warning sign/snow: entity 618 caused the extra clearing reversal.",
    5: "False Lich on dying Zombie, loot and chat overlay; retain Zombie identity.",
    7: "Recurring left warning-sign false positive with different camera offset and banner.",
    15: "False Lich on dying Zombie plus loot/chat; crowded and clipped Zombies.",
    17: "False Lich on portal-obscured Zombie at left edge.",
    19: "Real Lich and Zombies mostly behind bottom HUD; infer full boxes.",
    23: "Later climbing view of same HUD-obscured encounter; same split as frame 19.",
    28: "Warning-sign false positive in another run with stun banner.",
    29: "False Lich on dying Zombie; other dying and chat-obscured Zombies need review.",
    31: "Dying Zombie plus loot/pet confused with Lich; portal-obscured Zombie nearby.",
    33: "Real left-side Lich overlaps event UI icons.",
    38: "Real Lich casting skull spell behind overlapping Zombies.",
    40: "Same spell encounter at stronger effect occlusion; retained within same source split.",
    48: "Real Lich close to Hero/pet, more visible reference pose.",
    54: "Portal-obscured Zombie mistaken for Lich; top-left monster-card UI is not a sprite.",
    56: "Real Lich with open chat, pets and dying Zombie nearby.",
    58: "Real Lich under bright spell and MISS text; extra Lich proposal on loot/pet removed.",
    61: "Real Lich partially hidden by spell and Hero; feet repaired from visible robe.",
    70: "Real Lich attack pose with damage text, portal and dying Zombie.",
    73: "Ordinary Zombie mistaken for Lich; crowded upper platform and HUD occlusion.",
    76: "Left-edge portal/terrain false Lich; Hero overlaps previously missed Zombie.",
    78: "Real Lich overlapped by Zombie near bottom-left HUD; partial upper sprites.",
    87: "Real Lich overlapped by dying Zombie and loot.",
    92: "Real Lich and Hero behind bottom HUD/portal.",
    94: "Only Lich hat visible above HUD; body/feet inferred and must be reviewed.",
    98: "Real Lich and Zombie behind portal; contrasting example to portal false positives.",
    100: "False Lich on left-edge tree trunk; minimap hides a real Zombie's upper body.",
}


def main():
    inventory = json.loads((OUT / "inventory.json").read_text())
    predictions = {f["id"]: f["detections"] for f in json.loads((OUT / "predictions.json").read_text())}
    presets = json.loads((MAP / "dataset/review_settings.json").read_text())["box_presets"]
    reference = cv2.imread(str(MINIMAP_REFERENCE))
    annotations = MAP / "dataset/annotations.jsonl"
    manifest_path = MAP / "dataset/codex_manual_batch_v6_manifest.json"
    assert not manifest_path.exists(), "Batch already imported; preserve review work"
    additions, frames, copies = [], [], []
    with ReviewInstanceLock(annotations):
        before = annotations.read_bytes()
        current = read_jsonl(annotations)
        assert not any(f.frame_id.startswith(PREFIX) for f in current)
        hashes = {sha256_file(ROOT / f.image_path) for f in current}
        # Do not introduce run overlap with an earlier batch's split.
        old_provenance = before.decode() + "".join(p.read_text() for p in (MAP / "dataset").glob("*manifest.json"))
        for number in KEEP:
            event = inventory[number]
            assert event["id"] == number
            source = Path(event["screenshot"])
            digest = sha256_file(source)
            assert digest == event["sha256"] and digest not in hashes
            hashes.add(digest)
            run = source.parent.parent
            session = json.loads((run / "session.json").read_text())
            run_id = session["runId"]
            assert run_id[:8] not in old_provenance
            image = cv2.imread(str(source))
            assert image.shape[:2] == (1080, 1920)
            _, correlation, _, location = cv2.minMaxLoc(cv2.matchTemplate(
                image[:360, :400], reference, cv2.TM_CCOEFF_NORMED))
            assert correlation >= .90
            points = []
            for index in KEEP[number]:
                detection = predictions[number][index]
                x1, y1, x2, y2 = detection["bbox_xyxy"]
                preset = {"zombie": "1", "hero": "2", "lich": "3"}[detection["class_name"]]
                visibility = VISIBILITY.get(number, {}).get(index, 1.)
                if y1 < 4:
                    visibility = min(visibility, y2 / presets[int(preset)-1]["height"])
                points.append((preset, round((x1+x2)/2), round(y2), round(visibility, 2)))
            points.extend(EXTRA.get(number, []))
            monsters = []
            for preset, x, y, visibility in points:
                width, height = (presets[int(preset)-1][key] for key in ("width", "height"))
                monsters.append(MonsterAnnotation(bbox_xyxy=[x-width/2, y-height, x+width/2, y],
                    ground_position=[x, y], visibility=visibility, occluded=visibility < 1,
                    annotation_confidence=.75, review_required=True, source="codex_prelabel", box_preset=preset))
            positive = any(m.box_preset == "3" for m in monsters)
            kind = "lich_positive" if positive else "lich_hard_negative"
            split = "validation" if run_id.startswith(VALIDATION_RUN) else "train"
            start = datetime.fromisoformat(session["startedAtUtc"].replace("Z", "+00:00"))
            captured = datetime.fromisoformat(event["checkedAt"] + "+09:00")
            frame_id = f"{PREFIX}{number:04d}"
            destination = MAP / "dataset/images" / f"{frame_id}.png"
            assert not destination.exists()
            item = FrameAnnotation(frame_id=frame_id, timestamp=(captured-start).total_seconds(),
                image_path=destination.relative_to(ROOT).as_posix(), monsters=monsters, split=split,
                review_status="pending", category="codex_human_lich_evidence",
                conditions=["valid_minimap", kind, f"human_run_{run_id[:8]}", "batch6_run_grouped_split"],
                notes=NOTES[number] + " Codex model-assisted prelabels; human review required. "
                    "Full registered boxes include inferred hidden feet. Keep source-run split.")
            assert not item.validate(), (frame_id, item.validate())
            additions.append(item)
            copies.append((source, destination))
            frames.append(dict(frame_id=frame_id, image_path=item.image_path, image_sha256=digest,
                source_image=str(source), run_id=run_id, split=split, region_review=kind,
                review_status="pending", selection_note=NOTES[number], timestamp=item.timestamp,
                minimap_correlation=correlation, minimap_match_origin=list(location),
                original_detection_event=event, model_proposals=predictions[number],
                retained_proposal_indices=KEEP[number], added_ground_points=EXTRA.get(number, []),
                proposal_count=len(monsters), detection_log_sha256=sha256_file(run / "incidents/lich-detections.jsonl")))
        queue = ReviewSession(current + additions, split="all", queue="all", frame_id_prefixes=(PREFIX,))
        assert len(queue.items) == 29
        assert Counter(f.split for f in additions) == {"train": 23, "validation": 6}
        (OUT / "annotations-before-batch6.jsonl").write_bytes(before)
        for source, destination in copies:
            shutil.copy2(source, destination)
            assert sha256_file(source) == sha256_file(destination)
        appended = b"".join((json.dumps(f.to_dict(), separators=(",", ":"))+"\n").encode() for f in additions)
        temporary = annotations.with_suffix(".jsonl.tmp")
        temporary.write_bytes(before + (b"\n" if before and not before.endswith(b"\n") else b"") + appended)
        os.replace(temporary, annotations)
        manifest = dict(schema="specialized-visual-detection.codex-manual-review-batch.v1",
            created_at="2026-09-29", frame_id_prefix=PREFIX, review_status="pending", images_only=True,
            source={"kind": "still_images"},
            frames=frames, candidate_inventory=inventory,
            selection_policy="Visually screen all saved detection crops; select varied failure/occlusion families; omit repetitive frames.",
            candidate_count=len(inventory), source_run_count=len({r["run"] for r in inventory}),
            annotations_before_sha256=sha256_file(OUT / "annotations-before-batch6.jsonl"),
            prelabel_model="dataset/runs/combined-v3/training-640/weights/best.pt",
            prelabel_model_sha256=sha256_file(MAP / "dataset/runs/combined-v3/training-640/weights/best.pt"),
            prelabel_confidence=.05, prelabel_imgsz=640,
            split_policy=dict(unit="source_run", train_frames=23, validation_frames=6,
                validation_run_prefixes=[VALIDATION_RUN], prior_run_overlap=[], exact_image_duplicates=0,
                scope="Batch 6 only; previous annotations/splits preserved byte-for-byte.",
                limitation="Curated hard-case validation, not unbiased live precision/recall. Future frames from a reserved run must retain its split; augmentations inherit source split."))
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        assert annotations.read_bytes().startswith(before)
        print(json.dumps(dict(frames=len(additions), classes=dict(Counter(m.box_preset for f in additions for m in f.monsters)),
            split_kinds=dict(Counter(f'{f.split}/{f.conditions[1]}' for f in additions))), indent=2))


if __name__ == "__main__":
    main()
