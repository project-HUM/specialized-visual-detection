"""Visually proposed labels for saved positions 5-8, using reviewed positions 1-4.

Run without arguments to render proposals; pass --apply to save pending labels.
"""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monster_dataset.contact_sheet import write_contact_sheets
from monster_dataset.review_app import ReviewInstanceLock
from monster_dataset.schema import FrameAnnotation, MonsterAnnotation
from monster_dataset.validation import write_report
from perception.core import sha256_file

MAP = ROOT / "maps/timePassageOne"
EXPECTED_ANNOTATIONS = "ed599944872afca53d25fec5124d3d0393304fe5a76c415203e554dc8eca5110"
EXPECTED_SETTINGS = "b48bb07bddd89fef249cdd35246cc68b5f69b5f6bcc3e68ee5287b9d766a20fc"
# Ground X, full-box bottom Y, estimated visible fraction. These are manual
# visual proposals, not trained-model predictions. Preserve the user's sizes.
LABELS = {
    "tpo-sample0-0004": {
        "mob": [(614, 367, .8), (613, 734, .65), (678, 734, .45),
                (1018, 734, .85), (1230, 734, 1)],
        "hero": (760, 755, .25),
        "notes": "Check the overlapping pair left of the hero and the hero hidden by green attack effects.",
    },
    "tpo-sample0-0005": {
        "mob": [(720, 22, .18), (199, 389, .32), (249, 389, .5), (704, 391, 1),
                (418, 754, 1), (592, 754, .8), (671, 754, .8), (914, 755, .35), (1783, 754, 1)],
        "hero": (913, 773, .25),
        "notes": "Check the top-clipped mob, the two mobs behind the minimap, and the mob overlapping the hero/attack beam.",
    },
    "tpo-sample0-0006": {
        "mob": [(447, 24, .2), (916, 24, .2), (970, 24, .2),
                (466, 383, 1), (756, 383, 1), (1206, 383, 1), (1731, 383, 1),
                (7, 749, .5), (65, 751, .9), (695, 753, 1), (1951, 753, .18)],
        "hero": (194, 774, .45),
        "notes": "Check the three top-clipped mobs, two overlapping left-edge mobs, and the small visible slice at the right edge. "
                 "The tiny uncertain fragment behind the upper gold block was not labeled.",
    },
    "tpo-sample0-0007": {
        "mob": [(142, 284, .25), (211, 284, .35), (550, 283, .9),
                (433, 644, 1), (560, 644, .5), (577, 644, .65), (715, 644, .7),
                (852, 644, .85), (1072, 644, .85), (378, 1007, .4), (1618, 1007, .3)],
        "hero": (781, 665, .25),
        "notes": "Check the two faint mobs behind the translucent minimap, near-coincident mobs under loot, "
                 "the attack-covered hero, and the two mobs partly hidden by the bottom UI.",
    },
}


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    dataset = MAP / "dataset"
    annotations = dataset / "annotations.jsonl"
    settings_path = dataset / "review_settings.json"
    manifest_path = dataset / "codex_manual_batch_v0_manifest.json"
    backup = dataset / "runs/before-codex-labels-0"
    output = dataset / "prelabel_contacts/batch0"
    with ReviewInstanceLock(annotations):
        before = annotations.read_bytes()
        settings_before = settings_path.read_bytes()
        if hashlib.sha256(before).hexdigest() != EXPECTED_ANNOTATIONS or sha256_file(settings_path) != EXPECTED_SETTINGS:
            raise RuntimeError("Annotations or presets changed since visual inspection; inspect again before applying")
        if manifest_path.exists() or backup.exists():
            raise FileExistsError("Batch already applied; refusing to overwrite review work")
        lines = before.splitlines(keepends=True)
        items = [FrameAnnotation.from_dict(json.loads(line)) for line in lines]
        order_before = (dataset / "review_order.json").read_bytes()
        assert [item.frame_id for item in items] == json.loads(order_before)["frame_ids"]
        assert len(items) == 50 and all(item.review_status == "reviewed" for item in items[:4])
        selected = items[4:8]
        assert [item.frame_id for item in selected] == list(LABELS)
        assert all(item.review_status == "pending" and not item.monsters for item in selected)
        settings = json.loads(settings_before)
        presets = settings["box_presets"]
        assert [p["name"] for p in presets] == ["mob", "hero"]
        pilot = json.loads((dataset / "pilot_manifest.json").read_text())
        anchors = {f["frame_id"]: f for f in pilot["frames"]}
        output.mkdir(parents=True, exist_ok=True)
        frames = []
        for position, item in enumerate(selected, 5):
            assert sha256_file(ROOT / item.image_path) == anchors[item.frame_id]["image_sha256"]
            proposal = LABELS[item.frame_id]
            for preset_id, points in (("1", proposal["mob"]), ("2", [proposal["hero"]])):
                preset = presets[int(preset_id) - 1]
                width, height = preset["width"], preset["height"]
                for x, y, visibility in points:
                    item.monsters.append(MonsterAnnotation(
                        [x-width/2, y-height, x+width/2, y], [x, y],
                        visibility=visibility, occluded=visibility < 1,
                        annotation_confidence=.55 if visibility < .5 else .8,
                        review_required=True, source="codex_prelabel", box_preset=preset_id))
            item.conditions = [c for c in item.conditions if c != "unlabeled"] + ["codex_prelabel_pending"]
            item.notes = "Manual visual proposals using reviewed frames 1-4 and saved preset dimensions. " + proposal["notes"]
            assert not item.validate(1920, 1080), item.validate(1920, 1080)
            lines[position - 1] = (json.dumps(item.to_dict(), separators=(",", ":")) + "\n").encode()
            image = cv2.imread(str(ROOT / item.image_path))
            for number, monster in enumerate(item.monsters, 1):
                color = (50, 210, 255) if monster.box_preset == "1" else (255, 170, 70)
                x1, y1, x2, y2 = map(round, monster.bbox_xyxy)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label = f"{number}: {presets[int(monster.box_preset)-1]['name']}"
                cv2.putText(image, label, (max(0, min(x1, 1790)), max(16, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
            overlay = output / f"position-{position:02d}-{item.frame_id}.jpg"
            assert cv2.imwrite(str(overlay), image)
            anchor = anchors[item.frame_id]
            frames.append({"frame_id": item.frame_id, "display_position": position,
                           "source_frame": anchor["frame"], "timestamp": item.timestamp,
                           "source_video": anchor["source_video"], "source_video_sha256": anchor["source_video_sha256"],
                           "image_sha256": anchor["image_sha256"], "proposal_count": len(item.monsters),
                           "review_status": "pending", "notes": proposal["notes"],
                           "overlay": overlay.relative_to(ROOT).as_posix()})
        counts = dict(Counter(m.box_preset for item in selected for m in item.monsters))
        if args.apply:
            backup.mkdir(parents=True)
            (backup / "annotations.jsonl").write_bytes(before)
            (backup / "review_settings.json").write_bytes(settings_before)
            (backup / "review_order.json").write_bytes(order_before)
            # Compare again immediately before replacement, while holding the reviewer lock.
            assert annotations.read_bytes() == before and settings_path.read_bytes() == settings_before
            temporary = annotations.with_suffix(".jsonl.tmp")
            temporary.write_bytes(b"".join(lines))
            temporary.replace(annotations)
            after = annotations.read_bytes().splitlines(keepends=True)
            assert after[:4] == before.splitlines(keepends=True)[:4]
            assert after[8:] == before.splitlines(keepends=True)[8:]
            assert settings_path.read_bytes() == settings_before
            assert (dataset / "review_order.json").read_bytes() == order_before
            write_json(manifest_path, {
                "schema": "specialized-visual-detection.codex-manual-review-batch.v1",
                "map_id": "timePassageOne", "labeling_method": "manual_visual_proposals_pending_review",
                "class_mapping": {"1": "mob", "2": "hero"}, "preset_snapshot": settings,
                "reference_frame_ids": [i.frame_id for i in items[:4]],
                "before_annotations_sha256": EXPECTED_ANNOTATIONS,
                "after_annotations_sha256": sha256_file(annotations),
                "review": {"split": "pilot", "status": "pending", "images_only": True, "display_positions": [5, 8]},
                "counts_by_preset": counts, "frames": frames,
                "unchanged": "Positions 1-4 and 9-50, review settings, saved order, original images",
            })
            write_contact_sheets(items, dataset / "contacts", split="pilot", annotations_path=annotations)
            report = write_report(items, dataset / "dataset_report.json")
            report["note"] = "Empty annotations remain unlabeled, not confirmed negatives. Positions 5-8 contain pending visual proposals."
            write_json(dataset / "dataset_report.json", report)
        print(json.dumps({"applied": args.apply, "positions": [5, 8], "counts_by_preset": counts,
                          "frames": frames}, indent=2))


if __name__ == "__main__":
    main()
