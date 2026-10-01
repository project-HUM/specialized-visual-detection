"""Render manual proposals for positions 9-25; --apply saves pending annotations."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monster_dataset.contact_sheet import write_contact_sheets
from monster_dataset.review_app import ReviewInstanceLock
from monster_dataset.schema import FrameAnnotation, MonsterAnnotation
from monster_dataset.validation import write_report
from perception.core import sha256_file
from tools.prelabel_time_passage_one_first4 import LABELS as PREVIOUS_PROPOSALS

MAP = ROOT / "maps/timePassageOne"
EXPECTED_ANNOTATIONS = "5f5c060a0d471b74ec2e6d125570a70db989517b9eee768fed6543dbba2c4522"
EXPECTED_SETTINGS = "b48bb07bddd89fef249cdd35246cc68b5f69b5f6bcc3e68ee5287b9d766a20fc"


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def corrections(items):
    changes = []
    for position, item in enumerate(items[4:8], 5):
        previous = PREVIOUS_PROPOSALS[item.frame_id]
        points = [("1", *point[:2]) for point in previous["mob"]] + [("2", *previous["hero"][:2])]
        assert len(item.monsters) == len(points)
        for number, (monster, (preset, x, y)) in enumerate(zip(item.monsters, points), 1):
            assert monster.box_preset == preset
            x1, y1, x2, y2 = monster.bbox_xyxy
            updated = [(x1+x2)/2, y2]
            if abs(updated[0]-x) > 1e-5 or abs(updated[1]-y) > 1e-5:
                changes.append({"display_position": position, "box_number": number,
                                "preset": preset, "before_bottom_center": [x, y],
                                "after_bottom_center": updated,
                                "delta_xy": [updated[0]-x, updated[1]-y]})
    return {"adjusted_boxes": changes, "class_counts_unchanged": True,
            "hero_boxes_unchanged": not any(c["preset"] == "2" for c in changes),
            "preset_sizes_unchanged": True,
            "application": "Small per-sprite placement corrections; retain existing dimensions and visibility policy. No global offset."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    dataset = MAP / "dataset"
    annotations = dataset / "annotations.jsonl"
    settings_path = dataset / "review_settings.json"
    manifest_path = dataset / "codex_manual_batch_v1_manifest.json"
    proposals_path = dataset / "codex_manual_batch_v1_proposals.json"
    backup = dataset / "runs/before-codex-labels-1"
    output = dataset / "prelabel_contacts/batch1"
    proposals = json.loads(proposals_path.read_text())
    assert list(proposals) == [str(i) for i in range(9, 26)]
    with ReviewInstanceLock(annotations):
        before = annotations.read_bytes()
        settings_before = settings_path.read_bytes()
        order_before = (dataset / "review_order.json").read_bytes()
        if hashlib.sha256(before).hexdigest() != EXPECTED_ANNOTATIONS or sha256_file(settings_path) != EXPECTED_SETTINGS:
            raise RuntimeError("Annotations or presets changed after visual inspection; re-inspect before applying")
        if manifest_path.exists() or backup.exists():
            raise FileExistsError("Batch already applied; refusing to overwrite review work")
        lines = before.splitlines(keepends=True)
        items = [FrameAnnotation.from_dict(json.loads(line)) for line in lines]
        assert len(items) == 50
        assert [i.frame_id for i in items] == json.loads(order_before)["frame_ids"]
        assert all(i.review_status == "reviewed" for i in items[:8])
        selected = items[8:25]
        assert [i.frame_id for i in selected] == [f"tpo-sample0-{n:04d}" for n in range(8, 25)]
        assert all(i.review_status == "pending" and not i.monsters for i in selected)
        comparison = corrections(items)
        settings = json.loads(settings_before)
        presets = settings["box_presets"]
        assert [(p["name"], p["width"], p["height"]) for p in presets] == [("mob",104,116),("hero",108,148)]
        pilot = json.loads((dataset / "pilot_manifest.json").read_text())
        anchors = {f["frame_id"]: f for f in pilot["frames"]}
        output.mkdir(parents=True, exist_ok=True)
        frames, thumbnails = [], []
        for position, item in enumerate(selected, 9):
            anchor = anchors[item.frame_id]
            assert sha256_file(ROOT / item.image_path) == anchor["image_sha256"]
            proposal = proposals[str(position)]
            for preset_id, points in (("1", proposal["mob"]), ("2", [proposal["hero"]])):
                preset = presets[int(preset_id)-1]
                width, height = preset["width"], preset["height"]
                for x, y, visibility in points:
                    item.monsters.append(MonsterAnnotation(
                        [x-width/2, y-height, x+width/2, y], [x,y],
                        visibility=visibility, occluded=visibility < 1,
                        annotation_confidence=.5 if visibility < .5 else .8,
                        review_required=True, source="codex_prelabel", box_preset=preset_id))
            item.conditions = [c for c in item.conditions if c != "unlabeled"] + ["codex_prelabel_pending"]
            item.notes = "Manual visual proposals using reviewed positions 1-8, including user corrections to batch 0. " + proposal["notes"]
            assert not item.validate(1920,1080), (position,item.validate(1920,1080))
            lines[position-1] = (json.dumps(item.to_dict(), separators=(",", ":")) + "\n").encode()
            image = cv2.imread(str(ROOT / item.image_path))
            for number, m in enumerate(item.monsters, 1):
                color = (50,210,255) if m.box_preset == "1" else (255,170,70)
                x1,y1,x2,y2 = map(round,m.bbox_xyxy)
                cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
                cv2.putText(image,f"{number}:{presets[int(m.box_preset)-1]['name']}",
                            (max(0,min(x1,1790)),max(16,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
            cv2.putText(image,f"Position {position}: {len(proposal['mob'])} mob + hero / PENDING",
                        (520,1040),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
            overlay = output / f"position-{position:02d}-{item.frame_id}.jpg"
            assert cv2.imwrite(str(overlay),image)
            thumbnails.append(cv2.resize(image,(960,540),interpolation=cv2.INTER_AREA))
            frames.append({"frame_id":item.frame_id,"display_position":position,
                           "source_frame":anchor["frame"],"timestamp":item.timestamp,
                           "source_video":anchor["source_video"],"source_video_sha256":anchor["source_video_sha256"],
                           "image_sha256":anchor["image_sha256"],"proposal_count":len(item.monsters),
                           "counts_by_preset":dict(Counter(m.box_preset for m in item.monsters)),
                           "review_status":"pending","notes":proposal["notes"],
                           "overlay":overlay.relative_to(ROOT).as_posix()})
        for start in range(0,len(thumbnails),4):
            tiles = thumbnails[start:start+4]
            while len(tiles) < 4:
                tiles.append(np.zeros((540,960,3),np.uint8))
            assert cv2.imwrite(str(output/f"inspection-{start//4+1:02d}.jpg"),
                               cv2.vconcat([cv2.hconcat(tiles[:2]),cv2.hconcat(tiles[2:])]))
        counts = dict(Counter(m.box_preset for item in selected for m in item.monsters))
        if args.apply:
            backup.mkdir(parents=True)
            (backup / "annotations.jsonl").write_bytes(before)
            (backup / "review_settings.json").write_bytes(settings_before)
            (backup / "review_order.json").write_bytes(order_before)
            assert annotations.read_bytes() == before and settings_path.read_bytes() == settings_before
            temporary = annotations.with_suffix(".jsonl.tmp")
            temporary.write_bytes(b"".join(lines))
            temporary.replace(annotations)
            after = annotations.read_bytes().splitlines(keepends=True)
            assert after[:8] == before.splitlines(keepends=True)[:8]
            assert after[25:] == before.splitlines(keepends=True)[25:]
            assert settings_path.read_bytes() == settings_before
            assert (dataset / "review_order.json").read_bytes() == order_before
            write_json(manifest_path,{
                "schema":"specialized-visual-detection.codex-manual-review-batch.v1",
                "map_id":"timePassageOne","labeling_method":"manual_visual_proposals_pending_review",
                "class_mapping":{"1":"mob","2":"hero"},"preset_snapshot":settings,
                "reference_frame_ids":[i.frame_id for i in items[:8]],"reference_corrections":comparison,
                "before_annotations_sha256":EXPECTED_ANNOTATIONS,"after_annotations_sha256":sha256_file(annotations),
                "proposals_sha256":sha256_file(proposals_path),
                "review":{"split":"pilot","status":"pending","images_only":True,"display_positions":[9,25]},
                "counts_by_preset":counts,"frames":frames,
                "unchanged":"Positions 1-8 and 26-50, review settings, saved order, original images",
            })
            write_contact_sheets(items,dataset/"contacts",split="pilot",annotations_path=annotations)
            report = write_report(items,dataset/"dataset_report.json")
            report["note"] = "Positions 1-8 are reviewed; 9-25 have pending proposals; 26-50 remain unlabeled, not confirmed negatives."
            write_json(dataset/"dataset_report.json",report)
        print(json.dumps({"applied":args.apply,"positions":[9,25],"counts_by_preset":counts,
                          "reference_corrections":comparison,
                          "frame_counts":[{"position":f["display_position"],"counts":f["counts_by_preset"]} for f in frames]},indent=2))


if __name__ == "__main__":
    main()
