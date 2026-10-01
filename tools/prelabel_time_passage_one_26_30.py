"""Render manual proposals for positions 26-30; --apply saves pending labels."""
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

DATASET = ROOT / "maps/timePassageOne/dataset"
EXPECTED_ANNOTATIONS = "d7848cbbd945ffb4cc8526aded290576011a121f3bcee00dbb3031dc5a1934f3"
EXPECTED_SETTINGS = "b48bb07bddd89fef249cdd35246cc68b5f69b5f6bcc3e68ee5287b9d766a20fc"
EXPECTED_ORDER = "3443b6639aa32f54a947fed9f3c731fab84e9273f01674f4e2d3335221ef6b51"


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    annotations = DATASET / "annotations.jsonl"
    settings_path = DATASET / "review_settings.json"
    order_path = DATASET / "review_order.json"
    proposal_path = DATASET / "codex_manual_batch_v2_proposals.json"
    manifest_path = DATASET / "codex_manual_batch_v2_manifest.json"
    backup = DATASET / "runs/before-codex-labels-2"
    output = DATASET / "prelabel_contacts/batch2"
    proposals = json.loads(proposal_path.read_text())
    assert list(proposals) == [str(i) for i in range(26,31)]
    with ReviewInstanceLock(annotations):
        before = annotations.read_bytes()
        settings_before = settings_path.read_bytes()
        order_before = order_path.read_bytes()
        if (hashlib.sha256(before).hexdigest() != EXPECTED_ANNOTATIONS
                or sha256_file(settings_path) != EXPECTED_SETTINGS
                or sha256_file(order_path) != EXPECTED_ORDER):
            raise RuntimeError("Annotations, presets or order changed; re-inspect before applying")
        if manifest_path.exists() or backup.exists():
            raise FileExistsError("Batch already applied; refusing to overwrite review work")
        original_lines = before.splitlines(keepends=True)
        lines = list(original_lines)
        items = [FrameAnnotation.from_dict(json.loads(line)) for line in lines]
        assert len(items) == 50
        assert [i.frame_id for i in items] == json.loads(order_before)["frame_ids"]
        assert all(i.review_status == "reviewed" for i in items[:25])
        selected = items[25:30]
        assert [i.frame_id for i in selected] == [f"tpo-sample1-{n:04d}" for n in range(5)]
        assert all(i.review_status == "pending" for i in selected)
        assert all(not i.monsters for i in items[26:])
        saved_hero = json.loads(original_lines[25])["monsters"]
        assert len(saved_hero) == 1 and saved_hero[0]["box_preset"] == "2"
        assert saved_hero[0]["source"] == "visual_review"
        settings = json.loads(settings_before)
        presets = settings["box_presets"]
        assert [(p["name"],p["width"],p["height"]) for p in presets] == [("mob",104,116),("hero",108,148)]
        pilot = json.loads((DATASET / "pilot_manifest.json").read_text())
        anchors = {f["frame_id"]: f for f in pilot["frames"]}
        output.mkdir(parents=True, exist_ok=True)
        frames = []
        for position, item in enumerate(selected,26):
            anchor = anchors[item.frame_id]
            assert sha256_file(ROOT / item.image_path) == anchor["image_sha256"]
            proposal = proposals[str(position)]
            for preset_id, points in (("1",proposal["mob"]),("2",proposal["hero"])):
                preset = presets[int(preset_id)-1]
                width,height = preset["width"],preset["height"]
                for x,y,visibility in points:
                    item.monsters.append(MonsterAnnotation(
                        [x-width/2,y-height,x+width/2,y],[x,y],
                        visibility=visibility,occluded=visibility < 1,
                        annotation_confidence=.5 if visibility < .5 else .8,
                        review_required=True,source="codex_prelabel",box_preset=preset_id))
            item.conditions = [c for c in item.conditions if c != "unlabeled"] + ["codex_prelabel_pending"] + proposal.get("conditions",[])
            item.notes = "Manual visual proposals using reviewed frames 1-25 and the user's hero in frame 26. " + proposal["notes"]
            assert not item.validate(1920,1080), (position,item.validate(1920,1080))
            lines[position-1] = (json.dumps(item.to_dict(),separators=(",",":")) + "\n").encode()
            counts = dict(Counter(m.box_preset for m in item.monsters))
            image = cv2.imread(str(ROOT / item.image_path))
            for number,m in enumerate(item.monsters,1):
                color = (50,210,255) if m.box_preset == "1" else (255,170,70)
                x1,y1,x2,y2 = map(round,m.bbox_xyxy)
                cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
                cv2.putText(image,f"{number}:{presets[int(m.box_preset)-1]['name']}",
                            (max(0,min(x1,1790)),max(16,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
            cv2.putText(image,f"Position {position}: {counts.get('1',0)} mob + {counts.get('2',0)} hero / PENDING",
                        (520,1040),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
            overlay = output / f"position-{position:02d}-{item.frame_id}.jpg"
            assert cv2.imwrite(str(overlay),image)
            frames.append({"frame_id":item.frame_id,"display_position":position,
                           "source_frame":anchor["frame"],"timestamp":item.timestamp,
                           "source_video":anchor["source_video"],"source_video_sha256":anchor["source_video_sha256"],
                           "image_sha256":anchor["image_sha256"],"counts_by_preset":counts,
                           "new_proposal_count":len(proposal["mob"])+len(proposal["hero"]),
                           "review_status":"pending","notes":proposal["notes"],
                           "overlay":overlay.relative_to(ROOT).as_posix()})
        assert selected[0].to_dict()["monsters"][:1] == saved_hero
        counts = dict(Counter(m.box_preset for item in selected for m in item.monsters if m.source == "codex_prelabel"))
        if args.apply:
            backup.mkdir(parents=True)
            for name,data in (("annotations.jsonl",before),("review_settings.json",settings_before),("review_order.json",order_before)):
                (backup / name).write_bytes(data)
            assert annotations.read_bytes() == before and settings_path.read_bytes() == settings_before
            assert order_path.read_bytes() == order_before
            temporary = annotations.with_suffix(".jsonl.tmp")
            temporary.write_bytes(b"".join(lines))
            temporary.replace(annotations)
            after = annotations.read_bytes().splitlines(keepends=True)
            assert after[:25] == original_lines[:25] and after[30:] == original_lines[30:]
            assert json.loads(after[25])["monsters"][:1] == saved_hero
            assert settings_path.read_bytes() == settings_before and order_path.read_bytes() == order_before
            write_json(manifest_path,{
                "schema":"specialized-visual-detection.codex-manual-review-batch.v1",
                "map_id":"timePassageOne","labeling_method":"manual_visual_proposals_pending_review",
                "class_mapping":{"1":"mob","2":"hero"},"preset_snapshot":settings,
                "reference_frame_ids":[i.frame_id for i in items[:26]],
                "preserved_hero":saved_hero[0],
                "before_annotations_sha256":EXPECTED_ANNOTATIONS,"after_annotations_sha256":sha256_file(annotations),
                "proposals_sha256":sha256_file(proposal_path),
                "review":{"split":"pilot","status":"pending","images_only":True,"display_positions":[26,30]},
                "new_counts_by_preset":counts,"frames":frames,
                "unchanged":"Positions 1-25 and 31-50 byte-for-byte; frame-26 user hero exactly; presets, order and source images",
            })
            write_contact_sheets(items,DATASET/"contacts",split="pilot",annotations_path=annotations)
            report = write_report(items,DATASET/"dataset_report.json")
            report["note"] = "Positions 1-25 are reviewed; 26-30 have pending proposals with the frame-26 user hero preserved; 31-50 remain unlabeled, not confirmed negatives. Frame 27 is an off-map event; frame 30 has a fully inventory-occluded hero."
            write_json(DATASET/"dataset_report.json",report)
        print(json.dumps({"applied":args.apply,"positions":[26,30],"new_counts_by_preset":counts,
                          "preserved_user_hero":True,
                          "frame_counts":[{"position":f["display_position"],"counts":f["counts_by_preset"]} for f in frames]},indent=2))


if __name__ == "__main__":
    main()
