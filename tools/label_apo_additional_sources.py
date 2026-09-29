"""Append visually inspected sample1/sample3 proposals after the existing 50 frames."""
from collections import Counter
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

MAP = ROOT / "maps/APO"
# Display position -> (mob ground-X / visibility pairs, hero X/Y/visibility,
# special X/Y/visibility or None). Ground Y=734 for these APO mobs.
# No labels for the book-bubble NPC, pets, loot, fully hidden targets, or town NPCs.
LABELS = {
51: ([(704,.8),(760,.65),(807,.5),(1039,.65),(1072,.75),(1199,.85),(1322,.9),(1510,.8),(1668,.9)], (1264,739,1), (741,739,.8)),
52: ([(294,1),(486,1),(1124,.4),(1251,.4),(1301,.4),(1481,.4)], (1032,739,.25), (1010,739,.6)),
53: ([(229,.65),(276,.8),(766,.85),(895,.9),(983,.8),(1045,.9),(1492,.6),(1536,.75),(1590,.6),(1639,.75)], (792,423,1), (430,739,.95)),
54: ([(107,.7),(885,.8),(934,.8),(1125,.7),(1163,.7),(1350,.9),(1437,.7),(1707,.8),(1751,.5),(1790,.4)], (1238,739,.8), (1062,739,.85)),
55: ([(674,1),(884,.8),(1008,.85),(1059,.55),(1100,.85),(1459,.6),(1626,.95)], (1377,721,.85), (1135,739,.9)),
56: ([(210,.6),(287,.9),(420,1),(603,1),(982,.9),(1062,.95),(1398,.55),(1607,1),(1698,.8),(1735,.55)], (1332,739,.95), (240,739,.9)),
57: ([(244,.8),(703,1),(1023,.85),(1091,.75),(1147,.7),(1310,.65),(1522,.7),(1551,.55)], (1238,739,.3), (1125,739,.55)),
58: ([(324,.9),(381,.6),(417,.85),(727,1),(934,1),(1307,.95),(1457,.75),(1675,.95)], (832,371,.7), (193,739,.95)),
59: ([(294,1),(573,1),(1054,.6),(1112,.6),(1483,.65)], (1251,739,.3), (1063,739,.6)),
60: ([(229,.65),(275,.8),(1511,.45),(1605,.75),(1660,.55)], (829,371,.65), None),
61: ([(888,.8),(1000,1),(1114,.85),(1159,.65),(1329,1),(1454,.7),(1571,.8),(1641,.95)], (675,739,.9), (443,739,.95)),
62: ([(548,1),(689,1),(1045,.65),(1075,.6),(1098,.5),(1479,.65),(1540,.65),(1579,.75),(1641,.8)], (811,739,.75), (309,739,.95)),
63: ([(294,1),(441,1),(1099,.35),(1266,.3)], (1050,739,.35), (1002,739,.65)),
64: ([(405,.65),(437,.65),(556,.75),(596,.8),(1195,.85),(1277,.85),(1358,1),(1627,1)], (863,668,.7), (170,739,.9)),
65: ([(699,.75),(746,.85),(795,.5),(1026,.8),(1087,.8),(1149,.9),(1464,.7),(1690,.65)], (1283,739,1), (786,739,.8)),
66: ([], (1008,773,.7), None),
67: ([(491,.75),(528,.65),(859,.4),(975,.4),(1175,.7),(1268,.95),(1481,.7)], (1123,739,.3), None),
68: ([(1616,.6),(1646,.55)], (1371,739,.25), None),
69: ([(319,.95),(478,1),(593,1),(727,.7),(842,.7),(1091,.4),(1140,.4),(1325,1),(1445,.65),(1611,.55),(1682,.7)], (1507,739,.25), None),
70: ([(464,.7),(602,.65),(636,.65),(723,.85),(782,.55),(864,.6),(929,.8),(1305,1),(1390,.95),(1627,.65),(1665,.5)], (523,739,.85), None),
71: ([(345,1),(787,.6),(859,.6),(1126,1),(1455,.6)], (1365,739,.3), None),
72: ([(345,.9),(435,.8),(471,.65),(633,.95),(1169,.45),(1227,.65),(1323,.95),(1459,.55),(1515,.65)], (1128,739,.3), None),
73: ([(398,.75),(617,1),(770,.8),(874,.7),(966,1),(1107,1),(1295,.9),(1359,.9),(1461,.6),(1586,1)], (488,739,.3), None),
74: ([(789,.4),(1210,.9),(1312,1),(1418,.85),(1600,1),(1713,.95)], (1128,739,.3), None),
75: ([(189,.45),(251,.4),(294,.5),(468,.65),(643,.9),(734,1),(1297,.65),(1343,.7),(1785,.95)], (506,739,.3), None),
76: ([(395,1),(624,.9),(686,.75),(724,.55),(757,.6),(1273,.7),(1579,1),(1780,.5)], (1405,739,.3), None),
77: ([(345,1),(1344,1)], (540,739,.3), None),
78: ([(464,.7),(597,.75),(728,1),(910,.85),(980,1),(1255,1),(1418,.75),(1639,.5)], (523,739,.95), None),
79: ([(235,.85),(345,1),(478,.7),(523,.4),(933,.4),(1027,.4),(1372,.7),(1420,.65),(1734,.8)], (697,739,.3), None),
80: ([(174,.6),(345,1),(467,.35),(902,.8),(1372,.7),(1420,.65),(1694,.95)], (494,739,.3), None),
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


def main():
    dataset = MAP / "dataset"
    annotation_path = dataset / "annotations.jsonl"
    manifest_path = dataset / "codex_manual_batch_v3_manifest.json"
    backup = dataset / "runs/before-additional-sources"
    output = dataset / "prelabel_contacts/additional-sources"
    extraction = json.loads((dataset / "runs/additional-sources/extraction.json").read_text())
    with ReviewInstanceLock(annotation_path):
        if manifest_path.exists() or backup.exists():
            raise FileExistsError("Additional labels already appended; refusing overwrite")
        before = annotation_path.read_bytes()
        original = [FrameAnnotation.from_dict(json.loads(line)) for line in before.splitlines()]
        assert len(original) == 50
        order_path = dataset / "review_order.json"
        order = json.loads(order_path.read_text())
        assert [i.frame_id for i in original] == order["frame_ids"]
        settings_bytes = (dataset / "review_settings.json").read_bytes()
        settings = json.loads(settings_bytes)
        presets = settings["box_presets"]
        assert [p["name"] for p in presets] == ["mob","hero","special"]
        backup.mkdir(parents=True)
        (backup / "annotations.jsonl").write_bytes(before)
        (backup / "review_settings.json").write_bytes(settings_bytes)
        (backup / "review_order.json").write_bytes(order_path.read_bytes())
        output.mkdir(parents=True)
        appended, strips = [], []
        for entry in extraction["frames"]:
            position = entry["display_position"]
            assert sha256_file(ROOT / entry["image_path"]) == entry["image_sha256"]
            mobs, hero, special = LABELS[position]
            points = [(1,x,734,v) for x,v in mobs] + [(2,*hero)]
            if special:
                points.append((3,*special))
            item = FrameAnnotation(frame_id=entry["frame_id"], timestamp=entry["timestamp"],
                image_path=entry["image_path"], split="pilot", review_status="pending",
                category="random_original_video", conditions=[entry["source_name"],
                "codex_prelabel_pending", "book_bubble_npc_excluded"],
                notes="Manual visual proposals. Exclude NPCs, pets, loot, and fully hidden targets. Review effects and overlaps.")
            if position == 66:
                item.conditions.append("outside_APO_town")
                item.notes += " Random sample is in town; label only the user's hero, not other players/NPCs."
            if position == 60:
                item.conditions.append("dialogue_occlusion")
                item.notes += " Dialogue hides the special completely; no special box inferred."
            for preset_id,x,y,visibility in points:
                preset = presets[preset_id-1]
                width,height = preset["width"],preset["height"]
                item.monsters.append(MonsterAnnotation([x-width/2,y-height,x+width/2,y],[x,y],
                    visibility=visibility, occluded=visibility<1,
                    annotation_confidence=.55 if visibility<.5 else .8,
                    review_required=True,source="codex_prelabel",box_preset=str(preset_id)))
            assert not item.validate()
            appended.append(item)
            frame = cv2.imread(str(ROOT / entry["image_path"]))
            colors = [(50,210,255),(255,170,70),(255,70,210)]
            for box in item.monsters:
                idx = int(box.box_preset)-1
                x1,y1,x2,y2 = map(round,box.bbox_xyxy)
                cv2.rectangle(frame,(x1,y1),(x2,y2),colors[idx],2)
                cv2.putText(frame,presets[idx]["name"],(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,.55,colors[idx],2)
            overlay = output / f"position-{position:02d}-{item.frame_id}.jpg"
            assert cv2.imwrite(str(overlay),frame)
            title = np.zeros((30,1920,3),np.uint8)
            cv2.putText(title,f"{position} / {item.frame_id} / {len(mobs)} mob / {int(bool(special))} special",
                (8,22),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
            strips.append(cv2.resize(cv2.vconcat([title,frame[430:790]]),(1536,312)))
            entry.update(proposal_count=len(item.monsters), review_status="pending",
                         overlay=overlay.relative_to(ROOT).as_posix())
        for start in range(0,30,3):
            assert cv2.imwrite(str(output/f"inspection-{start//3+1:02d}.jpg"),cv2.vconcat(strips[start:start+3]))
        addition = b"".join((json.dumps(i.to_dict(),separators=(",",":"))+"\n").encode() for i in appended)
        assert before.endswith(b"\n")
        temporary = annotation_path.with_suffix(".jsonl.tmp")
        temporary.write_bytes(before + addition)
        temporary.replace(annotation_path)
        order["frame_ids"] += [i.frame_id for i in appended]
        order["source_groups"] = [dict(source="APO_sample2", display_positions=[1,50],existing_order_preserved=True)] + extraction["sources"]
        write_json(order_path,order)
        counts = dict(Counter(m.box_preset for i in appended for m in i.monsters))
        write_json(manifest_path,dict(schema="specialized-visual-detection.codex-manual-review-batch.v1",
            map_id="APO", labeling_method="manual_visual_proposals_pending_review",
            sources=extraction["sources"],frames=extraction["frames"],
            class_mapping={"1":"mob","2":"hero","3":"special"},counts_by_preset=counts,
            review=dict(split="pilot",status="pending",images_only=True,display_positions=[51,80]),
            sampling="15 indices per source without replacement; sequential VFR decode; PNG pixel equality checked",
            timestamp_basis="ffprobe best_effort_timestamp_time",source_size=[1920,1080],
            order="Existing 50 unchanged; sample1 then sample3; saved per-source shuffle",
            exclusion_policy="Book-bubble NPC, town NPCs, other players, pets, loot, and fully hidden objects remain unlabeled",
            preset_snapshot=settings))
        all_items = original + appended
        write_contact_sheets(all_items,dataset/"contacts",split="pilot",annotations_path=annotation_path)
        report = write_report(all_items,dataset/"dataset_report.json")
        assert annotation_path.read_bytes().startswith(before)
        assert (dataset/"review_settings.json").read_bytes() == settings_bytes
        assert not report["malformed_frames"]
        print(json.dumps(dict(appended=30,total=80,counts_by_preset=counts),indent=2))


if __name__ == "__main__":
    main()
