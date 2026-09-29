"""Pre-label saved APO display positions 21-50, excluding the book-bubble NPC."""
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
# Each mob tuple is (ground X, visible fraction). Y=734 follows the reference
# full-sprite box convention. Hero/special tuples are (X, Y, visible fraction).
# These are individual visual proposals, not a detector's predictions.
LABELS = {'0027': ([(443, 0.5),
           (478, 0.75),
           (514, 0.5),
           (638, 1),
           (832, 0.55),
           (1058, 0.8),
           (1470, 0.5),
           (1589, 0.75)],
          (1715, 739, 0.65),
          (1117, 739, 0.9),
          1728),
 '0014': ([(346, 0.75), (717, 0.9), (855, 0.5), (882, 0.5), (926, 0.8), (1341, 0.9), (1412, 0.7)],
          (603, 739, 1),
          (364, 739, 0.9),
          1792),
 '0042': ([(677, 0.75), (1587, 0.65), (1615, 0.5)], (870, 739, 0.35), (1715, 741, 0.8), 1755),
 '0006': ([(176, 0.35),
           (435, 1),
           (606, 1),
           (749, 0.9),
           (895, 0.7),
           (1045, 1),
           (1271, 0.7),
           (1322, 0.7),
           (1644, 0.8)],
          (1451, 739, 0.3),
          (215, 741, 0.85),
          1755),
 '0002': ([(218, 0.65),
           (349, 0.95),
           (414, 0.85),
           (529, 0.9),
           (773, 0.65),
           (1265, 0.55),
           (1294, 0.55),
           (1380, 1),
           (1470, 0.55)],
          (1195, 371, 0.8),
          (658, 739, 0.95),
          1668),
 '0023': ([(755, 0.8),
           (960, 0.75),
           (1001, 0.5),
           (1146, 0.75),
           (1231, 0.8),
           (1413, 0.65),
           (1477, 0.55),
           (1688, 0.65)],
          (1043, 739, 0.3),
          (484, 739, 0.95),
          1728),
 '0026': ([(1364, 0.75), (1464, 0.45), (1539, 0.65)], (1716, 739, 0.25), (948, 739, 0.85), 1728),
 '0038': ([(322, 1),
           (515, 0.85),
           (731, 0.7),
           (799, 0.45),
           (918, 1),
           (1041, 0.6),
           (1071, 0.65),
           (1238, 1),
           (1625, 0.75)],
          (643, 739, 0.3),
          (1567, 739, 0.8),
          1755),
 '0022': ([(489, 1), (1078, 0.7)], (1163, 739, 0.25), (282, 741, 0.9), 1728),
 '0041': ([(555, 0.85), (639, 0.8), (831, 0.55), (897, 0.65), (1406, 0.7), (1697, 0.55)],
          (1097, 739, 0.25),
          (1707, 739, 0.75),
          1755),
 '0017': ([(159, 0.35),
           (398, 1),
           (533, 1),
           (940, 1),
           (1399, 0.3),
           (1498, 0.6),
           (1597, 0.9),
           (1745, 0.4),
           (1760, 0.3)],
          (1377, 739, 0.25),
          (225, 741, 0.85),
          1728),
 '0015': ([(348, 0.8), (555, 1), (655, 1), (902, 0.8), (1034, 0.65), (1689, 0.5)],
          (1095, 739, 0.3),
          (224, 741, 0.9),
          1728),
 '0045': ([(1365, 1), (1587, 0.75)], (985, 739, 0.25), (1602, 741, 0.85), 1755),
 '0046': ([(1337, 1), (1587, 0.9)], (983, 739, 0.65), (1504, 741, 0.75), 1755),
 '0037': ([(250, 1),
           (518, 0.75),
           (751, 0.75),
           (825, 0.5),
           (993, 0.9),
           (1069, 0.75),
           (1126, 0.7),
           (1303, 1),
           (1669, 0.85)],
          (643, 739, 0.25),
          (1518, 741, 0.8),
          1755),
 '0020': ([(722, 1), (826, 0.45), (853, 0.4), (914, 0.8), (1181, 0.65), (1220, 0.65), (1298, 0.8)],
          (1408, 718, 0.3),
          (225, 741, 0.85),
          1728),
 '0001': ([(274, 0.55),
           (285, 0.7),
           (438, 0.85),
           (496, 0.85),
           (631, 0.75),
           (1031, 1),
           (1335, 1),
           (1494, 0.55),
           (1729, 0.6),
           (1752, 0.45)],
          (829, 371.5, 0.9),
          (645, 739, 0.95),
          1668),
 '0033': ([(433, 1),
           (605, 1),
           (732, 0.9),
           (865, 0.55),
           (1218, 0.8),
           (1279, 0.75),
           (1344, 0.75),
           (1558, 0.85)],
          (941, 712, 0.85),
          (1483, 742, 0.65),
          1740),
 '0005': ([(450, 0.7),
           (618, 1),
           (839, 0.55),
           (923, 0.9),
           (1077, 0.8),
           (1243, 0.75),
           (1417, 0.65),
           (1577, 0.65),
           (1615, 0.65),
           (1695, 0.45)],
          (1163, 739, 0.25),
          (454, 741, 0.9),
          1668),
 '0031': ([(428, 0.55), (473, 0.8), (734, 0.65), (766, 0.45), (837, 0.5), (1219, 1), (1344, 1), (1534, 0.75)],
          (996, 739, 1),
          (1483, 742, 0.65),
          1740),
 '0024': ([(1187, 1), (1298, 1), (1420, 0.55), (1446, 0.4), (1795, 0.6)],
          (1595, 739, 0.6),
          (909, 741, 0.65),
          1728),
 '0039': ([(400, 0.8),
           (491, 0.9),
           (694, 0.35),
           (754, 0.65),
           (830, 0.45),
           (958, 1),
           (1066, 1),
           (1153, 1),
           (1549, 0.85)],
          (643, 739, 0.6),
          (1655, 741, 0.9),
          1755),
 '0016': ([(345, 0.65),
           (375, 0.5),
           (655, 1),
           (934, 1),
           (1229, 0.8),
           (1382, 0.75),
           (1480, 0.55),
           (1637, 0.75),
           (1703, 0.55)],
          (1550, 739, 0.3),
          (225, 741, 0.85),
          1728),
 '0028': ([(364, 0.75), (435, 0.9), (686, 1), (934, 1), (1249, 1), (1356, 0.9), (1423, 0.45), (1460, 0.4)],
          (1153, 739, 0.85),
          (1562, 741, 0.9),
          1740),
 '0010': ([(179, 0.35), (203, 0.4), (990, 0.65), (1013, 0.45), (1090, 0.9), (1226, 0.25), (1387, 0.8)],
          (1207, 739, 0.55),
          (295, 741, 0.9),
          1755),
 '0030': ([(405, 0.55), (453, 0.8), (714, 0.7), (747, 0.45), (823, 0.5), (1218, 1), (1344, 1), (1508, 0.65)],
          (996, 739, 1),
          (1484, 741, 0.65),
          1740),
 '0043': ([(1420, 0.75), (1587, 0.8)], (1207, 739, 0.2), (1633, 741, 0.9), 1755),
 '0019': ([(613, 1), (799, 0.4), (829, 0.45), (946, 1), (1134, 0.6), (1155, 0.5), (1267, 0.8)],
          (1596, 739, 0.25),
          (315, 741, 0.9),
          1728),
 '0034': ([(433, 1), (605, 1), (726, 0.9), (958, 0.85), (1156, 0.9), (1218, 0.8), (1276, 0.65), (1437, 0.4)],
          (873, 739, 0.8),
          (1403, 741, 0.65),
          1750),
 '0025': ([(1310, 0.9), (1470, 0.5), (1545, 0.65)], (1715, 739, 0.25), (909, 741, 0.65), 1728)}


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    annotation_path = MAP / "dataset/annotations.jsonl"
    manifest_path = MAP / "dataset/codex_manual_batch_v2_manifest.json"
    backup = MAP / "dataset/runs/before-codex-labels-final30"
    output = MAP / "dataset/prelabel_contacts/final30"
    with ReviewInstanceLock(annotation_path):
        if manifest_path.exists() or backup.exists():
            raise FileExistsError("Final 30 labels already exist; refusing to overwrite review work")
        before = annotation_path.read_bytes()
        lines = before.splitlines(keepends=True)
        items = [FrameAnnotation.from_dict(json.loads(line)) for line in lines]
        order = json.loads((MAP / "dataset/review_order.json").read_text())["frame_ids"]
        assert [i.frame_id for i in items] == order
        selected = items[20:50]
        assert [i.frame_id for i in selected] == ["apo-random-" + suffix for suffix in LABELS]
        assert all(not i.monsters and i.review_status == "pending" for i in selected)
        settings = json.loads((MAP / "dataset/review_settings.json").read_text())
        presets = settings["box_presets"]
        assert [p["name"] for p in presets] == ["mob", "hero", "special"]
        pilot = json.loads((MAP / "dataset/pilot_manifest.json").read_text())
        anchors = {f["frame_id"]: f for f in pilot["frames"]}
        for item in selected:
            assert sha256_file(ROOT / item.image_path) == anchors[item.frame_id]["image_sha256"]
        backup.mkdir(parents=True)
        (backup / "annotations.jsonl").write_bytes(before)
        (backup / "review_settings.json").write_bytes((MAP / "dataset/review_settings.json").read_bytes())
        output.mkdir(parents=True, exist_ok=True)
        frames, strips = [], []
        colors = [(50,210,255), (255,170,70), (255,70,210)]
        for position, item in enumerate(selected, 21):
            mobs, hero, special, npc_x = LABELS[item.frame_id[-4:]]
            points = [(1,x,734,v) for x,v in mobs] + [(2,*hero),(3,*special)]
            for preset_id, x, y, visibility in points:
                preset = presets[preset_id-1]
                width, height = preset["width"], preset["height"]
                item.monsters.append(MonsterAnnotation(
                    [x-width/2,y-height,x+width/2,y], [x,y], visibility=visibility,
                    occluded=visibility<1, annotation_confidence=.6 if visibility<.5 else .8,
                    review_required=True, source="codex_prelabel", box_preset=str(preset_id)))
            item.conditions = [c for c in item.conditions if c != "unlabeled"] + [
                "codex_prelabel_pending", "book_bubble_npc_excluded"]
            item.notes = ("Manual visual proposals for saved display positions 21-50. "
                          "The ghost-like NPC with a book thought bubble is intentionally unlabeled; "
                          "neighboring mobs remain labeled. Review overlapping ghosts and effect/ladder occlusion.")
            assert not item.validate()
            lines[position-1] = (json.dumps(item.to_dict(), separators=(",", ":")) + "\n").encode()
            image = cv2.imread(str(ROOT / item.image_path))
            for monster in item.monsters:
                idx = int(monster.box_preset)-1
                x1,y1,x2,y2 = map(round, monster.bbox_xyxy)
                cv2.rectangle(image,(x1,y1),(x2,y2),colors[idx],2)
                cv2.putText(image,presets[idx]["name"],(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,.55,colors[idx],2)
            overlay = output / f"position-{position:02d}-{item.frame_id}.jpg"
            assert cv2.imwrite(str(overlay), image)
            strip = cv2.resize(image[450:760],(1440,233),interpolation=cv2.INTER_AREA)
            title = np.zeros((28,1440,3),np.uint8)
            cv2.putText(title,f"Position {position} / {item.frame_id} / {len(mobs)} mob + hero + special / book NPC excluded",
                        (8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
            strips.append(cv2.vconcat([title,strip]))
            frames.append({"frame_id":item.frame_id,"display_position":position,
                           "source_frame":anchors[item.frame_id]["frame"],
                           "image_sha256":anchors[item.frame_id]["image_sha256"],
                           "proposal_count":len(item.monsters),"review_status":"pending",
                           "excluded_npc_approx_ground_x":npc_x,
                           "overlay":overlay.relative_to(ROOT).as_posix()})
        temporary = annotation_path.with_suffix(".jsonl.tmp")
        temporary.write_bytes(b"".join(lines))
        temporary.replace(annotation_path)
        assert annotation_path.read_bytes().splitlines(keepends=True)[:20] == before.splitlines(keepends=True)[:20]
        assert annotation_path.read_bytes().splitlines(keepends=True)[50:] == before.splitlines(keepends=True)[50:]
        for start in range(0,30,3):
            assert cv2.imwrite(str(output/f"inspection-{start//3+1:02d}.jpg"),cv2.vconcat(strips[start:start+3]))
        counts = dict(Counter(m.box_preset for item in selected for m in item.monsters))
        write_json(manifest_path, {
            "schema":"specialized-visual-detection.codex-manual-review-batch.v1", "map_id":"APO",
            "labeling_method":"manual_visual_proposals_pending_review",
            "class_mapping":{"1":"mob","2":"hero","3":"special"},
            "exclusion_policy":"Ghost-like NPC bearing a book thought bubble is not a target; do not mask out neighboring mobs.",
            "source_video_sha256":pilot["source_video_sha256"],
            "review":{"split":"pilot","status":"pending","images_only":True,"display_positions":[21,50]},
            "prior_reviewed_frames_preserved":True,"note":"Upper-platform heroes in 0001/0002 are labeled at y=371; fully dissolved death effects in 0022 are not treated as visible mobs.","preset_snapshot":settings,"counts_by_preset":counts,"frames":frames})
        write_contact_sheets(items,MAP/"dataset/contacts",split="pilot",annotations_path=annotation_path)
        write_report(items,MAP/"dataset/dataset_report.json")
        print(json.dumps({"frames_added":30,"display_positions":[21,50],"counts_by_preset":counts,
                          "previous_reviewed_frames_preserved":True},indent=2))


if __name__ == "__main__":
    main()
