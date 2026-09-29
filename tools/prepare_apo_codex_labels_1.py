"""Pre-label saved APO display positions 6-20, excluding the book-bubble NPC."""
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
LABELS = {
    "0012": ([(525,.9),(696,.75),(879,.75),(1492,.55),(1546,.85)], (1200,739,.9), (634,737,.9), 1728),
    "0007": ([(185,.35),(306,.95),(731,1),(874,.7),(1157,.9),(1341,.8),(1614,.85)], (1009,739,.3), (166,739,.8), 1755),
    "0047": ([(1411,.75),(1589,.8)], (809,739,.65), (1655,739,.85), 1755),
    "0044": ([(1384,1),(1585,.7)], (1205,739,.25), (1618,739,.85), 1755),
    "0008": ([(186,.35),(270,.85),(771,.65),(916,.95),(1022,.5),(1157,.8),(1344,.85),(1588,.9)], (1086,739,.3), (199,739,.8), 1755),
    "0032": ([(433,1),(526,1),(786,.4),(815,.45),(1218,1),(1343,1),(1583,.9)], (996,739,1), (1485,742,.65), 1740),
    "0018": ([(177,.35),(475,1),(1009,.9),(1409,.4),(1529,.65),(1616,.9),(1740,.4),(1756,.3)], (1468,739,.25), (224,744,.85), 1728),
    "0021": ([(457,1),(1028,.85),(1351,.65),(1422,.75)], (1141,739,.3), (282,741,.9), 1728),
    "0036": ([(355,1),(563,.8),(795,.45),(870,.5),(899,.6),(1080,.9),(1195,.8),(1278,.8),(1471,.65)], (629,739,.4), (1346,740,.85), 1805),
    "0003": ([(312,.5),(344,.65),(410,.9),(1028,.75),(1058,.45),(1353,.9),(1480,.6),(1735,.8),(1780,.65)], (1622,739,.3), (230,739,.95), 1670),
    "0029": ([(237,.8),(326,1),(483,1),(587,1),(839,.5),(1249,.45),(1277,.8),(1313,.4),(1348,.65)], (1060,739,.8), (1483,742,.65), 1740),
    "0048": ([(343,.8),(455,.5),(484,.65),(587,.65),(1052,.6),(1100,.65),(1519,.75),(1584,.9)], (809,739,.65), (1741,739,.75), 1755),
    "0011": ([(524,.9),(665,.7),(879,.75),(1491,.55),(1545,.85)], (1165,739,1), (607,736,.9), 1728),
    "0009": ([(142,.45),(202,.45),(981,.6),(1002,.45),(1091,.95),(1220,.45),(1385,.8)], (1283,718,.55), (329,738,.9), 1755),
    "0049": ([(441,.9),(518,.8),(704,.9),(925,.85),(1005,.85),(1121,.8),(1170,.55),(1361,.95),(1680,.8)], (830,739,.3), (1630,741,.85), 1763),
}


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    annotation_path = MAP / "dataset/annotations.jsonl"
    manifest_path = MAP / "dataset/codex_manual_batch_v1_manifest.json"
    backup = MAP / "dataset/runs/before-codex-labels-1"
    output = MAP / "dataset/prelabel_contacts/batch1"
    with ReviewInstanceLock(annotation_path):
        if manifest_path.exists() or backup.exists():
            raise FileExistsError("Batch 1 already exists; refusing to overwrite review work")
        before = annotation_path.read_bytes()
        lines = before.splitlines(keepends=True)
        items = [FrameAnnotation.from_dict(json.loads(line)) for line in lines]
        order = json.loads((MAP / "dataset/review_order.json").read_text())["frame_ids"]
        assert [i.frame_id for i in items] == order
        selected = items[5:20]
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
        for position, item in enumerate(selected, 6):
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
            item.notes = ("Manual visual proposals for saved display positions 6-20. "
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
        assert annotation_path.read_bytes().splitlines(keepends=True)[:5] == before.splitlines(keepends=True)[:5]
        assert annotation_path.read_bytes().splitlines(keepends=True)[20:] == before.splitlines(keepends=True)[20:]
        for start in range(0,15,3):
            assert cv2.imwrite(str(output/f"inspection-{start//3+1:02d}.jpg"),cv2.vconcat(strips[start:start+3]))
        counts = dict(Counter(m.box_preset for item in selected for m in item.monsters))
        write_json(manifest_path, {
            "schema":"specialized-visual-detection.codex-manual-review-batch.v1", "map_id":"APO",
            "labeling_method":"manual_visual_proposals_pending_review",
            "class_mapping":{"1":"mob","2":"hero","3":"special"},
            "exclusion_policy":"Ghost-like NPC bearing a book thought bubble is not a target; do not mask out neighboring mobs.",
            "source_video_sha256":pilot["source_video_sha256"],
            "review":{"split":"pilot","status":"pending","images_only":True,"display_positions":[6,20]},
            "prior_reviewed_frames_preserved":True,"preset_snapshot":settings,"counts_by_preset":counts,"frames":frames})
        write_contact_sheets(items,MAP/"dataset/contacts",split="pilot",annotations_path=annotation_path)
        write_report(items,MAP/"dataset/dataset_report.json")
        print(json.dumps({"frames_added":15,"display_positions":[6,20],"counts_by_preset":counts,
                          "previous_reviewed_frames_preserved":True},indent=2))


if __name__ == "__main__":
    main()
