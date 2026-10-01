"""Freeze reviewed timePassageOne data for the agreed CPU experiment."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monster_dataset.annotation_io import export_yolo
from monster_dataset.review_app import ReviewInstanceLock
from monster_dataset.schema import read_jsonl, write_jsonl
from monster_dataset.validation import write_report
from tools.time_passage_one_training_runner import sha, write_json, inventory, verify, event

MAP = ROOT / "maps/timePassageOne"
EXPERIMENT = "pretrained-v1-20260930"
CLASSES = {"1":"mob", "2":"hero"}


def select_validation(items):
    rng = random.Random(20260930)
    selected = set()
    for prefix, count in (("tpo-sample0-",5),("tpo-sample1-",4)):
        eligible = sorted(i.frame_id for i in items if i.frame_id.startswith(prefix)
                          and "outside_timePassageOne_event" not in i.conditions)
        selected.update(rng.sample(eligible, count))
    return selected


def main():
    source = MAP / "dataset/annotations.jsonl"
    output = MAP / "dataset/runs" / EXPERIMENT
    tracked = MAP / "experiments" / EXPERIMENT
    if output.exists() or tracked.exists():
        raise FileExistsError("Experiment exists; refusing overwrite")
    with ReviewInstanceLock(source):
        original = source.read_bytes()
        settings = (MAP/"dataset/review_settings.json").read_bytes()
        order = (MAP/"dataset/review_order.json").read_bytes()
        items = read_jsonl(source)
        assert len(items) == len({i.frame_id for i in items}) == 50
        assert all(i.review_status == "reviewed" and not i.validate() and
                   all(not b.review_required and b.box_preset in CLASSES for b in i.monsters) for i in items)
        assert sum(len(i.monsters) for i in items) == 490
        assert json.loads(order)["frame_ids"] == [i.frame_id for i in items]
        pilot = json.loads((MAP/"dataset/pilot_manifest.json").read_text())
        anchors = {f["frame_id"]:f for f in pilot["frames"]}
        for item in items:
            a = anchors[item.frame_id]
            assert a["image_path"] == item.image_path and a["timestamp"] == item.timestamp
            assert sha(ROOT/item.image_path) == a["image_sha256"]
        validation = select_validation(items)
        assert [n for n,i in enumerate(items,1) if i.frame_id in validation] == [8,13,15,21,24,36,45,49,50]
        snapshot = copy.deepcopy(items)
        excluded = []
        for position, item in enumerate(snapshot,1):
            if "outside_timePassageOne_event" in item.conditions:
                item.split = "pilot"
                excluded.append(dict(frame_id=item.frame_id,display_position=position,reason="Off-map event scene"))
            else:
                item.split = "validation" if item.frame_id in validation else "train"
        assert [f["display_position"] for f in excluded] == [27,42,43]
        output.mkdir(parents=True)
        (output/"models").mkdir()
        provenance = output/"provenance"
        provenance.mkdir()
        (provenance/"annotations.canonical.jsonl").write_bytes(original)
        for name in ["review_settings.json","review_order.json","pilot_manifest.json",
                     *[f"codex_manual_batch_v{v}_manifest.json" for v in range(4)]]:
            src = MAP/"dataset"/name
            if src.exists(): shutil.copy2(src,provenance/name)
        for name in ["map.json","capture_manifest.json"]:
            shutil.copy2(MAP/name,provenance/name)
        shutil.copy2(Path(__file__),provenance/Path(__file__).name)
        shutil.copy2(ROOT/"tools/time_passage_one_training_runner.py",output/"runner.py")
        shutil.copy2(ROOT/"yolo11n.pt",output/"models/yolo11n.pt")
        reference = ROOT/"maps/APO/experiments/pretrained-v1-20260929/training-config.json"
        config = json.loads(reference.read_text())
        config.update(epochs=1000,patience=100,seed=20260930,val=True)
        write_json(output/"training-config.json",config)
        write_json(output/"stopping-policy.json",dict(min_epochs=200,max_epochs=1000,patience=100,
                   metric="validation mAP50-95",improvement="strictly higher; ties do not reset patience"))
        shutil.copy2(reference,provenance/"apo-training-config.json")
        exports = {split:export_yolo(snapshot,output/"dataset",split=split,image_root=ROOT,preset_classes=CLASSES)
                   for split in ["train","validation"]}
        assert (exports["train"]["frames"],exports["train"]["instances"]) == (38,407)
        assert (exports["validation"]["frames"],exports["validation"]["instances"]) == (9,80)
        frames, split_hashes = [], {"train":set(),"validation":set()}
        for position,item in enumerate(snapshot,1):
            a = anchors[item.frame_id]
            if item.split == "pilot":
                destination = output/"excluded"/Path(item.image_path).name
                destination.parent.mkdir(exist_ok=True)
                shutil.copy2(ROOT/item.image_path,destination)
                label = None
            else:
                destination = output/"dataset/images"/item.split/(item.frame_id+Path(item.image_path).suffix)
                label = output/"dataset/labels"/item.split/(item.frame_id+".txt")
                for line in label.read_text().splitlines():
                    c,x,y,w,h = map(float,line.split())
                    assert int(c)==c and 0<=c<2 and 0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1
                split_hashes[item.split].add(sha(destination))
            item.image_path = destination.relative_to(output).as_posix()
            frames.append(dict(frame_id=item.frame_id,display_position=position,split=item.split,
                source_video=a["source_video"],source_video_sha256=a["source_video_sha256"],
                source_frame=a["frame"],timestamp=item.timestamp,image_path=item.image_path,
                image_sha256=sha(destination),label_path=label.relative_to(output).as_posix() if label else None,
                label_sha256=sha(label) if label else None,boxes=len(item.monsters)))
        assert not split_hashes["train"] & split_hashes["validation"]
        write_jsonl(snapshot,output/"training_annotations.jsonl")
        report = write_report(snapshot,output/"dataset-report.json")
        assert report["training_ready"] and not report["malformed_frames"]
        (output/"dataset/dataset.yaml").write_text("path: .\ntrain: images/train\nval: images/validation\nnames:\n  0: mob\n  1: hero\n")
        import torch, ultralytics, cv2
        freeze = subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True)
        (output/"requirements-frozen.txt").write_text(freeze,encoding="utf-8")
        write_json(output/"environment.json",dict(python=sys.version,executable=sys.executable,
            torch=torch.__version__,ultralytics=ultralytics.__version__,opencv=cv2.__version__,
            cuda=torch.cuda.is_available(),threads=6))
        import importlib.metadata
        (output/"requirements-training.txt").write_text(
            f"torch=={torch.__version__}\nultralytics=={ultralytics.__version__}\nopencv-python=={importlib.metadata.version('opencv-python')}\n")
        for export in exports.values(): export["output"] = "dataset"
        manifest = dict(schema="timePassageOne.training-snapshot.v1",experiment=EXPERIMENT,
            created_utc=datetime.now(timezone.utc).isoformat(),class_names=list(CLASSES.values()),
            initialization="generic pretrained YOLO11n",initial_weights_sha256=sha(output/"models/yolo11n.pt"),
            canonical_sha256=sha(source),exports=exports,frames=frames,excluded=excluded,
            split_policy="Seed 20260930; 5 of 25 Sample0 and 4 of 22 eligible Sample1 frames held out; remaining 38 train",
            validation_positions=[8,13,15,21,24,36,45,49,50],
            evaluation_caution="Validation shares source videos with training; small development set",
            geometry_policy="Reviewed map scenes; no geometry calibration required for detector training",
            backup_directory=str(Path.home()/"Documents/Project HUM/Training Backups/timePassageOne"))
        write_json(output/"manifest.json",manifest)
        (output/"RECOVERY-START-HERE.md").write_text('''# timePassageOne CPU training recovery

This portable bundle includes frozen images, clipped YOLO labels, all reviewed
source annotations (including excluded event frames), pretrained weights and runner.
Original videos and the repository are not needed. Use the Python version in
environment.json. Install the recorded CPU PyTorch wheel from its CPU index,
then requirements-training.txt; requirements-frozen.txt records all dependencies.

    python runner.py verify
    python runner.py smoke --name smoke-640
    python runner.py train --name pretrained-640
    python runner.py train --name pretrained-640 --resume
    python runner.py evaluate --name pretrained-640

Use a new name for repeats. Resume requires an unfinished last.pt and its matching
runs/<name>/stopping/epoch-NNNNNN.json; copy the entire bundle. Stopping state
ahead of the checkpoint is ignored. Minimum 200 epochs, patience 100, cap 1000;
strict improvement in validation mAP50-95. Ties and zero plateaus do not reset
patience. The 200-epoch boundary does not reset the accumulated plateau.
CPU, six threads, workers 0, batch 16, 640px, AdamW lr=0.001, seed 20260930.
Schedule: cosine decay over the 1000-epoch cap; mosaic closes at epoch 991 if reached.

38 training frames and 9 validation frames share video sources. Event frames
27, 42 and 43 are excluded. Classes: 0 mob, 1 hero. Canonical data stays unchanged.
Evaluation reports mAP, per-class metrics and confidence .25 / matching IoU .50
counts with NMS IoU .70 and full-frame overlays. Inspect all nine comparisons and
write evaluation/<name>/visual-review.json, then run:

    python runner.py archive --name pretrained-640

The manifest records the local backup directory. Archive extraction is verified
by hash. This is a same-volume backup, not off-device protection. No automatic deployment.
''',encoding="utf-8")
        write_json(output/"payload-hashes.json",inventory(output))
        verify(output)
        event(output,"snapshot_created",train=38,validation=9,excluded=3,canonical_sha256=sha(source))
        assert source.read_bytes()==original
        assert (MAP/"dataset/review_settings.json").read_bytes()==settings
        assert (MAP/"dataset/review_order.json").read_bytes()==order
        tracked.mkdir(parents=True)
        for name in ["manifest.json","training-config.json","stopping-policy.json","environment.json",
                     "requirements-training.txt","requirements-frozen.txt","payload-hashes.json",
                     "dataset-report.json","RECOVERY-START-HERE.md"]:
            shutil.copy2(output/name,tracked/name)
        print(json.dumps(dict(snapshot=str(output),exports=exports),indent=2))


if __name__ == "__main__":
    main()
