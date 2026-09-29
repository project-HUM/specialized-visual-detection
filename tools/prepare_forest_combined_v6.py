"""Freeze reviewed batches 0-6 and inherited splits for pretrained CUDA training."""
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

import cv2
import torch
import ultralytics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monster_dataset.schema import read_jsonl, write_jsonl
from monster_dataset.annotation_io import export_yolo
from monster_dataset.review_app import ReviewInstanceLock
from perception.core import sha256_file
from tools.prepare_forest_codex_manual_batch_4 import MINIMAP_REFERENCE

MAP = ROOT / "maps/forest-of-dead-trees-2"
OUT = MAP / "dataset/runs/combined-v6-pretrained-gpu-20260929"
BACKUP = Path.home() / "Documents/Project HUM/Training Backups/forest-of-dead-trees-2"


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    assert torch.cuda.is_available(), "Preparation must capture the CUDA training environment"
    canonical = MAP / "dataset/annotations.jsonl"
    with ReviewInstanceLock(canonical):
        before = canonical.read_bytes()
        items = copy.deepcopy(read_jsonl(canonical))
        manifests = {p.name: p.read_bytes() for p in (MAP / "dataset").glob("*manifest.json")}
    assert len(items) == 112
    assert all(f.review_status == "reviewed" and not any(m.review_required for m in f.monsters)
               and not f.validate() for f in items)
    old_path = MAP / "dataset/runs/combined-v4-20260928/annotations.jsonl"
    inherited = {f.frame_id: f.split for f in read_jsonl(old_path)}
    assert len(inherited) == 82
    batch6 = json.loads(manifests["codex_manual_batch_v6_manifest.json"])
    batch5 = json.loads(manifests["codex_manual_batch_v5_manifest.json"])
    run_by_id = {}
    for batch in (batch5, batch6):
        for row in batch["frames"]:
            run_by_id[row["frame_id"]] = row.get("run_id", batch.get("source", {}).get("run_id"))
    selected, rows, excluded, hashes = [], [], [], set()
    groups = defaultdict(set)
    reference = cv2.imread(str(MINIMAP_REFERENCE))
    for item in items:
        path = ROOT / item.image_path
        digest = sha256_file(path)
        assert digest not in hashes, item.frame_id
        hashes.add(digest)
        image = cv2.imread(str(path))
        assert image is not None
        audit = cv2.resize(image, (1920, 1080))
        _, score, _, origin = cv2.minMaxLoc(cv2.matchTemplate(audit[:360, :400], reference, cv2.TM_CCOEFF_NORMED))
        if item.frame_id == "codex-random-0009":
            excluded.append(dict(frame_id=item.frame_id, reason="Known town/dialog image outside Forest map", minimap_correlation=score))
            continue
        assert score >= .9, (item.frame_id, score)
        if item.frame_id in inherited:
            item.split = inherited[item.frame_id]
        else:
            expected = next(f for f in batch6["frames"] if f["frame_id"] == item.frame_id)
            assert item.split == expected["split"]
        run = run_by_id.get(item.frame_id)
        if run:
            groups[run].add(item.split)
        selected.append(item)
        rows.append(dict(frame_id=item.frame_id, split=item.split, source_run=run,
            timestamp=item.timestamp, source_image=str(path), image_sha256=digest,
            image_dimensions_wh=[image.shape[1], image.shape[0]], annotation_dimensions_wh=[1920,1080],
            minimap_correlation=score, minimap_match_origin=list(origin), boxes=len(item.monsters),
            label_origin="latest_human_reviewed_canonical"))
    assert all(len(s) == 1 for s in groups.values())
    assert Counter(f.split for f in selected) == {"train":89,"validation":22}
    OUT.mkdir(parents=True)
    (OUT / "provenance").mkdir()
    (OUT / "models").mkdir()
    (OUT / "provenance/canonical-at-preparation.jsonl").write_bytes(before)
    for name, data in manifests.items():
        (OUT / "provenance" / name).write_bytes(data)
    shutil.copy2(old_path, OUT / "provenance/previous-split-annotations.jsonl")
    shutil.copy2(MINIMAP_REFERENCE, OUT / "provenance/minimap-reference.png")
    shutil.copy2(MAP / "dataset/review_settings.json", OUT / "provenance/review_settings.json")
    exports = {s: export_yolo(selected, OUT / "dataset", split=s, image_root=ROOT,
                             preset_classes={"1":"zombie","2":"hero","3":"lich"})
               for s in ("train","validation")}
    for item,row in zip(selected,rows):
        item.image_path = f"dataset/images/{item.split}/{item.frame_id}{Path(item.image_path).suffix.lower()}"
        row["image_path"] = item.image_path
        row["label_path"] = f"dataset/labels/{item.split}/{item.frame_id}.txt"
        row["label_sha256"] = sha256_file(OUT / row["label_path"])
        assert sha256_file(OUT / item.image_path) == row["image_sha256"]
    write_jsonl(selected, OUT / "annotations.jsonl")
    (OUT / "dataset/dataset.yaml").write_text("path: .\ntrain: images/train\nval: images/validation\nnames:\n  0: zombie\n  1: hero\n  2: lich\n")
    shutil.copy2(ROOT / "yolo11n.pt", OUT / "models/yolo11n.pt")
    shutil.copy2(MAP / "dataset/runs/combined-v3/training-640/weights/best.pt", OUT / "models/parent-best.pt")
    for source,destination in [(ROOT / "tools/forest_training_snapshot_runner.py", OUT / "run.py"),
                               (Path(__file__),OUT / "provenance/prepare.py"),
                               (ROOT / "tools/evaluate_forest_lich_snapshot.py", OUT / "evaluate_lich.py")]:
        shutil.copy2(source,destination)
    save(OUT / "training-configs.json", {"pretrained":dict(imgsz=640, batch=16, device=0, workers=0,
        seed=20260929, deterministic=True, optimizer="AdamW", lr0=.001, lrf=.01, cos_lr=True,
        epochs=200, patience=50, weight_decay=.0005, warmup_epochs=3., close_mosaic=10,
        amp=True, cache=False, save=True, save_period=10, plots=True, resume=False)})
    (OUT / "requirements-frozen.txt").write_text(subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True))
    save(OUT / "environment.json",dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
        torch=torch.__version__, cuda=torch.version.cuda, ultralytics=ultralytics.__version__,
        gpu=torch.cuda.get_device_name(0), gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
        nvidia_smi=subprocess.check_output(["nvidia-smi"],text=True)))
    for args,name in [(["diff","--binary"],"working-tree.patch"),(["status","--short"],"git-status.txt")]:
        (OUT / "provenance" / name).write_bytes(subprocess.check_output(["git",*args],cwd=ROOT))
    save(OUT / "manifest.json",dict(schema="forest.recoverable-training-snapshot.v1",experiment=OUT.name,
        created_utc=datetime.now(timezone.utc).isoformat(),frames=rows,exports=exports,excluded=excluded,
        canonical_annotations_sha256=sha256_file(OUT / "provenance/canonical-at-preparation.jsonl"),
        split_policy="Inherit 66/16 previous experiment IDs using latest labels; add batch6 source-run split 23/6; legacy videos retain temporal-group split.",
        incident_run_cross_split_overlap=[],validation_batch6_frames=[r["frame_id"] for r in rows if r["split"]=="validation" and r["frame_id"].startswith("codex-human-batch6-")],
        canonical_files_modified=False,initialization="original pretrained yolo11n.pt; no Forest weights loaded for training"))
    (OUT / "RECOVERY.md").write_text('''# Forest batches 0-6 CUDA experiment

Create a Python 3.12 environment, install requirements-frozen.txt with the extra
index https://download.pytorch.org/whl/cu128. An NVIDIA CUDA GPU is required.
This self-contained snapshot includes exact images, latest reviewed labels,
89/22 splits, original pretrained weights and the comparison model.

python run.py verify
python run.py smoke --mode pretrained --name smoke-640
python run.py train --mode pretrained --name pretrained-gpu-640
python run.py train --mode pretrained --name pretrained-gpu-640 --resume
python run.py evaluate --mode pretrained --name pretrained-gpu-640
python evaluate_lich.py --name pretrained-gpu-640

Use fresh run names; resume only an interrupted run. No automatic CPU fallback
or deployment. Payload hashes protect inputs; logs/checkpoints record progress.
The archive is a local same-volume backup, not off-device disaster recovery.
''')
    save(OUT / "payload-hashes.json",{p.relative_to(OUT).as_posix():dict(bytes=p.stat().st_size,sha256=sha256_file(p))
        for p in sorted(OUT.rglob("*")) if p.is_file()})
    BACKUP.mkdir(parents=True,exist_ok=True)
    archive=BACKUP / f"{OUT.name}-inputs.zip"
    with zipfile.ZipFile(archive,"x",zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in OUT.rglob("*"):
            if p.is_file():z.write(p,p.relative_to(OUT).as_posix())
    archive.with_suffix(".zip.sha256").write_text(sha256_file(archive)+"  "+archive.name+"\n")
    assert canonical.read_bytes()==before
    print(json.dumps(dict(snapshot=str(OUT),exports=exports,backup=str(archive)),indent=2))


if __name__ == "__main__":
    main()
