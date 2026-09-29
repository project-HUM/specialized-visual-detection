"""Freeze combined-v3 plus reviewed batch 5 into a self-contained recovery bundle."""
from __future__ import annotations
from collections import Counter
import copy
from datetime import datetime, timezone
import json
import importlib.metadata
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

import cv2
import ultralytics
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from monster_dataset.schema import read_jsonl, write_jsonl
from monster_dataset.annotation_io import export_yolo
from monster_dataset.review_app import ReviewInstanceLock
from perception.core import sha256_file
from tools.prepare_forest_codex_manual_batch_4 import _minimap_correlation, MINIMAP_REFERENCE

MAP = ROOT / "maps/forest-of-dead-trees-2"
PARENT = MAP / "dataset/runs/combined-v3"
OUT = MAP / "dataset/runs/combined-v4-20260928"
BACKUP = Path.home() / "Documents/Project HUM/Training Backups/forest-of-dead-trees-2"
NEW_VALIDATION = {f"codex-human-batch5-evidence-{n:04d}" for n in (4, 33, 44, 49)}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


def training_requirements():
    """Installed training dependency closure, without unrelated editable projects."""
    pending, versions = ["ultralytics"], {}
    while pending:
        name = pending.pop().lower().replace("_", "-")
        if name in versions:
            continue
        distribution = importlib.metadata.distribution(name)
        versions[name] = distribution.version
        for spec in distribution.requires or []:
            requirement = Requirement(spec)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    return "".join(f"{name}=={version}\n" for name, version in sorted(versions.items()))


def main():
    if OUT.exists():
        raise FileExistsError(f"Never replace an experiment: {OUT}")
    canonical_path = MAP / "dataset/annotations.jsonl"
    with ReviewInstanceLock(canonical_path):
        canonical_bytes = canonical_path.read_bytes()
        canonical = {i.frame_id: i for i in read_jsonl(canonical_path)}
        batch_bytes = (MAP / "dataset/codex_manual_batch_v5_manifest.json").read_bytes()
    batch = json.loads(batch_bytes)
    batch_frames = {f["frame_id"]: f for f in batch["frames"]}
    additions = [copy.deepcopy(canonical[k]) for k in batch_frames]
    assert len(additions) == 22 and NEW_VALIDATION <= set(batch_frames)
    assert all(i.review_status == "reviewed" and not any(m.review_required for m in i.monsters)
               and not i.validate() for i in additions), "Batch 5 must be confirmed before training"
    old = [copy.deepcopy(i) for i in read_jsonl(PARENT / "training_annotations.jsonl")
           if i.split in {"train", "validation"}]
    assert Counter(i.split for i in old) == {"train": 48, "validation": 12}
    assert not ({i.frame_id for i in old} & set(batch_frames))
    reference = cv2.imread(str(MINIMAP_REFERENCE))
    assert reference is not None
    validation_runs = {batch_frames[k].get("run_id", batch["source"]["run_id"]) for k in NEW_VALIDATION}
    for item in additions:
        run = batch_frames[item.frame_id].get("run_id", batch["source"]["run_id"])
        item.split = "validation" if run in validation_runs else "train"
    assert {i.frame_id for i in additions if i.split == "validation"} == NEW_VALIDATION
    OUT.mkdir(parents=True)
    for name in ("provenance", "models"):
        (OUT / name).mkdir()
    (OUT / "provenance/canonical-at-preparation.jsonl").write_bytes(canonical_bytes)
    (OUT / "provenance/batch5-manifest.json").write_bytes(batch_bytes)
    shutil.copy2(MINIMAP_REFERENCE, OUT / "provenance/minimap-reference.png")
    for name in ("manifest.json", "training_annotations.jsonl", "experiment-report.json"):
        shutil.copy2(PARENT / name, OUT / "provenance" / f"parent-{name}")
    for name in ("args.yaml", "results.csv"):
        shutil.copy2(PARENT / "training-640" / name, OUT / "provenance" / f"parent-{name}")
    for name in ("review_settings.json", "codex_manual_batch_v1_manifest.json",
                 "codex_manual_batch_v2_manifest.json", "codex_manual_batch_v3_manifest.json",
                 "codex_manual_batch_v4_manifest.json", "pilot_manifest.json"):
        shutil.copy2(MAP / "dataset" / name, OUT / "provenance" / name)
    rows, seen_hashes = [], set()
    all_items = old + additions
    for item in all_items:
        is_new = item.frame_id in batch_frames
        if is_new:
            source = ROOT / item.image_path
        else:
            candidates = list((PARENT / "yolo_dataset/images" / item.split).glob(item.frame_id + ".*"))
            assert len(candidates) == 1
            source = candidates[0]
        image_hash = sha256_file(source)
        assert image_hash not in seen_hashes, f"Duplicate image {item.frame_id}"
        seen_hashes.add(image_hash)
        image = cv2.imread(str(source))
        assert image is not None and image.shape[:2] in {(1080, 1920), (540, 960)}
        # Legacy exports include half-resolution JPEGs with normalized YOLO labels.
        # Preserve those exact training bytes; audit minimap at annotation coordinates.
        audit_image = cv2.resize(image, (1920, 1080)) if image.shape[:2] != (1080, 1920) else image
        score = _minimap_correlation(audit_image, reference)
        assert score >= .90, (item.frame_id, score)
        item.image_path = str(source)
        rows.append({"frame_id": item.frame_id, "split": item.split, "image_sha256": image_hash,
            "source_image": str(source), "minimap_correlation": score,
            "image_dimensions_wh": [image.shape[1], image.shape[0]], "annotation_dimensions_wh": [1920, 1080],
            "label_origin": "batch5_human_confirmed" if is_new else "frozen_combined_v3_snapshot",
            "source_run": batch_frames[item.frame_id].get("run_id", batch["source"]["run_id"]) if is_new else None,
            "timestamp": item.timestamp, "boxes": len(item.monsters)})
    classes = {"1": "zombie", "2": "hero", "3": "lich"}
    exports = {s: export_yolo(all_items, OUT / "dataset", split=s, image_root=ROOT,
                              preset_classes=classes) for s in ("train", "validation")}
    for item, row in zip(all_items, rows):
        extension = Path(item.image_path).suffix.lower()
        item.image_path = f"dataset/images/{item.split}/{item.frame_id}{extension}"
        row["image_path"] = item.image_path
        row["label_path"] = f"dataset/labels/{item.split}/{item.frame_id}.txt"
        row["label_sha256"] = sha256_file(OUT / row["label_path"])
        if row["label_origin"] == "frozen_combined_v3_snapshot":
            assert (OUT / row["label_path"]).read_bytes() == (
                PARENT / "yolo_dataset/labels" / item.split / f"{item.frame_id}.txt").read_bytes()
    write_jsonl(all_items, OUT / "annotations.jsonl")
    # Kept relative in the payload; the runner makes a relocated runtime YAML.
    (OUT / "dataset/dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/validation\nnames:\n  0: zombie\n  1: hero\n  2: lich\n")
    shutil.copy2(PARENT / "training-640/weights/best.pt", OUT / "models/parent-best.pt")
    shutil.copy2(ROOT / "yolo11n.pt", OUT / "models/yolo11n.pt")
    architecture = Path(ultralytics.__file__).parent / "cfg/models/11/yolo11.yaml"
    shutil.copy2(architecture, OUT / "models/yolo11n.yaml")
    shutil.copy2(ROOT / "tools/forest_training_snapshot_runner.py", OUT / "run.py")
    shutil.copy2(Path(__file__), OUT / "provenance/prepare_forest_combined_v4.py")
    common = dict(imgsz=640, batch=16, device="cpu", workers=0, seed=20260928,
                  deterministic=True, optimizer="AdamW", weight_decay=.0005, warmup_epochs=3.,
                  close_mosaic=10, save=True, save_period=10, plots=True, amp=False,
                  cache=False, cos_lr=True, lrf=.01, resume=False)
    configs = {"finetune": {**common, "epochs": 80, "patience": 20, "lr0": .0001},
               "pretrained": {**common, "epochs": 200, "patience": 50, "lr0": .001},
               "scratch": {**common, "epochs": 300, "patience": 75, "lr0": .001, "pretrained": False}}
    save_json(OUT / "training-configs.json", configs)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (OUT / "requirements-frozen.txt").write_text(freeze, encoding="utf-8")
    (OUT / "requirements-training.txt").write_text(training_requirements(), encoding="utf-8")
    save_json(OUT / "environment.json", {"python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "ultralytics": ultralytics.__version__,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    for args, name in [(["diff", "--binary"], "working-tree.patch"),
                       (["diff", "--cached", "--binary"], "staged.patch"),
                       (["status", "--short"], "git-status.txt")]:
        (OUT / "provenance" / name).write_bytes(subprocess.check_output(["git", *args], cwd=ROOT))
    manifest = {"schema": "forest.recoverable-training-snapshot.v1", "experiment": OUT.name,
        "created_utc": datetime.now(timezone.utc).isoformat(), "parent_experiment": "combined-v3",
        "parent_weights_sha256": sha256_file(OUT / "models/parent-best.pt"),
        "canonical_annotations_sha256": sha256_file(OUT / "provenance/canonical-at-preparation.jsonl"),
        "classes": classes, "exports": exports, "frames": rows,
        "split_policy": "Preserve all 48/12 legacy assignments; four whole new runs held out, 18 new train frames.",
        "new_validation_frame_ids": sorted(NEW_VALIDATION),
        "legacy_label_status": "Reuse the previously authorized combined-v3 training snapshot; no new canonical promotion.",
        "canonical_files_modified": False,
        "backup_directory": str(BACKUP), "backup_scope": "Independent local copy on the same physical volume; not an off-device backup."}
    save_json(OUT / "manifest.json", manifest)
    (OUT / "RECOVERY.md").write_text('''# Forest combined-v4 recovery

This folder is self-contained. The original videos, HUMAN logs and repository
are not needed to train. `manifest.json` lists each image, label and split.
`annotations.jsonl` retains full boxes before YOLO edge clipping. The `dataset`
directory contains all 66 training and 16 validation images with YOLO labels.
`provenance` preserves original labels, review state, parent metrics and code state.

Use the recorded Python version (see environment.json), preferably in a fresh
virtual environment. Install `requirements-training.txt` with pip; the full
`requirements-frozen.txt` is an audit record including unrelated local packages. The environment
was CPU PyTorch; if pip cannot find its +cpu wheel, use the PyTorch CPU index:
`python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-training.txt`
Package downloads still require availability of that index; wheels are not bundled.

From this recovered folder:

```powershell
python run.py verify
python run.py train --mode finetune --name repeat-finetune
python run.py train --mode pretrained --name fresh-pretrained
python run.py train --mode scratch --name random-initialization
python run.py evaluate --name repeat-finetune
```

`pretrained` starts a new training run from bundled original YOLO11n weights.
`scratch` builds the bundled architecture with random weights. It is a separate
experiment and may need more data/tuning. No old fine-tuned weights are used.
Never reuse a run name. An interrupted fine-tune can use
`python run.py train --mode finetune --name finetune-640 --resume`.
The runner verifies payload hashes before every action and resolves data paths
relative to this folder, so recovery at another directory works.

`history.jsonl`, per-run `args.yaml`, `results.csv`, plots, periodic checkpoints,
`last.pt`, and `best.pt` record training history. `evaluation` compares parent and
candidate on identical validation images. Fixed seeds/settings improve repeatability;
bitwise-identical training across different hardware/library versions is not promised.
Validation is small and selected, and is not an independent production test.
No model is automatically installed into HUMAN.
''', encoding="utf-8")
    payload = {p.relative_to(OUT).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
               for p in sorted(OUT.rglob("*")) if p.is_file()}
    save_json(OUT / "payload-hashes.json", payload)
    BACKUP.mkdir(parents=True, exist_ok=True)
    archive = BACKUP / f"{OUT.name}-retraining-inputs.zip"
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for p in OUT.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(OUT).as_posix())
    archive.with_suffix(".zip.sha256").write_text(sha256_file(archive)+"  "+archive.name+"\n")
    print(json.dumps({"snapshot": str(OUT), "backup": str(archive), "exports": exports}, indent=2))


if __name__ == "__main__":
    main()
