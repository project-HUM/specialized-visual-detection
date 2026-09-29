"""Clone verified v4 inputs for a separate, recoverable random-init experiment."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.forest_training_snapshot_runner import sha, verify


def main():
    map_dir = ROOT / "maps/forest-of-dead-trees-2"
    previous = map_dir / "dataset/runs/combined-v4-20260928"
    output = map_dir / "dataset/runs/combined-v5-scratch-20260928"
    verify(previous)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    payload = json.loads((previous / "payload-hashes.json").read_text())
    for name in payload:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(previous / name, target)
    for name in ["requirements-training.txt", "RECOVERY-START-HERE.md", "recovery-verification.json"]:
        shutil.copy2(previous / name, output / name)
    for name in ["experiment-report.json", "history.jsonl", "artifact-hashes.json", "archive-receipt.json"]:
        shutil.copy2(previous / name, output / "provenance" / f"v4-{name}")
    shutil.copy2(previous / "runs/finetune-640/weights/best.pt", output / "models/finetuned-best.pt")
    shutil.copy2(previous / "manifest.json", output / "provenance/v4-manifest.json")
    shutil.copy2(Path(__file__), output / "provenance/prepare_forest_scratch_comparison.py")
    manifest = json.loads((output / "manifest.json").read_text())
    manifest.update(experiment=output.name, created_utc=datetime.now(timezone.utc).isoformat(),
                    parent_experiment="combined-v4-20260928", initialization="random_weights_from_yolo11n_yaml",
                    initialization_file="models/yolo11n.yaml", initialization_sha256=sha(output / "models/yolo11n.yaml"),
                    comparison_models={"original": "models/parent-best.pt", "finetuned": "models/finetuned-best.pt"},
                    dataset_parent_manifest_sha256=sha(previous / "manifest.json"))
    for entry in manifest["exports"].values():
        entry["output"] = str(output / "dataset")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    # Preserve all images, labels, annotation metadata and split assignments exactly.
    assert (output / "annotations.jsonl").read_bytes() == (previous / "annotations.jsonl").read_bytes()
    for row in manifest["frames"]:
        assert sha(output / row["image_path"]) == row["image_sha256"]
        assert sha(output / row["label_path"]) == row["label_sha256"]
    guide = """# Recover the random-initialization comparison

All 82 images and labels are included with the same 66 train / 16 validation
split as combined-v4. Original and fine-tuned comparison checkpoints are also
bundled. Training scratch mode loads the YAML architecture and pretrained=False.
Neither comparison checkpoint nor the original pretrained weights initialize it.

Use the recorded Python version in a fresh environment, then from this folder:

```powershell
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-training.txt
python run.py verify
python run.py train --mode scratch --name scratch-640
python run.py evaluate --name scratch-640
```

Use a new run name to repeat. training-configs.json records the frozen settings:
YOLO11n, 640px, CPU, batch 16, AdamW lr=0.001, seed 20260928, at most 300 epochs,
patience 75. The runner records full effective options and versions in history.jsonl.
Full per-epoch history, plots and checkpoints live under runs/scratch-640.
The runner resolves relocated dataset paths without needing original videos/logs.
Requirements-frozen.txt is the full environment audit; install only the training
requirements above. Wheels are not bundled, so package index access is needed.

Provenance includes the previous experiment and original dataset source records.
Validation remains the same small selected set; it is not a new independent test.
No model is installed into HUMAN automatically.
"""
    (output / "RECOVERY-START-HERE.md").write_text(guide)
    (output / "RECOVERY.md").write_text(guide)
    hashes = {p.relative_to(output).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)}
              for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "payload-hashes.json").write_text(json.dumps(hashes, indent=2)+"\n")
    verify(output)
    backup = Path(manifest["backup_directory"])
    archive = backup / f"{output.name}-inputs.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for name in [*hashes, "payload-hashes.json"]:
            z.write(output / name, name)
    archive.with_suffix(".zip.sha256").write_text(sha(archive)+"  "+archive.name+"\n")
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        import hashlib
        for name, expected in hashes.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected["sha256"]
    tracked = map_dir / "experiments" / output.name
    tracked.mkdir()
    for name in ["manifest.json", "training-configs.json", "environment.json", "payload-hashes.json",
                 "requirements-training.txt", "RECOVERY-START-HERE.md"]:
        shutil.copy2(output / name, tracked / name)
    print(json.dumps({"snapshot": str(output), "input_archive": str(archive), "dataset_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
