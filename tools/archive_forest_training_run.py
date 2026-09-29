"""Seal a finished Forest training run and verify an independent recovery copy."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.forest_training_snapshot_runner import sha, verify, event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "maps/forest-of-dead-trees-2/dataset/runs/combined-v4-20260928")
    parser.add_argument("--name", default="finetune-640")
    parser.add_argument("--conclusion", default="Comparison recorded; no automatic deployment.")
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    verify(snapshot)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    comparison = json.loads((snapshot / "evaluation" / f"{args.name}-comparison.json").read_text())
    history = [json.loads(l) for l in (snapshot / "history.jsonl").read_text().splitlines()]
    assert any(e["event"] == "training_completed" and e["run"] == args.name for e in history)
    run = snapshot / "runs" / args.name
    rows = list(csv.DictReader((run / "results.csv").open()))
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    report = {"schema": "forest.training-result.v1", "experiment": manifest["experiment"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "parent_experiment": manifest.get("parent_experiment", "combined-v3"),
        "parent_weights_sha256": manifest.get("parent_weights_sha256", sha(snapshot / "models/parent-best.pt")),
        "best_weights": f"runs/{args.name}/weights/best.pt", "best_weights_sha256": sha(run / "weights/best.pt"),
        "epochs_completed": len(rows), "best_epoch_by_validation_map50_95": int(best["epoch"]),
        "best_epoch_metrics": best, "dataset": manifest["exports"],
        "comparison": comparison,
        "lich_fixed_thresholds": json.loads((snapshot / (
            "evaluation/lich-fixed-thresholds.json" if (snapshot / "evaluation/lich-fixed-thresholds.json").exists()
            else f"evaluation/{args.name}-lich-conf025/comparison.json")).read_text()),
        "conclusion": args.conclusion,
        "initialization": manifest.get("initialization", "combined-v3 fine-tune"),
        "deployed_to_human": False,
        "recovery_verification": json.loads((snapshot / (
            "scratch-recovery-verification.json" if (snapshot / "scratch-recovery-verification.json").exists()
            else "recovery-verification.json")).read_text()) if (snapshot / "recovery-verification.json").exists()
            or (snapshot / "scratch-recovery-verification.json").exists() else
            {"method": "Complete archive extracted and every artifact hash checked; see archive-receipt.json"}}
    (snapshot / "experiment-report.json").write_text(json.dumps(report, indent=2)+"\n")
    for name in ["archive_forest_training_run.py", "evaluate_forest_lich_thresholds.py"]:
        shutil.copy2(ROOT / "tools" / name, snapshot / "provenance" / name)
    if "finetuned" in comparison["models"]:
        shutil.copy2(ROOT / "tools/compare_forest_training_models.py", snapshot / "provenance/compare_forest_training_models.py")
    event(snapshot, "archive_prepared", report_sha256=sha(snapshot / "experiment-report.json"))
    artifacts = {p.relative_to(snapshot).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)}
                 for p in sorted(snapshot.rglob("*"))
                 if p.is_file() and p.name != "artifact-hashes.json" and "__pycache__" not in p.parts}
    (snapshot / "artifact-hashes.json").write_text(json.dumps(artifacts, indent=2)+"\n")
    backup = Path(manifest.get("backup_directory", Path.home() / "Documents/Project HUM/Training Backups/forest-of-dead-trees-2"))
    archive = backup / f"{snapshot.name}-{args.name}-complete.zip"
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for name in [*artifacts, "artifact-hashes.json"]:
            z.write(snapshot / name, name)
    archive.with_suffix(".zip.sha256").write_text(sha(archive)+"  "+archive.name+"\n")
    recovery = backup / f"{snapshot.name}-{args.name}-verified-restore"
    assert not recovery.exists()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for member in z.infolist():
            assert (recovery / member.filename).resolve().is_relative_to(recovery.resolve())
        z.extractall(recovery)
    for name, entry in artifacts.items():
        restored = recovery / name
        assert restored.stat().st_size == entry["bytes"] and sha(restored) == entry["sha256"], name
    verify(recovery)
    receipt = {"archive": str(archive), "sha256": sha(archive), "bytes": archive.stat().st_size,
        "verified_restoration": str(recovery), "verified_files": len(artifacts),
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Independent local copy on the same volume; not off-device disaster recovery."}
    # Receipt is external to the sealed archive so it can identify its final hash.
    (snapshot / "archive-receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    archive.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    tracked = ROOT / "maps/forest-of-dead-trees-2/experiments" / snapshot.name
    tracked.mkdir(parents=True, exist_ok=True)
    for name in ["experiment-report.json", "history.jsonl", "archive-receipt.json", "artifact-hashes.json",
                 "recovery-verification.json"]:
        if (snapshot / name).exists():
            shutil.copy2(snapshot / name, tracked / name)
    shutil.copy2(run / "results.csv", tracked / "training-results.csv")
    shutil.copy2(run / "args.yaml", tracked / "training-args.yaml")
    lich_report = snapshot / "evaluation/lich-fixed-thresholds.json"
    if not lich_report.exists():
        lich_report = snapshot / f"evaluation/{args.name}-lich-conf025/comparison.json"
    shutil.copy2(lich_report, tracked / "lich-fixed-thresholds.json")
    if (snapshot / "scratch-recovery-verification.json").exists():
        shutil.copy2(snapshot / "scratch-recovery-verification.json", tracked / "scratch-recovery-verification.json")
    print(json.dumps({"report": str(snapshot / "experiment-report.json"), **receipt}, indent=2))


if __name__ == "__main__":
    main()
