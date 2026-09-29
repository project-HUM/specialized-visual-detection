"""Portable runner copied into each immutable Forest experiment bundle."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(root):
    manifest = json.loads((root / "payload-hashes.json").read_text())
    for name, entry in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing/unsafe payload path: {name}")
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError(f"Payload changed: {name}")
    print(f"Verified {len(manifest)} payload files", flush=True)


def event(root, kind, **values):
    row = {"utc": datetime.now(timezone.utc).isoformat(), "event": kind, **values}
    with (root / "history.jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps(row) + "\n")
        out.flush()
        os.fsync(out.fileno())


def dataset_yaml(root):
    path = root / "runtime-dataset.yaml"
    path.write_text(f"path: {json.dumps((root/'dataset').as_posix())}\n"
                    "train: images/train\nval: images/validation\n"
                    "names:\n  0: zombie\n  1: hero\n  2: lich\n", encoding="utf-8")
    return path


def metrics(result):
    return {"overall": {k: float(v) for k, v in result.results_dict.items()},
            "per_class": {result.names[int(c)]: {"precision": float(result.box.p[n]),
                "recall": float(result.box.r[n]), "map50": float(result.box.ap50[n]),
                "map50_95": float(result.box.ap[n])}
                for n, c in enumerate(result.box.ap_class_index)}}


def evaluate(root, run_name):
    from ultralytics import YOLO
    data = dataset_yaml(root)
    target = root / "runs" / run_name / "weights/best.pt"
    if not target.is_file():
        raise FileNotFoundError(target)
    report = {"evaluation_set": "16 held-out frames; 12 legacy and 4 new run-isolated frames",
              "caution": "Small selected validation set; not a broad production accuracy estimate.",
              "models": {}}
    for name, weights in [("parent", root / "models/parent-best.pt"), ("candidate", target)]:
        result = YOLO(str(weights)).val(data=str(data), imgsz=640, batch=16, device="cpu",
            workers=0, plots=True, project=str(root / "evaluation"),
            name=f"{run_name}-{name}", exist_ok=False, verbose=False)
        report["models"][name] = {"weights": str(weights.relative_to(root)),
                                 "sha256": sha(weights), **metrics(result)}
    output = root / "evaluation" / f"{run_name}-comparison.json"
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    event(root, "evaluation_completed", report=str(output.relative_to(root)), sha256=sha(output))
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["verify", "train", "evaluate"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--mode", choices=["finetune", "pretrained", "scratch"], default="finetune")
    parser.add_argument("--name", default="finetune-640")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not args.name or Path(args.name).name != args.name or args.name in {".", ".."}:
        raise ValueError("Run name must be a single directory name")
    verify(root)
    if args.action == "verify":
        return
    if args.action == "evaluate":
        evaluate(root, args.name)
        return
    from ultralytics import YOLO
    import ultralytics
    import torch
    torch.set_num_threads(6)
    config = json.loads((root / "training-configs.json").read_text())[args.mode]
    run = root / "runs" / args.name
    if run.exists() and not args.resume:
        raise FileExistsError(f"Use a new run name; history is never overwritten: {run}")
    model_file = root / {"finetune": "models/parent-best.pt", "pretrained": "models/yolo11n.pt",
                         "scratch": "models/yolo11n.yaml"}[args.mode]
    if args.resume:
        if args.mode != "finetune":
            raise ValueError("Specify the original mode configuration before extending resume support")
        model_file = run / "weights/last.pt"
    effective = {**config, "data": str(dataset_yaml(root)), "project": str(root / "runs"),
                 "name": args.name, "exist_ok": False}
    event(root, "training_started", mode=args.mode, run=args.name, resume=args.resume,
          source_weights=str(model_file.relative_to(root)), source_sha256=sha(model_file),
          python=sys.version, torch=torch.__version__, ultralytics=ultralytics.__version__, args=effective)
    try:
        model = YOLO(str(model_file))
        if args.resume:
            model.train(resume=True, device="cpu")
        else:
            model.train(**effective)
        weights = {p.name: sha(p) for p in (run / "weights").glob("*.pt")}
        event(root, "training_completed", mode=args.mode, run=args.name, weights=weights,
              results_sha256=sha(run / "results.csv"))
    except BaseException:
        event(root, "training_failed", run=args.name, traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
