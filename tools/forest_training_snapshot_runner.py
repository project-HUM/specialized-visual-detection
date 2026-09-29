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


def require_device(device):
    """An explicitly requested CUDA device must never silently become CPU."""
    if str(device) != "cpu":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device {device}, but CUDA is unavailable")
        torch.cuda.get_device_properties(int(device))


def evaluate(root, run_name, config):
    from ultralytics import YOLO
    data = dataset_yaml(root)
    target = root / "runs" / run_name / "weights/best.pt"
    if not target.is_file():
        raise FileNotFoundError(target)
    count = len(list((root / "dataset/labels/validation").glob("*.txt")))
    require_device(config["device"])
    report = {"evaluation_set": f"{count} held-out frames from this frozen snapshot",
              "caution": "Small selected validation set; not a broad production accuracy estimate.",
              "models": {}}
    for name, weights in [("parent", root / "models/parent-best.pt"), ("candidate", target)]:
        result = YOLO(str(weights)).val(data=str(data), imgsz=config["imgsz"], batch=config["batch"], device=config["device"],
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
    parser.add_argument("action", choices=["verify", "smoke", "train", "evaluate"])
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
    config = json.loads((root / "training-configs.json").read_text())[args.mode]
    if args.action == "evaluate":
        evaluate(root, args.name, config)
        return
    from ultralytics import YOLO
    import ultralytics
    import torch
    torch.set_num_threads(6)
    require_device(config["device"])
    if args.action == "smoke":
        if args.resume:
            raise ValueError("Smoke checks cannot resume")
        config = {**config, "epochs": 1, "patience": 0, "save_period": -1}
    run = root / "runs" / args.name
    if run.exists() and not args.resume:
        raise FileExistsError(f"Use a new run name; history is never overwritten: {run}")
    model_file = root / {"finetune": "models/parent-best.pt", "pretrained": "models/yolo11n.pt",
                         "scratch": "models/yolo11n.yaml"}[args.mode]
    if args.resume:
        starts = [json.loads(line) for line in (root / "history.jsonl").read_text().splitlines()
                  if line.strip()]
        original = next(e for e in starts if e.get("event") == "training_started" and e.get("run") == args.name)
        if original["mode"] != args.mode or original["args"]["device"] != config["device"]:
            raise ValueError("Resume mode/device must match the original run")
        model_file = run / "weights/last.pt"
    effective = {**config, "data": str(dataset_yaml(root)), "project": str(root / "runs"),
                 "name": args.name, "exist_ok": False}
    event(root, "training_started", mode=args.mode, run=args.name, resume=args.resume,
          source_weights=str(model_file.relative_to(root)), source_sha256=sha(model_file),
          python=sys.version, torch=torch.__version__, ultralytics=ultralytics.__version__, args=effective)
    try:
        model = YOLO(str(model_file))
        if args.resume:
            model.train(resume=True, device=config["device"])
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
