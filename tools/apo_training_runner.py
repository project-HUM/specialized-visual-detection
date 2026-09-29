"""Portable APO training, validation, and recovery runner (copied into snapshots)."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import zipfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def inventory(root):
    return {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


def verify(root, filename="payload-hashes.json"):
    hashes = json.loads((root / filename).read_text())
    for name, entry in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing/unsafe payload path: {name}")
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError(f"Payload changed: {name}")
    print(f"Verified {len(hashes)} payload files", flush=True)
    return len(hashes)


def event(root, kind, **values):
    row = {"utc": datetime.now(timezone.utc).isoformat(), "event": kind, **values}
    with (root / "history.jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps(row) + "\n")
        out.flush()
        os.fsync(out.fileno())


def dataset_yaml(root):
    names = json.loads((root / "manifest.json").read_text())["class_names"]
    path = root / "runtime-dataset.yaml"
    path.write_text(f"path: {json.dumps((root/'dataset').as_posix())}\n"
                    "train: images/train\nval: images/validation\nnames:\n" +
                    "".join(f"  {i}: {json.dumps(n)}\n" for i, n in enumerate(names)), encoding="utf-8")
    return path


def iou(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0.


def match_counts(truth, predictions, threshold=.5):
    matched = set()
    fp = 0
    for confidence, box in sorted(predictions, key=lambda row: row[0], reverse=True):
        candidates = [(iou(box, gt), i) for i, gt in enumerate(truth) if i not in matched]
        score, index = max(candidates, default=(0., -1))
        if score >= threshold:
            matched.add(index)
        else:
            fp += 1
    return {"tp": len(matched), "fp": fp, "fn": len(truth)-len(matched)}


def configure_cpu():
    import torch
    import ultralytics.utils
    import ultralytics.utils.torch_utils
    # select_device() resets torch's pool from its imported NUM_THREADS constant.
    ultralytics.utils.NUM_THREADS = 6
    ultralytics.utils.torch_utils.NUM_THREADS = 6
    torch.set_num_threads(6)


def train(root, name, resume):
    from ultralytics import YOLO
    import ultralytics
    import torch
    configure_cpu()
    config = json.loads((root / "training-config.json").read_text())
    run = root / "runs" / name
    if run.exists() and not resume:
        raise FileExistsError(f"Use a new run name; existing history is preserved: {run}")
    model_file = run / "weights/last.pt" if resume else root / "models/yolo11n.pt"
    if not model_file.is_file():
        raise FileNotFoundError(model_file)
    effective = {**config, "data": str(dataset_yaml(root)), "project": str(root / "runs"),
                 "name": name, "exist_ok": False}
    event(root, "training_started", run=name, resume=resume, source_sha256=sha(model_file),
          python=sys.version, torch=torch.__version__, ultralytics=ultralytics.__version__, args=effective)
    try:
        model = YOLO(str(model_file))
        def check_threads(trainer):
            actual = torch.get_num_threads()
            if actual != 6:
                raise RuntimeError(f"Expected six CPU threads, got {actual}")
            event(root, "runtime_threads_verified", run=name, threads=actual)
        model.add_callback("on_train_start", check_threads)
        if resume:
            model.train(resume=True, data=effective["data"], device="cpu", workers=0)
        else:
            model.train(**effective)
        event(root, "training_completed", run=name,
              weights={p.name: sha(p) for p in (run / "weights").glob("*.pt")},
              results_sha256=sha(run / "results.csv"))
    except BaseException:
        event(root, "training_failed", run=name, traceback=traceback.format_exc())
        raise


def evaluate(root, name):
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    configure_cpu()
    manifest = json.loads((root / "manifest.json").read_text())
    names = manifest["class_names"]
    weights = root / "runs" / name / "weights/best.pt"
    output = root / "evaluation" / name
    if output.exists():
        raise FileExistsError(f"Evaluation already exists: {output}")
    model = YOLO(str(weights))
    result = model.val(data=str(dataset_yaml(root)), imgsz=640, batch=16, device="cpu",
                       workers=0, plots=True, project=str(root / "evaluation"), name=name,
                       exist_ok=False, verbose=False)
    report = {"weights": str(weights.relative_to(root)), "weights_sha256": sha(weights),
              "evaluation_set": "All 15 sample1 frames; video source held out from training",
              "caution": "Small development validation set; not production accuracy or HUMAN runtime behavior.",
              "overall": {k: float(v) for k, v in result.results_dict.items()},
              "per_class": {result.names[int(c)]: {"precision": float(result.box.p[n]),
                  "recall": float(result.box.r[n]), "map50": float(result.box.ap50[n]),
                  "map50_95": float(result.box.ap[n])}
                  for n, c in enumerate(result.box.ap_class_index)}}
    frames = [f for f in manifest["frames"] if f["split"] == "validation"]
    predictions = model.predict(source=[str(root / f["image_path"]) for f in frames],
                                imgsz=640, conf=.25, iou=.7, device="cpu", batch=16, verbose=False)
    fixed = {"confidence": .25, "matching_iou": .5, "nms_iou": .7,
             "method": "Class-aware greedy confidence-order matching against clipped YOLO labels",
             "per_class": {n: {"tp": 0, "fp": 0, "fn": 0} for n in names}, "frames": []}
    overlays = output / "overlays"
    overlays.mkdir()
    panels = []
    colors = [(50,210,255), (255,170,70), (255,70,210)]
    for frame, pred in zip(frames, predictions):
        truth = defaultdict(list)
        for line in (root / frame["label_path"]).read_text().splitlines():
            c, x, y, w, h = map(float, line.split())
            truth[int(c)].append([x-w/2,y-h/2,x+w/2,y+h/2])
        proposed = defaultdict(list)
        for c, conf, box in zip(pred.boxes.cls, pred.boxes.conf, pred.boxes.xyxyn):
            proposed[int(c)].append((float(conf),box.tolist()))
        counts = {n: match_counts(truth[c], proposed[c]) for c, n in enumerate(names)}
        for n in names:
            for key, value in counts[n].items():
                fixed["per_class"][n][key] += value
        fixed["frames"].append({"frame_id": frame["frame_id"], "counts": counts,
                                 "predictions": {names[c]: rows for c, rows in proposed.items()}})
        actual = cv2.imread(str(root / frame["image_path"]))
        height, width = actual.shape[:2]
        for c, boxes in truth.items():
            for box in boxes:
                x1,y1,x2,y2 = [round(v*scale) for v,scale in zip(box,[width,height,width,height])]
                cv2.rectangle(actual,(x1,y1),(x2,y2),colors[c],2)
                cv2.putText(actual,names[c],(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,.5,colors[c],1)
        predicted = pred.plot()
        assert cv2.imwrite(str(overlays / f'{frame["frame_id"]}-prediction.jpg'), predicted)
        # Full-frame GT/prediction pairs also expose UI and upper-platform false positives.
        pair = cv2.hconcat([cv2.resize(actual,(960,540)), cv2.resize(predicted,(960,540))])
        title = np.zeros((35,1920,3),np.uint8)
        cv2.putText(title,frame["frame_id"]+" | reviewed truth (left) / prediction conf=0.25 (right)",
                    (10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
        assert cv2.imwrite(str(overlays / f'{frame["frame_id"]}-comparison.jpg'),cv2.vconcat([title,pair]))
        crop = cv2.resize(predicted[430:790],(1536,288))
        caption = np.zeros((30,1536,3),np.uint8)
        cv2.putText(caption,frame["frame_id"]+" "+str(counts),(8,20),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
        panels.append(cv2.vconcat([caption,crop]))
    for start in range(0,len(panels),3):
        assert cv2.imwrite(str(output/f"inspection-{start//3+1:02d}.jpg"),cv2.vconcat(panels[start:start+3]))
    report["fixed_threshold"] = fixed
    write_json(output / "metrics.json", report)
    event(root,"evaluation_completed",run=name,report=str((output/"metrics.json").relative_to(root)),
          sha256=sha(output/"metrics.json"))
    print(json.dumps({k:v for k,v in report.items() if k!='fixed_threshold'},indent=2),flush=True)


def archive(root, name):
    manifest = json.loads((root/"manifest.json").read_text())
    evaluation = json.loads((root/"evaluation"/name/"metrics.json").read_text())
    visual = json.loads((root/"evaluation"/name/"visual-review.json").read_text())
    history = [json.loads(line) for line in (root/"history.jsonl").read_text().splitlines()]
    if not any(e["event"]=="training_completed" and e["run"]==name for e in history):
        raise ValueError("Cannot archive an incomplete training run")
    rows = list(csv.DictReader((root/"runs"/name/"results.csv").open()))
    best = max(rows,key=lambda row:float(row["metrics/mAP50-95(B)"]))
    report = dict(schema="apo.training-result.v1",experiment=manifest["experiment"],
        initialization="generic pretrained YOLO11n",epochs_completed=len(rows),
        best_epoch_by_validation_map50_95=int(best["epoch"]),best_epoch_metrics=best,
        exports=manifest["exports"],evaluation=evaluation,visual_review=visual,deployed_to_human=False)
    backup = Path(manifest["backup_directory"])
    backup.mkdir(parents=True,exist_ok=True)
    destination = backup/f"{root.name}-{name}-complete.zip"
    restored = backup/f"{root.name}-{name}-verified-restore"
    if destination.exists() or restored.exists() or (root/"artifact-hashes.json").exists():
        raise FileExistsError("Archive already exists; refusing overwrite")
    write_json(root/"experiment-report.json",report)
    event(root,"archive_prepared",run=name,report_sha256=sha(root/"experiment-report.json"))
    artifacts=inventory(root)
    write_json(root/"artifact-hashes.json",artifacts)
    with zipfile.ZipFile(destination,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for item in [*artifacts,"artifact-hashes.json"]:
            z.write(root/item,item)
    with zipfile.ZipFile(destination) as z:
        if z.testzip() is not None:
            raise ValueError("Archive integrity failure")
        for member in z.infolist():
            if not (restored/member.filename).resolve().is_relative_to(restored.resolve()):
                raise ValueError("Unsafe archive member")
        z.extractall(restored)
    verify(restored,"artifact-hashes.json")
    verify(restored)
    receipt=dict(archive=str(destination),sha256=sha(destination),bytes=destination.stat().st_size,
                 verified_restoration=str(restored),verified_files=len(artifacts),
                 verified_utc=datetime.now(timezone.utc).isoformat(),
                 note="Independent local copy on the same volume, not off-device backup")
    write_json(root/"archive-receipt.json",receipt)
    write_json(destination.with_suffix(".receipt.json"),receipt)
    destination.with_suffix(".zip.sha256").write_text(receipt["sha256"]+"  "+destination.name+"\n")
    print(json.dumps(receipt,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["verify","train","evaluate","archive"])
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--name",default="pretrained-640")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if not args.name or Path(args.name).name!=args.name or args.name in {".",".."}:
        raise ValueError("Run name must be a single directory name")
    root=args.root.resolve()
    verify(root)
    if args.action=="train": train(root,args.name,args.resume)
    elif args.action=="evaluate": evaluate(root,args.name)
    elif args.action=="archive": archive(root,args.name)


if __name__=="__main__":
    main()
