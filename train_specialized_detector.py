"""Optional reproducible map-scoped training entry point for the one-class detector.

This deliberately refuses to train while canonical labels are still pending.
Install the optional training extra (ultralytics) only on a training machine;
the capture-local inference contract does not depend on it.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from monster_dataset.schema import read_jsonl
from monster_dataset.annotation_io import validate_for_export
from perception.specialized_detector import train_yolo
from perception.workspace import MapWorkspace

def main() -> int:
    root=Path(__file__).resolve().parent
    selector=argparse.ArgumentParser(add_help=False); selector.add_argument("--map",default="rednose3")
    selected,_=selector.parse_known_args(); workspace=MapWorkspace.load(root,selected.map)
    parser=argparse.ArgumentParser(); parser.add_argument("--map",default=workspace.map_id); parser.add_argument("--annotations",type=Path,default=workspace.path("annotations")); parser.add_argument("--data",type=Path,default=workspace.path("yolo_dataset")/"dataset.yaml"); parser.add_argument("--output",type=Path,default=workspace.path("training_runs")); parser.add_argument("--model",default="yolo11n.pt",help="pretrained model name or local weights path"); parser.add_argument("--imgsz",type=int,nargs="+",default=[640,768,960]); parser.add_argument("--epochs",type=int,default=40); parser.add_argument("--device", default=None, help="Ultralytics device (auto when omitted, e.g. 0, cpu, mps)"); args=parser.parse_args()
    items=read_jsonl(args.annotations)
    try:
        validate_for_export(items,"train",source_size=workspace.source_size,image_root=workspace.repo_root); validate_for_export(items,"validation",source_size=workspace.source_size,image_root=workspace.repo_root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    results=[]
    for imgsz in args.imgsz:
        run_output=args.output.parent/f"{args.output.name}-{imgsz}"
        results.append(train_yolo(args.data,run_output,model=args.model,imgsz=imgsz,epochs=args.epochs,device=args.device))
    print(json.dumps(results,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
