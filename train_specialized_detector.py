"""Optional reproducible training entry point for the one-class detector.

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

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--annotations",type=Path,default=Path(__file__).parent/"monster_dataset"/"annotations.jsonl"); parser.add_argument("--data",type=Path,default=Path(__file__).parent/"monster_dataset"/"yolo_dataset"/"dataset.yaml"); parser.add_argument("--output",type=Path,default=Path(__file__).parent/"monster_dataset"/"runs"/"v0"); parser.add_argument("--imgsz",type=int,nargs="+",default=[640,768,960]); parser.add_argument("--epochs",type=int,default=40); parser.add_argument("--device", default=None, help="Ultralytics device (auto when omitted, e.g. 0, cpu, mps)"); args=parser.parse_args()
    items=read_jsonl(args.annotations)
    try:
        validate_for_export(items,"train"); validate_for_export(items,"validation")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    results=[]
    for imgsz in args.imgsz:
        run_output=args.output.parent/f"{args.output.name}-{imgsz}"
        results.append(train_yolo(args.data,run_output,imgsz=imgsz,epochs=args.epochs,device=args.device))
    print(json.dumps(results,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
