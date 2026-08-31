"""Optional reproducible training entry point for the one-class detector.

This deliberately refuses to train while canonical labels are still pending.
Install the optional training extra (ultralytics) only on a training machine;
the capture-local inference contract does not depend on it.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from monster_dataset.schema import read_jsonl
from perception.specialized_detector import train_yolo

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--annotations",type=Path,default=Path(__file__).parent/"monster_dataset"/"annotations.jsonl"); parser.add_argument("--data",type=Path,default=Path(__file__).parent/"monster_dataset"/"dataset.yaml"); parser.add_argument("--output",type=Path,default=Path(__file__).parent/"monster_dataset"/"runs"/"v0"); parser.add_argument("--imgsz",type=int,default=768); parser.add_argument("--epochs",type=int,default=40); args=parser.parse_args()
    items=read_jsonl(args.annotations); pending=[item.frame_id for item in items if item.review_status!="reviewed"]
    if pending: raise SystemExit(f"Refusing to train on pending annotations: {len(pending)} frames remain; review canonical JSONL first")
    print(json.dumps(train_yolo(args.data,args.output,imgsz=args.imgsz,epochs=args.epochs),indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
