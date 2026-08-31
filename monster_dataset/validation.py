from __future__ import annotations
import json
from pathlib import Path
from .schema import FrameAnnotation

def report(items: list[FrameAnnotation]) -> dict:
    reviewed=sum(i.review_status=="reviewed" for i in items); positives=sum(bool(i.monsters) for i in items)
    instances=[m for i in items for m in i.monsters]; counts={str(n):sum(len(i.monsters)==n for i in items) for n in sorted({len(i.monsters) for i in items})}
    sizes=[(m.bbox_xyxy[2]-m.bbox_xyxy[0],m.bbox_xyxy[3]-m.bbox_xyxy[1]) for m in instances]
    return {"schema_version":1,"frames":len(items),"reviewed_frames":reviewed,"pending_frames":len(items)-reviewed,"positive_frames":positives,"negative_frames":len(items)-positives,"monster_instances":len(instances),"occluded_instances":sum(m.occluded for m in instances),"review_required_instances":sum(m.review_required for m in instances),"count_distribution":counts,"box_size":{"mean_width":sum(x for x,_ in sizes)/max(1,len(sizes)),"mean_height":sum(y for _,y in sizes)/max(1,len(sizes))}}

def write_report(items: list[FrameAnnotation], path: Path) -> dict:
    result=report(items); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2),encoding="utf-8"); return result
