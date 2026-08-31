from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from .schema import FrameAnnotation

def _iou(a:list[float],b:list[float])->float:
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/max(1e-9,union)

def report(items:list[FrameAnnotation])->dict:
    instances=[m for i in items for m in i.monsters]; sizes=[(m.bbox_xyxy[2]-m.bbox_xyxy[0],m.bbox_xyxy[3]-m.bbox_xyxy[1]) for m in instances]
    overlaps=[_iou(a.bbox_xyxy,b.bbox_xyxy) for item in items for index,a in enumerate(item.monsters) for b in item.monsters[index+1:]]
    malformed={item.frame_id:item.validate() for item in items if item.validate()}
    development=[item for item in items if item.split in {"train","validation"}]
    return {"schema_version":1,"frames":len(items),"reviewed_frames":sum(i.review_status=="reviewed" for i in items),
            "pending_frames":sum(i.review_status!="reviewed" for i in items),"positive_frames":sum(bool(i.monsters) for i in items),
            "negative_frames":sum(not i.monsters for i in items),"monster_instances":len(instances),"occluded_instances":sum(m.occluded for m in instances),
            "review_required_instances":sum(m.review_required for m in instances),
            "count_distribution":dict(sorted(Counter(str(len(i.monsters)) for i in items).items())),
            "condition_distribution":dict(sorted(Counter(condition for i in items for condition in i.conditions).items())),
            "splits":{split:{"frames":sum(i.split==split for i in items),"reviewed":sum(i.split==split and i.review_status=="reviewed" for i in items),"pending":sum(i.split==split and i.review_status!="reviewed" for i in items)} for split in ("train","validation","test")},
            "malformed_frames":malformed,"training_ready":not malformed and all(i.review_status=="reviewed" and not any(m.review_required for m in i.monsters) for i in development),
            "box_size":{"mean_width":sum(x for x,_ in sizes)/max(1,len(sizes)),"mean_height":sum(y for _,y in sizes)/max(1,len(sizes))},
            "pairwise_iou":{"pairs":len(overlaps),"max":max(overlaps,default=0.),"ge_0_5":sum(value>=.5 for value in overlaps)}}

def write_report(items:list[FrameAnnotation],path:Path)->dict:
    result=report(items); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2),encoding="utf-8"); return result
