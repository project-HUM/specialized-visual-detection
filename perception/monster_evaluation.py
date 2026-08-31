"""Validation metrics for the specialized detector plus persistent tracker."""
from __future__ import annotations
import csv
import json
from math import hypot
from pathlib import Path
from typing import Any
import numpy as np
from monster_dataset.schema import FrameAnnotation, read_jsonl

def _match(gt: list[list[float]], predicted: list[list[float]], radius: float) -> tuple[int,int,int,list[tuple[int,int]]]:
    pairs=sorted((hypot(g[0]-p[0],g[1]-p[1]),gi,pi) for gi,g in enumerate(gt) for pi,p in enumerate(predicted))
    used_g:set[int]=set(); used_p:set[int]=set(); matches=[]
    for distance,gi,pi in pairs:
        if distance>radius or gi in used_g or pi in used_p: continue
        used_g.add(gi); used_p.add(pi); matches.append((gi,pi))
    return len(matches),len(predicted)-len(matches),len(gt)-len(matches),matches

def _summary(rows:list[dict[str,Any]])->dict[str,Any]:
    tp=sum(r["estimated_tp"] for r in rows); fp=sum(r["estimated_fp"] for r in rows); fn=sum(r["estimated_fn"] for r in rows)
    visual_tp=sum(r["visual_tp"] for r in rows); visual_fp=sum(r["visual_fp"] for r in rows); visual_fn=sum(r["visual_fn"] for r in rows)
    return {"frames":len(rows),"center_precision":tp/max(1,tp+fp),"center_recall":tp/max(1,tp+fn),
            "center_f1":2*tp/max(1,2*tp+fp+fn),"visual_center_precision":visual_tp/max(1,visual_tp+visual_fp),
            "visual_center_recall":visual_tp/max(1,visual_tp+visual_fn),
            "count_mae":float(np.mean([r["estimated_count_error"] for r in rows])) if rows else None,
            "visual_count_mae":float(np.mean([r["visual_count_error"] for r in rows])) if rows else None,
            "exact_count_accuracy":sum(r["estimated_count_error"]==0 for r in rows)/max(1,len(rows))}

def evaluate_monsters(annotations_path:Path,observations_path:Path,output_dir:Path,*,split:str="validation",radius:float=64.)->dict[str,Any]:
    if split not in {"train","validation"}: raise ValueError("Development evaluation permits only train or validation; sealed test remains untouched")
    annotations=[a for a in read_jsonl(annotations_path) if a.split==split]
    pending=[a.frame_id for a in annotations if a.review_status!="reviewed"]
    if pending: raise RuntimeError(f"Refusing evaluation: {len(pending)} {split} annotations are not reviewed")
    observations=[json.loads(line) for line in observations_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows=[]; failures=[]; all_track_ids:set[int]=set(); raw_count=0; suppressed_count=0; near_player_fp=0; total_estimated=0
    occluded_gt=0; occluded_matched=0
    for annotation in annotations:
        if not observations: break
        observation=min(observations,key=lambda o:abs(float(o.get("timestamp",-1))-annotation.timestamp))
        if abs(float(observation.get("timestamp",-1))-annotation.timestamp)>.15: continue
        gt=[m.ground_position for m in annotation.monsters]
        visual=observation["monsters"].get("visual_detections",[]); tracks=observation["monsters"].get("tracks",[])
        visual_points=[item.get("ground_position",item["center"]) for item in visual]; estimated_points=[item.get("ground_position",item["center"]) for item in tracks]
        etp,efp,efn,estimated_matches=_match(gt,estimated_points,radius); vtp,vfp,vfn,_=_match(gt,visual_points,radius)
        matched_gt_ids={gi for gi,_pi in estimated_matches}
        occluded_gt+=sum(monster.occluded for monster in annotation.monsters)
        occluded_matched+=sum(monster.occluded and index in matched_gt_ids for index,monster in enumerate(annotation.monsters))
        estimated_count=int(observation["monsters"].get("estimated_count",len(tracks))); visual_count=int(observation["monsters"].get("visual_detection_count",len(visual)))
        conditions=set(annotation.conditions)
        if annotation.category=="clean_gameplay": conditions.add("clean")
        if len(gt)>=2: conditions.add("crowded")
        if annotation.category=="heavy_effects": conditions.add("heavy_effects")
        if any(m.occluded and m.visibility>0 for m in annotation.monsters): conditions.add("partial_occlusion")
        if any(m.occluded and m.visibility==0 for m in annotation.monsters): conditions.add("full_overlap")
        if annotation.player_bbox_xyxy: conditions.add("near_player")
        debug=observation["monsters"].get("detector_debug",{}); raw_count+=len(debug.get("raw_detections",[])); suppressed_count+=len(debug.get("suppressed_duplicates",[]))
        all_track_ids.update(int(t["track_id"]) for t in tracks); total_estimated+=len(tracks)
        if annotation.player_bbox_xyxy:
            x1,y1,x2,y2=annotation.player_bbox_xyxy
            matched_prediction_ids={pi for _gi,pi in estimated_matches}
            near_player_fp+=sum(pi not in matched_prediction_ids and x1<=p[0]<=x2 and y1<=p[1]<=y2 for pi,p in enumerate(estimated_points))
        row={"frame_id":annotation.frame_id,"timestamp":annotation.timestamp,"ground_truth_count":len(gt),"visual_detection_count":visual_count,
             "estimated_count":estimated_count,"visual_count_error":abs(visual_count-len(gt)),"estimated_count_error":abs(estimated_count-len(gt)),
             "visual_tp":vtp,"visual_fp":vfp,"visual_fn":vfn,"estimated_tp":etp,"estimated_fp":efp,"estimated_fn":efn,"conditions":";".join(sorted(conditions))}
        rows.append(row)
        reasons=[]
        if visual_count!=len(gt): reasons.append("visual_count_mismatch")
        if estimated_count!=len(gt): reasons.append("estimated_count_mismatch")
        if vfp: reasons.append("visual_false_positive")
        if efp: reasons.append("track_false_positive")
        if reasons: failures.append({**row,"reasons":reasons})
    breakdown={condition:_summary([r for r in rows if condition in r["conditions"].split(";")]) for condition in ("clean","crowded","heavy_effects","partial_occlusion","full_overlap","near_player")}
    metrics={"schema_version":1,"split":split,"center_radius_px":radius,**_summary(rows),"breakdown":breakdown,
             "occluded_monster_recall":occluded_matched/max(1,occluded_gt),
             "player_false_positive_rate":near_player_fp/max(1,sum("near_player" in r["conditions"] for r in rows)),
             "duplicate_rate":suppressed_count/max(1,raw_count),
             "track_fragmentation_proxy":max(0,len(all_track_ids)-max([r["ground_truth_count"] for r in rows],default=0)),
             "runtime_ms":{"p50":float(np.percentile([o["runtime_ms"] for o in observations if "runtime_ms" in o],50)) if any("runtime_ms" in o for o in observations) else None,
                           "p95":float(np.percentile([o["runtime_ms"] for o in observations if "runtime_ms" in o],95)) if any("runtime_ms" in o for o in observations) else None}}
    output_dir.mkdir(parents=True,exist_ok=True); (output_dir/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    if rows:
        with (output_dir/"per_frame.csv").open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output_dir/"failures.jsonl").write_text("".join(json.dumps(row,separators=(",",":"))+"\n" for row in failures),encoding="utf-8")
    return metrics
