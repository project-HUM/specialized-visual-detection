"""Camera-compensated persistent monster tracking with supported occlusion."""
from __future__ import annotations
from dataclasses import dataclass
from math import hypot
from typing import Any, Sequence

Box = tuple[float, float, float, float]

def _center(box: Box) -> tuple[float, float]: return (box[0]+box[2])/2, (box[1]+box[3])/2
def _translate(box: Box, dx: float, dy: float) -> Box: return box[0]+dx, box[1]+dy, box[2]+dx, box[3]+dy
def _iou(a: Box, b: Box) -> float:
    w=max(0.,min(a[2],b[2])-max(a[0],b[0])); h=max(0.,min(a[3],b[3])-max(a[1],b[1])); inter=w*h
    aa=max(0.,a[2]-a[0])*max(0.,a[3]-a[1]); ba=max(0.,b[2]-b[0])*max(0.,b[3]-b[1])
    return inter/max(1e-6,aa+ba-inter) if inter else 0.
def _size_similarity(a: Box, b: Box) -> float:
    aw,ah=max(1.,a[2]-a[0]),max(1.,a[3]-a[1]); bw,bh=max(1.,b[2]-b[0]),max(1.,b[3]-b[1])
    return min(aw,bw)/max(aw,bw)*min(ah,bh)/max(ah,bh)

def deduplicate_detections(detections: Sequence[dict[str,Any]], *, iou_threshold: float=.88,
                           center_fraction: float=.22, size_similarity: float=.72) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    retained: list[dict[str,Any]]=[]; suppressed: list[dict[str,Any]]=[]
    for source_index,item in sorted(enumerate(detections),key=lambda x:float(x[1].get("score",0)),reverse=True):
        values=item.get("box")
        if not isinstance(values,Sequence) or len(values)!=4: continue
        box=tuple(float(v) for v in values)
        if box[2]<=box[0] or box[3]<=box[1]: continue
        duplicate=False
        for kept in retained:
            other=tuple(kept["box"]); cx,cy=_center(box); ox,oy=_center(other); diagonal=max(1.,hypot(other[2]-other[0],other[3]-other[1]))
            if _iou(box,other)>=iou_threshold and hypot(cx-ox,cy-oy)<=center_fraction*diagonal and _size_similarity(box,other)>=size_similarity:
                duplicate=True; break
        copy=dict(item); copy["source_index"]=source_index; (suppressed if duplicate else retained).append(copy)
    return retained,suppressed

@dataclass
class _Track:
    track_id: int
    map_box: Box
    score: float
    first_timestamp: float
    last_timestamp: float
    last_fresh_visual_timestamp: float
    last_support_timestamp: float
    min_hits: int
    hits: int=1
    vx: float=0.; vy: float=0.
    observed: bool=True; supported: bool=True
    state: str="tentative"
    occlusion_group_id: int|None=None
    last_visual_provenance: str="visual_unknown"
    @property
    def confirmed(self)->bool: return self.hits>=self.min_hits

@dataclass
class OcclusionGroup:
    group_id: int
    member_track_ids: tuple[int,...]
    first_timestamp: float
    last_support_timestamp: float
    supporting_detection: dict[str,Any]

class MonsterTracker:
    def __init__(self, *, min_hits:int=2, max_gap_s:float=.55, supported_occlusion_s:float=2.5,
                 base_match_distance:float=28., max_speed_px_s:float=180., overlap_distance:float=58.,
                 dedup_iou:float=.88, dedup_center_fraction:float=.22, dedup_size_similarity:float=.72):
        if min_hits<1 or max_gap_s<=0 or supported_occlusion_s<=0: raise ValueError("tracker lifetimes and min_hits must be positive")
        self.min_hits=min_hits; self.max_gap_s=max_gap_s; self.supported_occlusion_s=supported_occlusion_s
        self.base_match_distance=base_match_distance; self.max_speed_px_s=max_speed_px_s; self.overlap_distance=overlap_distance
        self.dedup_iou=dedup_iou; self.dedup_center_fraction=dedup_center_fraction; self.dedup_size_similarity=dedup_size_similarity
        self._next_id=1; self._next_group_id=1; self._tracks:list[_Track]=[]; self._groups:dict[int,OcclusionGroup]={}
        self.last_debug={"raw_detections":[],"suppressed_duplicates":[],"retained_detections":[],"occlusion_groups":[]}

    def reset(self)->None:
        self._next_id=1; self._next_group_id=1; self._tracks.clear(); self._groups.clear()
        self.last_debug={"raw_detections":[],"suppressed_duplicates":[],"retained_detections":[],"occlusion_groups":[]}

    def _predicted(self,t:_Track,now:float)->Box: return _translate(t.map_box,t.vx*max(0.,now-t.last_timestamp),t.vy*max(0.,now-t.last_timestamp))
    def _compatibility(self,t:_Track,box:Box,now:float)->tuple[bool,float]:
        predicted=self._predicted(t,now); px,py=_center(predicted); bx,by=_center(box); dt=max(0.,now-t.last_timestamp)
        gate=self.base_match_distance+self.max_speed_px_s*dt; distance=hypot(px-bx,py-by); overlap=_iou(predicted,box)
        return distance<=gate or overlap>=.08, distance/max(1.,gate)+(1.-overlap)*.35

    def _observe(self,t:_Track,item:dict[str,Any],now:float)->None:
        box=tuple(item["box"]); dt=max(1e-6,now-t.last_fresh_visual_timestamp); ox,oy=_center(t.map_box); nx,ny=_center(box)
        t.vx=.6*t.vx+.4*(nx-ox)/dt; t.vy=.6*t.vy+.4*(ny-oy)/dt; t.map_box=box
        t.score=.65*t.score+.35*float(item.get("score",0)); t.last_timestamp=now; t.last_fresh_visual_timestamp=now; t.last_support_timestamp=now
        t.hits+=1; t.observed=True; t.supported=True; t.state="visible" if t.hits>=self.min_hits else "tentative"; t.occlusion_group_id=None
        t.last_visual_provenance=str(item.get("provenance",f"visual_{item.get('backend','unknown')}"))

    def _support_group(self,group:OcclusionGroup,item:dict[str,Any],now:float,by_id:dict[int,_Track])->None:
        group.last_support_timestamp=now; group.supporting_detection=dict(item)
        for track_id in group.member_track_ids:
            t=by_id.get(track_id)
            if t is None: continue
            t.map_box=self._predicted(t,now); t.vx*=.85; t.vy*=.85; t.last_timestamp=now; t.last_support_timestamp=now
            t.observed=False; t.supported=True; t.state="occluded"; t.occlusion_group_id=group.group_id

    def update(self,detections:Sequence[dict[str,Any]],timestamp:float,*,camera_origin:tuple[float,float]=(0.,0.),
               exclusion_boxes:Sequence[Sequence[float]]=(),exclusion_iou:float=.35,include_tentative:bool=False)->list[dict[str,Any]]:
        if self._tracks and timestamp<max(t.last_timestamp for t in self._tracks): raise ValueError("Tracker timestamps must be monotonic")
        ox,oy=camera_origin; exclusions=[tuple(float(v) for v in box) for box in exclusion_boxes if len(box)==4]; transformed=[]
        for item in detections:
            values=item.get("box")
            if not isinstance(values,Sequence) or len(values)!=4: raise ValueError("Each detection must contain box=[x1,y1,x2,y2]")
            screen=tuple(float(v) for v in values)
            if screen[2]<=screen[0] or screen[3]<=screen[1] or any(_iou(screen,e)>=exclusion_iou for e in exclusions): continue
            transformed.append({**item,"box":_translate(screen,ox,oy)})
        retained,suppressed=deduplicate_detections(transformed,iou_threshold=self.dedup_iou,center_fraction=self.dedup_center_fraction,size_similarity=self.dedup_size_similarity)
        by_id={t.track_id:t for t in self._tracks}; matched_tracks:set[int]=set(); matched_detections:set[int]=set()

        # Persist an existing group while exactly one observation supports all members.
        for group in list(self._groups.values()):
            unmatched_confirmed = sum(t.confirmed and t.track_id not in matched_tracks for t in self._tracks)
            unmatched_detections = len(retained) - len(matched_detections)
            if unmatched_detections >= unmatched_confirmed:
                break
            members=[by_id[i] for i in group.member_track_ids if i in by_id]
            candidates=[]
            for di,item in enumerate(retained):
                if di in matched_detections: continue
                if members and all(self._compatibility(t,tuple(item["box"]),timestamp)[0] for t in members): candidates.append(di)
            if len(candidates)==1 and len(members)>=2:
                di=candidates[0]; self._support_group(group,retained[di],timestamp,by_id)
                matched_detections.add(di); matched_tracks.update(t.track_id for t in members)

        # Create groups for a new many-tracks-to-one-observation merge.
        for di,item in enumerate(retained):
            if di in matched_detections: continue
            unmatched_confirmed = sum(t.confirmed and t.track_id not in matched_tracks for t in self._tracks)
            unmatched_detections = len(retained) - len(matched_detections)
            if unmatched_detections >= unmatched_confirmed:
                break
            compatible=[t for t in self._tracks if t.confirmed and t.track_id not in matched_tracks and self._compatibility(t,tuple(item["box"]),timestamp)[0]]
            if len(compatible)<2: continue
            compatible.sort(key=lambda t:_center(self._predicted(t,timestamp))[0])
            member_ids=tuple(t.track_id for t in compatible); existing=next((g for g in self._groups.values() if g.member_track_ids==member_ids),None)
            group=existing or OcclusionGroup(self._next_group_id,member_ids,timestamp,timestamp,dict(item))
            if existing is None: self._groups[group.group_id]=group; self._next_group_id+=1
            self._support_group(group,item,timestamp,by_id); matched_detections.add(di); matched_tracks.update(member_ids)

        # Resolve groups or perform ordinary association. Greedy cost preserves predicted ordering.
        pairs=[]
        for di,item in enumerate(retained):
            if di in matched_detections: continue
            for t in self._tracks:
                if t.track_id in matched_tracks: continue
                ok,cost=self._compatibility(t,tuple(item["box"]),timestamp)
                if ok: pairs.append((cost,t.track_id,di))
        for _cost,track_id,di in sorted(pairs):
            if track_id in matched_tracks or di in matched_detections: continue
            self._observe(by_id[track_id],retained[di],timestamp); matched_tracks.add(track_id); matched_detections.add(di)

        for t in self._tracks:
            if t.track_id in matched_tracks: continue
            t.map_box=self._predicted(t,timestamp); t.last_timestamp=timestamp; t.observed=False; t.supported=False; t.occlusion_group_id=None
            if t.confirmed: t.state="temporal_hold"

        live=[]
        for t in self._tracks:
            lifetime=self.supported_occlusion_s if t.state=="occluded" and t.supported else self.max_gap_s
            reference=t.last_support_timestamp
            if timestamp-reference<=lifetime: live.append(t)
        self._tracks=live; live_ids={t.track_id for t in live}
        self._groups={gid:g for gid,g in self._groups.items() if set(g.member_track_ids)<=live_ids and any(t.occlusion_group_id==gid for t in live)}

        for di,item in enumerate(retained):
            if di in matched_detections: continue
            provenance=str(item.get("provenance",f"visual_{item.get('backend','unknown')}")); box=tuple(item["box"])
            self._tracks.append(_Track(self._next_id,box,float(item.get("score",0)),timestamp,timestamp,timestamp,timestamp,self.min_hits,
                                       state="visible" if self.min_hits==1 else "tentative",last_visual_provenance=provenance)); self._next_id+=1

        self.last_debug={"raw_detections":transformed,"suppressed_duplicates":suppressed,"retained_detections":retained,
                         "occlusion_groups":[{"group_id":g.group_id,"member_track_ids":list(g.member_track_ids),"first_timestamp":g.first_timestamp,"last_support_timestamp":g.last_support_timestamp,"supporting_detection":g.supporting_detection} for g in self._groups.values()]}
        output=[]
        for t in self._tracks:
            if t.state=="tentative" and not include_tentative: continue
            reference=t.last_support_timestamp; lifetime=self.supported_occlusion_s if t.state=="occluded" and t.supported else self.max_gap_s
            confidence=t.score if t.observed else t.score*max(0.,1.-(timestamp-reference)/lifetime)
            screen=_translate(t.map_box,-ox,-oy); flags=[] if t.observed else (["monster_overlap"] if t.supported else ["monster_temporal_hold"])
            output.append({"track_id":t.track_id,"state":t.state,"box":[round(v,2) for v in screen],"center":[round(v,2) for v in _center(screen)],
                           "ground_position":[round((screen[0]+screen[2])/2,2),round(screen[3],2)],
                           "confidence":round(max(0.,min(1.,confidence)),4),"confirmed":t.confirmed,"observed":t.observed,"supported":t.supported,
                           "occlusion_group_id":t.occlusion_group_id,"last_fresh_visual_timestamp":t.last_fresh_visual_timestamp,
                           "last_support_timestamp":t.last_support_timestamp,"provenance":t.last_visual_provenance if t.observed else "visual_track","quality_flags":flags})
        return sorted(output,key=lambda x:x["track_id"])

    def estimated_count(self)->int: return sum(t.confirmed for t in self._tracks)
    def count_bounds(self)->tuple[int,int]:
        confirmed=[t for t in self._tracks if t.confirmed]; fresh=sum(t.observed for t in confirmed)
        return (len(confirmed),len(confirmed)) if fresh==len(confirmed) else (fresh,len(confirmed))
