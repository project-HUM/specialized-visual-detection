"""Camera-compensated persistent monster tracking."""
from __future__ import annotations
from dataclasses import dataclass
from math import hypot
from typing import Any, Sequence

Box = tuple[float, float, float, float]

def _center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0

def _iou(left: Box, right: Box) -> float:
    w = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    h = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    inter = w * h
    if inter <= 0: return 0.0
    la = max(0.0, left[2]-left[0]) * max(0.0, left[3]-left[1])
    ra = max(0.0, right[2]-right[0]) * max(0.0, right[3]-right[1])
    return inter / max(1e-6, la + ra - inter)

def _size_similarity(left: Box, right: Box) -> float:
    lw, lh = max(1.0, left[2]-left[0]), max(1.0, left[3]-left[1])
    rw, rh = max(1.0, right[2]-right[0]), max(1.0, right[3]-right[1])
    return min(lw,rw)/max(lw,rw) * min(lh,rh)/max(lh,rh)

def deduplicate_detections(detections: Sequence[dict[str, Any]], *, iou_threshold: float = 0.88,
                           center_fraction: float = 0.22, size_similarity: float = 0.72) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Suppress only obvious duplicate boxes; preserve legitimate overlap."""
    ordered = sorted(enumerate(detections), key=lambda item: float(item[1].get("score", 0.0)), reverse=True)
    retained, suppressed = [], []
    for original_index, item in ordered:
        values = item.get("box")
        if not isinstance(values, Sequence) or len(values) != 4: continue
        box = tuple(float(value) for value in values)
        if box[2] <= box[0] or box[3] <= box[1]: continue
        duplicate = False
        for kept in retained:
            other = tuple(float(value) for value in kept["box"])
            cx, cy = _center(box); kx, ky = _center(other)
            diagonal = max(1.0, hypot(other[2]-other[0], other[3]-other[1]))
            if (_iou(box, other) >= iou_threshold and hypot(cx-kx, cy-ky) <= center_fraction*diagonal
                    and _size_similarity(box, other) >= size_similarity):
                duplicate = True; break
        copy = dict(item); copy["source_index"] = original_index
        (suppressed if duplicate else retained).append(copy)
    return retained, suppressed

@dataclass
class _Track:
    track_id: int
    map_box: Box
    score: float
    first_timestamp: float
    last_timestamp: float
    last_observed_timestamp: float
    hits: int = 1
    vx: float = 0.0
    vy: float = 0.0
    observed: bool = True
    state: str = "tentative"
    occlusion_group_id: int | None = None
    last_detection_id: int | None = None
    min_hits: int = 2
    @property
    def confirmed(self) -> bool:
        return self.hits >= self.min_hits

class MonsterTracker:
    """Track monsters in map coordinates and preserve IDs through overlap."""
    def __init__(self, *, min_hits: int = 2, max_gap_s: float = 0.55, base_match_distance: float = 28.0,
                 max_speed_px_s: float = 180.0, overlap_distance: float = 58.0):
        if min_hits < 1 or max_gap_s <= 0: raise ValueError("min_hits must be positive and max_gap_s must be greater than zero")
        self.min_hits, self.max_gap_s = min_hits, max_gap_s
        self.base_match_distance, self.max_speed_px_s, self.overlap_distance = base_match_distance, max_speed_px_s, overlap_distance
        self._next_id = 1; self._next_group_id = 1; self._tracks: list[_Track] = []
        self.last_debug = {"raw_detections": [], "suppressed_duplicates": [], "retained_detections": []}
    def reset(self):
        self._next_id = 1; self._next_group_id = 1; self._tracks.clear()
        self.last_debug = {"raw_detections": [], "suppressed_duplicates": [], "retained_detections": []}
    def _predicted(self, track: _Track, timestamp: float) -> Box:
        dt = max(0.0, timestamp-track.last_timestamp)
        return (track.map_box[0]+track.vx*dt, track.map_box[1]+track.vy*dt, track.map_box[2]+track.vx*dt, track.map_box[3]+track.vy*dt)
    def _compatible(self, track: _Track, box: Box, timestamp: float) -> tuple[bool,float]:
        predicted = self._predicted(track, timestamp); px,py = _center(predicted); cx,cy = _center(box)
        dt = max(0.0, timestamp-track.last_timestamp); gate = self.base_match_distance+self.max_speed_px_s*dt
        distance = hypot(cx-px, cy-py); overlap = _iou(predicted, box)
        return distance <= gate or overlap >= 0.08, distance/max(1.0,gate)+(1.0-overlap)*0.35
    def _observe(self, track: _Track, box: Box, score: float, timestamp: float, detection_id: int):
        dt = max(1e-6, timestamp-track.last_observed_timestamp); old_x,old_y = _center(track.map_box); new_x,new_y = _center(box)
        track.vx = .6*track.vx+.4*(new_x-old_x)/dt; track.vy = .6*track.vy+.4*(new_y-old_y)/dt
        track.map_box, track.score = box, .65*track.score+.35*score; track.last_timestamp = timestamp; track.last_observed_timestamp = timestamp
        track.hits += 1; track.observed = True; track.state = "visible"; track.occlusion_group_id = None; track.last_detection_id = detection_id
    def update(self, detections: Sequence[dict[str, Any]], timestamp: float, *, camera_origin: tuple[float,float] = (0.0,0.0),
               exclusion_boxes: Sequence[Sequence[float]] = (), exclusion_iou: float = .35, include_tentative: bool = False) -> list[dict[str,Any]]:
        if self._tracks and timestamp < max(t.last_timestamp for t in self._tracks): raise ValueError("Tracker timestamps must be monotonic")
        ox,oy = camera_origin; exclusions = [tuple(float(v) for v in x) for x in exclusion_boxes if len(x)==4]; transformed=[]
        for item in detections:
            values=item.get("box")
            if not isinstance(values, Sequence) or len(values)!=4: raise ValueError("Each detection must contain box=[x1,y1,x2,y2]")
            b=tuple(float(v) for v in values)
            if b[2]<=b[0] or b[3]<=b[1] or any(_iou(b,e)>=exclusion_iou for e in exclusions): continue
            transformed.append({**item,"box":(b[0]+ox,b[1]+oy,b[2]+ox,b[3]+oy)})
        retained,suppressed=deduplicate_detections(transformed); self.last_debug={"raw_detections":transformed,"suppressed_duplicates":suppressed,"retained_detections":retained}
        compat={di:[(self._compatible(t,item["box"],timestamp)[1],ti) for ti,t in enumerate(self._tracks) if self._compatible(t,item["box"],timestamp)[0]] for di,item in enumerate(retained)}
        matched_tracks=set(); matched_detections=set()
        for di,choices in compat.items():
            established=[ti for _cost,ti in choices if self._tracks[ti].confirmed]
            if len(established)<2: continue
            if max(self._compatible(self._tracks[ti],retained[di]["box"],timestamp)[1] for ti in established)>1.8: continue
            gid=self._next_group_id; self._next_group_id+=1
            for ti in established:
                t=self._tracks[ti]; t.map_box=self._predicted(t,timestamp); t.last_timestamp=timestamp; t.observed=False; t.state="occluded"; t.occlusion_group_id=gid; matched_tracks.add(ti)
            matched_detections.add(di)
        for _cost,ti,di in sorted((cost,ti,di) for di,choices in compat.items() if di not in matched_detections for cost,ti in choices):
            if ti in matched_tracks or di in matched_detections: continue
            item=retained[di]; self._observe(self._tracks[ti],item["box"],float(item.get("score",0.0)),timestamp,di); matched_tracks.add(ti); matched_detections.add(di)
        live=[]
        for ti,t in enumerate(self._tracks):
            if ti not in matched_tracks:
                t.map_box=self._predicted(t,timestamp); t.last_timestamp=timestamp; t.observed=False; t.occlusion_group_id=None
                if t.confirmed: t.state="temporal_hold"
            if timestamp-t.last_observed_timestamp<=self.max_gap_s: live.append(t)
        self._tracks=live
        for di,item in enumerate(retained):
            if di in matched_detections: continue
            b=item["box"]; self._tracks.append(_Track(self._next_id,b,float(item.get("score",0.0)),timestamp,timestamp,timestamp,state="visible" if self.min_hits <= 1 else "tentative",last_detection_id=di,min_hits=self.min_hits)); self._next_id+=1
        output=[]
        for t in self._tracks:
            if t.state=="tentative" and not include_tentative: continue
            gap=max(0.0,timestamp-t.last_observed_timestamp); conf=t.score if t.observed else t.score*max(0.0,1.0-gap/self.max_gap_s)
            b=tuple(t.map_box[i]-(ox if i%2==0 else oy) for i in range(4)); flags=[] if t.observed else (["monster_overlap"] if t.state=="occluded" else ["monster_temporal_hold"])
            output.append({"track_id":t.track_id,"state":t.state,"box":[round(v,2) for v in b],"center":[round(v,2) for v in _center(b)],"confidence":round(max(0.0,min(1.0,conf)),4),"confirmed":t.confirmed,"observed":t.observed,"occlusion_group_id":t.occlusion_group_id,"last_fresh_visual_timestamp":t.last_observed_timestamp,"provenance":["specialized_detector"] if t.observed else "visual_track","quality_flags":flags})
        return sorted(output,key=lambda x:x["track_id"])
    def estimated_count(self) -> int: return sum(t.confirmed for t in self._tracks)
    def count_bounds(self) -> tuple[int,int]:
        confirmed=[t for t in self._tracks if t.confirmed]; fresh=sum(t.observed for t in confirmed)
        return (len(confirmed),len(confirmed)) if fresh==len(confirmed) else (max(1,fresh),len(confirmed))
