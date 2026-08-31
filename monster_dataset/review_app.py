"""Small OpenCV annotation UI with VFR-aware temporal context."""
from __future__ import annotations
import bisect
from pathlib import Path
import cv2
import numpy as np
from perception.core import read_video_frame, video_frame_timestamps
from .schema import FrameAnnotation, MonsterAnnotation, read_jsonl, write_jsonl

CURRENT_ORIGIN=(0,270); CURRENT_SIZE=(960,540); SOURCE_SIZE=(1920,1080)

def display_box_to_source(start:tuple[int,int],end:tuple[int,int])->list[float]:
    x1,x2=sorted((start[0]-CURRENT_ORIGIN[0],end[0]-CURRENT_ORIGIN[0])); y1,y2=sorted((start[1]-CURRENT_ORIGIN[1],end[1]-CURRENT_ORIGIN[1]))
    x1=max(0,min(CURRENT_SIZE[0],x1)); x2=max(0,min(CURRENT_SIZE[0],x2)); y1=max(0,min(CURRENT_SIZE[1],y1)); y2=max(0,min(CURRENT_SIZE[1],y2))
    return [x1*2.,y1*2.,x2*2.,y2*2.]

class MonsterReviewApp:
    def __init__(self,annotations:Path,video:Path,*,split:str,start_id:str|None=None,delta_s:float=.2):
        if split not in {"train","validation"}: raise ValueError("Interactive review is limited to train and validation")
        self.annotations_path=annotations; self.video=video; self.delta_s=delta_s
        self.all_items=read_jsonl(annotations); self.items=[item for item in self.all_items if item.split==split]
        if not self.items: raise ValueError(f"No {split} annotations")
        self.index=next((i for i,item in enumerate(self.items) if item.frame_id==start_id),0); self.timestamps=video_frame_timestamps(video)
        self.drag_start:tuple[int,int]|None=None; self.last_index:int|None=None; self.window="Monster review"

    def _frame(self,timestamp:float)->np.ndarray:
        insertion=bisect.bisect_left(self.timestamps,timestamp); choices=[i for i in (insertion-1,insertion) if 0<=i<len(self.timestamps)]
        index=min(choices,key=lambda i:abs(self.timestamps[i]-timestamp)); return read_video_frame(self.video,frame_index=index)

    def _canvas(self)->np.ndarray:
        item=self.items[self.index]; panels=[]
        for label,timestamp in (("previous",max(0.,item.timestamp-self.delta_s)),("current",item.timestamp),("next",item.timestamp+self.delta_s)):
            frame=cv2.resize(self._frame(timestamp),(480,270),interpolation=cv2.INTER_AREA); cv2.rectangle(frame,(0,0),(480,25),(0,0,0),-1); cv2.putText(frame,label,(6,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA); panels.append(frame)
        current=cv2.imread(item.image_path)
        if current is None: current=cv2.resize(self._frame(item.timestamp),CURRENT_SIZE,interpolation=cv2.INTER_AREA)
        else: current=cv2.resize(current,CURRENT_SIZE,interpolation=cv2.INTER_AREA)
        for index,monster in enumerate(item.monsters):
            x1,y1,x2,y2=(int(v/2) for v in monster.bbox_xyxy); color=(0,180,255) if monster.occluded else (0,255,0)
            cv2.rectangle(current,(x1,y1),(x2,y2),color,2); cv2.circle(current,(int(monster.ground_position[0]/2),int(monster.ground_position[1]/2)),4,color,-1)
            cv2.putText(current,f"M{index+1} v={monster.visibility:.2f}{' REVIEW' if monster.review_required else ''}",(x1,max(16,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1,cv2.LINE_AA)
        help_panel=np.zeros((540,480,3),np.uint8); lines=[f"{item.frame_id}  {self.index+1}/{len(self.items)}",f"status: {item.review_status}  monsters: {len(item.monsters)}","drag: add box (ground=bottom center)","right click / Z: delete last","O: toggle occluded   V: cycle visibility","U: toggle review_required","R: reviewed + next   E: needs review + next","A/D: previous/next   S: save   Q: save/quit"]
        for row,line in enumerate(lines): cv2.putText(help_panel,line,(12,32+row*34),cv2.FONT_HERSHEY_SIMPLEX,.52,(230,230,230),1,cv2.LINE_AA)
        return cv2.vconcat([cv2.hconcat(panels),cv2.hconcat([current,help_panel])])

    def _mouse(self,event:int,x:int,y:int,_flags:int,_param:object)->None:
        inside=0<=x<CURRENT_SIZE[0] and CURRENT_ORIGIN[1]<=y<CURRENT_ORIGIN[1]+CURRENT_SIZE[1]
        if event==cv2.EVENT_LBUTTONDOWN and inside: self.drag_start=(x,y)
        elif event==cv2.EVENT_LBUTTONUP and self.drag_start and inside:
            box=display_box_to_source(self.drag_start,(x,y)); self.drag_start=None
            if box[2]-box[0]>=4 and box[3]-box[1]>=4:
                self.items[self.index].monsters.append(MonsterAnnotation(box,[(box[0]+box[2])/2,box[3]])); self.last_index=len(self.items[self.index].monsters)-1
        elif event==cv2.EVENT_RBUTTONDOWN and self.items[self.index].monsters:
            self.items[self.index].monsters.pop(); self.last_index=None

    def _last(self)->MonsterAnnotation|None:
        monsters=self.items[self.index].monsters
        return monsters[self.last_index if self.last_index is not None and self.last_index<len(monsters) else -1] if monsters else None

    def save(self)->None: write_jsonl(self.all_items,self.annotations_path)

    def run(self)->None:
        cv2.namedWindow(self.window,cv2.WINDOW_NORMAL); cv2.resizeWindow(self.window,1440,810); cv2.setMouseCallback(self.window,self._mouse)
        while True:
            cv2.imshow(self.window,self._canvas()); key=cv2.waitKey(30)&0xFF; last=self._last()
            if key in (ord("q"),27): self.save(); break
            if key==ord("s"): self.save()
            elif key in (ord("a"),81): self.save(); self.index=max(0,self.index-1); self.last_index=None
            elif key in (ord("d"),83): self.save(); self.index=min(len(self.items)-1,self.index+1); self.last_index=None
            elif key==ord("r"): self.items[self.index].review_status="reviewed"; self.save(); self.index=min(len(self.items)-1,self.index+1); self.last_index=None
            elif key==ord("e"): self.items[self.index].review_status="needs_review"; self.save(); self.index=min(len(self.items)-1,self.index+1); self.last_index=None
            elif key==ord("z") and self.items[self.index].monsters: self.items[self.index].monsters.pop(); self.last_index=None
            elif key==ord("o") and last: last.occluded=not last.occluded
            elif key==ord("u") and last: last.review_required=not last.review_required
            elif key==ord("v") and last:
                values=[1.,.75,.5,.25,0.]; last.visibility=values[(values.index(last.visibility)+1)%len(values)] if last.visibility in values else 1.
        cv2.destroyWindow(self.window)
