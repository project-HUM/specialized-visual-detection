"""Backend-independent one-class monster detector contracts and adapters."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
import cv2, numpy as np

@dataclass(frozen=True)
class MonsterDetection:
    bbox: tuple[float,float,float,float]
    detector_confidence: float
    ground_position: tuple[float,float]
    backend: str = "unknown"
    metadata: dict[str,Any] | None = None
    def to_dict(self): return {"box":list(self.bbox),"score":self.detector_confidence,"ground_position":list(self.ground_position),"backend":self.backend,"metadata":self.metadata or {}}

class SpecializedMonsterDetector(Protocol):
    def detect(self,image: np.ndarray,timestamp: float|None=None)->list[MonsterDetection]: ...

class TemplateMonsterDetector:
    def __init__(self,templates:list[np.ndarray],threshold:float=.42,backend:str="template"): self.templates=templates; self.threshold=threshold; self.backend=backend
    def detect(self,image,timestamp=None):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image; edge=cv2.Canny(gray,55,140); output=[]
        for template in self.templates:
            if template.shape[0]>edge.shape[0] or template.shape[1]>edge.shape[1]: continue
            scores=cv2.matchTemplate(edge,template,cv2.TM_CCOEFF_NORMED); maxima=scores==cv2.dilate(scores,np.ones((9,9),np.uint8)); ys,xs=np.where(maxima&(scores>=self.threshold))
            for y,x in zip(ys,xs):
                box=(float(x),float(y),float(x+template.shape[1]),float(y+template.shape[0])); output.append(MonsterDetection(box,float(scores[y,x]),((box[0]+box[2])/2,box[3]),self.backend,{"timestamp":timestamp}))
        return output

def train_yolo(dataset_yaml:Path,output_dir:Path,*,model:str="yolo11n.pt",imgsz:int=768,epochs:int=40)->dict[str,Any]:
    try: from ultralytics import YOLO
    except ImportError as exc: raise RuntimeError("Optional specialized training requires ultralytics") from exc
    output_dir.parent.mkdir(parents=True,exist_ok=True); run=YOLO(model).train(data=str(dataset_yaml),imgsz=imgsz,epochs=epochs,project=str(output_dir.parent),name=output_dir.name,workers=0,device="cpu")
    return {"backend":"ultralytics","model":model,"imgsz":imgsz,"epochs":epochs,"run":str(getattr(run,"save_dir",output_dir))}
