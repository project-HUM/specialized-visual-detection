from types import SimpleNamespace
import numpy as np
from perception.specialized_detector import YoloMonsterDetector

class _Array:
    def __init__(self,value): self.value=np.asarray(value)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value

class _Model:
    def __init__(self): self.kwargs=None
    def predict(self,**kwargs):
        self.kwargs=kwargs
        return [SimpleNamespace(boxes=SimpleNamespace(xyxy=_Array([[10,20,30,50]]),conf=_Array([.8])))]

def test_yolo_adapter_normalizes_backend_output_and_keeps_conservative_config():
    detector=object.__new__(YoloMonsterDetector); detector.model=_Model(); detector.weights="fake.pt"; detector.confidence=.21
    detector.nms_iou=.78; detector.input_resolution=768; detector.device="cpu"; detector.max_detections=100
    detections=detector.detect(np.zeros((100,100,3),np.uint8),1.5)
    assert detections[0].bbox==(10.,20.,30.,50.)
    assert detections[0].ground_position==(20.,50.)
    assert detections[0].backend=="yolo"
    assert detector.model.kwargs["iou"]==.78 and detector.model.kwargs["imgsz"]==768
