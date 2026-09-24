from inspect import signature
from types import SimpleNamespace
import numpy as np
from perception.specialized_detector import YoloMonsterDetector

class _Array:
    def __init__(self,value): self.value=np.asarray(value)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value

class _Model:
    def __init__(self): self.kwargs=None; self.names={0:"zombie",1:"hero",2:"lich"}
    def predict(self,**kwargs):
        self.kwargs=kwargs
        return [SimpleNamespace(
            boxes=SimpleNamespace(xyxy=_Array([[10,20,30,50]]),conf=_Array([.8]),cls=_Array([2])),
            names=self.names,
        )]


def test_yolo_nms_defaults_are_class_aware():
    assert signature(YoloMonsterDetector).parameters["nms_iou"].default == .80

def test_yolo_adapter_normalizes_backend_output_and_keeps_conservative_config():
    detector=object.__new__(YoloMonsterDetector); detector.model=_Model(); detector.weights="fake.pt"; detector.confidence=.21
    detector.nms_iou=.80; detector.class_nms_iou={1:.50,2:.50}; detector.input_resolution=768; detector.device="cpu"; detector.max_detections=100
    detector.class_box_sizes=None; detector.box_size_reference=None
    detections=detector.detect(np.zeros((100,100,3),np.uint8),1.5)
    assert detections[0].bbox==(10.,20.,30.,50.)
    assert detections[0].ground_position==(20.,50.)
    assert detections[0].backend=="yolo"
    assert detections[0].metadata["class_id"]==2
    assert detections[0].metadata["class_name"]=="lich"
    assert "classes" not in detector.model.kwargs
    assert detector.model.kwargs["iou"]==.80 and detector.model.kwargs["imgsz"]==768
    assert detections[0].metadata["class_nms_iou"]==.50


def test_yolo_adapter_replaces_prediction_with_scaled_registered_class_size():
    detector=object.__new__(YoloMonsterDetector); detector.model=_Model(); detector.weights="fake.pt"; detector.confidence=.21
    detector.nms_iou=.80; detector.class_nms_iou={1:.50,2:.50}; detector.input_resolution=640; detector.device="cpu"; detector.max_detections=100
    detector.class_box_sizes={2:(120.,200.)}; detector.box_size_reference=(1920,1080)
    detections=detector.detect(np.zeros((540,960,3),np.uint8),1.5)
    assert detections[0].bbox==(-10.,-50.,50.,50.)
    assert detections[0].ground_position==(20.,50.)
    assert detections[0].metadata["raw_yolo_box"]==[10.,20.,30.,50.]
    assert detections[0].metadata["registered_box_size"]==[120.,200.]
    assert detections[0].metadata["box_size_reference"]==[1920,1080]


def test_yolo_adapter_drops_class_without_registered_size():
    detector=object.__new__(YoloMonsterDetector); detector.model=_Model(); detector.weights="fake.pt"; detector.confidence=.21
    detector.nms_iou=.80; detector.class_nms_iou={1:.50,2:.50}; detector.input_resolution=640; detector.device="cpu"; detector.max_detections=100
    detector.class_box_sizes={0:(83.,131.)}; detector.box_size_reference=(1920,1080)
    assert detector.detect(np.zeros((540,960,3),np.uint8))==[]


def test_yolo_adapter_applies_stricter_hero_and_lich_nms():
    model = _Model()
    model.predict = lambda **kwargs: [SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=_Array([
                [0, 0, 10, 10], [1, 0, 11, 10],
                [20, 0, 30, 10], [23, 0, 33, 10],
                [40, 0, 50, 10], [43, 0, 53, 10],
            ]),
            conf=_Array([.90, .80, .90, .80, .90, .80]),
            cls=_Array([0, 0, 1, 1, 2, 2]),
        ), names=model.names,
    )]
    detector=object.__new__(YoloMonsterDetector); detector.model=model; detector.weights="fake.pt"; detector.confidence=.21
    detector.nms_iou=.80; detector.class_nms_iou={1:.50,2:.50}; detector.input_resolution=640; detector.device="cpu"; detector.max_detections=100
    detector.class_box_sizes=None; detector.box_size_reference=None

    detections=detector.detect(np.zeros((100,100,3),np.uint8))

    assert [item.metadata["class_name"] for item in detections] == ["zombie", "hero", "lich"]
