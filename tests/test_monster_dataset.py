from pathlib import Path
import cv2
import numpy as np
import pytest
from monster_dataset.schema import FrameAnnotation, MonsterAnnotation, read_jsonl, write_jsonl
from monster_dataset.validation import report
from monster_dataset.annotation_io import export_yolo
from monster_dataset.contact_sheet import write_temporal_context
from monster_dataset.review_app import display_box_to_source, MonsterReviewApp

def test_canonical_annotation_round_trip(tmp_path: Path):
    item=FrameAnnotation("f",1.25,"frame.jpg",[MonsterAnnotation([10,20,30,50],[20,50],occluded=True,visibility=.5)],category="heavy_effects")
    path=tmp_path/"annotations.jsonl"; write_jsonl([item],path); loaded=read_jsonl(path)
    assert loaded[0].monsters[0].ground_position == [20,50]
    assert loaded[0].monsters[0].occluded is True

def test_pending_annotations_are_not_exported_as_training_labels(tmp_path: Path):
    item=FrameAnnotation("f",0,"missing.jpg",review_status="pending")
    with pytest.raises(ValueError, match="Refusing train export"):
        export_yolo([item],tmp_path/"labels",split="train")

def test_export_is_split_aware_and_sealed_test_is_protected(tmp_path: Path):
    image=tmp_path/"frame.jpg"; cv2.imwrite(str(image),np.zeros((540,960,3),np.uint8))
    reviewed=FrameAnnotation("f",0,str(image),[MonsterAnnotation([10,20,30,50],[20,50])],split="train",review_status="reviewed")
    result=export_yolo([reviewed],tmp_path/"yolo",split="train")
    assert result["frames"]==1 and result["instances"]==1
    with pytest.raises(ValueError,match="sealed test"):
        export_yolo([reviewed],tmp_path/"yolo",split="test")

def test_malformed_annotation_blocks_export(tmp_path: Path):
    image=tmp_path/"frame.jpg"; cv2.imwrite(str(image),np.zeros((540,960,3),np.uint8))
    malformed=FrameAnnotation("f",0,str(image),[MonsterAnnotation([10,20,30,50],[100,100])],review_status="reviewed")
    with pytest.raises(ValueError,match="ground_position"):
        export_yolo([malformed],tmp_path/"yolo",split="train")

def test_report_exposes_review_and_overlap_statistics():
    items=[FrameAnnotation("a",0,"a.jpg",[MonsterAnnotation([0,0,20,20],[10,20]),MonsterAnnotation([5,0,25,20],[15,20])],review_status="reviewed"),FrameAnnotation("b",1,"b.jpg")]
    result=report(items)
    assert result["reviewed_frames"]==1
    assert result["monster_instances"]==2
    assert result["count_distribution"]=={"0":1,"2":1}

def test_temporal_context_cannot_reveal_sealed_test(tmp_path: Path):
    item=FrameAnnotation("sealed",10,"sealed.jpg",split="test")
    with pytest.raises(ValueError,match="train and validation"):
        write_temporal_context([item],tmp_path/"missing.mp4",tmp_path/"out",split="test")

def test_review_coordinate_transform_and_sealed_guard(tmp_path: Path):
    assert display_box_to_source((10,280),(110,380))==[20.,20.,220.,220.]
    with pytest.raises(ValueError,match="train and validation"):
        MonsterReviewApp(tmp_path/"missing.jsonl",tmp_path/"missing.mp4",split="test")
