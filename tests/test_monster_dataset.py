from pathlib import Path
import json
from monster_dataset.schema import FrameAnnotation, MonsterAnnotation, read_jsonl, write_jsonl
from monster_dataset.validation import report
from monster_dataset.annotation_io import export_yolo

def test_canonical_annotation_round_trip(tmp_path: Path):
    item=FrameAnnotation("f",1.25,"frame.jpg",[MonsterAnnotation([10,20,30,50],[20,50],occluded=True,visibility=.5)],category="heavy_effects")
    path=tmp_path/"annotations.jsonl"; write_jsonl([item],path); loaded=read_jsonl(path)
    assert loaded[0].monsters[0].ground_position == [20,50]
    assert loaded[0].monsters[0].occluded is True

def test_pending_annotations_are_not_exported_as_training_labels(tmp_path: Path):
    item=FrameAnnotation("f",0,"missing.jpg",review_status="pending")
    assert export_yolo([item],tmp_path/"labels")=={"exported":0,"skipped":1}

def test_report_exposes_review_and_overlap_statistics():
    items=[FrameAnnotation("a",0,"a.jpg",[MonsterAnnotation([0,0,20,20],[10,20]),MonsterAnnotation([5,0,25,20],[15,20])],review_status="reviewed"),FrameAnnotation("b",1,"b.jpg")]
    result=report(items)
    assert result["reviewed_frames"]==1
    assert result["monster_instances"]==2
    assert result["count_distribution"]=={"0":1,"2":1}
