import json
from pathlib import Path
import pytest
from monster_dataset.schema import FrameAnnotation, MonsterAnnotation, write_jsonl
from perception.monster_evaluation import evaluate_monsters

def _observation(timestamp,visual_count,estimated_count,visual_points,track_points):
    return {"timestamp":timestamp,"monsters":{"visual_detection_count":visual_count,"estimated_count":estimated_count,
            "visual_detections":[{"center":p,"ground_position":p} for p in visual_points],
            "tracks":[{"track_id":i+1,"center":p} for i,p in enumerate(track_points)],
            "detector_debug":{"raw_detections":visual_points,"suppressed_duplicates":[]}}}

def test_evaluation_separates_visual_and_estimated_counts(tmp_path:Path):
    annotations=tmp_path/"annotations.jsonl"
    write_jsonl([FrameAnnotation("f",1.0,"f.jpg",[MonsterAnnotation([0,0,20,20],[10,20]),MonsterAnnotation([20,0,40,20],[30,20])],split="validation",review_status="reviewed")],annotations)
    observations=tmp_path/"observations.jsonl"
    observations.write_text(json.dumps(_observation(1.0,1,2,[[20,20]],[[10,20],[30,20]]))+"\n",encoding="utf-8")
    result=evaluate_monsters(annotations,observations,tmp_path/"out")
    assert result["visual_count_mae"]==1
    assert result["count_mae"]==0
    assert result["exact_count_accuracy"]==1
    assert (tmp_path/"out"/"failures.jsonl").is_file()

def test_evaluation_refuses_pending_or_sealed_annotations(tmp_path:Path):
    annotations=tmp_path/"annotations.jsonl"; observations=tmp_path/"observations.jsonl"; observations.write_text("{}\n")
    write_jsonl([FrameAnnotation("f",1,"f.jpg",split="validation")],annotations)
    with pytest.raises(RuntimeError,match="not reviewed"): evaluate_monsters(annotations,observations,tmp_path/"out")
    with pytest.raises(ValueError,match="sealed test"): evaluate_monsters(annotations,observations,tmp_path/"out",split="test")
