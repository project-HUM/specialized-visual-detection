from pathlib import Path

import pytest

from perception.benchmark import evaluate


def test_evaluation_rejects_pending_weak_labels(tmp_path: Path):
    annotations = tmp_path / "annotations.csv"
    annotations.write_text(
        "sample_id,timestamp,split,category,review_status,character_x,character_y,character_visible,character_facing,monster_centers,a_wave_state,a_wave_direction,weak_a_wave_state,weak_a_wave_direction,notes\n"
        "x,1800,test,a_skill,pending,,,,,,,active,right,\n",
        encoding="utf-8",
    )
    observations = tmp_path / "observations.jsonl"
    observations.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="No reviewed test annotations"):
        evaluate(annotations, observations, tmp_path / "out")
