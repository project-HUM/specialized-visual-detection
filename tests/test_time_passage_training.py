import json
from pathlib import Path
import pytest
from tools.time_passage_one_training_runner import MinimumEpochStopping
from tools.prepare_time_passage_one_training import select_validation
from monster_dataset.schema import read_jsonl

POLICY = dict(min_epochs=200,max_epochs=1000,patience=100)


@pytest.mark.parametrize('score',[0.,.4])
def test_plateau_cannot_stop_before_minimum_even_for_zero(score):
    stop = MinimumEpochStopping(POLICY)
    for epoch in range(1,200): assert not stop(epoch,score)
    assert stop(200,score)
    assert stop.best_epoch == 1 and stop.reason == 'patience'


def test_improvement_resets_patience_but_ties_do_not():
    stop = MinimumEpochStopping(POLICY)
    for epoch in range(1,280):
        assert not stop(epoch,.5 if epoch < 180 else .6)
    assert stop(280,.6)
    assert stop.best_epoch == 180


def test_continued_improvements_stop_at_cap():
    stop = MinimumEpochStopping(POLICY)
    for epoch in range(1,1000): assert not stop(epoch,epoch/1000)
    assert stop(1000,1.) and stop.reason == 'epoch_cap'


def test_resume_uses_matching_checkpoint_epoch_not_ahead_state(tmp_path):
    original = MinimumEpochStopping(POLICY,tmp_path)
    for epoch in range(1,161): original(epoch,.2 if epoch<120 else .3)
    resumed = MinimumEpochStopping(POLICY,tmp_path)
    resumed.restore(100)
    assert resumed.best_epoch == 1 and resumed.best_fitness == .2
    for epoch in range(101,200): assert not resumed(epoch,.2)
    assert resumed(200,.2)
    assert json.loads((tmp_path/'epoch-000200.json').read_text())['reason']=='patience'


def test_resume_rejects_policy_change_and_missing_state(tmp_path):
    stop = MinimumEpochStopping(POLICY,tmp_path)
    stop(1,.1)
    with pytest.raises(ValueError,match='mismatch'):
        MinimumEpochStopping({**POLICY,'patience':101},tmp_path).restore(1)
    with pytest.raises(FileNotFoundError): MinimumEpochStopping(POLICY,tmp_path).restore(2)


def test_reject_nonfinite_or_skipped_validation():
    stop = MinimumEpochStopping(POLICY)
    for score in (None,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='finite'): stop(1,score)
    with pytest.raises(ValueError,match='consecutive'): stop(2,.4)


def test_approved_split_and_exclusions():
    items = read_jsonl(Path(__file__).resolve().parents[1]/'maps/timePassageOne/dataset/annotations.jsonl')
    selected = select_validation(items)
    assert [n for n,i in enumerate(items,1) if i.frame_id in selected]==[8,13,15,21,24,36,45,49,50]
    assert not any('outside_timePassageOne_event' in i.conditions for i in items if i.frame_id in selected)
