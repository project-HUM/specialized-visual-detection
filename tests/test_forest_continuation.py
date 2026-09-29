import pytest
from tools.continue_forest_training import continuation_lr, historical_best


def test_cosine_is_not_restarted_or_stretched_after_500():
    assert continuation_lr(0) == 1
    assert continuation_lr(250) == pytest.approx(.505)
    assert continuation_lr(490) > .01
    for epoch in [500, 664, 1000, 100000]:
        assert continuation_lr(epoch) == pytest.approx(.01)


def test_equal_score_keeps_original_patience_anchor():
    rows = [{'epoch': str(e), 'metrics/mAP50-95(B)': str(s)}
            for e,s in [(364,.75835),(490,.73),(510,.75835)]]
    assert historical_best(rows) == (364,.75835)
    rows.append({'epoch':'550','metrics/mAP50-95(B)':'.75836'})
    assert historical_best(rows) == (550,.75836)
