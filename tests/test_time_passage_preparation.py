import pytest

from tools.prepare_time_passage_one import midpoint_samples


def test_midpoints_use_actual_time_not_uniform_frame_indices():
    # Variable-rate source: midpoint targets are 2.5 and 7.5 seconds.
    assert midpoint_samples([0, 1, 2, 3, 7, 8, 9, 10], 2) == [(2, 2.5), (4, 7.5)]


def test_midpoints_cover_full_nonzero_timestamp_span():
    selected = midpoint_samples([10 + index for index in range(101)], 25)
    assert len(selected) == 25
    assert selected[0] == (2, 12)
    assert selected[-1] == (98, 108)
    assert [index for index, _ in selected] == list(range(2, 99, 4))


def test_midpoints_reject_duplicate_selection_in_sparse_timeline():
    with pytest.raises(ValueError, match="distinct frames"):
        midpoint_samples([0, 0.1, 0.2, 0.3, 100], 5)
