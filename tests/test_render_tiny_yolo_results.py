import numpy as np

from tools.render_tiny_yolo_results import TITLE_BAR_HEIGHT, _prepend_caption


def test_caption_bar_is_prepended_without_covering_source_frame():
    source = np.full((12, 20, 3), (11, 22, 33), dtype=np.uint8)

    rendered = _prepend_caption(source, "result")

    assert rendered.shape == (12 + TITLE_BAR_HEIGHT, 20, 3)
    np.testing.assert_array_equal(rendered[TITLE_BAR_HEIGHT:], source)
