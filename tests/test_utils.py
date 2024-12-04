import pytest

from granite_tools.utils import to_new_scale


class TestToNewScale:

    def test_simple(self):
        assert to_new_scale(83, 1, 272, newmin=1.0, newmax=5.0) == pytest.approx(
            2.2103321033210332
        )
