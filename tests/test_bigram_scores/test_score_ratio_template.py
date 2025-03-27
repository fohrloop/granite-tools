import pytest

from granite_tools.app_types import KeySeq
from granite_tools.bigram_scores.score_ratio_template import (
    get_group_sizes,
    select_every_nth_item,
)

ScoreRatioEntry = tuple[KeySeq, KeySeq, float]


@pytest.fixture
def ranks() -> list[KeySeq]:
    return [(1,), (2,), (3,), (4,), (5,)]


class TestGetGroupSizes:

    @pytest.mark.parametrize(
        "total, group_max_size, expected",
        [
            (7, 3, [2, 2, 3]),
            (11, 3, [2, 3, 3, 3]),
            (11, 4, [3, 4, 4]),
            # [5, 3, 3] would not be okay since there's difference of 2
            (11, 5, [3, 4, 4]),
            (11, 6, [5, 6]),
            (11, 7, [5, 6]),
            (11, 8, [5, 6]),
            (14, 8, [7, 7]),
            (15, 8, [7, 8]),
            (16, 8, [8, 8]),
            (17, 8, [5, 6, 6]),
            (18, 8, [6, 6, 6]),
            (19, 8, [6, 6, 7]),
            (20, 8, [6, 7, 7]),
            (21, 8, [7, 7, 7]),
            (22, 8, [7, 7, 8]),
            (23, 8, [7, 8, 8]),
            (24, 8, [8, 8, 8]),
        ],
    )
    def test_divide_to_groups(self, total, group_max_size, expected):

        group_sizes = list(get_group_sizes(total, group_max_size))
        assert sum(group_sizes) == total
        assert max(group_sizes) <= group_max_size
        assert group_sizes == expected


class TestSelectEveryNthItem:

    def test_12(self):

        lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        out = select_every_nth_item(lst, 4)
        # Groups are sized 3, 4, 4
        assert out == [1, 4, 8, 12]

    def test_8(self):

        lst = [1, 2, 3, 4, 5, 6, 7, 8]
        out = select_every_nth_item(lst, 3)
        # Groups are sized 2, 2, 3
        assert out == [1, 3, 5, 8]
