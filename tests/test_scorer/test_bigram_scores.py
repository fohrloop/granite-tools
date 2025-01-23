import pytest

from granite_tools.app_types import KeySeq
from granite_tools.scorer.bigram_scores import (
    _fit_selected_ngram_scores,
    _get_ngrams_present_in_score_ratios,
    fit_anchor_ngram_scores,
    get_group_sizes,
    select_every_nth_item,
)

ScoreRatioEntry = tuple[KeySeq, KeySeq, float]


@pytest.fixture
def score_ratios() -> list[ScoreRatioEntry]:
    return [
        ((2,), (1,), 1.1),  # x2/x1 = 1.1
        ((4,), (2,), 1.4),  # x4/x2 = 1.4
        ((5,), (2,), 1.5),  # x5/x2 = 1.5
        ((5,), (4,), 1.2),  # x5/x4 = 1.2
    ]


@pytest.fixture
def ranks() -> list[KeySeq]:
    return [(1,), (2,), (3,), (4,), (5,)]


class TestGetNgramsPresentInScoreRatios:

    def test_simple(self, score_ratios: list[ScoreRatioEntry]):

        out = _get_ngrams_present_in_score_ratios(
            score_ratios, [(1,), (4,), (2,), (3,), (5,)]
        )

        # There is no (3,). The output is ordered by the rank order.
        assert out == [(1,), (4,), (2,), (5,)]


class TestFitSelectedNgramScores:

    def test_simple(self, score_ratios: list[ScoreRatioEntry]):

        scores = _fit_selected_ngram_scores(
            first_ngram=(1,),
            ngrams_need_score=[(2,), (4,), (5,)],
            score_ratios=score_ratios,
        )

        # The first ngram is fixed to 1.0. The rest are calculated.
        assert scores[(1,)] == 1.0
        # x2/x1 = 1.1
        assert pytest.approx(scores[(2,)], rel=1e-3) == 1.1

        # We know that
        # x4/x2 = 1.4
        # x5/x2 = 1.5
        # x5/x4 = 1.2
        #
        # First, 1.1 * 1.4 = 1.54 (first quess for x4)
        # Then, 1.1 * 1.5 = 1.65 (first quess for x5)
        # Finally, the ratio of x5 and x4 must be close to 1.2, so
        # the scores are adjusted a bit.
        # 1.70 / 1.48 = 1.148 (closer to 1.2 than 1.65/1.54, which would be 1.07)
        assert pytest.approx(scores[(4,)], rel=1e-2) == 1.48
        assert pytest.approx(scores[(5,)], rel=1e-2) == 1.70


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
