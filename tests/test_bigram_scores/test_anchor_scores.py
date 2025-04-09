from __future__ import annotations

import typing

import pytest

from granite_tools.bigram_scores.anchor_scores import (
    _fit_selected_ngram_scores,
    _get_ngrams_present_in_score_ratios,
)

if typing.TYPE_CHECKING:
    from granite_tools.score_ratios import ScoreRatioEntry


@pytest.fixture
def score_ratios() -> list[ScoreRatioEntry]:

    return [
        {
            "score_ratio": 1.1,  # x2/x1
            "ref_keyseq": (1,),
            "ngram_keyseq": (2,),
        },
        {
            "score_ratio": 1.4,  # x4/x2
            "ref_keyseq": (2,),
            "ngram_keyseq": (4,),
        },
        {
            "score_ratio": 1.5,  # x5/x2
            "ref_keyseq": (2,),
            "ngram_keyseq": (5,),
        },
        {
            "score_ratio": 1.2,  # x5/x4
            "ref_keyseq": (4,),
            "ngram_keyseq": (5,),
        },
    ]


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
