import re

import pytest

from granite_tools.bigram_ranking import create_ngram_ranking

bigram_a = (5, 3)
bigram_b = (1, 2)
bigram_c = (18, 7)

unigram = (5,)
bigram_repeat = (5, 5)


class TestCreateNgramRanking:

    def test_simple(self):

        # In the comparison data (a,b) means a > b (a has higher score than b)
        # The ordering should be c > a > b
        comparisons_all = [
            (bigram_a, bigram_b),
            (bigram_c, bigram_b),
            (bigram_c, bigram_a),
        ]

        # The ranks are taken from the scores. At the top is the one with rank 1, which
        # is the one with lowerst score.
        # The rank order is then: b, a, c
        expected = [
            bigram_b,
            bigram_a,
            bigram_c,
        ]
        result = create_ngram_ranking(comparisons_all)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_unigram_warns(self):

        comparisons_all = [
            (bigram_a, unigram),
            (bigram_c, bigram_b),
            (bigram_c, bigram_a),
        ]

        with pytest.warns(
            UserWarning,
            match=re.escape(
                f"One of the ngrams, {unigram}, is a unigram. It will be ignored. Comparison pair: ((5, 3), (5,))"
            ),
        ):
            create_ngram_ranking(comparisons_all)

    def test_repeat_warns(self):

        comparisons_all = [
            (bigram_a, bigram_repeat),
            (bigram_c, bigram_b),
            (bigram_c, bigram_a),
        ]

        with pytest.warns(
            UserWarning,
            match=re.escape(
                f"One of the ngrams, {bigram_repeat}, is a repeat. It will be ignored. Comparison pair: ((5, 3), (5, 5))"
            ),
        ):
            create_ngram_ranking(comparisons_all)
