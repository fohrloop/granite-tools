import re

import pytest

from granite_tools.scripts.create_ngram_ranking import (
    NgramType,
    get_combined_order,
    get_order_for_types,
)


class TestGetCombinedOrder:

    def test_simple(self):

        unigrams = [
            (1,),
            (3,),
        ]
        repeats = [
            (2, 2),
            (4, 4),
        ]
        bigrams = [
            (5, 6),
            (7, 8),
            (10, 11),
        ]

        ngram_type_order = [
            NgramType.UNIGRAM,
            NgramType.BIGRAM,
            NgramType.UNIGRAM,
            NgramType.REPEAT,
            NgramType.REPEAT,
            NgramType.BIGRAM,
            NgramType.BIGRAM,
        ]

        result = get_combined_order(unigrams, repeats, bigrams, ngram_type_order)
        assert result == [
            (1,),
            (5, 6),
            (3,),
            (2, 2),
            (4, 4),
            (7, 8),
            (10, 11),
        ]

    def test_type_order_too_short(self):

        unigrams = [
            (1,),
        ]
        repeats = [
            (2, 2),
        ]
        bigrams = [
            (5, 6),
            (7, 8),
        ]

        ngram_type_order = [
            NgramType.UNIGRAM,
            NgramType.REPEAT,
            NgramType.BIGRAM,
        ]

        with pytest.raises(
            RuntimeError,
            match=re.escape("Too many ngrams (ngram_type_order too short)"),
        ):
            get_combined_order(unigrams, repeats, bigrams, ngram_type_order)

    def test_type_order_too_long(self):

        unigrams = [
            (1,),
        ]
        repeats = [
            (2, 2),
        ]
        bigrams = [
            (5, 6),
            (7, 8),
        ]

        ngram_type_order = [
            NgramType.UNIGRAM,
            NgramType.REPEAT,
            NgramType.BIGRAM,
            NgramType.BIGRAM,
            NgramType.BIGRAM,
        ]

        with pytest.raises(RuntimeError, match="Not enough bigrams"):
            get_combined_order(unigrams, repeats, bigrams, ngram_type_order)


class TestGetOrderForTypes:

    def test_simple(self):

        ordered_ngrams = [
            (1,),
            (1, 2),
            (5, 6),
            (3, 3),
            (3,),
        ]
        result = get_order_for_types(ordered_ngrams)
        assert result == [
            NgramType.UNIGRAM,
            NgramType.BIGRAM,
            NgramType.BIGRAM,
            NgramType.REPEAT,
            NgramType.UNIGRAM,
        ]
