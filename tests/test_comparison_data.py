from granite_tools.comparison_data import (
    get_bigram_pairs,
    get_bigram_unigram_pairs,
    get_ordered_unigrams_and_repeats,
    get_used_key_indices,
)


class TestGetBigramUnigramPairs:

    def test_simple(self):

        pairs = [
            ((1, 2), (3,)),
            ((1,), (3, 4)),
            ((1, 2), (3, 4)),
            ((1,), (3,)),
        ]
        result = get_bigram_unigram_pairs(pairs)
        assert result == [((1, 2), (3,)), ((1,), (3, 4))]


class TestGetBigramPairs:

    def test_simple(self):

        pairs = [
            ((1, 2), (3,)),
            ((1,), (3, 4)),
            ((1, 2), (3, 4)),
            ((1,), (3,)),
            ((1, 2), (5, 6)),
        ]
        result = get_bigram_pairs(pairs)
        assert result == [((1, 2), (3, 4)), ((1, 2), (5, 6))]


class TestGetUsedKeyIndices:

    def test_simple(self):

        pairs = [
            ((1, 2), (3,)),
            ((1,), (3, 4)),
            ((1, 2), (3, 4)),
            ((1,), (3,)),
        ]
        result = get_used_key_indices(pairs)
        assert result == [1, 2, 3, 4]

    def test_just_bigrams(self):

        pairs = [
            ((1, 2), (3, 4)),
            ((1, 2), (5, 6)),
        ]
        result = get_used_key_indices(pairs)
        assert result == [1, 2, 3, 4, 5, 6]


class TestGetOrderedUnigramsAndRepeats:

    def test_simple(self):

        out = get_ordered_unigrams_and_repeats({1: 0.5, 2: 0.1, 3: 0.3})

        assert out == (
            [(2,), (3,), (1,)],
            [(2, 2), (3, 3), (1, 1)],
        )
