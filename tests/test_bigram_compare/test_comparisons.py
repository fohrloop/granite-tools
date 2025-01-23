import pytest

from granite_tools.bigram_compare.comparisons import keyseqs_to_comparisons


class TestKeySeqsToComparisons:

    def test_simple(self):
        keyseqs = [(0,), (1,), (2,), (3,)]

        assert keyseqs_to_comparisons(keyseqs) == [
            ((3,), (2,)),
            ((2,), (1,)),
            ((1,), (0,)),
        ]

    def test_single_keyseq(self):
        with pytest.raises(ValueError):
            keyseqs_to_comparisons([(0,)])
