import pytest

from granite_tools import Ngram, NgramList, NormalizationWarning


class TestNgramList:

    def test_simple_list(self):
        ngramtext = """
        92 a
        7 b
        1 c
        """

        ngrams = NgramList(ngramtext)
        assert len(ngrams) == 3
        assert list(ngrams.iter_tuples()) == [
            (92, "a"),
            (7, "b"),
            (1, "c"),
        ]

    def test_ignore_case(self):
        ngramtext = """
        40 a
        20 b
        40 A
        """

        ngrams = NgramList(ngramtext, ignore_case=True)
        assert list(ngrams.iter_tuples()) == [
            (80, "a"),
            (20, "b"),
        ]

    def test_not_ignore_case(self):
        ngramtext = """
        41 a
        20 b
        39 A
        """

        ngrams = NgramList(ngramtext, ignore_case=False)
        assert list(ngrams.iter_tuples()) == [
            (41, "a"),
            (39, "A"),
            (20, "b"),
        ]

    def test_normalize(self):
        ngramtext = """
        30 a
        20 b
        """

        ngrams = NgramList(ngramtext, normalize=True)
        assert list(ngrams.iter_tuples()) == [
            (60, "a"),
            (40, "b"),
        ]

    def test_not_normalize(self):
        ngramtext = """
        30 a
        20 b
        """

        with pytest.warns(NormalizationWarning):
            ngrams = NgramList(ngramtext, normalize=False)
        assert list(ngrams.iter_tuples()) == [
            (30, "a"),
            (20, "b"),
        ]
