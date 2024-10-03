import pytest

from granite_tools import Ngram, NgramList, NormalizationWarning
from granite_tools.chargram import CHAR_PRINT_MAPPING


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
        assert ngrams.normalized is True

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
        assert ngrams.normalized is False

    def test_ignore_whitespace(self):
        ngramtext = """
        30 a b
        10  c
        10 d 
        25 e
        25 f
        """

        ngrams = NgramList(ngramtext, ignore_whitespace=True)
        assert list(ngrams.iter_tuples()) == [
            (50, "e"),
            (50, "f"),
        ]

    def test_ignore_whitespace_no_normalize(self):
        ngramtext = """
        30 a b
        10  c
        10 d 
        25 e
        25 f
        """

        with pytest.warns(NormalizationWarning):
            ngrams = NgramList(ngramtext, ignore_whitespace=True, normalize=False)
        assert list(ngrams.iter_tuples()) == [
            (25, "e"),
            (25, "f"),
        ]

    def test_not_ignore_whitespace(self):
        ngramtext = """
        30 a b
        10  c
        10 d 
        25 e
        25 f
        """

        ngrams = NgramList(ngramtext, ignore_whitespace=False)
        space = CHAR_PRINT_MAPPING[" "]

        expected = [
            (30, f"a{space}b"),
            (25, "e"),
            (25, "f"),
            (10, f"{space}c"),
            (10, f"d{space}"),
        ]
        assert list(ngrams.iter_tuples()) == expected
