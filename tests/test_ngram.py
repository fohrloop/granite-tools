import pytest

from granite_tools.ngram_tools import NgramList, NormalizationWarning
from granite_tools.ngram_tools.ngram import CHAR_PRINT_MAPPING, PositionedNgramDiff


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
            ("a", 92),
            ("b", 7),
            ("c", 1),
        ]

    def test_ignore_case(self):
        ngramtext = """
        40 a
        20 b
        40 A
        """

        ngrams = NgramList(ngramtext, ignore_case=True)
        assert list(ngrams.iter_tuples()) == [
            ("a", 80),
            ("b", 20),
        ]

    def test_not_ignore_case(self):
        ngramtext = """
        41 a
        20 b
        39 A
        """

        ngrams = NgramList(ngramtext, ignore_case=False)
        assert list(ngrams.iter_tuples()) == [
            ("a", 41),
            ("A", 39),
            ("b", 20),
        ]

    def test_normalize(self):
        ngramtext = """
        30 a
        20 b
        """

        ngrams = NgramList(ngramtext, normalize=True)
        assert list(ngrams.iter_tuples()) == [
            ("a", 60),
            ("b", 40),
        ]
        assert ngrams.normalized is True

    def test_explicit_not_normalize(self):
        ngramtext = """
        30 a
        20 b
        """

        with pytest.warns(NormalizationWarning):
            ngrams = NgramList(ngramtext, normalize=False)
        assert list(ngrams.iter_tuples()) == [
            ("a", 30),
            ("b", 20),
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
            ("e", 50),
            ("f", 50),
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
            ("e", 25),
            ("f", 25),
        ]

    def test_exclude_chars(self):
        ngramtext = """
        30 ab
        10 bc
        10 cd
        25 ef
        25 ae
        """

        ngrams = NgramList(
            ngramtext,
            exclude_chars="a",
        )

        expected = [
            ("ef", 55.55555555555556),
            ("bc", 22.22222222222222),
            ("cd", 22.22222222222222),
        ]
        assert list(ngrams.iter_tuples()) == expected

    def test_exclude_chars_different_case(self):
        ngramtext = """
        30 ab
        10 bc
        10 cd
        25 ef
        25 ae
        """

        ngrams = NgramList(
            ngramtext, exclude_chars="A", ignore_case=False, normalize=False
        )

        expected = [
            ("ab", 30),
            ("ae", 25),
            ("ef", 25),
            ("bc", 10),
            ("cd", 10),
        ]
        assert list(ngrams.iter_tuples()) == expected

    def test_exclude_chars_different_case2(self):

        # Now, ignore case
        ngramtext = """
        30 ab
        10 bc
        10 cd
        25 ef
        25 ae
        """

        with pytest.warns(NormalizationWarning):
            ngrams = NgramList(
                ngramtext, exclude_chars="A", ignore_case=True, normalize=False
            )

        expected = [
            ("ef", 25),
            ("bc", 10),
            ("cd", 10),
        ]
        assert list(ngrams.iter_tuples()) == expected

    def test_exclude_chars_different_case3(self):

        # Now, ignore case
        ngramtext = """
        30 AB
        10 BC
        10 CD
        25 EF
        25 AE
        """

        with pytest.warns(NormalizationWarning):
            ngrams = NgramList(
                ngramtext, exclude_chars="a", ignore_case=True, normalize=False
            )

        expected = [
            ("ef", 25),
            ("bc", 10),
            ("cd", 10),
        ]
        assert list(ngrams.iter_tuples()) == expected

    def test_include_chars(self):

        ngramtext = """
        30 ab
        10 bc
        10 cd
        25 ef
        25 ae
        """

        with pytest.warns(NormalizationWarning):
            ngrams = NgramList(
                ngramtext, include_chars="bcd", ignore_case=True, normalize=False
            )

        expected = [
            ("bc", 10),
            ("cd", 10),
        ]
        assert list(ngrams.iter_tuples()) == expected

    def test_include_chars_ignore_case(self):

        ngramtext = """
        30 ab
        10 bc
        10 cd
        25 Cd
        25 ae
        """

        with pytest.warns(NormalizationWarning):
            ngrams = NgramList(
                ngramtext, include_chars="bcd", ignore_case=True, normalize=False
            )

        expected = [
            ("cd", 10 + 25),
            ("bc", 10),
        ]
        assert list(ngrams.iter_tuples()) == expected

    def test_include_chars_ignore_case_false(self):

        ngramtext = """
        30 ab
        10 bc
        10 cd
        25 Cd
        25 ae
        """

        with pytest.warns(NormalizationWarning):
            ngrams = NgramList(
                ngramtext, include_chars="bcd", ignore_case=False, normalize=False
            )

        expected = [
            ("bc", 10),
            ("cd", 10),
        ]
        assert list(ngrams.iter_tuples()) == expected

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
            (f"a{space}b", 30),
            ("e", 25),
            ("f", 25),
            (f"d{space}", 10),
            (f"{space}c", 10),
        ]
        assert list(ngrams.iter_tuples()) == expected


class TestNgramDiff:
    """Test difference calculations between two NgramLists."""

    def test_diff_simple(self):
        with pytest.warns(NormalizationWarning):
            ngrams1 = NgramList(
                """
            10 a
            9 b
            5 c
            0.1 e
            """,
                normalize=False,
            )
            ngrams2 = NgramList(
                """
            12 a
            6 e
            4 b
            0.3 c
            """,
                normalize=False,
            )

        diff = ngrams1.diff(ngrams2, n=3)
        assert list(diff.iter_ngrams("ref")) == [
            PositionedNgramDiff("a", 10, 1, 0, 0),
            PositionedNgramDiff("b", 9, 2, 0, 0),
            PositionedNgramDiff("c", 5, 3, 0, 0),
            PositionedNgramDiff("e", 0.1, 4, 0, 0),
        ]
        assert list(diff.iter_ngrams("other")) == [
            PositionedNgramDiff("a", 12, 1, freq_diff=+2, rank_diff=0),
            PositionedNgramDiff("e", 6, 2, freq_diff=+5.9, rank_diff=+2),
            PositionedNgramDiff("b", 4, 3, freq_diff=-5, rank_diff=-1),
            PositionedNgramDiff("c", 0.3, 4, freq_diff=-4.7, rank_diff=-1),
        ]

    def cannot_mix_normalized_and_non_normalized(self):
        ngrams1 = NgramList(
            """
        10 a
        """,
            normalize=True,
        )
        ngrams2 = NgramList(
            """
        12 a
        """,
            normalize=False,
        )

        with pytest.raises(RuntimeError, match="different `.normalized` value"):
            ngrams1.diff(ngrams2, n=1)
