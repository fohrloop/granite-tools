from __future__ import annotations

import typing
import warnings
from dataclasses import dataclass
from typing import Literal

import plotext

if typing.TYPE_CHECKING:
    from typing import Iterable, Optional

CHAR_PRINT_MAPPING = {" ": "␣", "\\n": "⏎", "\\\\": "\\"}

WHITESPACE_CHARS = (" ", "\\n")
"""Whitespace characters (as in the original ngram files)"""

FreqType = Literal["absolute", "cumulative" "both"]


@dataclass
class Ngram:
    chars: str
    """The characters in the ngram."""

    freq: float
    """The "frequency" of the ngram. The unit may be arbitrary, but typically
    8.52 would mean 8.52% (if the ngram file has been normalized)"""


@dataclass
class PositionedNgram(Ngram):
    """Ngram which also know it's position (rank, cumulative frequency)"""

    cumulative_freq: float
    """The cumulative frequency of the ngram up to this point"""

    rank: int
    """The rank of the ngram (1=most frequent, and the larger the number is the
    less frequent the ngram is)"""


class NormalizationWarning(UserWarning):
    pass


NORMALIZE_DEFAULT: bool = True


class NgramList:

    _ngrams: dict[str, Ngram]
    """A list of Ngram objects."""

    normalized: bool
    """True if the ngrams have been normalized to sum to 100.0 (with some
    tolerance since the frequencies are floating point numbers)."""

    def __init__(
        self,
        ngramtext: str,
        ignore_case: bool = False,
        ignore_whitespace: bool = False,
        normalize: Optional[bool] = None,
    ):
        """Create a new NgramList from a string of ngrams

        Parameters
        ----------
        ngramtext : str
            A string of ngrams, one per line, in the format <freq> <ngram>
        ignore_case: bool
            If True, ignore case when creating ngrams. This would mean that
            ngrams 'ab', 'Ab', 'aB', and 'AB' would all be considered the same.
            Default: False
        ignore_whitespace: bool
            If True, ignore ngrams that contain whitespace. Default: False
        normalize: bool
            If True, normalize the ngrams so that the sum of all frequencies is
            100.0. If False, and the input `ngramtext` is not normalized, a
            NormalizationWarning is issued. Default: True.

        Example
        -------
        >>> ngramtext = "1 a\n2 b\n3 c"
        >>> ngrams = NgramList(ngramtext)

        """
        if normalize is None:
            normalize = NORMALIZE_DEFAULT

        self._ngrams: dict[str, Ngram] = dict()
        for freq, chars in iter_lines(ngramtext, ignore_whitespace=ignore_whitespace):
            if ignore_case:
                chars = chars.lower()
            if chars in self._ngrams:
                self._ngrams[chars].freq += freq
            else:
                self._ngrams[chars] = Ngram(chars, freq)

        total = sum(ngram.freq for ngram in self._ngrams.values())
        if not total_is_normalized(total):
            if normalize:
                for ngram in self._ngrams.values():
                    ngram.freq = 100 * (ngram.freq / total)
                self.normalized = True
            else:
                warnings.warn(
                    f"Total frequency is {total}. This is not normalized.",
                    NormalizationWarning,
                )
                self.normalized = False
        else:
            self.normalized = True

    def iter_ngrams(self, ngrams_count: int = 2**32) -> Iterable[PositionedNgram]:
        total: float = 0
        for rank, (freq, chars) in enumerate(self.iter_tuples(), start=1):
            if rank > ngrams_count:
                break
            total += freq
            yield PositionedNgram(
                chars=chars, freq=freq, cumulative_freq=total, rank=rank
            )

    def iter_tuples(self) -> Iterable[tuple[float, str]]:
        """Iterates over the ngrams in the list, the most frequent ngram first,
        yielding a tuple of (freq, chars)"""
        for ngram in sorted(self._ngrams.values(), key=lambda x: x.freq, reverse=True):
            yield ngram.freq, ngram.chars

    def iter_txt(
        self, ngrams_count: int, resolution=2, freq_type: FreqType = "absolute"
    ):
        freq_width = 3 + resolution

        def fmt(freq):
            return f"{freq:{freq_width}.{resolution}f}"

        for ngram in self.iter_ngrams(ngrams_count):
            if freq_type == "absolute":
                freq_str = fmt(ngram.freq)
            elif freq_type == "cumulative":
                freq_str = fmt(ngram.cumulative_freq)
            elif freq_type == "both":
                cumulative_str = f"({ngram.cumulative_freq:.{resolution}f})".rjust(
                    freq_width + 2
                )
                freq_str = f"{fmt(ngram.freq)} {cumulative_str}"

            yield f"{ngram.rank:4} {ngram.chars} {freq_str}"

    def to_barplot(
        self, ngrams_count: int, title: str, freq_type: FreqType = "absolute"
    ):
        freqs = []
        ylabels = []
        cumfreqs = []
        width = 50
        for ngram in self.iter_ngrams(ngrams_count):
            freqs.append(ngram.freq)
            ylabels.append(f"{ngram.rank:4}: {ngram.chars}")
            cumfreqs.append(ngram.cumulative_freq)

        if freq_type == "both":
            plotext.simple_multiple_bar(
                ylabels,
                [freqs, cumfreqs],
                labels=["absolute", "cumulative"],
                width=width,
                title=title,
            )
        else:
            freqs = cumfreqs if freq_type == "cumulative" else freqs
            plotext.simple_bar(ylabels, freqs, width=width, title=title)
        plotext.show()

    def __len__(self):
        return len(self._ngrams)


def total_is_normalized(total: float) -> bool:
    return abs(total - 100.0) < 1e-4


def iter_lines(text, ignore_whitespace: bool = False):
    lines = text.split("\n")
    for line_ in lines:
        line = line_.lstrip()
        if not line:
            continue
        freq, ngram = line.split(" ", 1)
        if ignore_whitespace and any(char in WHITESPACE_CHARS for char in ngram):
            continue
        for char, new_char in CHAR_PRINT_MAPPING.items():
            ngram = ngram.replace(char, new_char)
        yield float(freq), ngram


def calculate_sum(text: str):
    total = 0
    for freq, _ in iter_lines(text):
        total += freq
    return total
