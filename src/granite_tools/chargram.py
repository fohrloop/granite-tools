import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

CHAR_PRINT_MAPPING = {" ": "␣", "\\n": "⏎", "\\\\": "\\"}


@dataclass
class Ngram:
    chars: str
    """The characters in the ngram."""

    freq: float
    """The "frequency" of the ngram. The unit may be arbitrary, but typically
    8.52 would mean 8.52% (if the ngram file has been normalized)"""


class NormalizationWarning(UserWarning):
    pass


class NgramList:

    _ngrams: dict[str, Ngram]
    """A list of Ngram objects."""

    def __init__(
        self, ngramtext: str, ignore_case: bool = False, normalize: bool = False
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
        normalize: bool
            If True, normalize the ngrams so that the sum of all frequencies is
            100.0. If False, and the input `ngramtext` is not normalized, a
            NormalizationWarning is issued. Default: False.

        Example
        -------
        >>> ngramtext = "1 a\n2 b\n3 c"
        >>> ngrams = NgramList(ngramtext)

        """

        self._ngrams = dict()
        for freq, chars in iter_lines(ngramtext):
            if ignore_case:
                chars = chars.lower()
            if chars in self._ngrams:
                self._ngrams[chars].freq += freq
            else:
                self._ngrams[chars] = Ngram(chars, freq)

        total = sum(ngram.freq for ngram in self)
        if not total_is_normalized(total):
            if normalize:
                for ngram in self:
                    ngram.freq = 100 * (ngram.freq / total)
            else:
                warnings.warn(
                    f"Total frequency is {total}. This is not normalized.",
                    NormalizationWarning,
                )

    def iter_tuples(self):
        """Iterates over the ngrams in the list, the most frequent ngram first,
        yielding a tuple of (freq, chars)"""
        for ngram in self:
            yield ngram.freq, ngram.chars

    def iter_txt(self):
        for freq, chars in self.iter_tuples():
            yield f"{freq} {chars}"

    def __iter__(self):
        return iter(sorted(self._ngrams.values(), key=lambda x: x.freq, reverse=True))

    def __len__(self):
        return len(self._ngrams)


def total_is_normalized(total: float) -> bool:
    return abs(total - 100.0) < 1e-6


def iter_lines(text):
    lines = text.split("\n")
    for line_ in lines:
        line = line_.lstrip()
        if not line:
            continue
        freq, ngram = line.split(" ", 1)
        for char, new_char in CHAR_PRINT_MAPPING.items():
            ngram = ngram.replace(char, new_char)
        yield float(freq), ngram


def calculate_sum(text: str):
    total = 0
    for freq, _ in iter_lines(text):
        total += freq
    return total


if __name__ == "__main__":
    filename = sys.argv[1]
    file_contents = Path(filename).read_text()
    with warnings.catch_warnings(record=True) as recorded_warnings:
        ngrams = NgramList(file_contents)
    for w in recorded_warnings:
        print("WARNING:", w.message)
    for rank, ngram in enumerate(ngrams, start=1):
        print(f"{rank}. {ngram.chars} {ngram.freq}")
