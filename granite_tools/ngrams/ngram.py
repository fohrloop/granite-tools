from __future__ import annotations

import io
import sys
import typing
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Literal

import plotext  # type:ignore

if typing.TYPE_CHECKING:
    from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

CHAR_PRINT_MAPPING = {" ": "␣", "\\n": "⏎", "\t": "⇥", "\\\\": "\\"}

WHITESPACE_CHARS = (" ", "\\n", "\t")
"""Whitespace characters (as in the original ngram files)"""

FreqType = Literal["absolute", "cumulative", "both"]
NgramListType = Literal["ref", "other"]
OutType = Literal["table", "plaintext"]


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


@dataclass
class PositionedNgramDiff:
    chars: str
    """The characters of an ngram."""

    freq: float | None
    """The "frequency" of the ngram. Unit is either percents (if the Ngrams
    used to calculate the diff were normalized), or arbitrary (otherwise).
    Value of None means that the `chars` were not present in the reference
    NgramList"""

    rank: int | None
    """The rank of the reference ngram (1=most frequent, and the larger the
    number is the less frequent the ngram is). Value of None means that the
    `chars` were not present in the reference NgramList"""

    freq_diff: float | None
    """The "frequency" difference of this and the reference ngram. Value of
    None means that the `chars` were not present in the "ref" or the "other"
    NgramList"""

    rank_diff: int | None
    """The rank difference between this and the reference ngram. Negative
    values mean thank this ngram is more frequent than the reference ngram.
    None means missing value (ngram was missing in one or the other list)."""


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
        exclude_chars: str = "",
        include_chars: Optional[str] = None,
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
        exclude_chars: str
            Exclude ngrams that contain any of the characters in the string.
            May not be used with include_chars. Default: "".
        include_chars: str | None
            If given, only include ngrams that consists of characters in
            this string. If None, all ngrams are included. May not be used with
            exclude_chars. Default: None.

        Example
        -------
        >>> ngramtext = "1 a\n2 b\n3 c"
        >>> ngrams = NgramList(ngramtext)

        """

        if normalize is None:
            normalize = NORMALIZE_DEFAULT

        self._ngrams: dict[str, Ngram] = dict()
        for freq, chars in iter_lines(
            ngramtext,
            ignore_whitespace=ignore_whitespace,
            ignore_case=ignore_case,
            exclude_chars=exclude_chars,
            include_chars=include_chars,
        ):

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

    def iter_ngrams(
        self, ngrams_count: int | None = None, subset: Set[str] | None = None
    ) -> Iterable[PositionedNgram]:
        total: float = 0
        if not ngrams_count:
            ngrams_count = sys.maxsize

        for rank, (chars, freq) in enumerate(self.iter_tuples(), start=1):
            if subset and chars not in subset:
                continue
            if rank > ngrams_count:
                break
            total += freq
            yield PositionedNgram(
                chars=chars, freq=freq, cumulative_freq=total, rank=rank
            )

    def iter_tuples(self) -> Iterable[tuple[str, float]]:
        """Iterates over the ngrams in the list, the most frequent ngram first,
        yielding a tuple of (chars, freq,)"""

        # The sorting is done first by frequency (descending) and then by the
        # characters in the ngram (ascending).
        for ngram in sorted(
            self._ngrams.values(), key=lambda x: (-x.freq, x.chars), reverse=False
        ):
            yield ngram.chars, ngram.freq

    def iter_txt(
        self, ngrams_count: int, resolution=2, freq_type: FreqType = "absolute"
    ):

        yield from iter_txt(
            ngrams=self.iter_ngrams(ngrams_count),
            resolution=resolution,
            freq_type=freq_type,
        )

    def to_printable(
        self,
        title: str,
        ngrams_count: int,
        resolution=2,
        freq_type: FreqType = "absolute",
        out: OutType = "table",
    ) -> str:
        iterable = self.iter_txt(
            ngrams_count=ngrams_count,
            resolution=resolution,
            freq_type=freq_type,
        )
        return to_printable(iterable, out, title)

    def to_barplot(
        self, ngrams_count: int, title: str, freq_type: FreqType = "absolute"
    ) -> str:
        freqs = []
        ylabels = []
        cumfreqs = []
        width = 50
        for ngram in self.iter_ngrams(ngrams_count):
            freqs.append(ngram.freq)
            ylabels.append(f"{ngram.rank:4}: {ngram.chars}")
            cumfreqs.append(ngram.cumulative_freq)

        return to_barplot(
            freqs=freqs,
            cumfreqs=cumfreqs,
            ylabels=ylabels,
            width=width,
            title=title,
            freq_type=freq_type,
        )

    def to_dict(
        self, n: int | None = None, subset: Set[str] | None = None
    ) -> Dict[str, PositionedNgram]:
        d = {}
        for ngram in self.iter_ngrams(ngrams_count=n, subset=subset):
            d[ngram.chars] = ngram
        return d

    def diff(self, other: NgramList, n: int) -> NgramDiffList:
        return NgramDiffList.from_ngrams(self, other, n)

    def __len__(self):
        return len(self._ngrams)


class NgramDiffList:
    """Represents a diff of two NgramLists (a difference between two corpora).

    Parameters
    ----------
    ref: list[PositionedNgramDiff]
        The ngrams from the reference NgramList (the "ref" corpus).
    other: list[PositionedNgramDiff]
        The ngrams from the other NgramList (the "other" corpus).
    n: int
        The number of ngrams used from both of the original NgramLists
        to form the diff. The number of ngrams in the diff is >= n.
    """

    def __init__(
        self, ref: list[PositionedNgramDiff], other: list[PositionedNgramDiff], n: int
    ):
        self.ref = sorted(
            ref, key=lambda x: x.rank if x.rank else float("inf"), reverse=False
        )
        self.other = sorted(
            other, key=lambda x: x.rank if x.rank else float("inf"), reverse=False
        )
        self.n = n

    @classmethod
    def from_ngrams(
        cls,
        ref: NgramList,
        other: NgramList,
        n: int,
    ):

        if other.normalized != ref.normalized:
            raise RuntimeError(
                "The two NgramLists have different `.normalized` value! "
                "Cannot calculate a meaningful diff."
            )
        ref_chars = set(ng.chars for ng in ref.iter_ngrams(n))
        other_chars = set(ng.chars for ng in other.iter_ngrams(n))
        chars_union = ref_chars | other_chars

        ngrams_ref = ref.to_dict(subset=chars_union)
        ngrams_other = other.to_dict(subset=chars_union)

        ref_pruned = []
        other_pruned = []
        for chars in chars_union:

            freq_other = ngrams_other[chars].freq if chars in ngrams_other else None
            freq_ref = ngrams_ref[chars].freq if chars in ngrams_ref else None
            rank_other = ngrams_other[chars].rank if chars in ngrams_other else None
            rank_ref = ngrams_ref[chars].rank if chars in ngrams_ref else None

            if freq_other is not None and freq_ref is not None:
                freq_diff = freq_other - freq_ref
            else:
                freq_diff = None

            if rank_other is not None and rank_ref is not None:
                rank_diff = rank_ref - rank_other
            else:
                rank_diff = None

            ref_pruned.append(
                PositionedNgramDiff(
                    chars=chars,
                    freq=freq_ref,
                    rank=rank_ref,
                    freq_diff=0,
                    rank_diff=0,
                )
            )
            other_pruned.append(
                PositionedNgramDiff(
                    chars=chars,
                    freq=freq_other,
                    rank=rank_other,
                    freq_diff=freq_diff,
                    rank_diff=rank_diff,
                )
            )

        return cls(ref_pruned, other_pruned, n=n)

    def iter_ngrams(
        self, which: NgramListType = "ref"
    ) -> Iterable[PositionedNgramDiff]:
        if which not in ("ref", "other"):
            raise ValueError("which must be either 'ref' or 'other'")
        diffs = self.ref if which == "ref" else self.other
        yield from iter(diffs)

    def to_barplots(
        self, title_ref: str, title_other: str, freq_type: FreqType = "absolute"
    ) -> Tuple[str, str]:

        return (
            self._to_barplot("ref", title=title_ref, freq_type=freq_type),
            self._to_barplot("other", title=title_other, freq_type=freq_type),
        )

    def _to_barplot(
        self,
        which: NgramListType,
        title: str,
        freq_type: FreqType = "absolute",
    ) -> str:

        freqs = []
        ylabels = []
        cumfreqs = []
        cumfreq: float = 0
        base_width = 50

        diffs = list(self.iter_ngrams(which))
        rank_width = get_required_rank_width(diffs)
        if which == "ref":
            width = base_width
        else:
            width = base_width + rank_width + 4  # 2 spaces and 1 "()""

        for diff in diffs:
            cumfreq += diff.freq if diff.freq is not None else 0
            freqs.append(diff.freq if diff.freq is not None else 0)
            if which == "ref":
                ylabels.append(f"{diff.rank or '???':{rank_width}}: {diff.chars}")
            else:
                rank_diff_str = (
                    f"{diff.rank_diff :+{rank_width}d}"
                    if diff.rank_diff is not None
                    else "???"
                )
                ylabels.append(
                    f"{diff.rank or '???':{rank_width}} ({rank_diff_str}): {diff.chars}"
                )
            cumfreqs.append(cumfreq)

        return to_barplot(
            freqs=freqs,
            cumfreqs=cumfreqs,
            ylabels=ylabels,
            width=width,
            title=title,
            freq_type=freq_type,
        )

    def to_printable(
        self,
        title_ref: str,
        title_other: str,
        resolution=2,
        freq_type: FreqType = "absolute",
    ) -> Tuple[str, str]:
        table_ref = to_table(
            self._iter_table_rows(
                which="ref",
                resolution=resolution,
                freq_type=freq_type,
            ),
            title=title_ref,
        )
        table_other = to_table(
            self._iter_table_rows(
                which="other",
                resolution=resolution,
                freq_type=freq_type,
            ),
            title=title_other,
        )
        return table_ref, table_other

    def _iter_table_rows(
        self,
        which: NgramListType = "ref",
        resolution=2,
        freq_type: FreqType = "absolute",
    ):

        yield from self.iter_txt(
            which,
            resolution=resolution,
            freq_type=freq_type,
        )

    def iter_txt(
        self,
        which: NgramListType = "ref",
        resolution=2,
        freq_type: FreqType = "absolute",
    ):
        freq_width = 3 + resolution

        diffs = list(self.iter_ngrams(which))
        rank_width = get_required_rank_width(diffs)

        def fmt(freq):
            if freq is None:
                return "???"
            return f"{freq:{freq_width}.{resolution}f}"

        for diff in diffs:
            if freq_type == "absolute":
                freq_str = fmt(diff.freq)
            elif freq_type == "cumulative":
                freq_str = fmt(diff.cumulative_freq)
            elif freq_type == "both":
                cumulative_str = f"({diff.cumulative_freq:.{resolution}f})".rjust(
                    freq_width + 2
                )
                freq_str = f"{fmt(diff.freq)} {cumulative_str}"

            if diff.rank is None:
                rank_str = "???".rjust(rank_width)
            else:
                rank_str = f"{diff.rank:{rank_width}}"
            if diff.rank_diff is None:
                rank_diff_str = "???".rjust(rank_width)
            else:
                rank_diff_str = f"{diff.rank_diff:+{rank_width}d}"

            if which == "ref":
                rank_chars = f"{rank_str}: {diff.chars}"
            else:
                rank_chars = f"{rank_str} ({rank_diff_str}): {diff.chars}"

            yield f"{rank_chars} {freq_str}"


def get_required_rank_width(diffs: list[PositionedNgramDiff]) -> int:
    max_abs_rank = max(
        abs(diff.rank_diff) for diff in diffs if diff.rank_diff is not None
    )
    return len(str(max_abs_rank)) + 1


def to_table(rows: Iterable[str], title: str):
    rows = list(rows)
    max_width = max(len(row) for row in rows)
    rows.insert(0, f"{title.center(max_width)}")
    rows.insert(1, "-" * (max_width))
    return "\n".join(rows)


def to_printable(iterable: Iterable[str], out: OutType, title: str) -> str:
    if out == "table":
        return to_table(
            iterable,
            title=title,
        )
    elif out == "plaintext":
        return to_plaintext(iterable)


def to_plaintext(iterable: Iterable[str]) -> str:
    out_str_parts = []
    for row in iterable:
        parts = row.split()
        if not parts:
            continue
        out_str_parts.append(f"{parts[2]} {parts[1]}")
    return "\n".join(out_str_parts)


def iter_txt(
    ngrams: Iterable[PositionedNgram],
    resolution=2,
    freq_type: FreqType = "absolute",
):
    freq_width = 3 + resolution

    def fmt(freq):
        return f"{freq:{freq_width}.{resolution}f}"

    for ngram in ngrams:
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
    freqs: Sequence[float],
    ylabels: Sequence[str],
    freq_type: FreqType,
    title: str,
    width: int = 50,
    cumfreqs: Sequence[float] | None = None,
) -> str:
    if cumfreqs is None and freq_type in ("both", "cumulative"):
        raise RuntimeError(f"cumfreqs is required when freq_type is {freq_type}")
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
    with redirect_stdout(io.StringIO()) as f:
        plotext.show()
        return f.getvalue()


def total_is_normalized(total: float) -> bool:
    return abs(total - 100.0) < 1e-4


def iter_lines(
    text,
    ignore_whitespace: bool = False,
    ignore_case: bool = False,
    exclude_chars: str = "",
    include_chars: Optional[str] = None,
):
    if include_chars and exclude_chars:
        raise ValueError("include_chars and exclude_chars cannot be used together.")

    include_chars = include_chars or ""

    lines = text.split("\n")

    if ignore_case:
        exclude_chars = exclude_chars.lower()
        include_chars = include_chars.lower()

    include_chars_set = set(include_chars)

    for line_ in lines:
        line = line_.lstrip()
        if not line:
            continue
        freq, ngram = line.split(" ", 1)

        skip_ngram = False

        if ignore_case:
            ngram = ngram.lower()

        for char in exclude_chars:
            if char in ngram:
                skip_ngram = True
                break
        if include_chars and not all(char in include_chars_set for char in ngram):
            skip_ngram = True

        if skip_ngram:
            continue

        if ignore_whitespace and any(char in ngram for char in WHITESPACE_CHARS):
            continue
        for char, new_char in CHAR_PRINT_MAPPING.items():
            ngram = ngram.replace(char, new_char)
        yield float(freq), ngram


def calculate_sum(text: str):
    total = 0
    for freq, _ in iter_lines(text):
        total += freq
    return total
