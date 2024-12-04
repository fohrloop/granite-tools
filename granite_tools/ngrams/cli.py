from __future__ import annotations

import re
import warnings
from enum import Enum
from functools import partial
from itertools import zip_longest
from pathlib import Path
from typing import Optional

import typer

try:
    from typing import Annotated
except ImportError:
    # For older python versions
    from typing_extensions import Annotated  # type: ignore

from granite_tools.ngrams.ngram import NgramList


def get_printable_for_ngrams(
    filename: Path,
    ngrams_count: int,
    ignore_case: bool,
    ignore_whitespace: bool,
    resolution: int,
    freq_type: FrequencyType,
    out: ResultType,
    raw: bool = False,
    exclude_chars: str = "",
    include_chars: Optional[str] = None,
):

    foldername = filename.parent.name
    ngrams = _get_ngramlist(
        filename,
        ignore_case=ignore_case,
        ignore_whitespace=ignore_whitespace,
        raw=raw,
        exclude_chars=exclude_chars,
        include_chars=include_chars,
    )
    if out == "plot":
        printable = ngrams.to_barplot(
            ngrams_count=ngrams_count,
            freq_type=freq_type,
            title=foldername,
        )
    else:
        printable = ngrams.to_printable(
            ngrams_count=ngrams_count,
            resolution=resolution,
            freq_type=freq_type,
            title=foldername,
            out=out,
        )
    return printable


class NgramSize(str, Enum):
    unigram = "1"
    bigram = "2"
    trigram = "3"
    all = "all"


class FrequencyType(str, Enum):
    absolute = "absolute"
    cumulative = "cumulative"
    both = "both"


class ResultType(str, Enum):
    table = "table"
    plot = "plot"
    plaintext = "plaintext"


class DiffResultType(str, Enum):
    table = "table"
    plot = "plot"


NGRAM_SIZE_MAP = {
    NgramSize.unigram: "1",
    NgramSize.bigram: "2",
    NgramSize.trigram: "3",
    NgramSize.all: "*",
}


def iter_ngram_files(folder: str | Path, ngram_size: NgramSize):
    size = NGRAM_SIZE_MAP[ngram_size]
    folder = Path(folder)
    for file in sorted(folder.glob(f"{size}-grams.txt")):
        if file.exists():
            yield file


DEFAULT_MAX_RANK = 40
ARG_NAME_NGRAM_SIZE = "--ngram-size"
ARG_NAME_NGRAM_SIZE_SHORT = "-s"

ARG_NGRAM_SRC = Annotated[
    Path,
    typer.Argument(
        help="Path to a folder of ngram files or to a single *-gram.txt file.",
        show_default=False,
    ),
]
ARG_NGRAMS_COUNT = Annotated[
    int,
    typer.Option(
        "--ngrams-count",
        "-n",
        help="The number of ngrams to show (most common first). To include all the ngrams, use 0.",
    ),
]

ARG_NGRAM_SIZE = Annotated[
    NgramSize,
    typer.Option(
        ARG_NAME_NGRAM_SIZE,
        ARG_NAME_NGRAM_SIZE_SHORT,
        help="Which ngram size to show.",
    ),
]
ARG_IGNORE_CASE = Annotated[
    bool,
    typer.Option(
        "--ignore-case",
        "-i",
        help="Ignore case when comparing ngrams (i.e. consider 'ab', 'aB', 'Ab', and 'AB' to be the same).",
    ),
]

ARG_IGNORE_WHITESPACE = Annotated[
    bool,
    typer.Option(
        "--ignore-whitespace",
        "-w",
        help="Ignore all ngrams which contain whitespace (i.e. Drop ngrams with whitespace).",
    ),
]

ARG_RESOLUTION = Annotated[
    int,
    typer.Option(
        "--resolution",
        help="The resolution for printed numbers. Example with resolution of 3: 0.234. This only affects the tabular form (not plots).",
    ),
]
ARG_FREQ_TYPE = Annotated[
    FrequencyType,
    typer.Option(
        "--freq",
        help="Type of frequency (ngram score) to show.",
    ),
]

ARG_OUT = Annotated[
    ResultType,
    typer.Option(
        "--type",
        help="The output presentation type. 'plot' draws a barplot. 'plaintext' is a plain text list. This format is supported by dariogoetz/keyboard_layout_optimizer. The 'table' shows a simple table.",
    ),
]

ARG_RAW = Annotated[
    bool,
    typer.Option(
        "--raw",
        help="Use raw values of the ngram frequencies/counts, instead of normalizing them.",
    ),
]

ARG_EXCLUDE_WITH_CHARS = Annotated[
    str,
    typer.Option(
        "--exclude-chars",
        help='Exclude ngrams which contain any of the given characters. Example: --exclude-chars "äö€". May not be used together with --include-chars.',
    ),
]

ARG_INCLUDE_WITH_CHARS = Annotated[
    Optional[str],
    typer.Option(
        "--include-chars",
        help='Include only ngrams which consist of the given characters. Example: --include-chars "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ". May not be used together with --exclude-chars.',
    ),
]


def show_ngrams(
    ngram_src: ARG_NGRAM_SRC,
    ngrams_count: ARG_NGRAMS_COUNT = DEFAULT_MAX_RANK,
    ngram_size: ARG_NGRAM_SIZE = NgramSize.all,
    ignore_case: ARG_IGNORE_CASE = False,
    ignore_whitespace: ARG_IGNORE_WHITESPACE = False,
    resolution: ARG_RESOLUTION = 2,
    freq_type: ARG_FREQ_TYPE = FrequencyType.absolute,
    out: ARG_OUT = ResultType.table,
    raw: ARG_RAW = False,
    exclude_chars: ARG_EXCLUDE_WITH_CHARS = "",
    include_chars: ARG_INCLUDE_WITH_CHARS = None,
):
    """Show ngrams from a folder or a file."""

    if exclude_chars and include_chars:
        raise typer.BadParameter("Cannot use --exclude-chars with --include-chars!")

    files = _get_file_iterator(ngram_src, ngram_size)

    for file in files:
        printable = get_printable_for_ngrams(
            file,
            ngrams_count=ngrams_count,
            ignore_case=ignore_case,
            ignore_whitespace=ignore_whitespace,
            resolution=resolution,
            freq_type=freq_type,
            out=out,
            raw=raw,
            exclude_chars=exclude_chars,
            include_chars=include_chars,
        )
        print(printable)


ARG_NGRAM_SRC_REF = Annotated[
    Path,
    typer.Argument(
        help="Path to a folder of ngram files or to a single *-gram.txt file. This is used as the reference.",
        show_default=False,
    ),
]
ARG_NGRAM_SRC_OTHER = Annotated[
    Path,
    typer.Argument(
        help="Path to a folder of ngram files or to a single *-gram.txt file. This is compared to the reference (=other)",
        show_default=False,
    ),
]

ARG_NGRAMS_COUNT_COMPARE = Annotated[
    int,
    typer.Option(
        "--ngrams-count",
        "-n",
        help="The number of ngrams to show (most common first). If using with --diff option, then this number of ngrams is taken from both corpora, and the union of the top ngrams is shown. To include all the ngrams, use 0.",
    ),
]

ARG_DIFF = Annotated[
    bool,
    typer.Option(
        "--diff",
        help="Show difference using first ngram source as reference.",
    ),
]

ARG_SWAP = Annotated[
    bool,
    typer.Option(
        "-S",
        "--swap",
        help='Swap "ref" and "other" input arguments (`ngram_src_ref` and `ngram_src_other`).',
    ),
]


def compare_ngrams(
    ngram_src_ref: ARG_NGRAM_SRC_REF,
    ngram_src_other: ARG_NGRAM_SRC_OTHER,
    ngrams_count: ARG_NGRAMS_COUNT_COMPARE = DEFAULT_MAX_RANK,
    ngram_size: ARG_NGRAM_SIZE = NgramSize.all,
    ignore_case: ARG_IGNORE_CASE = False,
    ignore_whitespace: ARG_IGNORE_WHITESPACE = False,
    resolution: ARG_RESOLUTION = 2,
    freq_type: ARG_FREQ_TYPE = FrequencyType.absolute,
    out: ARG_OUT = ResultType.table,
    raw: ARG_RAW = False,
    diff: ARG_DIFF = False,
    swap: ARG_SWAP = False,
):
    """Compare ngrams from two folders or files."""
    if swap:
        ngram_src_other, ngram_src_ref = ngram_src_ref, ngram_src_other

    if diff:
        compare_diff_ngrams(
            ngram_src_ref=ngram_src_ref,
            ngram_src_other=ngram_src_other,
            ngrams_count=ngrams_count,
            ngram_size=ngram_size,
            ignore_case=ignore_case,
            ignore_whitespace=ignore_whitespace,
            resolution=resolution,
            freq_type=freq_type,
            out=out,
            raw=raw,
        )
        return

    files_ref = _get_file_iterator(ngram_src_ref, ngram_size)
    files_other = _get_file_iterator(ngram_src_other, ngram_size)

    func = partial(
        get_printable_for_ngrams,
        ngrams_count=ngrams_count,
        ignore_case=ignore_case,
        ignore_whitespace=ignore_whitespace,
        resolution=resolution,
        freq_type=freq_type,
        out=out,
        raw=raw,
    )
    for file_ref, file_other in zip(files_ref, files_other):
        out_ref = func(file_ref)
        out_other = func(file_other)
        width = _get_width(plot=out == "plot", resolution=resolution)
        print_side_by_side(out_ref, out_other, size=width)


def compare_diff_ngrams(
    ngram_src_ref: ARG_NGRAM_SRC_REF,
    ngram_src_other: ARG_NGRAM_SRC_OTHER,
    ngrams_count: ARG_NGRAMS_COUNT = DEFAULT_MAX_RANK,
    ngram_size: ARG_NGRAM_SIZE = NgramSize.all,
    ignore_case: ARG_IGNORE_CASE = False,
    ignore_whitespace: ARG_IGNORE_WHITESPACE = False,
    resolution: ARG_RESOLUTION = 2,
    freq_type: ARG_FREQ_TYPE = FrequencyType.absolute,
    out: ARG_OUT = ResultType.table,
    raw: ARG_RAW = False,
):

    if freq_type != "absolute":
        raise typer.BadParameter(
            'Using other "--freq-type" than "absolute" with "--diff" is not supported. '
        )
    if out == "plaintext":
        raise typer.BadParameter(
            'Cannot use "plaintext" output format with "--diff" option.'
        )
    plot = out == "plot"

    files_ref = _get_file_iterator(ngram_src_ref, ngram_size)
    files_other = _get_file_iterator(ngram_src_other, ngram_size)
    width = _get_width(plot=plot, resolution=resolution)

    for file_ref, file_other in zip(files_ref, files_other):

        ngrams_ref = _get_ngramlist(
            file_ref,
            ignore_case=ignore_case,
            ignore_whitespace=ignore_whitespace,
            raw=raw,
        )
        ngrams_other = _get_ngramlist(
            file_other,
            ignore_case=ignore_case,
            ignore_whitespace=ignore_whitespace,
            raw=raw,
        )
        diff = ngrams_ref.diff(ngrams_other, n=ngrams_count)

        if plot:
            printable_ref, printable_other = diff.to_barplots(
                title_ref=file_ref.parent.name,
                title_other=file_other.parent.name,
                freq_type=freq_type,
            )
        else:
            printable_ref, printable_other = diff.to_printable(
                title_ref=file_ref.parent.name,
                title_other=file_other.parent.name,
                freq_type=freq_type,
            )
        print_side_by_side(printable_ref, printable_other, size=width)


def _get_width(plot: bool, resolution: int) -> int:
    return 50 if plot else 15 + resolution


def _get_ngramlist(
    filename: str | Path,
    ignore_case: bool,
    ignore_whitespace: bool,
    raw: bool = False,
    exclude_chars: str = "",
    include_chars: Optional[str] = None,
) -> NgramList:
    file_contents = Path(filename).read_text()

    with warnings.catch_warnings(record=True) as recorded_warnings:
        ngrams = NgramList(
            file_contents,
            ignore_case=ignore_case,
            ignore_whitespace=ignore_whitespace,
            normalize=not raw,
            exclude_chars=exclude_chars,
            include_chars=include_chars,
        )
    for w in recorded_warnings:
        print("WARNING:", w.message)
    return ngrams


def _make_title_row_narrower(title_line: str, size: int):
    dash = "\u2500"
    pattern = rf"([^{dash}]+)[{dash}]+ (.*?) [{dash}]+([^{dash}]+)"
    match = re.findall(pattern, title_line)
    if not match:
        return title_line
    start, name, end = match[0]
    return start + name.center(size, dash) + end


def print_side_by_side(left: str, right: str, size=40):
    lines1 = left.splitlines()
    lines2 = right.splitlines()
    lines1[0] = _make_title_row_narrower(lines1[0], size)
    lines2[0] = _make_title_row_narrower(lines2[0], size)

    side_by_side = [
        f"{line1}{_get_required_padding(line1, size)} {line2}"
        for line1, line2 in zip_longest(lines1, lines2, fillvalue="")
    ]

    print("\n".join(side_by_side))


def _get_required_padding(line: str, target_width: int, padchar=" "):
    printable_chars = remove_ansi_escape_sequences(line)
    padlength = max(0, target_width - len(printable_chars))
    return padchar * padlength


# Remove escape codes (used for coloring, etc.)
# From: https://stackoverflow.com/a/14693789/3015186
ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def remove_ansi_escape_sequences(string: str) -> str:
    return ansi_escape.sub("", string)


def _get_file_iterator(ngram_src: Path, ngram_size: NgramSize):
    if not ngram_src.exists():
        print(f"{ngram_src} does not exist!")
        raise typer.Exit(1)

    if ngram_src.is_file():
        if ngram_size != NgramSize.all:
            raise typer.BadParameter(
                f'Using "{ARG_NAME_NGRAM_SIZE}/{ARG_NAME_NGRAM_SIZE_SHORT}" when '
                "argument is a single file is not permitted. Either give a folder path "
                f'as an input, or remove the "{ARG_NAME_NGRAM_SIZE}/{ARG_NAME_NGRAM_SIZE_SHORT}" from arguments.',
            )
        files = iter((ngram_src,))
    else:
        files = iter_ngram_files(ngram_src, ngram_size=ngram_size)
    return files


def cli_show_ngrams():
    typer.run(show_ngrams)


def cli_compare_ngrams():
    typer.run(compare_ngrams)


if __name__ == "__main__":
    typer.run(show_ngrams)
ARG_NGRAMS_COUNT
