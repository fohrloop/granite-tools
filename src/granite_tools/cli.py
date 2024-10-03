from __future__ import annotations

import io
import re
import warnings
from contextlib import redirect_stdout
from enum import Enum
from functools import partial
from itertools import zip_longest
from pathlib import Path

import typer

try:
    from typing import Annotated
except ImportError:
    # For older python versions
    from typing_extensions import Annotated  # type: ignore

from granite_tools.chargram import NgramList


def show_ngrams_for_file(
    filename: str | Path,
    ngrams_count: int,
    ignore_case: bool,
    ignore_whitespace: bool,
    resolution: int,
    freq_type: FrequencyType,
    plot: bool = False,
    raw: bool = False,
):
    file_contents = Path(filename).read_text()
    foldername = filename.parent.name
    with warnings.catch_warnings(record=True) as recorded_warnings:
        ngrams = NgramList(
            file_contents,
            ignore_case=ignore_case,
            ignore_whitespace=ignore_whitespace,
            normalize=not raw,
        )
    for w in recorded_warnings:
        print("WARNING:", w.message)

    if plot:
        ngrams.to_barplot(
            ngrams_count=ngrams_count,
            freq_type=freq_type,
            title=foldername,
        )
    else:
        lines = list(
            ngrams.iter_txt(
                ngrams_count=ngrams_count,
                resolution=resolution,
                freq_type=freq_type,
            )
        )
        max_width = max(len(l) for l in lines)
        print(" " * 3 + foldername)
        print(" " * 3 + "-" * (max_width - 3))
        print("\n".join(lines))


class NgramSize(str, Enum):
    unigram = "1"
    bigram = "2"
    trigram = "3"
    all = "all"


class FrequencyType(str, Enum):
    absolute = "absolute"
    cumulative = "cumulative"
    both = "both"


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
        help="The number of ngrams to show (most common first).",
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
        "--type",
        help="Type of frequency (ngram score) to show.",
    ),
]

ARG_PLOT = Annotated[
    bool,
    typer.Option(
        "--plot",
        help="Draw a barplot instead of showing a table.",
    ),
]

ARG_RAW = Annotated[
    bool,
    typer.Option(
        "--raw",
        help="Use raw values of the ngram frequencies/counts, instead of normalizing them.",
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
    plot: ARG_PLOT = False,
    raw: ARG_RAW = False,
):
    """Show ngrams from a folder or a file."""
    files = _get_file_iterator(ngram_src, ngram_size)

    for file in files:
        show_ngrams_for_file(
            file,
            ngrams_count=ngrams_count,
            ignore_case=ignore_case,
            ignore_whitespace=ignore_whitespace,
            resolution=resolution,
            freq_type=freq_type,
            plot=plot,
            raw=raw,
        )


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


def compare_ngrams(
    ngram_src_ref: ARG_NGRAM_SRC_REF,
    ngram_src_other: ARG_NGRAM_SRC_OTHER,
    ngrams_count: ARG_NGRAMS_COUNT = DEFAULT_MAX_RANK,
    ngram_size: ARG_NGRAM_SIZE = NgramSize.all,
    ignore_case: ARG_IGNORE_CASE = False,
    ignore_whitespace: ARG_IGNORE_WHITESPACE = False,
    resolution: ARG_RESOLUTION = 2,
    freq_type: ARG_FREQ_TYPE = FrequencyType.absolute,
    plot: ARG_PLOT = False,
    raw: ARG_RAW = False,
):
    """Compare ngrams from two folders or files."""
    files_ref = _get_file_iterator(ngram_src_ref, ngram_size)
    files_other = _get_file_iterator(ngram_src_other, ngram_size)

    func = partial(
        show_ngrams_for_file,
        ngrams_count=ngrams_count,
        ignore_case=ignore_case,
        ignore_whitespace=ignore_whitespace,
        resolution=resolution,
        freq_type=freq_type,
        plot=plot,
        raw=raw,
    )
    for file_ref, file_other in zip(files_ref, files_other):
        with redirect_stdout(io.StringIO()) as f:
            func(file_ref)
            out_ref = f.getvalue()
        with redirect_stdout(io.StringIO()) as f:
            func(file_other)
            out_other = f.getvalue()

        width = 50 if plot else 15 + resolution
        print_side_by_side(out_ref, out_other, size=width)


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


# Remove escape codes (used for coloring, etc.)
# From: https://stackoverflow.com/a/14693789/3015186
ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _get_required_padding(line: str, target_width: int, padchar=" "):
    printable_chars = remove_ansi_escape_sequences(line)
    padlength = max(0, target_width - len(printable_chars))
    return padchar * padlength


def remove_ansi_escape_sequences(string: str) -> str:
    return ansi_escape.sub("", string)


def _get_file_iterator(ngram_src: Path, ngram_size: NgramSize):
    if not ngram_src.exists():
        raise typer.Exit(f"{ngram_src} does not exist!")

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
