from __future__ import annotations

import warnings
from enum import Enum
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
        for text in ngrams.iter_txt(
            ngrams_count=ngrams_count,
            resolution=resolution,
            freq_type=freq_type,
        ):
            print(text)


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


def show_ngrams(
    ngram_src: Annotated[
        Path,
        typer.Argument(
            help="Path to a folder of ngram files or to a single *-gram.txt file.",
            show_default=False,
        ),
    ],
    ngrams_count: Annotated[
        int,
        typer.Option(
            "--ngrams-count",
            "-n",
            help="The number of ngrams to show (most common first).",
        ),
    ] = DEFAULT_MAX_RANK,
    ngram_size: Annotated[
        NgramSize,
        typer.Option(
            ARG_NAME_NGRAM_SIZE,
            ARG_NAME_NGRAM_SIZE_SHORT,
            help="Which ngram size to show.",
        ),
    ] = NgramSize.all,
    ignore_case: Annotated[
        bool,
        typer.Option(
            "--ignore-case",
            "-i",
            help="Ignore case when comparing ngrams (i.e. consider 'ab', 'aB', 'Ab', and 'AB' to be the same).",
        ),
    ] = False,
    ignore_whitespace: Annotated[
        bool,
        typer.Option(
            "--ignore-whitespace",
            "-w",
            help="Ignore all ngrams which contain whitespace (i.e. Drop ngrams with whitespace).",
        ),
    ] = False,
    resolution: Annotated[
        int,
        typer.Option(
            "--resolution",
            help="The resolution for printed numbers. Example with resolution of 3: 0.234. This only affects the tabular form (not plots).",
        ),
    ] = 2,
    freq_type: Annotated[
        FrequencyType,
        typer.Option(
            "--type",
            help="Type of frequency (ngram score) to show.",
        ),
    ] = FrequencyType.absolute,
    plot: Annotated[
        bool,
        typer.Option(
            "--plot",
            help="Draw a barplot instead of showing a table.",
        ),
    ] = False,
    raw: Annotated[
        bool,
        typer.Option(
            "--raw",
            help="Use raw values of the ngram frequencies/counts, instead of normalizing them.",
        ),
    ] = False,
):
    """Show ngrams from a folder or a file."""

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


def cli_show_ngrams():
    typer.run(show_ngrams)


if __name__ == "__main__":
    typer.run(show_ngrams)
