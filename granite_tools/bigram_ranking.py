import pickle
import sys
import warnings
from pathlib import Path

import typer

from granite_tools.app_types import KeySeq
from granite_tools.bigram_compare.fitting import get_scores
from granite_tools.bigram_compare.scorer import is_bigram, is_repeat, is_unigram
from granite_tools.comparison_data import get_bigram_pairs, get_ordered_keys

try:
    from typing import Annotated
except ImportError:
    # For older python versions
    from typing_extensions import Annotated  # type: ignore


ARG_NGRAM_COMPARE_FILE = Annotated[
    Path,
    typer.Argument(
        help="The path to the bigram.compare.pickle file (from: granite-bigram-compare). NOTE: The output file will have the same name but with .ranking extension.",
        show_default=False,
    ),
]


class NotBigramWarning(UserWarning):
    """Used to warn when a ngram in comparison is not a (non-repeating) bigram."""


def create_bigram_ranking_cli():
    typer.run(create_bigram_ranking_cli_)


def create_bigram_ranking_cli_(
    bigram_compare_file: ARG_NGRAM_COMPARE_FILE,
):
    """Used to create estimate for ngram ranking scores using .compare.pickle file from the
    granite-bigram-cfompare application."""

    with open(bigram_compare_file, "rb") as f:
        data = pickle.load(f)

    comparisons_all = data["comparisons_all"]
    ngram_ranking = create_ngram_ranking(comparisons_all)

    if bigram_compare_file.endswith(".compare.pickle"):
        outfile = Path(bigram_compare_file[:-15] + ".ranking")
    else:
        outfile = Path(bigram_compare_file).with_suffix(".ranking")

    if outfile.exists():
        print(f"ERROR: {outfile} already exists! Aborting")
        sys.exit(1)

    print("Saving to", outfile)
    save_ranking_to_file(outfile, ngram_ranking)


def create_ngram_ranking(comparisons_all: list[tuple[KeySeq, KeySeq]]) -> list[KeySeq]:
    for pair in comparisons_all:
        for keyseq in pair:
            if is_repeat(keyseq):
                warnings.warn(
                    f"One of the ngrams, {keyseq}, is a repeat. It will be ignored. Comparison pair: {pair}",
                    NotBigramWarning,
                    stacklevel=2,
                )
            if is_unigram(keyseq):
                warnings.warn(
                    f"One of the ngrams, {keyseq}, is a unigram. It will be ignored. Comparison pair: {pair}",
                    NotBigramWarning,
                    stacklevel=2,
                )
    bigram_pairs = get_bigram_pairs(comparisons_all)
    bigram_scores = get_scores(bigram_pairs)
    bigrams_ordered = get_ordered_keys(bigram_scores)
    return bigrams_ordered


def save_ranking_to_file(file, ngram_ranking):
    with open(file, "w") as f:
        for ngram in ngram_ranking:
            f.write(",".join(map(str, ngram)) + "\n")
