"""Used to create estimate for ngram ranking scores using .compare.pickle file from the
granite-bigram-compare application.


Usage: python granite_tools/scripts/create_ngram_ranking.py <some.compare.pickle>

NOTE: This saves the ranking to a file with the same name as the input file, but with
the extension changed to .ranking (any file with the same name will be overwritten).
"""

import pickle
import sys
from enum import Enum
from pathlib import Path

from granite_tools.bigram_compare.fitting import get_scores
from granite_tools.bigram_compare.scorer import is_bigram, is_repeat, is_unigram
from granite_tools.comparison_data import (
    get_bigram_pairs,
    get_ordered_keys,
    get_ordered_unigrams_and_repeats,
    get_used_key_indices,
)
from granite_tools.unigram_scores import calculate_unigram_scores

KeySeq = tuple[int, ...]
Unigram = tuple[int]
Bigram = tuple[int, int]


class NgramType(Enum):
    UNIGRAM = 1
    REPEAT = 2
    BIGRAM = 3


def get_combined_order(
    unigrams: list[Unigram],
    repeats: list[Bigram],
    bigrams: list[Bigram],
    ngram_type_order: list[NgramType],
):

    combined_data: list[KeySeq] = []
    for ngram_type in ngram_type_order:
        if ngram_type == NgramType.UNIGRAM:
            if not unigrams:
                raise RuntimeError("Not enough unigrams")
            combined_data.append(unigrams.pop(0))
        elif ngram_type == NgramType.REPEAT:
            if not repeats:
                raise RuntimeError("Not enough repeats")
            combined_data.append(repeats.pop(0))
        elif ngram_type == NgramType.BIGRAM:
            if not bigrams:
                raise RuntimeError("Not enough bigrams")
            combined_data.append(bigrams.pop(0))
    if unigrams or repeats or bigrams:
        raise RuntimeError("Too many ngrams (ngram_type_order too short)")
    return combined_data


def get_order_for_types(ordered_ngrams: list[KeySeq]) -> list[NgramType]:
    ngram_types = []
    for ngram in ordered_ngrams:
        if is_unigram(ngram):
            ngram_types.append(NgramType.UNIGRAM)
        elif is_repeat(ngram):
            ngram_types.append(NgramType.REPEAT)
        elif is_bigram(ngram):
            ngram_types.append(NgramType.BIGRAM)
        else:
            raise ValueError(f"Unknown ngram type: {ngram}")
    return ngram_types


def create_ngram_ranking(comparisons_all: list[tuple[KeySeq, KeySeq]]) -> list[KeySeq]:
    bigram_pairs = get_bigram_pairs(comparisons_all)

    bigram_scores = get_scores(bigram_pairs)
    used_key_indices = get_used_key_indices(bigram_pairs)
    unigram_scores = calculate_unigram_scores(bigram_scores, used_key_indices)

    unigrams_ordered, repeats_ordered = get_ordered_unigrams_and_repeats(unigram_scores)
    bigrams_ordered = get_ordered_keys(bigram_scores)

    scores_all = get_scores(comparisons_all)
    ngrams_ordered_all = get_ordered_keys(scores_all)
    ngram_types = get_order_for_types(ngrams_ordered_all)

    ngram_ranking = get_combined_order(
        unigrams_ordered, repeats_ordered, bigrams_ordered, ngram_types
    )
    return ngram_ranking


def save_ranking_to_file(file, ngram_ranking):
    with open(file, "w") as f:
        for ngram in ngram_ranking:
            f.write(",".join(map(str, ngram)) + "\n")


if __name__ == "__main__":

    compare_file = sys.argv[1]
    with open(compare_file, "rb") as f:
        data = pickle.load(f)

    comparisons_all = data["comparisons_all"]
    ngram_ranking = create_ngram_ranking(comparisons_all)

    if compare_file.endswith(".compare.pickle"):
        outfile = Path(compare_file[:-15] + ".ranking")
    else:
        outfile = Path(compare_file).with_suffix(".ranking")

    if outfile.exists():
        print(f"ERROR: {outfile} already exists! Aborting")
        sys.exit(1)

    print("Saving to", outfile)
    save_ranking_to_file(outfile, ngram_ranking)
