from __future__ import annotations

import typing

from granite_tools.bigram_compare.scorer import is_bigram

if typing.TYPE_CHECKING:
    KeySeq = tuple[int, ...]


def get_bigram_pairs(
    comparisons_all: list[tuple[KeySeq, KeySeq]],
) -> list[tuple[KeySeq, KeySeq]]:
    return [x for x in comparisons_all if is_bigram(x[0]) and is_bigram(x[1])]


def get_used_key_indices(comparisons: list[tuple[KeySeq, KeySeq]]) -> list[int]:
    """Get the indices of the keys that are used in the comparisons."""
    key_indices: set[int] = set()
    for pair in comparisons:
        for key_seq in pair:
            key_indices.update(key_seq)
    return sorted(key_indices)


# TODO: check if needed
def get_ordered_unigrams_and_repeats(unigram_scores: dict[int, float]):
    """Get the unigrams and repeats ordered by their score (lowest to highest)"""
    unigram_indices = get_ordered_keys(unigram_scores)
    unigrams = [(x,) for x in unigram_indices]
    repeats = [(x, x) for x in unigram_indices]
    return unigrams, repeats


def get_ordered_keys(dct):
    return sorted(dct, key=lambda x: dct[x], reverse=False)
