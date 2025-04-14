from __future__ import annotations

import itertools
import typing

if typing.TYPE_CHECKING:
    from typing import Iterable

    from granite_tools.hands import Hand


def create_permutations(
    left: Hand, right: Hand, sequence_lengths: tuple[int, ...] = (2,)
) -> list[tuple[int, ...]]:
    """This creates permutations of given sequence lengths that can be typed with at
    least one of the hands. The returned list contains tuples of key indices."""
    key_indices = get_union_of_keys(left, right)

    permutations_lst = []
    for seq in iterate_permutations(key_indices, sequence_lengths):
        if not permutation_is_typable(left, right, seq):
            continue
        permutations_lst.append(seq)

    return permutations_lst


def iterate_permutations(
    key_indices: list[int], sequence_lengths: tuple[int, ...] = (2,)
) -> Iterable[tuple[int, ...]]:
    for seq_length in sequence_lengths:
        for seq in itertools.product(key_indices, repeat=seq_length):
            yield seq


def permutation_is_typable(
    left: Hand, right: Hand, permutation: tuple[int, ...]
) -> bool:
    """Check if permutation is typable with at least one hand"""
    for hand in (left, right):
        can_be_typed = True

        for key in permutation:
            if key not in hand.symbols_visualization:
                can_be_typed = False
                break

        if can_be_typed:
            return True

    return False


def get_union_of_keys(left: Hand, right: Hand) -> list[int]:
    """Gets the union of key indices from both hands."""
    return sorted(set(left.symbols_visualization.keys()) | set(right.symbols_visualization.keys()))  # type: ignore
