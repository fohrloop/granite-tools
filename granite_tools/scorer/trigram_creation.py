from __future__ import annotations

import random
import re
import typing
from pathlib import Path

from granite_tools.app_types import HandType
from granite_tools.hands import get_hands_data

if typing.TYPE_CHECKING:

    from granite_tools.config import Config
    from granite_tools.hands import Hand, Hands


def create_score_template_(n_trigram_sets: int, config: Config, outfile: Path):
    """Creates a trigram scoring template.

    Parameters
    ----------
    n_trigram_sets: int
        The number of trigram sets to create. Each trigram set is a subset of a trigram
        family and contains 4 trigrams
    config: Config
        The keyboard configuration.
    """
    hands = get_hands_data(config)

    all_indices = hands.get_common_indices()
    trigram_sets: set[tuple[int, int, int]] = set()

    while len(trigram_sets) < n_trigram_sets:
        indices = random.sample(all_indices, 3)
        if len({hands.left.get_finger(x) for x in indices}) < 3:
            # Skip trigrams with SFB or SFT
            continue
        trigram_sets.add(typing.cast(tuple[int, int, int], tuple(sorted(indices))))

    generated_trigrams = []
    for trigram_set in trigram_sets:
        trigram, trigram_indices, handtype = create_random_onehand_trigram(
            trigram_set, hands
        )
        main_hand = hands.left if handtype == "Left" else hands.right
        other_hand = hands.right if handtype == "Left" else hands.left
        redir = create_random_redir_trigram(trigram_indices, main_hand)
        balanced = create_random_balanced_trigram(
            trigram_indices, main_hand, other_hand
        )
        skipgram = create_random_skipgram_trigram(
            trigram_indices, main_hand, other_hand
        )
        generated_trigrams.append([trigram, balanced, skipgram, redir])

    old_scores = _read_reference_trigrams(outfile)
    if Path(outfile).exists():
        with open(outfile, "r") as f:
            text_old = f.read().rstrip()
            if text_old:
                text_old += "\n"
    else:
        text_old = ""

    with open(outfile, "w") as f:
        if text_old:
            f.write(text_old)
        for trigrams in generated_trigrams:
            if old_scores and trigrams[0] in old_scores:
                print(f"Skipping reference trigram {trigrams[0]} as it already exists.")
                continue
            for i, trigram in enumerate(trigrams):
                if i % 4 == 0:
                    f.write(f'["{trigram}"]\n')
                else:
                    f.write(f'"{trigram}" = \n')
            f.write("\n")

    with open(outfile, "r") as f:
        print(f"The file {outfile} has now {len(f.readlines())} lines.")


def _read_reference_trigrams(scorefile: str | Path) -> list[str]:
    if not Path(scorefile).exists():
        return []
    out = []
    with open(scorefile, "r") as f:
        for line in f:
            ref = _get_reference_trigram(line)
            if not ref:
                continue
            out.append(ref)
    return out


def _get_reference_trigram(line: str) -> str:
    match = re.match(r"\[(.*)\]\s*", line)
    if match is None:
        return ""
    ref = match.group(1)
    if len(ref) == 5:
        if ref.startswith('"') and ref.endswith('"'):
            return ref[1:-1]
        if ref.startswith("'") and ref.endswith("'"):
            return ref[1:-1]
    assert len(ref) == 3
    return ref


def create_random_onehand_trigram(
    indices: tuple[int, int, int], hands: Hands
) -> tuple[str, tuple[int, int, int], HandType]:
    rolldirection = random.choice(["in", "out"])
    side = typing.cast(HandType, random.choice(["Left", "Right"]))
    return *create_onehand_trigram(indices, rolldirection, side, hands), side


def create_onehand_trigram(
    indices: tuple[int, int, int],
    rolldirection: str,
    side: HandType,
    hands: Hands,
) -> tuple[str, tuple[int, int, int]]:

    if side == "Left":
        hand = hands.left
    elif side == "Right":
        hand = hands.right
    else:
        raise ValueError("Unknown side")

    if rolldirection not in ("in", "out"):
        raise ValueError("Unknown rolldirection: must be 'in' or 'out'")

    left_to_right = (rolldirection == "in" and side == "Left") or (
        rolldirection == "out" and side == "Right"
    )
    indices_out: list[int] = sorted(
        indices, key=lambda x: hand.matrix_positions[x], reverse=not left_to_right
    )

    return hand.get_symbols_visualization(indices_out), typing.cast(
        tuple[int, int, int], tuple(indices_out)
    )


def create_random_redir_trigram(
    onehand_indices: tuple[int, int, int], hand: Hand
) -> str:
    """
    Parameters
    ----------
    onehand_indices:
        The indices of the onehand trigram. These should be ordered (as rolling inwards
        or outwards). These will not be checked for validity."""
    start_index = random.choice([1, 2])
    indices = create_redir_trigram(onehand_indices, start_index)
    return hand.get_symbols_visualization(indices)


def create_redir_trigram(
    onehand_indices: tuple[int, int, int], start_index: int
) -> tuple[int, int, int]:
    if start_index not in (1, 2):
        raise ValueError("start_index must be 1 or 2")
    if start_index == 1:
        return (onehand_indices[1], onehand_indices[2], onehand_indices[0])
    return (onehand_indices[2], onehand_indices[0], onehand_indices[1])


def create_random_balanced_trigram(
    onehand_indices: tuple[int, int, int],
    hand_main: Hand,
    hand_other: Hand,
) -> str:
    """
    Parameters
    ----------
    onehand_indices:
        The indices of the onehand trigram. These should be ordered (as rolling inwards
        or outwards). These will not be checked for validity."""
    idx_from_other_hand = random.choice([0, 2])
    return create_balanced_trigram(
        onehand_indices, hand_main, hand_other, idx_from_other_hand
    )


def create_balanced_trigram(
    onehand_indices: tuple[int, int, int],
    hand_main: Hand,
    hand_other: Hand,
    idx_from_other_hand: int,
) -> str:
    if idx_from_other_hand not in (0, 2):
        raise ValueError("idx_from_other_hand must be 0 or 2")
    indices_main = (
        (onehand_indices[1], onehand_indices[2])
        if idx_from_other_hand == 0
        else (onehand_indices[0], onehand_indices[1])
    )

    if idx_from_other_hand == 0:
        return hand_other.get_symbols_visualization(
            (onehand_indices[0],)
        ) + hand_main.get_symbols_visualization(indices_main)
    return hand_main.get_symbols_visualization(
        indices_main
    ) + hand_other.get_symbols_visualization((onehand_indices[idx_from_other_hand],))


def create_random_skipgram_trigram(
    onehand_indices: tuple[int, int, int],
    hand_main: Hand,
    hand_other: Hand,
) -> str:
    """
    Parameters
    ----------
    onehand_indices:
        The indices of the onehand trigram. These should be ordered (as rolling inwards
        or outwards). These will not be checked for validity."""
    idx_from_other_hand = random.choice([0, 1, 2])
    return create_skipgram_trigram(
        onehand_indices, hand_main, hand_other, idx_from_other_hand
    )


def create_skipgram_trigram(
    onehand_indices: tuple[int, int, int],
    hand_main: Hand,
    hand_other: Hand,
    idx_from_other_hand: int,
) -> str:
    if idx_from_other_hand not in (0, 1, 2):
        raise ValueError("idx_from_other_hand must be 0, 1 or 2")
    idx_other = (onehand_indices[idx_from_other_hand],)

    if idx_from_other_hand == 0:
        idx_main = (onehand_indices[1], onehand_indices[2])
    elif idx_from_other_hand == 1:
        idx_main = (onehand_indices[0], onehand_indices[2])
    else:
        idx_main = (onehand_indices[0], onehand_indices[1])

    return (
        hand_main.get_symbols_visualization((idx_main[0],))
        + hand_other.get_symbols_visualization(idx_other)
        + hand_main.get_symbols_visualization((idx_main[1],))
    )
