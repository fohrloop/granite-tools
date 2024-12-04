"""
Running:

uv run granite_tools/scripts/get_low_bigram_sum_trigrams.py <config_file> <bigram_ranking_file> <trigram_scoring_file>

Example:

uv run granite_tools/scripts/get_low_bigram_sum_trigrams.py examples/keyseq_effort.yml tmp/efforts.compare.ranking tmp/trigram.relative.scoring
"""

from __future__ import annotations

import random
import sys
import typing

from granite_tools.app_types import HandType
from granite_tools.config import read_config
from granite_tools.hands import get_hands_data
from granite_tools.permutations import create_permutations
from granite_tools.scorer.scorer import (
    load_bigram_and_unigram_scores,
    load_trigram_scores,
)
from granite_tools.scorer.trigram_creation import (
    create_onehand_trigram,
    create_random_balanced_trigram,
    create_random_redir_trigram,
    create_random_skipgram_trigram,
)

if typing.TYPE_CHECKING:
    from granite_tools.hands import Hands

TRESHOLD_REF_BIGRAM_SUM = 3.75


def get_three_finger_trigrams(hands: Hands) -> list[tuple[int, int, int]]:
    # Get all trigrams
    all_trigrams = create_permutations(hands.left, hands.right, sequence_lengths=(3,))
    # Get only those trigram which are typed with three different fingers (skip SFB & SFT)
    three_finger_trigrams: list[tuple[int, int, int]] = []
    for trigram in all_trigrams:
        if len({hands.left.get_finger(x) for x in trigram}) < 3:
            continue
        three_finger_trigrams.append(
            typing.cast(tuple[int, int, int], tuple(sorted(trigram)))
        )

    return three_finger_trigrams


def get_onehand_trigrams(
    hands: Hands, side: HandType = "Left"
) -> list[tuple[int, int, int]]:
    """Gets all unique the onehand type of trigrams for a single hand"""

    three_finger_trigrams = get_three_finger_trigrams(hands)
    onehand_trigrams = set()
    for trigram in three_finger_trigrams:
        for rolldirection in ("in", "out"):
            _, onehand_indices = create_onehand_trigram(
                trigram, rolldirection, side=side, hands=hands
            )
            onehand_trigrams.add(onehand_indices)
    return sorted(onehand_trigrams)


if __name__ == "__main__":
    random.seed(83582)
    config_file = sys.argv[1]
    bigram_ranking_file = sys.argv[2]
    trigram_scoring_file = sys.argv[3]

    config = read_config(config_file)
    hands = get_hands_data(config)

    bigram_scores = load_bigram_and_unigram_scores(bigram_ranking_file)
    trigram_scores = load_trigram_scores(trigram_scoring_file)

    onehand_trigrams = get_onehand_trigrams(hands, side="Left")
    matching_trigrams = []
    bigram_sums = []
    for trigram in onehand_trigrams:
        bigram1 = bigram_scores[(trigram[0], trigram[1])]
        bigram2 = bigram_scores[(trigram[1], trigram[2])]
        bigram_sum = bigram1 + bigram2
        if bigram_sum >= TRESHOLD_REF_BIGRAM_SUM:
            continue
        bigram_sums.append(bigram_sum)
        matching_trigrams.append(trigram)

    print("Number of all onehand trigrams:", len(onehand_trigrams))

    selected_trigrams = []
    for bigram_sum, trigram in zip(bigram_sums, matching_trigrams):
        trigram_left = hands.get_symbols_visualization("Left", trigram)
        trigram_right = hands.get_symbols_visualization("Right", trigram)
        if trigram_left in trigram_scores or trigram_right in trigram_scores:
            continue
        print(
            f"{trigram_left} (={trigram_right}), score={bigram_sum:.2f}, indices={trigram}"
        )
        selected_trigrams.append(trigram)
    print(
        f"Number of onehand trigrams with bigram_sum < {TRESHOLD_REF_BIGRAM_SUM}:",
        len(matching_trigrams),
    )
    print("Number of trigram families not yet scored:", len(selected_trigrams))
    print("\n\n")

    generated_trigrams: list[str] = []
    for trigram_indices in selected_trigrams:
        handtype = random.choice(["Left", "Right"])
        main_hand = hands.left if handtype == "Left" else hands.right
        other_hand = hands.right if handtype == "Left" else hands.left

        onehand = main_hand.get_symbols_visualization(trigram_indices)
        redir = create_random_redir_trigram(trigram_indices, main_hand)
        balanced = create_random_balanced_trigram(
            trigram_indices, main_hand, other_hand
        )
        skipgram = create_random_skipgram_trigram(
            trigram_indices, main_hand, other_hand
        )
        generated_trigrams.extend([onehand, balanced, skipgram, redir])

    for i, trigram_str in enumerate(generated_trigrams):
        score = "1.0" if i % 4 == 0 else ""  # give vanilla onehands a score of 1.0
        print(trigram_str + f" = {score}")
