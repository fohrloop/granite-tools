from __future__ import annotations

import itertools
import random
import sys
import typing
from pathlib import Path

import typer

from granite_tools.bigram_scores.rankings import load_bigram_rankings
from granite_tools.bigram_scores.score_ratio_template import (
    save_score_ratios,
    select_every_nth_item,
)
from granite_tools.config import read_config
from granite_tools.hands import get_hands_data

try:
    from typing import Annotated
except ImportError:
    # For older python versions
    from typing_extensions import Annotated  # type: ignore

if typing.TYPE_CHECKING:
    from typing import Sequence

    from granite_tools.app_types import KeySeq
    from granite_tools.hands import Hand, Hands


def create_bigram_score_ratio_template():
    typer.run(create_bigram_score_ratio_template_)


ARG_CONFIG = Annotated[
    Path,
    typer.Argument(
        help="The path to the Granite configuration (YAML) file.",
        show_default=False,
    ),
]

ARG_NGRAM_RANKING_FILE = Annotated[
    Path,
    typer.Argument(
        help="The path to the input ngram ranking file.",
        show_default=False,
    ),
]

ARG_NGRAM_TEMPLATE_OUTFILE = Annotated[
    Path,
    typer.Argument(
        help="The path to the output (ngram score ratio) file.",
        show_default=False,
    ),
]

ARG_SCORE_RATIO_NGRAM_GAP = Annotated[
    int,
    typer.Argument(
        help="The 'n' for selecting every 'nth' ngram (optional).",
        show_default=True,
    ),
]


def create_bigram_score_ratio_template_(
    ranking_file: ARG_NGRAM_RANKING_FILE,
    config_file: ARG_CONFIG,
    outfile: ARG_NGRAM_TEMPLATE_OUTFILE,
    n: ARG_SCORE_RATIO_NGRAM_GAP = 8,
):
    if outfile.exists():
        print(f"Output file {outfile} already exists. Aborting.")
        sys.exit(1)

    config = read_config(config_file)
    bigram_rankings = load_bigram_rankings(ranking_file)
    hands = get_hands_data(config)

    selected_key_sequences = select_every_nth_item(bigram_rankings, n=n)

    # Create all possible key sequence pairs
    keyseq_pairs = list(itertools.combinations(selected_key_sequences, 2))

    ngram_pairs = get_ngrams_randomly_left_or_right(hands, keyseq_pairs)

    random.shuffle(ngram_pairs)
    refs, ngrams = zip(*ngram_pairs)

    save_score_ratios(outfile, ngrams, refs)

    print(f"Created bigram score ratio template file: {outfile}")
    print(f"\n Number of anchor ngrams: {len(selected_key_sequences)}")
    print(f" Number of score ratios: {len(ngram_pairs)}")

    print(
        "\nYour task is to fill the score_ratio values. score_ratio = score(ngram) / score(ref)"
    )


def get_ngrams_randomly_left_or_right(
    hands: Hands,
    keyseq_pairs: Sequence[tuple[KeySeq, KeySeq]],
) -> list[tuple[str, str]]:
    """Gets ngrams for score ratios. Left is the reference ngram, right is the ngram."""
    ngram_pairs = []
    fallback = "__not_found__"

    for keyseq_pair in keyseq_pairs:
        hand = random.choice((hands.left, hands.right))
        other_hand = hands.left if hand is hands.right else hands.right

        ngram_left, hand = _get_left_or_right(
            hand, other_hand, keyseq_pair[0], fallback=fallback
        )
        other_hand = hands.left if hand is hands.right else hands.right

        ngram_right, _ = _get_left_or_right(
            hand, other_hand, keyseq_pair[1], fallback=fallback
        )
        ngram_pairs.append((ngram_left, ngram_right))

    if fallback in ngram_pairs:
        raise RuntimeError("Some ngrams were not found.")

    return ngram_pairs


def _get_left_or_right(
    hand: Hand, fallback_hand: Hand, keyseq: KeySeq, fallback: str = "__not_found__"
) -> tuple[str, Hand]:

    ngram = hand.get_symbols_visualization(keyseq, fallback=fallback)
    if ngram != fallback:
        return ngram, hand
    ngram = fallback_hand.get_symbols_visualization(keyseq, fallback=fallback)
    if ngram != fallback:
        return ngram, fallback_hand
    raise RuntimeError(f"Key sequence {keyseq} not found in both hands.")
