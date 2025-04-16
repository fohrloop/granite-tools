from __future__ import annotations

import itertools
import json
import random
import sys
import time
import typing
from collections import defaultdict
from pathlib import Path

import pandas as pd
import typer
from matplotlib import pyplot as plt

from granite_tools.app_types import BigramScoreDict
from granite_tools.bigram_scores.anchor_scores import (
    fit_anchor_ngram_scores,
    read_raw_anchor_scores_json,
)
from granite_tools.bigram_scores.bigram_scores import read_bigram_scores
from granite_tools.bigram_scores.plotting import (
    BIGRAM_DUMBBELL_PLOT_LEGEND_INFO,
    bigram_scores_dumbbell_plot,
    plot_anchor_scores,
    plot_bigram_scores,
)
from granite_tools.bigram_scores.rankings import load_bigram_rankings
from granite_tools.bigram_scores.score_ratio_template import (
    save_score_ratios,
    select_every_nth_item,
)
from granite_tools.bigram_scores.score_ratios import get_worst_score_ratios
from granite_tools.bigram_scores.spline_smoothing import get_spline_scores
from granite_tools.config import read_config
from granite_tools.hands import Hands, get_hands_data
from granite_tools.score_ratios import load_score_ratio_entries
from granite_tools.unigram_scores import calculate_unigram_scores
from granite_tools.utils import DATA_FOLDER

try:
    from typing import Annotated
except ImportError:
    # For older python versions
    from typing_extensions import Annotated  # type: ignore

if typing.TYPE_CHECKING:
    from typing import Sequence

    from granite_tools.app_types import BigramOrUnigram, KeySeq
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


def bigram_scores_fit():
    typer.run(bigram_scores_fit_)


ARG_BIGRAM_TEMPLATE_FILE = Annotated[
    Path,
    typer.Argument(
        help="The path to the input bigram score ratio file.",
        show_default=False,
    ),
]

ARG_BIGRAM_SCORE_OUTFILE = Annotated[
    Path,
    typer.Argument(
        help="The path to the output bigram score (JSON) file ",
        show_default=True,
    ),
]

ARG_ANCHOR_BIGRAM_RAW_SCORE_OUTFILE = Annotated[
    Path,
    typer.Argument(
        help="The path to the output anchor bigram RAW score (JSON) file. This file is useful for debugging but it's not needed later on (after you've settled with the bigram scoring).",
        show_default=True,
    ),
]

DEFAULT_SCORE_FILE_OUT = DATA_FOLDER / "bigram.scores.json"
DEFAULT_ANCHOR_RAW_SCORE_FILE_OUT = DATA_FOLDER / "bigram-anchor-scores-raw.json"


def bigram_scores_fit_(
    config_file: ARG_CONFIG,
    bigram_ranking_file: ARG_NGRAM_RANKING_FILE,
    bigram_scoreratio_file: ARG_BIGRAM_TEMPLATE_FILE,
    bigram_score_file_out: ARG_BIGRAM_SCORE_OUTFILE = DEFAULT_SCORE_FILE_OUT,
    anchor_bigram_raw_score_file_out: ARG_ANCHOR_BIGRAM_RAW_SCORE_OUTFILE = DEFAULT_ANCHOR_RAW_SCORE_FILE_OUT,
):
    """Create bigram.scores.json (and the raw anchor score json) files based on the
    bigram.ranking and bigram.scoreratios.yml files."""
    config = read_config(config_file)
    hands = get_hands_data(config)

    bigram_keyseq_ordered = load_bigram_rankings(bigram_ranking_file)
    score_ratio_entries = load_score_ratio_entries(bigram_scoreratio_file, hands)

    t0 = time.time()
    print("Fitting anchor ngram scores.. (this might take a few minutes)")
    # raw_anchor_scores = fit_anchor_ngram_scores(score_ratio_entries, ngrams_ordered)
    raw_anchor_scores = read_raw_anchor_scores_json(anchor_bigram_raw_score_file_out)
    bigram_scores_ordered, _ = get_spline_scores(
        bigram_keyseq_ordered, raw_anchor_scores
    )

    bigram_scores_dct = dict(zip(bigram_keyseq_ordered, bigram_scores_ordered))
    unigram_scores_dct = calculate_unigram_scores(bigram_scores_dct, config)
    u_scores = {(k,): v for (k, v) in unigram_scores_dct.items()}

    bigram_and_unigram_scores = make_bigram_score_dicts(
        u_scores | bigram_scores_dct, hands
    )

    worst = get_worst_score_ratios(score_ratio_entries, raw_anchor_scores, hands)
    n_show = 100
    with pd.option_context("display.max_rows", n_show):
        print(worst.tail(n_show))

    print(f"Fitting scores took {time.time() - t0:.2f}s")
    plot_anchor_scores(bigram_keyseq_ordered, raw_anchor_scores)
    plot_bigram_scores(bigram_and_unigram_scores)

    with open(bigram_score_file_out, "w") as f:
        json.dump(bigram_and_unigram_scores, f, indent=2)
    print(f"Bigram scores written to {bigram_score_file_out}")

    with open(anchor_bigram_raw_score_file_out, "w") as f:
        json.dump({str(k): v for k, v in raw_anchor_scores.items()}, f, indent=2)
    print(f"Raw bigram anchor scores written to {anchor_bigram_raw_score_file_out}")

    plt.show()


def make_bigram_score_dicts(
    scores: dict[KeySeq, float],
    hands: Hands,
) -> list[BigramScoreDict]:

    scores = dict(sorted(scores.items(), key=lambda item: item[1]))
    scoredcts = []
    unigram_rank = 0
    bigram_rank = 0
    ngram_type: BigramOrUnigram
    for rank, (key_indices, score) in enumerate(scores.items(), start=1):

        if len(key_indices) == 1:
            ngram_type = "unigram"
            unigram_rank += 1
            rank_type = unigram_rank
        elif len(key_indices) == 2:
            ngram_type = "bigram"
            bigram_rank += 1
            rank_type = bigram_rank
        else:
            raise ValueError(
                f"Invalid ngram length: {len(key_indices)}. Expected 1 or 2."
            )

        dct = BigramScoreDict(
            key_indices=key_indices,
            type=ngram_type,
            score=round(score, 3),
            rank=rank,
            rank_type=rank_type,
            __comment__left=hands.get_symbols_visualization("Left", key_indices),
            __comment__right=hands.get_symbols_visualization("Right", key_indices),
        )

        scoredcts.append(dct)
    return scoredcts


def bigram_scores_plot():
    typer.run(bigram_scores_plot_)


ARG_BIGRAM_SCORE_FILE = Annotated[
    Path,
    typer.Argument(
        help="The path to the bigram score (JSON) file ",
        show_default=True,
    ),
]

ARG_SHORT_OR_LONG_ANNOTATIONS = Annotated[
    bool,
    typer.Option(
        "--short/--long",
        help="Whether to show short or long form of the annotations.",
        show_default=True,
    ),
]


def bigram_scores_plot_(
    bigram_score_file: ARG_BIGRAM_SCORE_FILE = DEFAULT_SCORE_FILE_OUT,
    short_annotations: ARG_SHORT_OR_LONG_ANNOTATIONS = True,  # True means "--short"
):
    """Plot bigram scores from a bigram score JSON file."""
    bigram_scores = read_bigram_scores(bigram_score_file)
    plot_bigram_scores(bigram_scores, short_annotations=short_annotations)
    plt.show()


ARG_BIGRAM_DUMBBELL_PLOT_OUTFILE = Annotated[
    Path,
    typer.Argument(
        help="The path to the dumbbell plot output (.svg) file.",
        show_default=True,
    ),
]


def bigram_scores_dumbbell_plot_cli():
    typer.run(bigram_scores_dumbbell_plot_cli_)


def bigram_scores_dumbbell_plot_cli_(
    bigram_score_file: ARG_BIGRAM_SCORE_FILE,
    config_file: ARG_CONFIG,
    save_to: ARG_BIGRAM_DUMBBELL_PLOT_OUTFILE,
):

    config = read_config(config_file)
    hands = get_hands_data(config)

    bigram_scores = read_bigram_scores(bigram_score_file)

    data: defaultdict[str, list[str | float]] = defaultdict(list)

    for score in bigram_scores:
        if not score["type"] == "bigram":
            continue
        keyseq = tuple(score["key_indices"])
        data["rank"].append(score["rank_type"])
        data["score"].append(score["score"])
        data["ngram_left"].append(hands.get_symbols_visualization("Left", keyseq))
        data["ngram_right"].append(hands.get_symbols_visualization("Right", keyseq))
        data["repeats"].append(str(hands.get_repeats_text(keyseq)))
        data["rowdiff"].append(str(hands.get_rowdiff_text(keyseq)))

    bigram_scores_dumbbell_plot(data)
    print(BIGRAM_DUMBBELL_PLOT_LEGEND_INFO)
    plt.savefig(str(save_to), dpi=150)
    print("Plot saved to", save_to)
    plt.close()
