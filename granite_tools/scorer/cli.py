from __future__ import annotations

import itertools
import random
import sys
import typing
from enum import Enum
from pathlib import Path

import typer

from granite_tools.config import read_config
from granite_tools.hands import get_hands_data
from granite_tools.scorer.bigram_scores import (
    load_bigram_and_unigram_scores,
    load_ranking,
    save_score_ratios,
    select_every_nth_item,
)
from granite_tools.scorer.scorer import (
    TrigramModelParameters,
    TrigramScoreSets,
    get_initial_params_and_bounds,
    get_trigram_params_error_fun,
    group_trigram_scores,
    iter_trigrams_scores,
    load_trigram_scores,
    max_abs_error,
    optimize_parameters,
)

from .plotting import plot_trigram_scores
from .trigram_creation import create_score_template_

try:
    from typing import Annotated
except ImportError:
    # For older python versions
    from typing_extensions import Annotated  # type: ignore

if typing.TYPE_CHECKING:
    from typing import Sequence

    from granite_tools.app_types import KeySeq
    from granite_tools.hands import Hand, Hands


def create_score_template_cli():
    typer.run(create_score_template)


ARG_CONFIG = Annotated[
    Path,
    typer.Argument(
        help="The path to the keyboard configuration (YAML) file.",
        show_default=False,
    ),
]

ARG_OUTFILE = Annotated[
    Path,
    typer.Argument(
        help="The path to the output (trigram scoring) file.",
        show_default=False,
    ),
]


def create_score_template(
    config_file: ARG_CONFIG,
    outfile: ARG_OUTFILE,
    n_trigram_sets: ARG_N_TRIGRAM_SETS = 50,
):
    config = read_config(config_file)
    create_score_template_(n_trigram_sets, config, outfile)


def fit_parameters_cli():
    typer.run(fit_parameters)


ARG_BIGRAM_RANKING_FILE = Annotated[
    Path,
    typer.Argument(
        help="The path to the bigram (+unigram) ranking file. Created with create_ngram_ranking.py",
        show_default=False,
    ),
]

ARG_TRIGRAM_SCORE_FILE = Annotated[
    Path,
    typer.Argument(
        help="The path to the trigram scoring (input) file. You may create of template of such file with granite-trigram-score-template.",
        show_default=False,
    ),
]


def fit_parameters(
    config_file: ARG_CONFIG,
    bigram_ranking_file: ARG_BIGRAM_RANKING_FILE,
    trigram_score_file: ARG_TRIGRAM_SCORE_FILE,
):
    config_base = read_config(str(config_file))
    hands = get_hands_data(config_base)
    bigram_scores = load_bigram_and_unigram_scores(bigram_ranking_file)
    x0, bounds = get_initial_params_and_bounds()
    trigram_scores = load_trigram_scores(trigram_score_file)
    scoresets = TrigramScoreSets.from_trigram_scores(trigram_scores, hands)
    scorefunc = get_trigram_params_error_fun(scoresets, hands, bigram_scores)
    params = optimize_parameters(scorefunc, x0, bounds, hands.config)

    for name, value in zip(
        (
            "vert2u_coeff",
            "dirchange_coeff",
            "balanced_b_coeff",
            "unigram_coeff",
            "skipgram_b_coeff",
            "easy_rolling_coeff",
            "bigram_raw_range_max",
            "bigram_scaling_exponent",  # TODO: needs fix
        ),
        params,
    ):
        print(f"{name}: {value:.3f}")


def fit_check_cli():
    app = typer.Typer(pretty_exceptions_enable=False)
    app.command()(fit_check)
    app()


ARG_N_TRIGRAM_SETS = Annotated[
    int,
    typer.Argument(
        help="Number of trigram families used. Each trigram set consists of four trigrams from a trigram family.",
        show_default=True,
    ),
]

ARG_TRIGRAM_SCORING_INPUT_FILE = Annotated[
    Path,
    typer.Argument(
        help="The path to the trigram scoring (input) file.",
        show_default=False,
    ),
]


class TrigramType(str, Enum):
    all = "all"
    onehand = "onehand"
    balanced = "balanced"
    redir = "redir"
    skipgram = "skipgram"


ARG_TRIGRAM_TYPE = Annotated[
    TrigramType,
    typer.Argument(
        help="""The type for the trigrams to plot. 'all' plots all. Note that only the
        non-reference trigrams are plotted since the reference trigrams are always
        scores 1.0 without any error. The reference trigrams are the FIRST trigrams of
        each trigram family in the trigram scoring file.""",
        show_default=False,
    ),
]


def fit_check(
    config_file: ARG_CONFIG,
    bigram_ranking_file: ARG_BIGRAM_RANKING_FILE,
    trigram_score_file: ARG_TRIGRAM_SCORING_INPUT_FILE,
    trigram_type: ARG_TRIGRAM_TYPE = TrigramType.all,
):
    config = read_config(config_file)
    hands = get_hands_data(config)
    scoresets = TrigramScoreSets.from_file(trigram_score_file, hands)
    bigram_scores = load_bigram_and_unigram_scores(bigram_ranking_file)
    params = TrigramModelParameters.from_config(config)
    trigram_scores_iter = iter_trigrams_scores(params, scoresets, hands, bigram_scores)
    groups = group_trigram_scores(trigram_scores_iter, group_sort_by=max_abs_error)
    plot_trigram_scores(
        groups,
        hands,
        outfile=trigram_score_file.parent / (trigram_score_file.name + ".svg"),
        trigram_type=trigram_type,
    )


def create_ngram_score_ratio_template():
    typer.run(create_ngram_score_ratio_template_)


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


def create_ngram_score_ratio_template_(
    ranking_file: ARG_NGRAM_RANKING_FILE,
    config_file: ARG_CONFIG,
    outfile: ARG_NGRAM_TEMPLATE_OUTFILE,
    n: ARG_SCORE_RATIO_NGRAM_GAP = 8,
):
    if outfile.exists():
        print(f"Output file {outfile} already exists. Aborting.")
        sys.exit(1)

    config = read_config(config_file)
    bigram_rankings = load_ranking(ranking_file)
    hands = get_hands_data(config)

    selected_key_sequences = select_every_nth_item(bigram_rankings, n=n)

    # Create all possible key sequence pairs
    keyseq_pairs = list(itertools.combinations(selected_key_sequences, 2))

    ngram_pairs = get_ngrams_randomly_left_or_right(hands, keyseq_pairs)

    random.shuffle(ngram_pairs)
    refs, ngrams = zip(*ngram_pairs)

    save_score_ratios(outfile, ngrams, refs)

    print(f"Created ngram score ratio template file: {outfile}")
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
