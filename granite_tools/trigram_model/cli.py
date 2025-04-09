from __future__ import annotations

import typing
from enum import Enum
from pathlib import Path

import typer

from granite_tools.bigram_scores import load_bigram_and_unigram_scores
from granite_tools.config import read_config
from granite_tools.hands import get_hands_data
from granite_tools.score_ratios import load_score_ratio_entries
from granite_tools.trigram_model import (
    TrigramModelParameters,
    create_optimization_target_function,
    get_initial_params,
    get_trigram_scores,
    max_abs_error,
    optimize_parameters,
)

from .plotting import plot_trigram_scores

try:
    from typing import Annotated
except ImportError:
    # For older python versions
    from typing_extensions import Annotated  # type: ignore


if typing.TYPE_CHECKING:
    from granite_tools.score_ratios import ScoreRatioEntry

ARG_CONFIG = Annotated[
    Path,
    typer.Argument(
        help="The path to the Granite configuration (YAML) file.",
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


def fit_parameters_cli():
    typer.run(fit_parameters)


ARG_BIGRAM_RANKING_FILE = Annotated[
    Path,
    typer.Argument(
        help="The path to the bigram (+unigram) ranking file. Created with create_ngram_ranking.py",
        show_default=False,
    ),
]

TRIGRAM_SCORE_RATIO_HELP = (
    "The path to the trigram score ratio (input) YAML file.\n"
    "The file should contain a list of dictionaries with the following keys: "
    "(1) ngram: The ngram to score. "
    "(2) ref: The reference ngram. "
    "(3) score_ratio: The score ratio (score_ngram/score_ref), which should be more than 1.0. "
    " (if not, swap the ref and ngram)."
)
ARG_TRIGRAM_SCORE_RATIO_FILE = Annotated[
    Path,
    typer.Argument(
        help=TRIGRAM_SCORE_RATIO_HELP,
        show_default=False,
    ),
]

ARG_ANCHOR_SCORES_FILE = Annotated[
    Path,
    typer.Argument(
        help="The path to the bigram anchor scores file. Created with granite_tools/scripts/bigram_anchor_scores_fit.py",
        show_default=False,
    ),
]


def fit_parameters(
    config_file: ARG_CONFIG,
    bigram_ranking_file: ARG_BIGRAM_RANKING_FILE,
    bigram_anchor_scores_file: ARG_ANCHOR_SCORES_FILE,
    trigram_score_ratio_file: ARG_TRIGRAM_SCORE_RATIO_FILE,
):
    config_base = read_config(str(config_file))
    hands = get_hands_data(config_base)
    bigram_scores = load_bigram_and_unigram_scores(
        bigram_ranking_file, bigram_anchor_scores_file
    )

    x0 = get_initial_params(hands.config)
    if len(x0) == 0:
        print(f"No parameters to optimize (all parameters fixed in {config_file})")
        return

    trigram_score_ratios = load_score_ratio_entries(trigram_score_ratio_file, hands)

    scorefunc = create_optimization_target_function(
        trigram_score_ratios, hands, bigram_scores
    )

    params_tuple = optimize_parameters(scorefunc, x0, hands.config)
    params = TrigramModelParameters.from_tuple(params_tuple, hands.config)
    fitted_params = []
    fixed_params = []
    for name, value in params.iter_names_and_values():
        if getattr(hands.config, name) is None:
            fitted_params.append((name, value))
        else:
            fixed_params.append((name, value))

    print("\nFixed parameters:")
    for name, value in fixed_params:
        print(f"{name}: {value:.3f}")

    print("\nFitted parameters:")
    for name, value in fitted_params:
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
    alternating = "alternating"


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
    raw_anchor_ngram_scores_file: ARG_ANCHOR_SCORES_FILE,
    trigram_type: ARG_TRIGRAM_TYPE = TrigramType.all,
):
    config = read_config(config_file)
    hands = get_hands_data(config)
    scoresets = TrigramScoreSets.from_file(trigram_score_file, hands)
    bigram_scores = load_bigram_and_unigram_scores(
        bigram_ranking_file, raw_anchor_ngram_scores_file
    )

    params = TrigramModelParameters.from_config(config)
    trigram_scores_iter = get_trigram_scores(params, scoresets, hands, bigram_scores)
    groups = group_trigram_scores(trigram_scores_iter, group_sort_by=max_abs_error)
    plot_trigram_scores(
        groups,
        hands,
        outfile=trigram_score_file.parent / (trigram_score_file.name + ".svg"),
        trigram_type=trigram_type,
    )
