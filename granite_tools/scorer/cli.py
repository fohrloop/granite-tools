from __future__ import annotations

from enum import Enum
from pathlib import Path

import typer

from granite_tools.bigram_scores import load_bigram_and_unigram_scores
from granite_tools.config import read_config
from granite_tools.hands import get_hands_data
from granite_tools.scorer.scorer import (
    TrigramModelParameters,
    TrigramScoreSets,
    create_optimization_target_function,
    get_initial_params_and_bounds,
    group_trigram_scores,
    iter_trigrams_scores,
    load_trigram_relative_scores,
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


def create_score_template_cli():
    typer.run(create_score_template)


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

ARG_ANCHOR_SCORES_FILE = Annotated[
    Path,
    typer.Argument(
        help="The path to the anchor scores file. Created with granite_tools/scripts/scoreratios_fit.py",
        show_default=False,
    ),
]


def fit_parameters(
    config_file: ARG_CONFIG,
    bigram_ranking_file: ARG_BIGRAM_RANKING_FILE,
    trigram_score_file: ARG_TRIGRAM_SCORE_FILE,
    raw_anchor_ngram_scores_file: ARG_ANCHOR_SCORES_FILE,
):
    config_base = read_config(str(config_file))
    hands = get_hands_data(config_base)
    bigram_scores = load_bigram_and_unigram_scores(
        bigram_ranking_file, raw_anchor_ngram_scores_file
    )
    x0, bounds = get_initial_params_and_bounds()
    trigram_scores = load_trigram_relative_scores(trigram_score_file)

    scoresets = TrigramScoreSets.from_relative_trigram_scores(trigram_scores, hands)
    scorefunc = create_optimization_target_function(scoresets, hands, bigram_scores)

    print("Note: bounds not used.")
    params = optimize_parameters(scorefunc, x0, None, hands.config)

    for name, value in zip(
        (
            "vert2u_coeff",
            "dirchange_coeff",
            "balanced_b_coeff",
            "unigram_coeff",
            "skipgram_b_coeff",
            "easy_rolling_coeff",
            "bigram_range_max",
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
    trigram_scores_iter = iter_trigrams_scores(params, scoresets, hands, bigram_scores)
    groups = group_trigram_scores(trigram_scores_iter, group_sort_by=max_abs_error)
    plot_trigram_scores(
        groups,
        hands,
        outfile=trigram_score_file.parent / (trigram_score_file.name + ".svg"),
        trigram_type=trigram_type,
    )
