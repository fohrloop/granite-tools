"""Train with data without thumb keys, validate with data with thumb keys."""

import argparse
import datetime as dt
from statistics import mean

import pandas as pd
import seaborn as sns  # type: ignore
from matplotlib import pyplot as plt

from granite_tools.app_types import FingerType
from granite_tools.bigram_scores import load_bigram_and_unigram_scores
from granite_tools.config import read_config
from granite_tools.hands import get_hands_data
from granite_tools.scorer import get_trigram_data
from granite_tools.scorer.scorer import (
    TrigramModelParameters,
    TrigramScoreSets,
    create_optimization_target_function,
    get_initial_params_and_bounds,
    get_scoredict,
    load_trigram_relative_scores,
    optimize_parameters,
)
from granite_tools.scorer.trigramtype import get_easy_rolling_type_mapping

pd.set_option("display.max_rows", 1000)


def split_has_thumbs_and_no_thumbs(
    trigram_scores: dict[str, dict[str, float]],
):
    training_trigram_scores = dict()
    validation_trigram_scores = dict()

    for trigram in trigram_scores:
        fingers, _ = hands.get_fingers(trigram)
        has_thumb = FingerType.T in fingers
        if not has_thumb:
            training_trigram_scores[trigram] = trigram_scores[trigram]
        else:
            validation_trigram_scores[trigram] = trigram_scores[trigram]

    print(
        f"{len(training_trigram_scores)} training trigrams, {len(validation_trigram_scores)} validation trigrams"
    )
    return training_trigram_scores, validation_trigram_scores


def max_abs(items):
    return max(abs(x) for x in items)


def rmse(items):
    return (sum(x**2 for x in items) / len(items)) ** 0.5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", help="Granite Config YAML file")
    parser.add_argument("bigram_ranking_file", help="Bigram Ranking file")
    parser.add_argument("trigram_score_file", help="Relative Trigram Scores file")
    parser.add_argument("anchor_ngram_scores_file", help="Raw Anchor Ngram Scores file")
    args = parser.parse_args()

    trigram_scores = load_trigram_relative_scores(args.trigram_score_file)
    config = read_config(args.config_file)
    bigram_scores = load_bigram_and_unigram_scores(
        args.bigram_ranking_file, args.anchor_ngram_scores_file
    )

    hands = get_hands_data(config)
    x0, bounds = get_initial_params_and_bounds()

    print("Started at:", dt.datetime.now())

    trig_scores_nothumb, trig_scores_thumb = split_has_thumbs_and_no_thumbs(
        trigram_scores
    )
    scoresets_train = TrigramScoreSets.from_relative_trigram_scores(
        trig_scores_nothumb, hands
    )
    scoresets_valid = TrigramScoreSets.from_relative_trigram_scores(
        trig_scores_thumb, hands
    )
    mapping = get_easy_rolling_type_mapping(hands.config.easy_rolling_trigrams, hands)

    func_optimize = create_optimization_target_function(
        scoresets_train, hands, bigram_scores, mapping=mapping
    )

    params_tuple = optimize_parameters(func_optimize, x0, bounds, hands.config)

    params = TrigramModelParameters.from_tuple(params_tuple)
    print(params)

    scores_train = get_scoredict(
        params_tuple, scoresets_train, hands, bigram_scores, mapping=mapping
    )
    scores_valid = get_scoredict(
        params_tuple, scoresets_valid, hands, bigram_scores, mapping=mapping
    )
    print("Finished at:", dt.datetime.now())
    print("")

    print(
        f"Training score (no thumbs) - ave: {mean(scores_train.values())}, rmse: {rmse(scores_train.values())}, max: {max_abs(scores_train.values())}",
    )
    print(
        f"Validation score (thumbs) - ave: {mean(scores_valid.values())},  rmse: {rmse(scores_valid.values())}, max: {max_abs(scores_valid.values())}",
    )

    df_train = get_trigram_data(
        config, bigram_scores, trig_scores_nothumb, params=params
    ).reset_index()
    df_valid = get_trigram_data(
        config, bigram_scores, trig_scores_thumb, params=params
    ).reset_index()

    df = pd.concat((df_train, df_valid)).reset_index(drop=True)
    sns.relplot(
        x="ref_bigram_sum",
        y="score_ratio_resid",
        hue="has_thumb",
        size="score_ratio_resid",
        sizes=(40, 400),
        alpha=0.5,
        palette="muted",
        height=6,
        data=df,
    )
    plt.grid(ls="--", lw=0.5, color="lightgray")
    plt.tight_layout()
    plt.show()
    breakpoint()
