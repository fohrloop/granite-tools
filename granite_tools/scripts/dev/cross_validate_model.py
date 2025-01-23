"""Splits trigram model data into training and validation sets a few times and
calculates trigram model error with cross validation.

Example:

    uv run granite_tools/scripts/dev/cross_validate_model.py examples/config.yml tmp/granite.ranking tmp/trigram.relative.toml --folds=4
"""

import argparse
import copy
import datetime as dt
import random

import numpy as np

from granite_tools.config import read_config
from granite_tools.hands import get_hands_data
from granite_tools.scorer.bigram_scores import load_bigram_and_unigram_scores
from granite_tools.scorer.scorer import (
    TrigramModelParameters,
    TrigramScoreSets,
    create_optimization_target_function,
    get_initial_params_and_bounds,
    load_trigram_relative_scores,
    optimize_parameters,
)


def iter_training_validation_sets(
    trigram_scores: dict[str, dict[str, float]], folds: int
):
    trigram_scores_all = list(trigram_scores)
    random.shuffle(trigram_scores_all)
    validation_size = int(len(trigram_scores_all) / folds)
    for i in range(folds):
        start_idx = i * validation_size
        end_idx = (
            (i + 1) * validation_size if i < folds - 1 else len(trigram_scores_all)
        )

        selection = slice(start_idx, end_idx)

        training_trigram_scores = copy.deepcopy(trigram_scores)
        validation_trigram_scores = dict()

        for ref_trigram in trigram_scores_all[selection]:
            validation_trigram_scores[ref_trigram] = training_trigram_scores.pop(
                ref_trigram
            )
        print(
            f"Cross validation round {i+1}/{folds}, [{start_idx}:{end_idx}], {len(training_trigram_scores)} training trigrams, {len(validation_trigram_scores)} validation trigrams"
        )
        yield training_trigram_scores, validation_trigram_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", help="Granite Config YAML file")
    parser.add_argument("bigram_ranking_file", help="Bigram Ranking file")
    parser.add_argument("trigram_score_file", help="Relative Trigram Scores file")
    parser.add_argument("anchor_ngram_scores_file", help="Raw Anchor Ngram Scores file")
    parser.add_argument(
        "--folds", type=float, default=10, help="Folds (rounds) in cross validation"
    )
    args = parser.parse_args()

    trigram_scores = load_trigram_relative_scores(args.trigram_score_file)
    config = read_config(args.config_file)
    bigram_scores = load_bigram_and_unigram_scores(
        args.bigram_ranking_file, args.anchor_ngram_scores_file
    )
    folds = int(args.folds)

    hands = get_hands_data(config)
    x0, bounds = get_initial_params_and_bounds()
    bounds = None
    print("Started at:", dt.datetime.now())

    scores_train = []
    scores_valid = []
    for trig_scores_train, trig_scores_valid in iter_training_validation_sets(
        trigram_scores, folds
    ):
        scoresets_train = TrigramScoreSets.from_relative_trigram_scores(
            trig_scores_train, hands
        )
        scoresets_valid = TrigramScoreSets.from_relative_trigram_scores(
            trig_scores_valid, hands
        )
        func_train = create_optimization_target_function(
            scoresets_train, hands, bigram_scores
        )
        func_valid = create_optimization_target_function(
            scoresets_valid, hands, bigram_scores
        )

        params = optimize_parameters(func_train, x0, bounds, hands.config)
        print(TrigramModelParameters.from_tuple(params))
        score_train = func_train(params)
        score_valid = func_valid(params)
        scores_train.append(score_train)
        scores_valid.append(score_valid)
    print("Finished at:", dt.datetime.now())
    print("")

    print("Training scores:", scores_train)
    print("Validation scores:", scores_valid)

    print("")
    print(f"Training score mean: {np.mean(scores_train):.4f}")
    print(f"Training score std: {np.std(scores_train):.4f}")
    print(f"Validation score mean: {np.mean(scores_valid):.4f}")
    print(f"Validation score std: {np.std(scores_valid):.4f}")

    breakpoint()
