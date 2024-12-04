"""Creates a plot ngram scores based on score ratios data.

Usage:

    uv run granite_tools/scripts/scoreratios_plot.py [config_file] [bigram_ranking_file] [scoreratio_file] [scores_raw_out_file]

Example:

    uv run granite_tools/scripts/scoreratios_plot.py examples/keyseq_effort.yml tmp/granite.ranking tmp/granite.scoreratios-fixed.yml tmp/granite.scores-raw.json
"""

from __future__ import annotations

import sys
import typing

from matplotlib import pyplot as plt

from granite_tools.config import read_config
from granite_tools.hands import get_hands_data
from granite_tools.scorer.bigram_scores import load_ranking
from granite_tools.scorer.smooth_scores import (
    create_monotone_bspline,
    read_raw_scores_json,
    scores_to_training_data,
)

if typing.TYPE_CHECKING:
    from typing import Sequence

    from granite_tools.app_types import KeySeq

    ScoreRatioEntry = tuple[KeySeq, KeySeq, float]


def plot_score_ratios(score_ratio_entries: Sequence[ScoreRatioEntry]):
    score_ratios = [x[2] for x in score_ratio_entries]
    x = list(range(1, len(score_ratios) + 1))
    plt.plot(x, score_ratios, marker="o", ls="", alpha=0.3)
    plt.xlabel("entry (same order as in scoreratio file)")
    plt.ylabel("score ratio")
    plt.grid(ls="--", lw=0.5, color="lightgray")
    plt.show()


if __name__ == "__main__":

    try:
        config_file = sys.argv[1]
        bigram_ranking_file = sys.argv[2]
        scoreratio_file = sys.argv[3]
        scores_raw_out_file = sys.argv[4]
    except IndexError:
        print(__doc__)
        sys.exit(1)

    config = read_config(config_file)
    hands = get_hands_data(config)
    ngrams_ordered = load_ranking(bigram_ranking_file)
    scores = read_raw_scores_json(scores_raw_out_file)
    x_train, y_train, x_all = scores_to_training_data(ngrams_ordered, scores)

    bspline = create_monotone_bspline(
        x_train,
        y_train,
        bspline_degree=2,
        knot_segments=35,
        lambda_smoothing=1,
        kappa_penalty=1e6,
    )

    plt.plot(x_train, y_train, marker="*", ls="", markersize=8, label="raw", alpha=0.4)
    plt.plot(
        x_train,
        bspline(x_train),
        marker="o",
        ls="",
        markersize=5,
        color="red",
        label="smoothed (anchor)",
    )
    plt.plot(
        x_all,
        bspline(x_all),
        marker=".",
        markersize=3,
        ls="",
        color="red",
        label="smoothed (all)",
    )
    plt.legend()
    plt.xlabel("rank")
    plt.ylabel("score")
    plt.grid(ls="--", lw=0.5, color="lightgray")
    plt.show()
