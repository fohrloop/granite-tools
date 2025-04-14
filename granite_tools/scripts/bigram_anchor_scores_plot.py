"""Creates a plot ngram scores based on score ratios data.

Usage:

    uv run granite_tools/scripts/bigram_anchor_scores_plot.py [config_file] [bigram_ranking_file] [bigram_raw_anchor_json]

Example:

    uv run granite_tools/scripts/bigram_anchor_scores_plot.py examples/config.yml data/granite.bigram.ranking data/bigram-anchor-scores-raw.json
"""

from __future__ import annotations

import sys

from matplotlib import pyplot as plt

from granite_tools.bigram_scores.anchor_scores import read_raw_anchor_scores_json
from granite_tools.bigram_scores.plotting import plot_anchor_scores
from granite_tools.bigram_scores.rankings import load_bigram_rankings
from granite_tools.config import read_config
from granite_tools.hands import get_hands_data

if __name__ == "__main__":

    try:
        config_file = sys.argv[1]
        bigram_ranking_file = sys.argv[2]
        anchor_scores_raw_json_file = sys.argv[3]
    except IndexError:
        print(__doc__)
        sys.exit(1)

    config = read_config(config_file)
    hands = get_hands_data(config)
    ngrams_ordered = load_bigram_rankings(bigram_ranking_file)
    scores = read_raw_anchor_scores_json(anchor_scores_raw_json_file)
    plot_anchor_scores(ngrams_ordered, scores)
    plt.show()
