"""Shows the fit quality of ngram (raw) scores.

Usage:

    uv run granite_tools/scripts/scoreratios_show_worst_fit.py CONFIG_FILE BIGRAM_RANKING_FILE BIGRAM_SCORERATIO_FILE BIGRAM_ANCHOR_SCORES_JSON [N_SHOW]

Example:

    uv run granite_tools/scripts/scoreratios_show_worst_fit.py examples/config.yml data/granite.bigram.ranking data/granite.bigram.scoreratios.yml data/bigram-anchor-scores-raw.json
"""

from __future__ import annotations

import sys

import pandas as pd

from granite_tools.bigram_scores.anchor_scores import read_raw_anchor_scores_json
from granite_tools.bigram_scores.rankings import load_bigram_rankings
from granite_tools.bigram_scores.score_ratios import get_worst_score_ratios
from granite_tools.config import read_config
from granite_tools.hands import get_hands_data
from granite_tools.score_ratios import load_score_ratio_entries

if __name__ == "__main__":
    try:
        config_file = sys.argv[1]
        bigram_ranking_file = sys.argv[2]
        scoreratio_file = sys.argv[3]
        anchor_scores_raw_file = sys.argv[4]
        n_show = int(sys.argv[5]) if len(sys.argv) > 5 else 50
    except IndexError:
        print(__doc__)
        sys.exit(1)
    config = read_config(config_file)
    hands = get_hands_data(config)

    ngrams_ordered = load_bigram_rankings(bigram_ranking_file)
    score_ratio_entries = load_score_ratio_entries(scoreratio_file, hands)
    raw_anchor_scores = read_raw_anchor_scores_json(anchor_scores_raw_file)

    worst = get_worst_score_ratios(score_ratio_entries, raw_anchor_scores, hands)
    with pd.option_context("display.max_rows", n_show):
        print(worst.tail(n_show))
