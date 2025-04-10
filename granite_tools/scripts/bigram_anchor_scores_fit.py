"""Creates a fit of raw bigram anchor scores based on bigram (and unigram) score ratios

Usage:

    uv run granite_tools/scripts/bigram_anchor_scores_fit.py [config_file] [bigram_ranking_file] [scoreratio_file] [output_file.json]

Example:

    uv run granite_tools/scripts/bigram_anchor_scores_fit.py examples/config.yml tmp/granite.bigram.ranking tmp/granite.bigram.scoreratios.yml tmp/bigram-anchor-scores-raw.json
"""

from __future__ import annotations

import json
import sys
import time

from granite_tools.bigram_scores.anchor_scores import fit_anchor_ngram_scores
from granite_tools.bigram_scores.rankings import load_bigram_rankings
from granite_tools.config import read_config
from granite_tools.hands import get_hands_data
from granite_tools.score_ratios import load_score_ratio_entries

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

    ngrams_ordered = load_bigram_rankings(bigram_ranking_file)
    score_ratio_entries = load_score_ratio_entries(scoreratio_file, hands)

    t0 = time.time()
    print("Fitting anchor ngram scores.. (this might take a few minutes)")
    scores = fit_anchor_ngram_scores(score_ratio_entries, ngrams_ordered)
    print(f"Fitting scores took {time.time() - t0:.2f}s")

    print(scores)

    with open(scores_raw_out_file, "w") as f:
        json.dump({str(k): v for k, v in scores.items()}, f)

    print(f"Scores written to {scores_raw_out_file}")
