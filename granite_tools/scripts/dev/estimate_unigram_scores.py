"""Used to estimate unigram scores using the ngram (unigram and bigram) ranking file and
the (raw) anchor ngram scores json

Usage:

    python granite_tools/scripts/dev/estimate_unigram_scores.py <bigram_ranking_file> <anchor_bigram_raw_score_file.json> <config_file.yml>

where the
    - bigram_ranking_file is the file containing the bigram rankings (from create_ngram_ranking.py)
    - anchor_bigram_raw_score_file.json is created with granite-bigram-scores-fit
    - config_file.yml is the Granite Config YAML file (optional)
"""

# TODO: Is this file needed?

from __future__ import annotations

import sys

from granite_tools.bigram_scores.anchor_scores import read_raw_anchor_scores_json
from granite_tools.bigram_scores.rankings import load_bigram_rankings
from granite_tools.bigram_scores.spline_smoothing import get_spline_scores
from granite_tools.config import read_config
from granite_tools.unigram_scores import calculate_unigram_scores, print_results

if __name__ == "__main__":

    try:
        bigram_ranking_file = sys.argv[1]
        anchor_bigram_raw_score_file = sys.argv[2]
        config_file = sys.argv[3]
        config = read_config(config_file)
        ngrams_ordered = load_bigram_rankings(bigram_ranking_file)
        raw_anchor_scores = read_raw_anchor_scores_json(anchor_bigram_raw_score_file)
    except Exception:
        print(__doc__)
        sys.exit(1)

    bigram_scores, _ = get_spline_scores(ngrams_ordered, raw_anchor_scores)
    bigram_scores_dct = dict(zip(ngrams_ordered, bigram_scores))
    unigram_scores = calculate_unigram_scores(bigram_scores_dct, config)

    print_results(unigram_scores, config_file)
