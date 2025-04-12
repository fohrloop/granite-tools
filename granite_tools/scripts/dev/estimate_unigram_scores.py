"""Used to estimate unigram scores using the ngram (unigram and bigram) ranking file and
the (raw) anchor ngram scores json

Usage:

    python granite_tools/scripts/dev/estimate_unigram_scores.py <bigram_ranking_file> <raw_anchor_scores_file.json> <config_file.yml>

where the
    - bigram_ranking_file is the file containing the bigram and unigram rankings (from create_ngram_ranking.py)
    - raw_anchor_scores_file.json is created with granite-bigram-scores-fit
    - config_file.yml is the Granite Config YAML file (optional)
"""

from __future__ import annotations

import sys

from granite_tools.bigram_scores.anchor_scores import read_raw_anchor_scores_json
from granite_tools.bigram_scores.rankings import load_bigram_rankings
from granite_tools.bigram_scores.spline_smoothing import get_spline_scores
from granite_tools.unigram_scores import calculate_unigram_scores, print_results
from granite_tools.utils import get_linear_scaling_function

if __name__ == "__main__":

    try:
        bigram_ranking_file = sys.argv[1]
        scores_raw_out_file = sys.argv[2]
        config_file = sys.argv[3]
        ngrams_ordered = load_bigram_rankings(bigram_ranking_file)
        scores = read_raw_anchor_scores_json(scores_raw_out_file)
    except Exception:
        print(__doc__)
        sys.exit(1)

    y_all, ranks = get_spline_scores(ngrams_ordered, scores)
    s = get_linear_scaling_function(
        oldmin=min(y_all), oldmax=max(y_all), newmin=1, newmax=5
    )
    scores_arr = s(y_all)
    bigram_scores = dict(zip(ngrams_ordered, (float(x) for x in scores_arr)))
    bigram_scores = {k: v for k, v in bigram_scores.items() if len(k) == 2}
    used_key_indices = list(set(ks for pair in ngrams_ordered for ks in pair))
    unigram_scores = calculate_unigram_scores(bigram_scores, used_key_indices)
    print_results(unigram_scores, config_file)
