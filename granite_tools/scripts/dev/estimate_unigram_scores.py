"""Used to estimate unigram scores using the ngram (unigram and bigram) ranking file and
the (raw) anchor ngram scores json

Usage:

    python granite_tools/scripts/dev/estimate_unigram_scores.py <bigram_ranking_file> <raw_anchor_scores_file.json> <config_file.yml>

where the
    - bigram_ranking_file is the file containing the bigram and unigram rankings (from create_ngram_ranking.py)
    - raw_anchor_scores_file.json is created with scoreratios_fit.py
    - config_file.yml is the Granite Config YAML file (optional)
"""

from __future__ import annotations

import sys

from granite_tools.scorer.bigram_scores import get_spline_scores, load_ranking
from granite_tools.scorer.smooth_scores import read_raw_anchor_scores_json
from granite_tools.unigram_scores import calculate_unigram_scores, print_results
from granite_tools.utils import get_linear_scaling_function

if __name__ == "__main__":

    try:
        bigram_ranking_file = sys.argv[1]
        scores_raw_out_file = sys.argv[2]
        config_file = sys.argv[3]
        ngrams_ordered = load_ranking(bigram_ranking_file)
        scores = read_raw_anchor_scores_json(scores_raw_out_file)
    except:
        print(__doc__)
        sys.exit(1)

    y_all, x_all = get_spline_scores(ngrams_ordered, scores)
    s = get_linear_scaling_function(
        oldmin=min(y_all), oldmax=max(y_all), newmin=1, newmax=5
    )
    scores = s(y_all)
    bigram_scores = dict(zip(ngrams_ordered, (float(x) for x in scores)))
    bigram_scores = {k: v for k, v in bigram_scores.items() if len(k) == 2}
    used_key_indices = list(set(ks for pair in ngrams_ordered for ks in pair))
    unigram_scores = calculate_unigram_scores(bigram_scores, used_key_indices)
    print_results(unigram_scores, config_file)
