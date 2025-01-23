"""Used to estimate unigram scores using .compare.pickle file from the
granite-scorer-compare application.


Usage: python granite_tools/scripts/estimate_unigram_scores.py <some.compare.pickle> [config_file.yml]
"""

from __future__ import annotations

import pickle
import sys

from granite_tools.compare.fitting import get_scores
from granite_tools.comparison_data import get_bigram_pairs, get_used_key_indices
from granite_tools.unigram_scores import calculate_unigram_scores, print_results

if __name__ == "__main__":

    with open(sys.argv[1], "rb") as f:
        data = pickle.load(f)

    comparisons_all = data["comparisons_all"]
    bigram_pairs = get_bigram_pairs(comparisons_all)

    bigram_scores = get_scores(bigram_pairs)
    used_key_indices = get_used_key_indices(bigram_pairs)

    unigram_scores = calculate_unigram_scores(bigram_scores, used_key_indices)
    config_file = sys.argv[2] if len(sys.argv) > 2 else None
    print_results(unigram_scores, config_file)
