"""Compare few bigram scoring models.


Usage:

    python granite_tools/scripts/dev/compare_bigram_scoring_models.py BIGRAM_RANKING_FILE SCORES_RAW_OUT_FILE

Example:
    python granite_tools/scripts/dev/compare_bigram_scoring_models.py data/granite.bigram.ranking data/bigram-anchor-scores-raw.json

"""

from __future__ import annotations

import sys

import numpy as np
from matplotlib import pyplot as plt

from granite_tools.bigram_scores.anchor_scores import read_raw_anchor_scores_json
from granite_tools.bigram_scores.rankings import load_bigram_rankings
from granite_tools.bigram_scores.spline_smoothing import (
    SPLINE_KWARGS,
    create_monotone_bspline,
    scores_to_training_data,
)
from granite_tools.scaling import get_linear_scaling_function

try:
    bigram_ranking_file = sys.argv[1]
    scores_raw_out_file = sys.argv[2]
except IndexError:
    print(__doc__)
    sys.exit(1)


def get_spline_data(
    bigram_ranking_file: str, scores_raw_out_file: str
) -> tuple[list[int], np.ndarray]:

    ngrams_ordered = load_bigram_rankings(bigram_ranking_file)
    scores = read_raw_anchor_scores_json(scores_raw_out_file)
    x_train, y_train, x_all = scores_to_training_data(ngrams_ordered, scores)
    bspline = create_monotone_bspline(
        x_train,
        y_train,
        **SPLINE_KWARGS,  # type: ignore[arg-type]
    )

    y_all = bspline(x_all)
    return x_all, np.array(y_all)


bigram_scaling_exponent1 = 0.242
bigram_raw_range_max1 = 36165.713

# Previos model. Newest parameter values.
bigram_raw_range_max2 = 69.482
bigram_scaling_exponent2 = 0.631

n = 18**2 + 18  # 18 keys

y_linear = np.linspace(1, 5, n)

y_old1 = np.linspace(1, bigram_raw_range_max1, n) ** bigram_scaling_exponent1
y_old2 = np.linspace(1, bigram_raw_range_max2, n) ** bigram_scaling_exponent2

x = list(range(len(y_old1)))
x_spline, y_spline = get_spline_data(bigram_ranking_file, scores_raw_out_file)

s = get_linear_scaling_function(
    oldmin=min(y_spline), oldmax=max(y_spline), newmin=1, newmax=5
)
s2 = get_linear_scaling_function(
    oldmin=min(y_spline), oldmax=max(y_spline), newmin=1, newmax=13
)

plt.plot(x, y_old1, label="Old (older params)", alpha=0.5, ls=":", color="lightgray")
plt.plot(x, y_old2, label="Old (newest params)", alpha=0.5, ls="-.", color="darkgray")
plt.plot(x, y_linear, label="Linear (1 to 5)", alpha=0.5, ls="--", color="black")
plt.plot(x_spline, y_spline, label="Spline (unscaled)", color="tab:red")
plt.plot(x_spline, s(y_spline), label="Spline (1 to 5)", color="tab:green")
plt.plot(x_spline, s2(y_spline), label="Spline (1 to 13)", color="tab:blue")

plt.ylabel("score")
plt.xlabel("rank of unigram/bigram")
plt.legend()
plt.grid(ls="--", lw=0.5, color="lightgray")
plt.show()
