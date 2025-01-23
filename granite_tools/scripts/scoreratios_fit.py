"""Creates a fit of raw ngram scores based on ngram score ratios

Usage:

    uv run granite_tools/scripts/scoreratios_fit.py [config_file] [bigram_ranking_file] [scoreratio_file] [scores_raw_out_file]

Example:

    uv run granite_tools/scripts/scoreratios_fit.py examples/keyseq_effort.yml tmp/granite.ranking tmp/granite.scoreratios-fixed.yml tmp/granite.scores-raw.json
"""

from __future__ import annotations

import json
import sys
import time
import typing

from granite_tools.config import read_config
from granite_tools.hands import get_hands_data
from granite_tools.scorer.bigram_scores import (
    fit_ngram_scores,
    load_ranking,
    make_score_ratio_entries,
)

if typing.TYPE_CHECKING:

    from granite_tools.app_types import KeySeq

    ScoreRatioEntry = tuple[KeySeq, KeySeq, float]


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
    score_ratio_entries = make_score_ratio_entries(scoreratio_file, hands)

    t0 = time.time()
    scores = fit_ngram_scores(score_ratio_entries, ngrams_ordered)
    print(f"Fitting scores took {time.time() - t0:.2f}s")

    print(scores)

    with open(scores_raw_out_file, "w") as f:
        json.dump({str(k): v for k, v in scores.items()}, f)

    print(f"Scores written to {scores_raw_out_file}")
