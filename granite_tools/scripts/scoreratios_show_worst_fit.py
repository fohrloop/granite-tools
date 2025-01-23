"""Shows the fit quality of ngram (raw) scores.

Usage:

    uv run granite_tools/scripts/scoreratios_show_worst_fit.py CONFIG_FILE BIGRAN_RANKING_FILE SCORERATIO_FILE SCORES_RAW_OUT_FILE [N_SHOW]

Example:

    uv run granite_tools/scripts/scoreratios_show_worst_fit.py examples/config.yml tmp/granite.ranking tmp/granite.scoreratios-fixed.yml tmp/bigram-anchor-scores-raw.json
"""

# mypy: ignore-errors
from __future__ import annotations

import ast
import json
import sys
import typing
from collections import defaultdict

import numpy as np
import pandas as pd

from granite_tools.config import read_config
from granite_tools.hands import get_hands_data
from granite_tools.scorer.bigram_scores import load_ranking, make_score_ratio_entries

if typing.TYPE_CHECKING:
    from typing import Sequence

    from granite_tools.app_types import KeySeq
    from granite_tools.hands import Hands

    ScoreRatioEntry = tuple[KeySeq, KeySeq, float]


def get_worst_score_ratios(
    score_ratio_entries: Sequence[ScoreRatioEntry],
    scores: dict[KeySeq, float],
    hands: Hands,
) -> pd.DataFrame:
    data = defaultdict(list)

    for score_ratio_entry in score_ratio_entries:
        ngram, ref, score_ratio = score_ratio_entry
        ngram_score = scores[ngram]
        ref_score = scores[ref]
        score_ratio_calc = ngram_score / ref_score
        error = score_ratio_calc - score_ratio

        symbols_ngram_left = hands.get_symbols_visualization("Left", ngram)
        symbols_ref_left = hands.get_symbols_visualization("Left", ref)
        symbols_ngram_right = hands.get_symbols_visualization("Right", ngram)
        symbols_ref_right = hands.get_symbols_visualization("Right", ref)

        log2err = np.log2(score_ratio_calc / score_ratio)

        rel_err = 2 ** abs(log2err) - 1
        signed_rel_err = np.sign(log2err) * rel_err

        data["ratio_name"].append(f"{symbols_ngram_left}/{symbols_ref_left}")
        data["ref"].append(symbols_ref_left)
        data["ngram"].append(symbols_ngram_left)
        data["ref_right"].append(symbols_ref_right)
        data["ngram_right"].append(symbols_ngram_right)
        data["score_ratio"].append(score_ratio)
        data["score_ratio_calc"].append(score_ratio_calc)
        data["score_ratio_err"].append(error)
        data["score_ref"].append(ref_score)
        data["score_ngram"].append(ngram_score)
        data["log2err"].append(log2err)
        data["score_ratio_resid"].append(signed_rel_err)
    df = (
        pd.DataFrame(data)
        .sort_values(by="score_ratio_resid", key=abs)
        .set_index("ratio_name")
    )

    return df


if __name__ == "__main__":
    try:
        config_file = sys.argv[1]
        bigram_ranking_file = sys.argv[2]
        scoreratio_file = sys.argv[3]
        scores_raw_out_file = sys.argv[4]
        n_show = int(sys.argv[5]) if len(sys.argv) > 5 else 50
    except IndexError:
        print(__doc__)
        sys.exit(1)
    config = read_config(config_file)
    hands = get_hands_data(config)

    ngrams_ordered = load_ranking(bigram_ranking_file)
    score_ratio_entries = make_score_ratio_entries(scoreratio_file, hands)

    with open(scores_raw_out_file) as f:
        scores_json = json.load(f)
        scores = {ast.literal_eval(k): v for k, v in scores_json.items()}

    worst = get_worst_score_ratios(score_ratio_entries, scores, hands)
    with pd.option_context("display.max_rows", n_show):
        print(worst.tail(n_show))
