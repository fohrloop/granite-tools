from __future__ import annotations

import json
import typing
from pathlib import Path

from granite_tools.app_types import BigramScoreDict
from granite_tools.bigram_scores.anchor_scores import read_raw_anchor_scores_json
from granite_tools.bigram_scores.rankings import load_bigram_rankings
from granite_tools.bigram_scores.spline_smoothing import get_spline_scores
from granite_tools.utils import get_linear_scaling_function

if typing.TYPE_CHECKING:

    from granite_tools.app_types import KeySeq


def read_bigram_scores(file: str | Path) -> list[BigramScoreDict]:
    """Read bigram scores from a bigram scores JSON file."""
    with open(file, "r") as f:
        bigram_scores = json.load(f)
    return bigram_scores


def load_bigram_and_unigram_scores(
    bigram_ranking_file: str | Path, raw_anchor_scores_file: str | Path
) -> dict[KeySeq, float]:
    """Load bigram (and unigram) scores.

    Parameters
    ----------
    bigram_ranking_file:
        The file with key sequences on each row. Top of file: easiest, end of file:
        most difficult. The key sequences are ints (e.g 10) or pairs of ints (e.g. 6,7)
    raw_anchor_scores_file:
        The json file containing the raw anchor scores. These are the absolute scores
        for certain anchor bigrams (and unigrams). Example key-value pair:
        "(12, 6)": 1.8170985207530244
    """
    ngrams_ordered = load_bigram_rankings(bigram_ranking_file)
    scores = read_raw_anchor_scores_json(raw_anchor_scores_file)
    y_all, ranks = get_spline_scores(ngrams_ordered, scores)
    scores = dict()
    for x, y in zip(ranks, y_all):
        ngram = ngrams_ordered[int(x - 1)]
        scores[ngram] = float(y)
    return scores


def get_scaled_bigram_scores(
    bigram_scores: dict[KeySeq, float], newmax: float, newmin: float = 1
) -> dict[KeySeq, float]:
    """Creates scaled bigram scores based on non-scaled ones. Uses linear scaling."""

    scaled_scores = dict()
    s = get_linear_scaling_function(
        oldmin=min(bigram_scores.values()),
        oldmax=max(bigram_scores.values()),
        newmin=newmin,
        newmax=newmax,
    )
    for keyseq, score in bigram_scores.items():
        scaled_scores[keyseq] = s(score)
    return scaled_scores
