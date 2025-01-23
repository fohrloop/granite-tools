"""This module provides functions to load bigram and unigram scores from bigram
(+unigram) ranking files"""

from __future__ import annotations

import math
import typing
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize

from granite_tools.scorer.modelparams import SPLINE_KWARGS
from granite_tools.scorer.smooth_scores import (
    create_monotone_bspline,
    read_raw_anchor_scores_json,
    scores_to_training_data,
)
from granite_tools.utils import get_linear_scaling_function

if typing.TYPE_CHECKING:
    from typing import Sequence, TypeVar

    from granite_tools.app_types import KeySeq
    from granite_tools.hands import Hands

    # ScoreRatioEntry: (ngram, ref_ngram, scoreratio)
    # score(ngram)/score(ref_ngram) = scoreratio
    ScoreRatioEntry = tuple[KeySeq, KeySeq, float]
    T = TypeVar("T")


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
    ngrams_ordered = load_ranking(bigram_ranking_file)
    scores = read_raw_anchor_scores_json(raw_anchor_scores_file)
    y_all, ranks = get_spline_scores(ngrams_ordered, scores)
    scores = dict()
    for x, y in zip(ranks, y_all):
        ngram = ngrams_ordered[int(x - 1)]
        scores[ngram] = float(y)
    return scores


def load_ranking(file) -> list[KeySeq]:
    with open(file) as f:
        return [tuple(map(int, line.strip().split(","))) for line in f]


def fit_anchor_ngram_scores(
    score_ratios: Sequence[ScoreRatioEntry], ranks: Sequence[KeySeq]
) -> dict[KeySeq, float]:
    """Fits ngram scores based on score ratioss. The ranks must be a sequence sorted
    from least to most effort key sequence."""

    ngrams = _get_ngrams_present_in_score_ratios(score_ratios, ranks)
    first_ngram = ngrams[0]
    ngrams_need_score = ngrams[1:]
    scores = _fit_selected_ngram_scores(first_ngram, ngrams_need_score, score_ratios)

    return scores


def _fit_selected_ngram_scores(
    first_ngram: KeySeq,
    ngrams_need_score: Sequence[KeySeq],
    score_ratios: Sequence[ScoreRatioEntry],
) -> dict[KeySeq, float]:
    def err_func(x: Sequence[float] | np.ndarray) -> float:
        s = {
            first_ngram: 1.0,
            **{ngram: x[i] for i, ngram in enumerate(ngrams_need_score)},
        }
        total_error = 0.0
        for ngram1, ngram2, score_ratio_expected in score_ratios:

            score_ratio_calc = s[ngram1] / s[ngram2]
            err = np.log2(score_ratio_calc / score_ratio_expected) ** 2
            total_error += err
        return total_error / len(score_ratios)

    x0 = np.linspace(1, 10, len(ngrams_need_score))
    res = minimize(err_func, x0, method="nelder-mead", options=dict(maxiter=1e6))  # type: ignore

    print("Error before optimization:", err_func(x0))
    print("Error after optimization:", err_func(res.x))

    if not res.success:
        raise ValueError(f"Optimization failed: {res.message}")

    return {first_ngram: 1.0} | {
        ngram: float(res.x[i]) for i, ngram in enumerate(ngrams_need_score)
    }


def _get_ngrams_present_in_score_ratios(
    score_ratios: Sequence[ScoreRatioEntry], ranks: Sequence[KeySeq]
) -> list[KeySeq]:
    """Gets the list of ngrams which are present in the `score_ratios`. The returned
    list is sorted based on the order of `ranks`."""
    ngrams_set = set()

    for score_ratio_entry in score_ratios:
        ngram1, ngram2, _ = score_ratio_entry

        if ngram1 not in ranks:
            raise ValueError(f"ngram1 {ngram1} not in ranks")
        if ngram2 not in ranks:
            raise ValueError(f"ngram2 {ngram2} not in ranks")

        ngrams_set.add(ngram1)
        ngrams_set.add(ngram2)

    if ranks[0] not in ngrams_set:
        raise ValueError(f"Ngram ranked first ({ranks[0]}) not in score_ratios!")

    return sorted(ngrams_set, key=lambda ngram: ranks.index(ngram))


def get_group_sizes(total_items: int, group_max_size: int) -> list[int]:
    """Get the sizes of groups when dividing `total_items` into groups of `group_max_size`
    or smaller. The groups are divided in a balanced way, i.e. the sizes of the groups
    are as equal as possible (maximum difference of one).

    If some groups are smaller than others, they are placed at the beginning of the
    list"""
    groups = math.ceil(total_items / group_max_size)
    parts, remainder = divmod(total_items, groups)
    return [parts] * (groups - remainder) + [parts + 1] * remainder


def select_every_nth_item(items: Sequence[T], n: int) -> Sequence[T]:
    """Selects every "nth" item from the list of items, if possible. If not possible
    directly, selects "as closely as possible", by first selecting every "n-1"th, and
    then every "nth". The first and last items are always included."""
    group_sizes = get_group_sizes(len(items) - 1, n)
    out = [items[0]]
    cur = 0
    for group_size in group_sizes:
        cur += group_size
        out.append(items[cur])
    return out


score_ratio_entry_template = """- ref: "{ref}"
  ngram: "{ngram}"
  score_ratio: {score_ratio}

"""


def save_score_ratios(
    outfile: str | Path,
    ngrams: Sequence[str],
    ref_ngrams: Sequence[str],
    score_ratios: Sequence[float] | None = None,
):
    """Save the score ratios to a YAML file. The ngrams and ref_ngrams are lists of
    ngrams and their references. The score_ratios is a list of score ratios, which is
    optional. If not provided, the score ratio is set to 1.0 for all ngram pairs."""
    with open(outfile, "w") as f:
        for i, (ngram, ref) in enumerate(zip(ngrams, ref_ngrams)):
            score_ratio = score_ratios[i] if score_ratios is not None else 1

            txt = score_ratio_entry_template.format(
                ngram=ngram, ref=ref, score_ratio=score_ratio
            )
            f.write(txt)


def make_score_ratio_entries(
    scoreratio_file: str, hands: Hands
) -> list[ScoreRatioEntry]:

    out = []
    with open(scoreratio_file) as f:
        d = yaml.safe_load(f)

        for item in d:
            ref_keyseq, _ = hands.where(item["ref"])
            ngram_keyseq, _ = hands.where(item["ngram"])
            out.append((tuple(ngram_keyseq), tuple(ref_keyseq), item["score_ratio"]))

    return out


def get_spline_scores(
    ngrams_ordered: list[KeySeq], scores: dict[KeySeq, float]
) -> tuple[np.ndarray, list[int]]:
    x_train, y_train, ranks = scores_to_training_data(ngrams_ordered, scores)
    bspline = create_monotone_bspline(
        x_train,
        y_train,
        **SPLINE_KWARGS,
    )

    bspline_scores = bspline(ranks)
    return np.array(bspline_scores), ranks
