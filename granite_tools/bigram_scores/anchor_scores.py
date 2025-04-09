"""Fitting bigram anchor (raw) scores based on bigram score ratios."""

from __future__ import annotations

import ast
import json
import typing

import numpy as np
from scipy.optimize import minimize

if typing.TYPE_CHECKING:
    from pathlib import Path
    from typing import Sequence, TypeVar

    from granite_tools.app_types import KeySeq
    from granite_tools.score_ratios import ScoreRatioEntry

    T = TypeVar("T")


def read_raw_anchor_scores_json(
    raw_anchor_scores_file: str | Path,
) -> dict[KeySeq, float]:
    with open(raw_anchor_scores_file) as f:
        scores_json = json.load(f)
        scores = {ast.literal_eval(k): v for k, v in scores_json.items()}
    return scores


def fit_anchor_ngram_scores(
    score_ratios: Sequence[ScoreRatioEntry], ranks: Sequence[KeySeq]
) -> dict[KeySeq, float]:
    """Fits ngram scores based on score ratios. The ranks must be a sequence sorted
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
        for entry in score_ratios:
            ngram_keyseq, ref_keyseq, score_ratio_expected = (
                entry["ngram_keyseq"],
                entry["ref_keyseq"],
                entry["score_ratio"],
            )

            score_ratio_calc = s[ngram_keyseq] / s[ref_keyseq]
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
        ngram1, ngram2 = (
            score_ratio_entry["ref_keyseq"],
            score_ratio_entry["ngram_keyseq"],
        )

        if ngram1 not in ranks:
            raise ValueError(f"ngram1 {ngram1} not in ranks")
        if ngram2 not in ranks:
            raise ValueError(f"ngram2 {ngram2} not in ranks")

        ngrams_set.add(ngram1)
        ngrams_set.add(ngram2)

    if ranks[0] not in ngrams_set:
        raise ValueError(
            f"Ngram ranked first ({ranks[0]}) not used in any of the score ratios! Check that the score ratio file contains at least one entry with ngram corresponding to those key_sequence indices (defined in the used granite config file)."
        )

    return sorted(ngrams_set, key=lambda ngram: ranks.index(ngram))
