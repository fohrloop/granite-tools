from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from typing import Callable, TypeVar, Union

    import numpy as np

    T = TypeVar("T")

    Number = Union[int, float]
    NumericArray = np.ndarray

    NumericLike = TypeVar("NumericLike", Number, NumericArray)


def get_linear_scaling_function(
    oldmin: float, oldmax: float, newmin: float = 1, newmax: float = 5
) -> Callable[[NumericLike], NumericLike]:
    oldrange = oldmax - oldmin
    newrange = newmax - newmin
    k = newrange / oldrange

    def _scale_a_score(score: NumericLike) -> NumericLike:
        return newmin + (score - oldmin) * k

    return _scale_a_score


def get_scaled_scores(
    original_scores: dict[T, float], newmax: float, newmin: float = 1
) -> dict[T, float]:
    """Creates scaled ngram scores based on non-scaled ones. Uses linear scaling."""

    scaled_scores: dict[T, float] = dict()
    s = get_linear_scaling_function(
        oldmin=min(original_scores.values()),
        oldmax=max(original_scores.values()),
        newmin=newmin,
        newmax=newmax,
    )
    for key, score in original_scores.items():
        scaled_scores[key] = s(score)
    return scaled_scores
