from typing import TypeVar

T = TypeVar("T")


def get_linear_scaling_function(oldmin, oldmax, newmin=1, newmax=5):
    oldrange = oldmax - oldmin
    newrange = newmax - newmin
    k = newrange / oldrange

    def _scale_a_score(score: T) -> T:
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
