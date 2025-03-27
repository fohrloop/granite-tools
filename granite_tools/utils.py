from typing import TypeVar

T = TypeVar("T")


def to_new_scale(
    val: float, oldmin: float, oldmax: float, newmin=1.0, newmax=5.0
) -> float:
    valrange = oldmax - oldmin
    newvalrange = newmax - newmin
    k = newvalrange / valrange
    return newmin + (val - oldmin) * k


def get_linear_scaling_function(oldmin, oldmax, newmin=1, newmax=5):
    oldrange = oldmax - oldmin
    newrange = newmax - newmin
    k = newrange / oldrange

    def _scale_a_score(score: T) -> T:
        return newmin + (score - oldmin) * k

    return _scale_a_score
