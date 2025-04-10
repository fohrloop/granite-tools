from __future__ import annotations

import typing
from typing import Iterable

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize

from granite_tools.easy_rolling import get_easy_rolling_type_mapping
from granite_tools.trigram_model.params import TrigramModelParameters
from granite_tools.trigram_model.scorer import _get_error, get_trigram_scores

if typing.TYPE_CHECKING:
    from typing import Iterable, Sequence

    from granite_tools.app_types import KeySeq
    from granite_tools.config import Config
    from granite_tools.easy_rolling import EasyRollingTrigramsMap
    from granite_tools.hands import Hands
    from granite_tools.score_ratios import ScoreRatioEntry


def create_optimization_target_function(
    trigram_score_ratios: list[ScoreRatioEntry],
    hands: Hands,
    bigram_scores: dict[KeySeq, float],
    mapping: EasyRollingTrigramsMap | None = None,
):
    """Creates a function that can be used as the target for optimizing the trigram
    model parameters.

    bigram_scores: The non-scaled bigram scores; original bigram scores."""

    # Calculate this mapping only once as it's always the same. About 85% speed improvement.
    mapping = mapping or (
        get_easy_rolling_type_mapping(hands.config.easy_rolling_trigrams, hands)
        if hands.config.easy_rolling_trigrams is not None
        else None
    )
    get_lower_limit, get_upper_limit = get_limit_funcs(hands.config.limit_multipliers)

    def func(x: tuple[float], *_: typing.Any) -> float:
        """Function that can be used as optimization target. Takes the model parameters
        as a tuple (or: Sequence) and returns a single float."""
        sum_: float = 0
        model_params = TrigramModelParameters.from_tuple(x, hands.config)

        trigram_scores = get_trigram_scores(
            model_params, trigram_score_ratios, hands, bigram_scores, mapping=mapping
        )

        for i, trigram_scoredct in enumerate(trigram_scores):
            err = _get_error(trigram_scoredct)
            r = trigram_scoredct["score_ratio_actual"]
            limit_func = get_lower_limit if err < 0 else get_upper_limit
            err_limit = limit_func((r,))[0] - r
            scaled_err = err / err_limit
            sum_ += scaled_err**2

        return (sum_ / (i + 1)) ** 0.5

    return func


def get_initial_params(
    config: Config,
) -> tuple[float, ...]:
    """Create initial params for optimization.

    Each parameter set in configuration is FIXED to that value during optimization.
    Each parameter missing from configuration is OPTIMIZED."""

    x0 = []
    for param in TrigramModelParameters.names:
        if getattr(config, param) is None:
            default = TrigramModelParameters.defaults[param]
            x0.append(default)

    return tuple(x0)


def optimize_parameters(
    scorefunc: typing.Callable[[list[float]], float],
    x0: Sequence[float],
    config: Config,
) -> tuple[float, ...]:

    res = minimize(
        scorefunc,
        x0=np.array(x0),
        method="Nelder-Mead",
        args=(config,),
        options=dict(
            maxiter=10000,
            maxfev=10000,
            disp=True,
        ),
    )  # type: ignore

    return tuple(float(x) for x in res.x)


def create_log_m_func(
    limit_multipliers: dict[float, float],
) -> typing.Callable[[float], float]:

    # TODO: Check if this is needed?
    mlist_sorted = sorted(
        ((k, v) for k, v in limit_multipliers.items()), key=lambda x: x[0]
    )

    score_ratios, multipliers = map(np.array, zip(*mlist_sorted))
    log_m = np.log(multipliers)

    spline = PchipInterpolator(score_ratios, log_m, extrapolate=False)

    def get_log_m(r):

        if r <= score_ratios[0]:
            return log_m[0]
        elif r >= score_ratios[-1]:
            return log_m[-1]
        return spline(r)

    return get_log_m


def get_limit_funcs(
    limit_multipliers: dict[float, float],
) -> tuple[
    typing.Callable[[Iterable[float]], np.ndarray],
    typing.Callable[[Iterable[float]], np.ndarray],
]:
    """Creates functions that return the lower and upper limits for a given score ratio
    The functions accept an array of score ratios (r) as input and return the limits.
    To calculate residual limits, subtract "r" from the calculated limits."""

    get_log_m = create_log_m_func(limit_multipliers)

    def get_lower_limit(r: Iterable[float]) -> np.ndarray:
        log_m = np.array([get_log_m(ri) for ri in r])
        m = np.exp(log_m)
        return r / m

    def get_upper_limit(r: Iterable[float]) -> np.ndarray:
        log_m = np.array([get_log_m(ri) for ri in r])
        m = np.exp(log_m)
        return r * m

    return get_lower_limit, get_upper_limit
