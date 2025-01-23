from __future__ import annotations

import typing
from typing import TypeVar

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex
from sklearn.linear_model import LinearRegression  # type: ignore

from granite_tools.config import read_config
from granite_tools.hands import get_hands_data

if typing.TYPE_CHECKING:
    KeySeq = tuple[int, ...]
    from typing import Callable


def create_data_for_fitting(
    used_key_indices: list[int],
    scores: dict[KeySeq, float],
) -> tuple[np.ndarray, np.ndarray]:
    y_all = []
    x_all = []
    for pair, score in scores.items():
        x = []
        for idx in used_key_indices:
            if idx in pair:
                x.append(1)
            else:
                x.append(0)
        x_all.append(x)
        y_all.append(score)
    return np.array(x_all), np.array(y_all)


def calculate_unigram_scores(
    scores: dict[KeySeq, float], used_key_indices: list[int]
) -> dict[int, float]:
    """Fit a linear regression model which estimates unigram scores from bigram
    comparison data

    Returns
    -------
    dict[int, float]
        A dictionary mapping key indices to the estimated unigram scores
    """
    X, y = create_data_for_fitting(used_key_indices, scores)
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    coefs = model.coef_

    out = dict()
    for idx, coef in zip(used_key_indices, coefs):
        out[idx] = float(coef)
    return out


T = TypeVar("T")


def scale_scores(
    unigram_scores: dict[T, float], newmin: float = 1.0, newmax: float = 5.0
) -> dict[T, float]:
    scores = unigram_scores.copy()
    minval = min(scores.values())
    maxval = max(scores.values())
    valrange = maxval - minval
    newvalrange = newmax - newmin
    k = newvalrange / valrange
    for idx in scores:
        score = scores[idx]
        scaled_score = newmin + (score - minval) * k
        scores[idx] = float(scaled_score)
    return scores


def print_results(unigram_scores: dict[int, float], config_file: str | None):
    config = read_config(config_file) if config_file is not None else None
    hands = get_hands_data(config) if config else None

    scores = scale_scores(unigram_scores, 1.0, 5.0)
    colorfunc = get_hex_func(min(scores.values()), max(scores.values()))

    for idx, score in scores.items():
        score = round(score, 3)
        color = colorfunc(score)
        if hands is None:
            print(f"{idx}: {score}, {color}")
        else:
            print(f"{hands.get_symbols_visualization('Left', [idx])}: {score}, {color}")


def get_hex_func(
    min_value: float,
    max_value: float,
    min_color=(1, 1, 1),
    max_color=(0.542, 0.211, 0.973),
    n_bins=100,
) -> Callable[[float], str]:
    colors = [min_color, max_color]
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_color_map", colors, N=n_bins
    )

    norm = Normalize(vmin=min_value, vmax=max_value)

    return lambda x: to_hex(custom_cmap(norm(x)))
