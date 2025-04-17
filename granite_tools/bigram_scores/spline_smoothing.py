"""Smoothing of bigram scores"""

from __future__ import annotations

import typing

import numpy as np
from scipy.interpolate import BSpline

if typing.TYPE_CHECKING:
    from typing import Sequence

    from granite_tools.app_types import KeySeq


SPLINE_BSPLINE_DEGREE = 3
SPLINE_KNOT_SEGMENTS = 35
SPLINE_LAMBDA_SMOOTHING = 1
SPLINE_KAPPA_PENALTY = 1e6

"""For fitting the cubic spline model (bigram/unigram scores smoothing and interpolation)"""
SPLINE_KWARGS = {
    "bspline_degree": SPLINE_BSPLINE_DEGREE,
    "knot_segments": SPLINE_KNOT_SEGMENTS,
    "lambda_smoothing": SPLINE_LAMBDA_SMOOTHING,
    "kappa_penalty": SPLINE_KAPPA_PENALTY,
}


def get_spline_scores(
    ngrams_ordered: list[KeySeq], scores: dict[KeySeq, float]
) -> tuple[list[float], list[int]]:
    """Turn raw anchor bigram scores into smooth monotonic scores.

    Parameters
    ----------
    ngrams_ordered:
        The list of ngrams ordered by difficulty. The first ngram is the easiest.
    scores:
        The dictionary with ANCHOR bigram RAW scores. The keys are the ngrams (key
        sequences) and the values are the raw scores.

    Returns
    -------
    bspline_scores:
        The smoothed scores for the bigrams. The scores are in the same order as the
        output ranks.
    ranks:
        The ranks of the bigrams. Basically a list of integers.
    """
    x_train, y_train, ranks = scores_to_training_data(ngrams_ordered, scores)
    bspline = create_monotone_bspline(
        x_train,
        y_train,
        **SPLINE_KWARGS,  # type: ignore[arg-type]
    )

    bspline_scores = bspline(ranks)
    ngram_scores = [float(x) for x in bspline_scores]

    return ngram_scores, ranks


def create_monotone_bspline(
    rank_train: Sequence[float],
    score_train: Sequence[float],
    bspline_degree: int = 3,
    knot_segments: int = 10,
    lambda_smoothing: float = 0.1,
    kappa_penalty: float = 10**6,
) -> BSpline:

    alphas, knots = fit_iter_pspline_smooth(
        rank_train,
        score_train,
        bspline_degree,
        knot_segments,
        lambda_smoothing,
        kappa_penalty,
    )
    b_spline = BSpline(knots, alphas, bspline_degree, extrapolate=False)
    return b_spline


def fit_iter_pspline_smooth(
    x_train: Sequence[float],
    y_train: Sequence[float],
    bspline_degree: int = 3,
    knot_segments: int = 10,
    lambda_smoothing: float = 0.1,
    kappa_penalty: float = 10**6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    bspline_degree: int
        The degree of the B-spline (which is also the degree of the fitted spline
        function). The order of the splines is degree + 1.
    knot_segments: int
        number of inter-knot segments between min(x) and max(x). Defines the number of
        knots. Will be limited to len(x)-2. (giving too high number will result in
        len(x)-2).
    lambda_smoothing: float
        The smoothing parameter. Higher values will result in smoother curves.
    kappa_penalty: float
        The penalty parameter for enforcing monotonicity. Higher values will result in
        more monotonic curves.
    """

    knot_interval = (max(x_train) - min(x_train)) / knot_segments

    # You need to add deg knots on each side of the interval. See, for example,
    # De Leeuw (2017) Computing and Fitting Monotone Splines
    # The basic interval is [min(x_train), max(x_train)], and
    # the extended interval is [min(knots), max(knots)].
    # You may only ask for values within the basic interval, as there are always m
    # (=deg+1) non-zero B-splines. Outside the basic interval, there are less B-splines
    # with non-zero values and the model is extrapolating.
    knots = np.linspace(
        min(x_train) - (bspline_degree + 1) * knot_interval,
        max(x_train) + (bspline_degree + 1) * knot_interval,
        bspline_degree * 2 + knot_segments + 1,
    )
    alphas = np.ones(len(x_train))

    B = BSpline.design_matrix(x=x_train, t=knots, k=bspline_degree).toarray()
    n_base_funcs = B.shape[1]
    I = np.eye(n_base_funcs)  # noqa: E741
    D3 = np.diff(I, n=3, axis=0)
    D1 = np.diff(I, n=1, axis=0)

    # Monotone smoothing
    V = np.zeros(n_base_funcs - 1)

    # Some terms for the equation
    # (BᵀB + λD3ᵀD3 + κD1ᵀVD1)α = Bᵀy
    #   A = BᵀB + λD3ᵀD3 + κD1ᵀVD1
    #   BTy = Bᵀy
    #
    # Therefore, we have
    #  Aα = Bᵀy
    B_gram = B.T @ B
    A_part1 = B_gram + lambda_smoothing * D3.T @ D3
    BTy = B.T @ y_train

    # The system of equations is
    # [ AᵀA   Cᵀ] [ α ] = [ Bᵀy ]
    # [  C    0 ] [ µ ]   [  d  ]
    #
    # where µ is the lagrange multiplier (λ means the smoothing parameter)
    # And Cα = d is the constraint enforced with the system.

    # Add the constraint for the first point
    C = B[0, :].reshape(1, -1)
    d = np.array([y_train[0]])
    C_and_zero = np.hstack([C, np.zeros((1, 1))])
    right_side = np.concatenate([BTy, d])
    for _ in range(30):

        W = np.diag(V * kappa_penalty)

        # Modify the system to include the constraint
        A = A_part1 + D1.T @ W @ D1
        A_aug = np.vstack(
            [
                np.hstack([A, C.T]),
                C_and_zero,
            ]
        )

        solution = np.linalg.solve(A_aug, right_side)
        # Exclude the Lagrange multiplier(s) from the solution
        alphas = solution[: A.shape[0]]

        V_new = (D1 @ alphas < 0) * 1
        dv = np.sum(V != V_new)
        V = V_new
        if dv == 0:
            break
    else:
        print("Max iteration reached")

    return alphas, knots


def scores_to_training_data(
    ngrams_ordered: Sequence[KeySeq], scores: dict[KeySeq, float]
) -> tuple[list[int], list[float], list[int]]:
    x_train, y_train = [], []
    for i, keyseq in enumerate(ngrams_ordered, start=1):
        if keyseq in scores:
            x_train.append(i)
            y_train.append(scores[keyseq])
    x_all = list(range(1, len(ngrams_ordered) + 1))

    return x_train, y_train, x_all
