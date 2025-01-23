from __future__ import annotations

import ast
import json
import typing

import numpy as np
from scipy.interpolate import BSpline

if typing.TYPE_CHECKING:
    from granite_tools.app_types import KeySeq


def create_monotone_bspline(
    rank_train,
    score_train,
    bspline_degree=3,
    knot_segments=10,
    lambda_smoothing=0.1,
    kappa_penalty=10**6,
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
    x_train,
    y_train,
    bspline_degree=3,
    knot_segments=10,
    lambda_smoothing=0.1,
    kappa_penalty=10**6,
):
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
    I = np.eye(n_base_funcs)
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


def read_raw_scores_json(scores_raw_out_file: str) -> dict[KeySeq, float]:
    with open(scores_raw_out_file) as f:
        scores_json = json.load(f)
        scores = {ast.literal_eval(k): v for k, v in scores_json.items()}
    return scores


def scores_to_training_data(
    ngrams_ordered: list[KeySeq], scores: dict[KeySeq, float]
) -> tuple[list[float], list[float]]:
    x_train, y_train = [], []
    for i, keyseq in enumerate(ngrams_ordered, start=1):
        if keyseq in scores:
            x_train.append(i)
            y_train.append(scores[keyseq])
    x_all = list(range(1, len(ngrams_ordered) + 1))

    return x_train, y_train, x_all
