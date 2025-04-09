"""Script for manual testing the smooth_scores module and the spline parameters"""

# mypy: ignore-errors
import numpy as np
from matplotlib import pyplot as plt

from granite_tools.trigram_model.smooth_scores import create_monotone_bspline

if __name__ == "__main__":
    scores = np.array([0.7, 2.0, 2.5, 3.4, 3.0, 3.6, 3.33, 4.1])
    x = np.array([1, 8, 15, 22, 30, 38, 46, 54])
    x_valid = np.linspace(min(x), max(x), 1000)
    y_est_train, y_est_valid = create_monotone_bspline(
        x, scores, x_valid, lambda_smoothing=100000.0, knot_segments=120
    )

    plt.plot(x, scores, label="raw scores", marker="o", ls="", color="lightgray")
    color_p_spline = "red"
    plt.plot(
        x_valid,
        y_est_valid,
        label="spline scores (valid)",
        ls="--",
        color=color_p_spline,
    )
    plt.plot(
        x,
        y_est_train,
        label="spline scores (train)",
        marker="o",
        ls="",
        color=color_p_spline,
    )
    plt.xlabel("rank of ngram")
    plt.ylabel("score")
    plt.legend()
    plt.grid(ls="--", lw=0.5, color="lightgray")
    plt.show()
