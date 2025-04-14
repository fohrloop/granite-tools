from __future__ import annotations

from matplotlib import pyplot as plt

from granite_tools.app_types import BigramScoreDict, KeySeq
from granite_tools.bigram_scores.spline_smoothing import (
    SPLINE_KWARGS,
    create_monotone_bspline,
    scores_to_training_data,
)


def plot_anchor_scores(ngrams_ordered: list[KeySeq], scores: dict[KeySeq, float]):
    """Plot the bigram (and unigram) raw scores. These scores are not final as they're
    not yet put into scale. This plot is for debugging the bigram score fitting process
    and its easier to spot problems in an unscaled plot."""
    x_train, y_train, x_all = scores_to_training_data(ngrams_ordered, scores)

    bspline = create_monotone_bspline(x_train, y_train, **SPLINE_KWARGS)

    plt.plot(x_train, y_train, marker="*", ls="", markersize=8, label="raw", alpha=0.4)
    plt.plot(
        x_train,
        bspline(x_train),
        marker="o",
        ls="",
        markersize=5,
        color="red",
        label="smoothed (anchor)",
    )
    plt.plot(
        x_all,
        bspline(x_all),
        marker=".",
        markersize=3,
        ls="",
        color="red",
        label="smoothed (all)",
    )
    plt.legend()
    plt.xlabel("rank")
    plt.ylabel("score")
    plt.grid(ls="--", lw=0.5, color="lightgray")
    plt.title("Bigram scores (raw, unscaled)")


def plot_bigram_scores(scores: list[BigramScoreDict]):

    bigrams = [s for s in scores if s["type"] == "bigram"]
    unigrams = [s for s in scores if s["type"] == "unigram"]

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    ax_unigram = axes[0]
    ax_bigram = axes[1]

    ax_unigram.scatter(
        [s["rank_type"] for s in unigrams],
        [s["score"] for s in unigrams],
        marker=".",
        s=34,
        color="tab:blue",
    )
    ax_unigram.set_xlabel("Rank of unigram")
    ax_unigram.set_ylabel("Score")
    ax_unigram.set_title("Unigrams")

    ax_bigram.scatter(
        [s["rank_type"] for s in bigrams],
        [s["score"] for s in bigrams],
        marker=".",
        s=5,
        color="tab:red",
    )
    ax_bigram.set_xlabel("Rank of bigram")
    ax_bigram.set_ylabel("Score")

    ax_bigram.set_title("Bigrams")
    ax_unigram.grid(ls="--", lw=0.5, color="lightgray")
    ax_bigram.grid(ls="--", lw=0.5, color="lightgray")
    plt.tight_layout()
