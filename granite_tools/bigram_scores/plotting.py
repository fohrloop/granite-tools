from __future__ import annotations

from math import ceil

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from mplcursors import Selection, cursor  # type: ignore[import]
from rich.color import Color

from granite_tools.app_types import BigramScoreDict, KeySeq
from granite_tools.bigram_scores.spline_smoothing import (
    SPLINE_KWARGS,
    create_monotone_bspline,
    scores_to_training_data,
)
from granite_tools.hands import Hands


def plot_anchor_scores(ngrams_ordered: list[KeySeq], scores: dict[KeySeq, float]):
    """Plot the anchor bigram raw scores. These scores are not final as they're
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
    """Plots bigram (and unigram) scores."""
    bigrams = [s for s in scores if s["type"] == "bigram"]
    unigrams = [s for s in scores if s["type"] == "unigram"]

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    ax_unigram = axes[0]
    ax_bigram = axes[1]

    scatter_unigram = ax_unigram.scatter(
        [s["rank_type"] for s in unigrams],
        [s["score"] for s in unigrams],
        marker=".",
        s=122,
        color="tab:blue",
        zorder=10,
    )
    ax_unigram.set_xlabel("Rank of unigram")
    ax_unigram.set_title("Unigrams")

    scatter_bigram = ax_bigram.scatter(
        [s["rank_type"] for s in bigrams],
        [s["score"] for s in bigrams],
        marker=".",
        s=5,
        color="tab:red",
        zorder=10,
    )
    ax_bigram.set_xlabel("Rank of bigram")
    ax_bigram.set_title("Bigrams")

    for ax in axes:
        ax.set_ylabel("Score")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(ls="--", lw=0.5, color="lightgray", zorder=-10)

    plt.tight_layout()
    cur = cursor(figure, hover=True)

    def set_annotation_text(sel: Selection):
        add_relative_score = False
        if sel.artist == scatter_unigram:
            scores = unigrams
        elif sel.artist == scatter_bigram:
            scores = bigrams
            add_relative_score = True
        else:
            return
        dct = scores[sel.index]

        labels = [
            f"left: {dct['__comment__left']}",
            f"right: {dct['__comment__right']}",
            f"score: {dct['score']:.2f}",
            f"rank: {dct['rank_type']}",
            f"key_indices: {dct['key_indices']}",
        ]

        if add_relative_score:
            ref_score = scores[0]["score"]
            labels.insert(3, f"relative_score: {dct['score']/ref_score:.2f}")

        sel.annotation.set_text("\n".join(labels))

    cur.connect("add", set_annotation_text)


DUMBBELL_PLOT_ONE_ROW_HEIGHT = 0.27

BIGRAM_DUMBBELL_PLOT_LEGEND_INFO = """Legend:
-------
SFB: Single Finger Bigram
REP: Repeated key

mi2u: Middle below index finger (2u)
mp1u: Middle below pinky (1u)
ip2u: Index below pinky (2u)
mr2u: Middle below ring finger (2u)
pr2u: Pinky below ring finger (2u)
mp2u: Middle below pinky (2u)
rp1u: Ring below pinky (1u)
rp2u: Ring below pinky (2u)
"""


ROWDIFF_MAPPING = {v: k for (k, v) in Hands.rowdiff_names.items()}


def bigram_scores_dumbbell_plot(data: dict[str, list[float | str]]):
    """Plots ngram scores and saves the plot to a file.

    Parameters
    ----------
    data :
        The data to plot. Should contain keys "rank", "score", "ngram_left",
        "ngram_right", "repeats", and "rowdiff". The rank is list of int, the score is
        list of float and the rest are list of strings.
    """

    fig, ax = plt.subplots()
    ax2 = ax.twiny()
    plt.sca(ax)

    y_tick_labels = []

    df = pd.DataFrame(data)
    df["length"] = df.ngram_left.str.len()
    df = df[df["length"] == 2]

    df.index = pd.Index(range(1, len(df) + 1), dtype=int)
    n_items = len(df)
    y_ticks = []
    shown_labels = set()

    for row in df.itertuples():

        ngram_left, ngram_right, score, index, repeats, rowdiff = (
            row.ngram_left,
            row.ngram_right,
            row.score,
            row.Index,
            row.repeats,
            row.rowdiff,
        )
        score = float(score)  # type: ignore[arg-type]
        idx = int(index)  # type: ignore[arg-type]
        y = n_items - idx + 1
        linecolor = "gray" if idx % 2 == 0 else "darkgray"

        y_ticks.append(y)
        ngram = str(ngram_left if ngram_left else ngram_right)

        y_tick_labels.append(f"{idx}: {ngram}")

        label, marker_color, marker = _get_label_color_marker(
            str(repeats), str(rowdiff)
        )
        if label not in shown_labels:
            shown_labels.add(label)
        else:
            label = None

        ax.scatter(
            score,
            y,
            color=marker_color,
            label=label,
            marker=marker,
        )

        ax.plot(
            [0, score],
            (y, y),
            color=linecolor,
            lw=1.0,
            zorder=-10,
        )

    ax.set_yticks(y_ticks, labels=y_tick_labels)

    # Make the secondary x-axis at the top same as the main x-axis at the bottom
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xlim(ax.get_xlim())

    ax.set_ylim(0, n_items + 1)
    plt.legend(loc="upper left")
    ax.grid(axis="x", linestyle="--", color="lightgray")

    # Add background shading
    max_score_int = ceil(df["score"].max())
    ax.set_xticks(range(0, max_score_int))
    major_ticks = ax.get_xticks()
    for i in range(len(major_ticks) - 1):
        if i % 2 == 0:
            ax.axvspan(
                major_ticks[i],
                major_ticks[i + 1],
                color="lightgray",
                alpha=0.1,
                zorder=-10,
            )
    if major_ticks[-1] % 2 == 0:
        # The rightmost shaded area needs special treatment as it may not be between
        # two ticks, but between the last tick and the end of the plot.
        xlim = ax.get_xlim()
        ax.axvspan(
            major_ticks[-1],
            major_ticks[-1] + 1,
            color="lightgray",
            alpha=0.1,
            zorder=-10,
        )
        ax.set_xlim(*xlim)

    fig.set_size_inches(20, 4 + DUMBBELL_PLOT_ONE_ROW_HEIGHT * n_items)
    plt.tight_layout()


def _get_label_color_marker(repeats: str, rowdiff: str):
    if repeats.startswith("SFB"):
        return "SFB", "tab:red", ">"
    elif repeats.startswith("REP"):
        return "REP", "orange", "p"
    if rowdiff == "" or rowdiff == "2u":
        return "normal", "darkgray", "o"

    rd = ROWDIFF_MAPPING[rowdiff]
    richcolor = Hands.rowdiff_colors[rd]
    rgb_tuple = tuple(Color.parse(richcolor).get_truecolor())
    hex_color = "#{:02x}{:02x}{:02x}".format(*rgb_tuple)
    return rowdiff, hex_color, "x"
