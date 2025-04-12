"""Plots bigram scores and saves the plot to a file.

Usage:

    python granite_tools/scripts/plot_ngram_scores.py CONFIG_FILE BIGRAM_RANKING_FILE ANCHOR_SCORES_RAW_FILE OUTFILE_FIGURE

Example:
    python granite_tools/scripts/plot_ngram_scores.py examples/config.yml tmp/granite.bigram.ranking tmp/bigram-anchor-scores-raw.json tmp/granite.scores-plot.svg

where

    CONFIG_FILE is the path to the granite configuration YAML file.
    BIGRAM_RANKING_FILE is the path to the bigram (+unigram) ranking file (e.g., granite.bigram.ranking).
    ANCHOR_SCORES_RAW_FILE is the path to the anchor scores raw file (e.g., bigram-anchor-scores-raw.json).
    OUTFILE_FIGURE is the path to save the plot figure (e.g., granite.scores-plot.svg).
"""

from __future__ import annotations

import sys
import typing
from collections import defaultdict
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from rich.color import Color

from granite_tools.bigram_scores.anchor_scores import read_raw_anchor_scores_json
from granite_tools.bigram_scores.rankings import load_bigram_rankings
from granite_tools.bigram_scores.spline_smoothing import get_spline_scores
from granite_tools.config import read_config
from granite_tools.hands import Hands, get_hands_data
from granite_tools.utils import get_linear_scaling_function

# mypy: ignore-errors
DUMBBELL_PLOT_ONE_ROW_HEIGHT = 0.27

LEGEND_INFO = """Legend:
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


class NgramDataDict(typing.TypedDict):
    x: list[float]
    y: list[float]
    ngram_left: list[str]
    ngram_right: list[str]


ROWDIFF_MAPPING = {v: k for (k, v) in Hands.rowdiff_names.items()}


def get_label_color_marker(repeats: str, rowdiff: str):
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


def plot_trigram_scores(
    data: NgramDataDict,
    outfile: Path,
):
    """Plots ngram (unigram and bigram) scores and saves the plot to a file.

    Parameters
    ----------
    data : NgramDataDict
        The data to plot.
    outfile : Path
        Where to save the plot.
    """
    fig, ax = plt.subplots()
    ax2 = ax.twiny()
    plt.sca(ax)

    y_tick_labels = []

    df = pd.DataFrame(data)
    df["length"] = df.ngram_left.str.len()
    df = df[df["length"] == 2]

    s = get_linear_scaling_function(
        oldmin=min(df["y"]), oldmax=max(df["y"]), newmin=1, newmax=5
    )
    df["y"] = df["y"].apply(s)

    df.index = range(1, len(df) + 1)
    n_items = len(df)
    y_ticks = []
    shown_labels = set()

    for row in df.itertuples():

        ngram_left, ngram_right, score, index, repeats, rowdiff = (
            row.ngram_left,
            row.ngram_right,
            row.y,
            row.Index,
            row.repeats,
            row.rowdiff,
        )
        y = n_items - index + 1
        linecolor = "gray" if index % 2 == 0 else "darkgray"

        y_ticks.append(y)
        ngram = ngram_left if ngram_left else ngram_right

        y_tick_labels.append(f"{index}: {ngram}")

        label, marker_color, marker = get_label_color_marker(repeats, rowdiff)
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
    ax.set_xticks(range(0, 6))
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

    fig.set_size_inches(20, 4 + DUMBBELL_PLOT_ONE_ROW_HEIGHT * n_items)
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=150)
    print("Plot saved to", outfile)
    plt.close()

    plt.plot(
        df.index,
        df.y,
        marker=".",
        ls="",
        markersize=5,
        color="blue",
    )
    plt.xlabel("Rank of bigram")
    plt.ylabel("Score of bigram")
    plt.show()


if __name__ == "__main__":

    try:
        config_file = sys.argv[1]
        bigram_ranking_file = sys.argv[2]
        scores_raw_out_file = sys.argv[3]
        outfile_figure = sys.argv[4]
    except IndexError:
        print(__doc__)
        sys.exit(1)

    config = read_config(config_file)
    hands = get_hands_data(config)
    ngrams_ordered = load_bigram_rankings(bigram_ranking_file)
    scores = read_raw_anchor_scores_json(scores_raw_out_file)

    y_all, ranks = get_spline_scores(ngrams_ordered, scores)
    data = defaultdict(list)
    data["y"] = list(y_all)
    data["x"] = list(ranks)

    for keyseq in ngrams_ordered:
        data["ngram_left"].append(hands.get_symbols_visualization("Left", keyseq))
        data["ngram_right"].append(hands.get_symbols_visualization("Right", keyseq))
        data["repeats"].append(str(hands.get_repeats_text(keyseq)))
        data["rowdiff"].append(str(hands.get_rowdiff_text(keyseq)))

    print(LEGEND_INFO)
    plot_trigram_scores(data, Path(outfile_figure))
