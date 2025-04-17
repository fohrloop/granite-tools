from __future__ import annotations

import typing
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from granite_tools.trigram_features import TrigramFeatures

if typing.TYPE_CHECKING:
    from granite_tools.hands import Hands
    from granite_tools.trigram_model.scorer import TrigramScoreDict

DUMBBELL_PLOT_ONE_ROW_HEIGHT = 0.8

TRIGRAM_TYPE_COLORS = {
    "balanced": "#008f5d",
    "easy-rolling": "#de3d82",
    "onehand": "black",
    "alternating": "#cb5d00",
}


def plot_trigram_scores(
    scores: list[TrigramScoreDict],
    hands: Hands,
    outfile: Path,
) -> None:
    """Plots trigram scores and saves the plot to a file.

    Parameters
    ----------
    scores : list[TrigramScoreDict]
        The trigram score data.
    hands : Hands
        The hands data (from configuration file).
    outfile : Path
        Where to save the plot.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax_left, ax_right = axes
    ax_left_upper = ax_left.twiny()
    ax_right_upper = ax_right.twiny()
    plt.sca(ax_left)

    yticklabels_left = []
    yticklabels_right = []
    yticks = []
    gap_between_bars = 0.1
    width_of_bars = 0.25
    bar_displacement = (width_of_bars + gap_between_bars) / 2

    for i, scoredct in enumerate(scores[::-1], start=1):
        trigram_score = scoredct["trigram_score_details"]["score"]
        ref_score = scoredct["reference_score_details"]["score"]
        score_ratio_act = scoredct["score_ratio_actual"]
        score_ratio_pred = scoredct["score_ratio_pred"]
        trigram = scoredct["trigram"]
        ref_trigram = scoredct["reference_trigram"]

        features = TrigramFeatures.from_string(trigram, hands, None, True)
        ref_features = TrigramFeatures.from_string(ref_trigram, hands, None, True)

        ax_left.scatter(
            score_ratio_act,
            i,
            facecolors="none",
            edgecolor="black",
            s=30,
            marker="o",
            zorder=10,
        )
        ax_left.scatter(score_ratio_pred, i, color="red", s=15, zorder=20)

        ax_right.barh(
            i + bar_displacement,
            trigram_score,
            color="darkgray",
            height=0.25,
            zorder=10,
        )
        ax_right.barh(
            i - bar_displacement,
            ref_score,
            color="gray",
            height=0.25,
            zorder=10,
        )
        yticklabels_left.append(
            f"{trigram}/{ref_trigram}\n({features.trigram_subtype}/{ref_features.trigram_subtype})"
        )
        yticklabels_right.append(
            f"{trigram} ({features.trigram_subtype})\n\n{ref_trigram} ({ref_features.trigram_subtype})"
        )
        yticks.append(i)

    ax_left.set_yticks(yticks, yticklabels_left)

    fig.set_size_inches(12, 4 + DUMBBELL_PLOT_ONE_ROW_HEIGHT * len(scores))
    plt.grid(axis="x", linestyle="--", color="lightgray", zorder=-100)
    plt.axvline(x=1.0, color="black", linestyle="-", lw=0.5, zorder=2)
    # Make the secondary x-axis at the top same as the main x-axis at the bottom
    ax_left_upper.set_xlim(ax_left.get_xlim())
    ax_left.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_left_upper.set_xticks(ax_left.get_xticks(), ax_left.get_xticklabels())
    ax_left.set_xticks(ax_left_upper.get_xticks(), ax_left_upper.get_xticklabels())
    ax_left.set_xlabel("Score ratio")

    ax_right.grid(axis="x", linestyle="--", color="lightgray", zorder=-100)
    ax_right.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_right_upper.set_xticks(ax_right.get_xticks(), ax_right.get_xticklabels())
    ax_right.set_yticks(yticks, yticklabels_right)
    ax_right.set_xlabel("Score")
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=150)
    print("Plot saved to", outfile)
    plt.close()
