from __future__ import annotations

import typing
from pathlib import Path

from matplotlib import pyplot as plt

from granite_tools.easy_rolling import get_easy_rolling_type_mapping
from granite_tools.trigram_types import get_trigram_type

if typing.TYPE_CHECKING:
    from granite_tools.hands import Hands
    from granite_tools.trigram_model.scorer import TrigramScoreDict

DUMBBELL_PLOT_ONE_ROW_HEIGHT = 0.27

TRIGRAM_TYPE_COLORS = {
    "balanced": "#008f5d",
    "easy-rolling": "#de3d82",
    "onehand": "black",
    "alternating": "#cb5d00",
}


def plot_trigram_scores(
    groups: dict[str, list[TrigramScoreDict]],
    hands: Hands,
    outfile: Path,
    trigram_type: str = "all",
):
    """Plots trigram scores and saves the plot to a file.

    Parameters
    ----------
    groups : dict[str, list[TrigramScoreDict]]
        A dictionary with group names as keys and lists of trigram score dictionaries as values.
    hands : Hands
        The hands data (from configuration file).
    outfile : Path
        Where to save the plot.
    trigram_type : str
        The trigram type to plot. Default is "all". Other options are
        "balanced", "onehand", "redir", and "alternating". Note that only the non-reference
        trigrams are plotted since the reference trigrams are always scores 1.0 without
        any error. The reference trigrams are the FIRST trigrams of each trigram family
        in the trigram scoring file.
    """
    fig, ax = plt.subplots()
    ax2 = ax.twiny()
    plt.sca(ax)

    n_items = 0
    y_tick_labels: list[str] = []
    # TODO: is the mapping needed anymore?
    mapping = get_easy_rolling_type_mapping(hands.config.easy_rolling_trigrams, hands)
    for i, (_, scoredicts) in enumerate(groups.items()):
        row_is_odd = i % 2 == 1
        for d in reversed(scoredicts):

            type_of_trigram = get_trigram_type(d["trigram"], hands)
            linecolor = TRIGRAM_TYPE_COLORS[type_of_trigram]
            if trigram_type != "all" and trigram_type != type_of_trigram:
                continue
            n_items += 1
            y = n_items
            y_tick_labels.append(
                f'{d["trigram"]} ({type_of_trigram}, {d['reference_trigram']})'
            )
            marker = "o" if row_is_odd else "s"

            ax.scatter(
                d["score_ratio_actual"],
                y,
                color="cornflowerblue" if row_is_odd else "tab:blue",
                label="Target (manual)" if n_items == 1 else None,
                marker=marker,
            )
            ax.scatter(
                d["score_ratio_pred"],
                y,
                color="tomato" if row_is_odd else "tab:red",
                label="Model" if n_items == 1 else None,
                marker=marker,
            )

            ax.plot(
                (d["score_ratio_actual"], d["score_ratio_pred"]),
                (y, y),
                color=linecolor,
                lw=1.0,
                zorder=-10,
            )

    ax.set_yticks(list(range(1, n_items + 1)), labels=y_tick_labels)

    # Make the secondary x-axis at the top same as the main x-axis at the bottom
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xlim(ax.get_xlim())

    ax.set_ylim(0, n_items + 1)
    plt.legend(loc="upper left")
    ax.grid(axis="x", linestyle="--", color="lightgray")
    fig.set_size_inches(20, 4 + DUMBBELL_PLOT_ONE_ROW_HEIGHT * n_items)
    plt.tight_layout()
    plt.savefig(str(outfile), dpi=150)
    print("Plot saved to", outfile)
    plt.close()
