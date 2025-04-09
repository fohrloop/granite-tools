"""Draw a scatter plot from the data.

Running:

uv run granite_tools/scripts/check_trigram_scores_plot.py examples/config.yml <bigram.ranking> <trigram.relative.scoring>

where <bigram.ranking> is a bigram (+unigram) ranking file and
      <trigram.relative.scoring> a trigram relative scoring file.
"""

import sys

import pandas as pd
import seaborn as sns  # type: ignore
from matplotlib import pyplot as plt

from granite_tools.trigram_model import get_trigram_data_from_files

pd.set_option("display.max_rows", 1000)


def get_rmse(df):
    return float((df["score_ratio_resid"] ** 2).mean() ** 0.5)


if __name__ == "__main__":
    sns.set_theme(style="ticks")
    config = sys.argv[1]
    bigram_ranking_file = sys.argv[2]
    trigram_score_file = sys.argv[3]
    raw_anchor_ngram_scores_file = sys.argv[4]

    df = get_trigram_data_from_files(
        config,
        bigram_ranking_file,
        trigram_score_file,
        raw_anchor_ngram_scores_file,
    ).reset_index()
    df["score_ratio_abs_resid"] = df["score_ratio_resid"].abs()

    print("RMSE\n----")
    print("  Total:", round(get_rmse(df), 3))
    for group, group_df in df.groupby("trigram_type"):
        print(" ", group, round(get_rmse(group_df), 3))

    sns.relplot(
        x="ref_bigram_sum",
        y="score_ratio_resid",
        hue="ref_trigram_type",
        size="score_ratio_abs_resid",
        sizes=(40, 400),
        alpha=0.5,
        palette="muted",
        height=6,
        data=df,
    )
    plt.tight_layout()
    plt.show()

    print(len(df))
