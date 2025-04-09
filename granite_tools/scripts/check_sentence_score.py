"""Draw a scatter plot from the data.

Running:

uv run granite_tools/scripts/check_sentence_score.py <config-file.yaml> <bigram.ranking> <trigram.relative.scoring>

where <config-file.yaml> is the configuration file, for example:  examples/config.yml
      <bigram.ranking> is a bigram (+unigram) ranking file

The <config-file.yaml> must contain the fitted model parameters.
"""

import sys

import numpy as np
import pandas as pd

from granite_tools.bigram_scores import load_bigram_and_unigram_scores
from granite_tools.config import read_config
from granite_tools.hands import get_hands_data
from granite_tools.trigram_model.scorer import TrigramModelParameters, get_trigram_score

pd.set_option("display.max_rows", 1000)


def get_score_df(text: str) -> pd.DataFrame:
    data = []
    for i in range(len(text) - 2):
        ngram = text[i : i + 3]
        try:
            scoredct = get_trigram_score(
                ngram, hands, params, bigram_and_unigram_scores
            )
            if scoredct.get("untypable", False):
                continue
            d = dict(
                trigram=ngram,
                score=scoredct["score"],
                trigramtype=scoredct["trigramtype"],
                vert2u=scoredct.get("vert2u", np.nan),
                # TODO: fix
                sfb_in_onehand=scoredct.get("sfb_in_onehand", np.nan),
                sft=scoredct.get("sft", np.nan),
            )
            data.append(d)
        except Exception as e:
            print(f"ERROR when processing {ngram}: ", e)
            raise e
    df = pd.DataFrame(data)
    breakpoint()  # TODO: fix later
    return df


if __name__ == "__main__":
    config_file = sys.argv[1]
    bigram_ranking_file = sys.argv[2]
    raw_anchor_ngram_scores_file = sys.argv[3]

    bigram_and_unigram_scores = load_bigram_and_unigram_scores(
        bigram_ranking_file, raw_anchor_ngram_scores_file
    )

    config = read_config(config_file)
    params = TrigramModelParameters.from_config(config)
    hands = get_hands_data(config)
    while True:
        some_text = input("type some text: ")
        df = get_score_df(some_text)
        print(df)
