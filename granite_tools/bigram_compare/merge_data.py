"""Module for granite-bigram-compare-merge

Used to add ranking data to and existing comparison (.pickle) file by creating a new
comparison (.pickle) file.

The .pickle files are created by granite-bigram-compare and the ranking data file is
created by granite-bigram-ranking-initial or granite-bigram-ranking-view."""

import sys

from granite_tools.app_types import KeySeq
from granite_tools.bigram_compare.comparisons import keyseqs_to_comparisons
from granite_tools.bigram_compare.fitting import get_scores
from granite_tools.bigram_compare.scorer import (
    CompareDataDict,
    load_compare_pickle,
    save_compare_pickle,
    verify_data,
)
from granite_tools.bigram_scores.rankings import load_bigram_rankings


def main() -> None:
    """
    Merge new rankings data ("new.ranking", from granite-bigram-ranking-initial or
    granite-bigram-ranking-view) with existing comparison data (old.compare.pickle) to create a new
    updated comparison file (new.compare.pickle) to be used in  granite-bigram-compare.

    This is useful if your granite config has changed and you want to add new comparisons.

    Usage:
          <old.compare.pickle> <new.ranking> <new.compare.pickle>
    """
    if len(sys.argv) != 4:
        print(main.__doc__)
        sys.exit(1)

    old_compare_file = sys.argv[1]
    new_ranking_file = sys.argv[2]
    new_compare_file = sys.argv[3]

    old_compare = load_compare_pickle(old_compare_file)
    new_rankings: list[KeySeq] = load_bigram_rankings(new_ranking_file)

    new_compare = merge_compare_to_rankings(old_compare, new_rankings)
    save_compare_pickle(new_compare_file, new_compare)
    print("Saved to:", new_compare_file)


def merge_compare_to_rankings(
    compare: CompareDataDict, new_rankings: list[KeySeq]
) -> CompareDataDict:
    verify_data(compare)

    # number of automatic comparisons
    n_auto = len(compare["initial_order"]) - 1

    manual_comparisons = compare["comparisons_all"][n_auto:]
    new_automatic_comparisons = keyseqs_to_comparisons(new_rankings)

    comparisons_all = new_automatic_comparisons + manual_comparisons

    current_scores = get_scores(comparisons_all)
    current_order = sorted(new_rankings, key=lambda x: current_scores[x])
    new_compare = CompareDataDict(
        comparisons_all=comparisons_all,
        initial_order=new_rankings,
        current_order=current_order,
        processed_key_sequences=compare["processed_key_sequences"].copy(),
        current_scores=current_scores,
        p95_fraction=compare["p95_fraction"],
        pairs_per_sequence=compare["pairs_per_sequence"],
    )
    verify_data(new_compare)
    return new_compare


if __name__ == "__main__":
    main()
