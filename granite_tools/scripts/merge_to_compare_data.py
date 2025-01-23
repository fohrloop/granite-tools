import sys
from pathlib import Path

from granite_tools.app_types import KeySeq
from granite_tools.compare.merge_data import merge_compare_to_rankings
from granite_tools.compare.scorer import (
    load_compare_pickle,
    save_compare_pickle,
    verify_data,
)
from granite_tools.order import load_key_seqs_from_file


def main() -> None:
    """
    Merge new rankings data ("new.ranking", from granite-scorer-baseline or
    granite-scorer-view) with existing comparison data (old.compare.pickle) to create a new
    updated comparison file (new.compare.pickle) to be used in  granite-scorer-compare.

    This is useful if your granite config has changed and you want to add new comparisons.

    Usage:
    python granite_tools/scripts/merge_to_compare_data.py  <old.compare.pickle> <new.ranking> <new.compare.pickle>

    Example:
    python granite_tools/scripts/merge_to_compare_data.py  tmp/old.compare.pickle tmp/granite-new.ranking tmp/granite.compare.pickle
    """
    if len(sys.argv) != 4:
        print(main.__doc__)
        sys.exit(1)

    old_compare_file = sys.argv[1]
    new_ranking_file = sys.argv[2]
    new_compare_file = sys.argv[3]

    if Path(new_compare_file).exists():
        print(f"{new_compare_file} exists already! Aborting")
        sys.exit(1)

    old_compare = load_compare_pickle(old_compare_file)
    new_rankings: list[KeySeq] = load_key_seqs_from_file(new_ranking_file)

    new_compare = merge_compare_to_rankings(old_compare, new_rankings)

    verify_data(new_compare)

    save_compare_pickle(new_compare_file, new_compare)

    print(f"Saved to: {new_compare_file}")


if __name__ == "__main__":
    main()
