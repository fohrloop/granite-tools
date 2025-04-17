from __future__ import annotations

import itertools
import typing
from collections import defaultdict
from pathlib import Path

import pandas as pd
import typer

from granite_tools.app_types import HandType
from granite_tools.config import read_config
from granite_tools.easy_rolling import get_easy_rolling_type_mapping
from granite_tools.hands import get_hands_data
from granite_tools.trigram_features import TrigramFeatures

try:
    from typing import Annotated
except ImportError:
    # For older python versions
    from typing_extensions import Annotated  # type: ignore


def opposite_hand(hand: HandType) -> HandType:
    """Return the opposite hand."""
    if hand == "Left":
        return "Right"
    elif hand == "Right":
        return "Left"
    raise ValueError(f"Invalid hand: {hand}. Must be 'Left' or 'Right'.")


def get_canonical_hand_permutation(
    hand1: HandType, hand2: HandType, hand3: HandType
) -> tuple[HandType, HandType, HandType]:
    """Canonical hand permutation is the one with more Left hands keys in it."""
    left_count = sum([h == "Left" for h in (hand1, hand2, hand3)])
    right_count = 3 - left_count

    if left_count > right_count:
        return (hand1, hand2, hand3)
    out = typing.cast(
        tuple[HandType, HandType, HandType],
        tuple(opposite_hand(h) for h in (hand1, hand2, hand3)),
    )
    return out


ARG_CONFIG = Annotated[
    Path,
    typer.Argument(
        help="The path to the Granite configuration (YAML) file.",
        show_default=False,
    ),
]

ARG_OUTFILE = Annotated[
    Path | None,
    typer.Argument(
        help="The path to the output csv file (optional).",
        show_default=False,
    ),
]


ARG_NUMBER = Annotated[
    int,
    typer.Argument(
        help="Number of trigrams to show per group.",
        show_default=True,
    ),
]


def create_trigram_table_cli() -> None:
    typer.run(create_trigram_table)


def create_trigram_table(
    config_file: ARG_CONFIG,
    outfile: ARG_OUTFILE = None,
    n: ARG_NUMBER = 4,
) -> None:

    config = read_config(config_file)
    hands = get_hands_data(config)

    all_key_indices = sorted(
        hands.left.symbols_visualization.keys()
        | hands.right.symbols_visualization.keys()
    )

    hand_permutations = typing.cast(
        tuple[tuple[HandType, HandType, HandType]],
        tuple(itertools.product(["Left", "Right"], repeat=3)),
    )
    index_permutations = itertools.product(all_key_indices, repeat=3)

    identifiers = set()
    rows: defaultdict[object, list[str | None | tuple]] = defaultdict(list)
    mapping = get_easy_rolling_type_mapping(hands.config.easy_rolling_trigrams, hands)

    for indices in index_permutations:
        for h1, h2, h3 in hand_permutations:
            canonical_permutation = get_canonical_hand_permutation(h1, h2, h3)
            identifier = tuple(indices) + canonical_permutation

            try:
                trigram = (
                    hands.get_symbols_visualization(h1, [indices[0]])
                    + hands.get_symbols_visualization(h2, [indices[1]])
                    + hands.get_symbols_visualization(h3, [indices[2]])
                )
                flags = TrigramFeatures.from_string(
                    trigram, hands, mapping, use_visualization_symbols=True
                )
            except Exception:
                # Skip untypable trigrams
                print(f"Skipping untypable trigram: {identifier}")
                continue

            # The "mirrored" trigrams are the same. For example (Left, Right, Right)
            # and (Right, Left, Left)
            if identifier in identifiers:
                continue
            identifiers.add(identifier)

            rows["trigram"].append(trigram)
            rows["main_type"].append(flags.main_type)
            rows["subtype"].append(flags.trigram_subtype)
            rows["single_finger_pattern"].append(flags.single_finger_pattern)
            rows["vert2u"].append(flags.vert2u)
            rows["redir"].append(flags.redir)
            rows["identifier"].append(identifier)

    df = (
        pd.DataFrame(rows)
        .sort_values(by=["main_type", "subtype", "trigram"])
        .reset_index(drop=True)
    )

    groups = defaultdict(list)
    for i, (subtype, df_sub) in enumerate(df.groupby("subtype"), start=1):
        n_sub = len(df_sub)
        print(f"\n\n{subtype} ({n_sub} trigrams)")
        df_examples = df_sub.sample(n) if n_sub > n else df_sub
        print(df_examples)

        groups["subtype"].append(subtype)
        groups["n"].append(n_sub)

    df_groups = (
        pd.DataFrame(groups).sort_values(by="n", ascending=False).reset_index(drop=True)
    )
    df_groups.index += 1
    print("\n\nSubtypes")
    print(df_groups)

    if outfile is not None:
        df.to_csv(outfile, index=False, sep="\t")
        print(f"\nSaved to {outfile}")
    else:
        print("\nNo output file specified. Not saving.")
    breakpoint()


if __name__ == "__main__":
    create_trigram_table_cli()
