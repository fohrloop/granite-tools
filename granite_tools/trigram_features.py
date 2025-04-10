from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Literal, Sequence

from granite_tools.app_types import FingerType, HandOrKey, Vert2uPenaltyConfig
from granite_tools.easy_rolling import get_easy_rolling_type_mapping
from granite_tools.trigram_types import get_trigram_type

if typing.TYPE_CHECKING:
    from granite_tools.easy_rolling import EasyRollingTrigramsMap
    from granite_tools.hands import Hand, Hands
    from granite_tools.trigram_types import TrigramMainType

SingleFingerPatternFlag = Literal["SFB", "SFT", "SFTb", "SFS", "SFSb"] | None
Vert2uFeatureFlag = Literal["v1x", "v2x"] | None
RedirFeatureFlag = Literal["redir"] | None
EasyRollingFeatureFlag = Literal["easy-rolling"] | None


@dataclass(frozen=True)
class TrigramFeatures:
    """Contains all information about a trigram."""

    main_type: TrigramMainType
    single_finger_pattern: SingleFingerPatternFlag = None
    vert2u: Vert2uFeatureFlag = None
    redir: RedirFeatureFlag = None
    easy_rolling: EasyRollingFeatureFlag = None

    @classmethod
    def from_string(
        cls,
        trigram: str,
        hands: Hands,
        mapping: EasyRollingTrigramsMap | None = None,
        use_visualization_symbols: bool = False,
    ) -> TrigramFeatures:
        """Create a TrigramFeatures object from a string.

        Parameters
        ----------
        trigram : str
            The trigram to calculate the features for.
        hands : Hands
            The hands object to use for the trigram type checking. Created with the
            get_hands_data() function.
        mapping : EasyRollingTrigramsMap | None
            The mapping to use for the trigram type checking. If None, the mapping will
            be created from the hands object. Required for the easy rolling
            trigram type. Since creating of this mapping is expensive, it is recommended
            to create it once (with get_easy_rolling_type_mapping()) and pass it to this
            function.
        use_visualization_symbols : bool
            Whether to use the `symbols_scoring` (False, default) or the
            `symbols_visualization` (True) from the configuration as the source of
            truth. Has only effect on special symbols like space.
        """
        return cls(
            main_type=get_trigram_type(
                trigram,
                hands,
                use_visualization_symbols=use_visualization_symbols,
            ),
            single_finger_pattern=get_single_finger_pattern(
                trigram, hands, use_visualization_symbols=use_visualization_symbols
            ),
            vert2u=get_vert2u_feature(
                trigram, hands, use_visualization_symbols=use_visualization_symbols
            ),
            redir=get_redir_feature(
                trigram, hands, use_visualization_symbols=use_visualization_symbols
            ),
            easy_rolling=get_easy_rolling_feature(
                trigram,
                hands,
                mapping,
                use_visualization_symbols=use_visualization_symbols,
            ),
        )

    @property
    def trigram_subtype(self) -> str:
        """Get the name of the trigram subtype."""
        if (
            self.single_finger_pattern is None
            and self.vert2u is None
            and self.redir is None
            and self.easy_rolling is None
        ):
            return self.main_type

        if self.main_type != "onehand":
            raise RuntimeError(
                'Expecting only "onehand" trigram type to have any additional feature flags.'
            )
        parts = (
            pt
            for pt in (
                self.easy_rolling,
                self.single_finger_pattern,
                self.redir,
                self.vert2u,
            )
            if pt
        )
        return "|".join(parts)


def get_single_finger_pattern(
    trigram: str,
    hands: Hands,
    use_visualization_symbols: bool = False,
) -> SingleFingerPatternFlag:
    """Get the single finger patter flag for a trigram.

    Parameters
    ----------
    trigram : str
        The trigram to get the single finger pattern for.
    hands : Hands
        Contains the configuration.
    """
    indices, keytypes = hands.where(trigram, visualization=use_visualization_symbols)
    onehand = (keytypes[0] == keytypes[1]) and (keytypes[0] == keytypes[2])
    if not onehand or keytypes[0] == "Untypable":
        return None
    hand = hands.left if keytypes[0] == "Left" else hands.right

    if len(indices) != 3:
        raise ValueError(f"Not a trigram! Indices: '{indices}'")

    fingers_ = [hand.get_finger(idx) for idx in indices]

    if None in fingers_:
        raise ValueError(
            f"One of the keys in the trigram is not assigned to any finger!: '{indices=}'"
        )
    fingers = typing.cast(Sequence[FingerType], fingers_)

    if _is_sftb(indices, fingers):
        return "SFTb"
    elif _is_sft(indices, fingers):
        return "SFT"
    elif _is_sfsb(indices, fingers):
        return "SFSb"
    elif _is_sfs(indices, fingers):
        return "SFS"
    elif _is_sfb(indices, fingers):
        return "SFB"
    return None


def _is_sfb(indices: Sequence[int], fingers: Sequence[FingerType]) -> bool:
    """Checks if a trigram contains a single finger bigram (SFB)."""
    first_pair = fingers[0] == fingers[1] and indices[0] != indices[1]
    second_pair = fingers[1] == fingers[2] and indices[1] != indices[2]

    # One of the pairs must be a SFB, but if both are, it is SFT.
    return first_pair != second_pair


def _is_sft(indices: Sequence[int], fingers: Sequence[FingerType]) -> bool:
    """Checks if a trigram is a single finger trigram (SFT).
    * The trigram is a onehand (LLL or RRR),
    * All the three keys are typed with the same finger,
    * All the three keys are different from each other.
    """
    first_pair = fingers[0] == fingers[1] and indices[0] != indices[1]
    second_pair = fingers[1] == fingers[2] and indices[1] != indices[2]

    keys_different = len(set(indices)) == len(indices)

    return first_pair and second_pair and keys_different


def _is_sftb(indices: Sequence[int], fingers: Sequence[FingerType]) -> bool:
    """Checks if a trigram is a single finger boomerang trigram (SFTb).
    * The trigram is a onehand (LLL or RRR),
    * All the three keys are typed with the same finger,
    * First and third key are the same, but other keys are different from each other,
       and
    * Second key is different from the first (and last) key (if they're the same, it
      would be a three key repeat)
    """
    first_pair = fingers[0] == fingers[1] and indices[0] != indices[1]
    second_pair = fingers[1] == fingers[2] and indices[1] != indices[2]

    first_and_last_same = indices[0] == indices[2]

    return first_pair and second_pair and first_and_last_same


def _is_sfs(indices: Sequence[int], fingers: Sequence[FingerType]) -> bool:
    """Checks if a trigram is a single finger skipgram (SFS). If trigram is SFS:
    * The trigram is a onehand (LLL or RRR),
    * First and last key are typed with the same finger, but are NOT the same key
      (otherwise, it is a SFSb), and
    * The second key is typed with different finger than the others
    """
    first_and_last_same = fingers[0] == fingers[2] and indices[0] != indices[2]
    second_key_different = fingers[1] != fingers[2] and fingers[1] != fingers[0]

    return first_and_last_same and second_key_different


def _is_sfsb(indices: Sequence[int], fingers: Sequence[FingerType]) -> bool:
    """Checks if a trigram is a single finger boomerang skipgram (SFSb). If trigram is
      SFSb:
    * The trigram is a onehand (LLL or RRR),
    * First and last key are the same key, and
    * The second key is typed with different finger than the others (if it would be the
       same finger, it would be a SFTb)
    """
    first_and_last_same_key = indices[0] == indices[2]
    second_key_different = fingers[1] != fingers[2] and fingers[1] != fingers[0]

    return first_and_last_same_key and second_key_different


def get_vert2u_feature(
    trigram: str, hands: Hands, use_visualization_symbols: bool = False
) -> Vert2uFeatureFlag:
    """Get the feature flag for vert2u feature for the trigram.

    A trigram with a vert2u is a onehand (LLL or RRR), and contains a "vert2u" bigram, which

    Has 2u upwards or downwards movement,
    contains no thumb key and
    is not part of the vert2u_penalties.exceptions list.


    Parameters
    ----------
    trigram : str
        The trigram to get the single finger pattern for.
    hands : Hands
        Contains the configuration.

    Returns
    -------
    Vert2uFeatureFlag
        "v1x" if the trigram has a vert2u feature with 1x penalty,
        "v2x" if the trigram has a vert2u feature with 2x penalty,
        None if the trigram does not have a vert2u feature.
    """
    indices, keytypes = hands.where(trigram, visualization=use_visualization_symbols)
    onehand = (keytypes[0] == keytypes[1]) and (keytypes[0] == keytypes[2])
    if not onehand or keytypes[0] == "Untypable":
        return None
    hand = hands.left if keytypes[0] == "Left" else hands.right

    first = bigram_is_vert2u(indices[0:2], hand, hands.config.vert2u_penalties)
    second = bigram_is_vert2u(indices[1:3], hand, hands.config.vert2u_penalties)

    if not any((first, second)):
        return None
    if all((first, second)):
        return "v2x"
    return "v1x"


def bigram_is_vert2u(
    indices: Sequence[int], hand: Hand, vert2u_penalty_config: Vert2uPenaltyConfig
) -> bool:
    """Checks if a bigram is a vert2u bigram. A vert2u bigram is:

    * written with single hand
    * Has 2u upwards or downwards movement,
    * contains no thumb key and
    * is not part of the vert2u_penalties.exceptions list in the configuration

    Parameters
    ----------
    bigram: str
        The bigram to check.
    hands: Hands
        Contains the configuration (and the vert2u_penalties section)

    Returns
    -------
    bool
        True if the bigram is a vert2u bigram, False otherwise.
    """
    if len(indices) != 2:
        raise ValueError(f"Not a bigram! Indices: '{indices}'")

    fingers = [hand.get_finger(idx) for idx in indices]
    if FingerType.T in fingers:
        return False  # Thumb key present

    matrix_positions = [hand.matrix_positions[idx] for idx in indices]
    row1, key_index1 = matrix_positions[0][1], indices[0]
    row2, key_index2 = matrix_positions[1][1], indices[1]
    if abs(row1 - row2) != 2:
        return False  # not 2u

    if row1 < row2:  # lowest row number is at the top.
        top = key_index1  # row1
        bottom = key_index2  # row2
    else:
        top = key_index2
        bottom = key_index1

    # Special cases from exceptions list
    for exception in vert2u_penalty_config.get("exceptions", []):
        exc_bottom, exc_tops = exception
        for exc_top in exc_tops:
            if (bottom, top) == (exc_bottom, exc_top):
                return False

    return True


def get_redir_feature(
    trigram: str, hands: Hands, use_visualization_symbols: bool = False
) -> RedirFeatureFlag:
    """Get the feature flag for redir feature for the trigram.

    A trigram has the redir feature if:

    * The trigram is a onehand (LLL or RRR),
    * All keys must be located on a different column,
    * None of the keys is a thumb key, AND
    * Column numbering is not monotonically increasing or decreasing
      (there must be a direction change)

    Returns
    -------
    RedirFeatureFlag
        "redir" if the trigram has the redir feature, None otherwise.
    """
    indices, keytypes = hands.where(trigram, visualization=use_visualization_symbols)
    onehand = (keytypes[0] == keytypes[1]) and (keytypes[0] == keytypes[2])
    if not onehand or keytypes[0] == "Untypable":
        return None
    hand = hands.left if keytypes[0] == "Left" else hands.right

    # All keys are typed with different finger
    fingers = [hand.get_finger(idx) for idx in indices]

    # All keys are on same row
    matrix_positions = [hand.matrix_positions[idx] for idx in indices]
    cols = [pos[0] for pos in matrix_positions]

    # None of the keys is a thumb
    if FingerType.T in fingers:
        return None  # Thumb key present

    if len(set(cols)) != 3:
        return None  # All keys are not on different columns

    if (cols[2] > cols[1] > cols[0]) or (cols[0] > cols[1] > cols[2]):
        # Direction is not changing
        return None

    return "redir"


def get_easy_rolling_feature(
    trigram: str,
    hands: Hands,
    mapping: EasyRollingTrigramsMap | None = None,
    use_visualization_symbols: bool = False,
) -> EasyRollingFeatureFlag:
    """Get the easy-rolling feature flag for a trigram.

    A trigram has the easy-rolling feature if the trigram is a onehand and defined in
    the easy_rolling_trigrams section in the granite configuration.

    Returns
    -------
    EasyRollingFeatureFlag
        "easy-rolling" if the trigram is easy-rolling. Otherwise None.
    """

    if len(trigram) != 3:
        raise ValueError(f"Not a trigram! Trigram: '{trigram}'")

    indices, keytypes = typing.cast(
        tuple[tuple[int, int, int], tuple[HandOrKey, HandOrKey, HandOrKey]],
        hands.where(trigram, visualization=use_visualization_symbols),
    )

    onehand = (keytypes[0] == keytypes[1]) and (keytypes[0] == keytypes[2])

    if not onehand or "Untypable" in keytypes:
        return None

    mapping = mapping or get_easy_rolling_type_mapping(
        hands.config.easy_rolling_trigrams, hands
    )

    if not mapping:
        return None
    if mapping.get(indices):
        return "easy-rolling"
    return None
