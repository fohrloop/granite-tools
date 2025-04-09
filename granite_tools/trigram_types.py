from __future__ import annotations

import typing

if typing.TYPE_CHECKING:

    from granite_tools.app_types import TrigramMainType
    from granite_tools.hands import Hands

    Coord = tuple[int, int]
    TrigramShapeCoords = tuple[Coord, Coord, Coord]


class UnTypableTrigramError(RuntimeError):
    """Exception raised when a trigram is untypable."""

    def __init__(self, trigram: str):
        super().__init__(f"Trigram '{trigram}' is untypable.")


def get_trigram_type(
    trigram: str,
    hands: Hands,
    use_visualization_symbols: bool = False,
) -> TrigramMainType:
    """Get a trigram type. The trigram mapping can be provided as an argument which
    makes the trigram type checking much faster.

    Parameters
    ----------
    trigram : str
        The trigram to get the type for.
    hands : Hands
        The hands object to use for the trigram type checking. Created with the
        get_hands_data() function.
    use_visualization_symbols : bool
        Whether to use the `symbols_scoring` (False, default) or the
        `symbols_visualization` (True) from the configuration as the source of
        truth. Has only effect on special symbols like space.

    Raises
    ------
    ValueError
        If the `trigram` is not a valid trigram.
    UnTypableTrigramError
        If the `trigram` is untypable on the keyboard with the configuration defined
        in the `hands` object.
    RuntimeError
        If the trigram type is not recognized. (This should never happen.)
    """
    if len(trigram) != 3:
        raise ValueError(f"Not a trigram! '{trigram}'")

    indices, keytypes = hands.where(trigram, visualization=use_visualization_symbols)
    indices = typing.cast(tuple[int, int, int], tuple(indices))

    if "Untypable" in keytypes:
        raise UnTypableTrigramError(trigram)

    if keytypes[0] == keytypes[1] == keytypes[2]:
        return "onehand"
    elif keytypes[0] == keytypes[1] or keytypes[1] == keytypes[2]:
        return "balanced"
    elif keytypes[0] == keytypes[2] and keytypes[0] != keytypes[1]:
        return "alternating"
    raise RuntimeError(f"Trigram type not recognized for '{trigram}'")
