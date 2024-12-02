from __future__ import annotations

import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    from typing import Callable, Literal

    KeySeq = tuple[int, ...]

    SideName = Literal["left", "right"]


@dataclass
class NgramPlacementState:

    highest_low: KeySeq | None
    lowest_high: KeySeq | None
    low_effort_sequences: list[KeySeq]
    hight_effort_sequences: list[KeySeq]


class NgramPlacementManager:
    """Manages the placement of ngrams in the ordered ngrams list. The user can move
    the current ngram using the methods of this class."""

    all_ngrams: list[KeySeq]
    """The permutations of the key sequences (corresponding to ngrams)."""

    _current_index: int
    """The current index of the permutations list."""

    current_ngram: KeySeq

    ordered_ngrams: list[KeySeq]
    """Holds the user selected order of key sequences. This gets a new member in the
    list (in a specific place; not necessarily to the end of the list) each time a key
    sequence is added. The items are taken from the all_key_sequences."""

    _current_ngram_movement_history: list[NgramPlacementState]

    def __init__(
        self, permutations: list[KeySeq], callback: None | Callable[[], None] = None
    ) -> None:
        self.all_ngrams = permutations
        self._current_index = 0
        self.current_ngram = tuple()
        self._current_ngram_movement_history = []
        self.ordered_ngrams = []
        self._callback = callback
        self._start_placing_next_ngram()

    def _start_placing_next_ngram(self) -> None:
        self.current_ngram = self.all_ngrams[self._current_index]

        self.placement_state = NgramPlacementState(
            *split_ordered_ngrams_into_two_halfs(
                self.ordered_ngrams, larger_size="right"
            )
        )
        self._current_ngram_movement_history = []
        self.refresh_callback()

    def place_current_ngram(self) -> KeySeq | None:
        """Place the current ngram in the ordered ngrams to the position determined by
        the `left_of_current` and `right_of_current` properties."""

        if self.is_finished():
            return None

        if self.left_of_current is not None:
            idx_left = self.ordered_ngrams.index(self.left_of_current)
            self.ordered_ngrams.insert(idx_left + 1, self.current_ngram)
        elif self.right_of_current is not None:
            idx_right = self.ordered_ngrams.index(self.right_of_current)
            self.ordered_ngrams.insert(idx_right, self.current_ngram)
        else:
            self.ordered_ngrams.append(self.current_ngram)

        ngram_placed = self.current_ngram
        self._current_index += 1

        if not self.is_finished():
            self._start_placing_next_ngram()
        else:
            self.refresh_callback()

        return ngram_placed

    def move_left(self):
        if self.left_of_current is None or self.is_finished():
            return
        self._current_ngram_movement_history.append(self.placement_state)
        self.placement_state = NgramPlacementState(
            *split_ordered_ngrams_into_two_halfs(
                self.ngrams_left_side_of_current, larger_size="right"
            )
        )
        self.refresh_callback()

    def move_right(self):
        if self.right_of_current is None or self.is_finished():
            return
        self._current_ngram_movement_history.append(self.placement_state)
        self.placement_state = NgramPlacementState(
            *split_ordered_ngrams_into_two_halfs(
                self.ngrams_right_side_of_current,
                larger_size="left",
            ),
        )
        self.refresh_callback()

    def move_back(self):
        """Move the ngram being currently placed 'back' in the movement history (if it
        has any)"""
        if self.is_finished():
            return
        if self._current_ngram_movement_history:
            self.placement_state = self._current_ngram_movement_history.pop()
        self.refresh_callback()

    def reset_current_ngram(self):
        """Resets the current ngram placement process."""
        if self.is_finished():
            return
        self._start_placing_next_ngram()
        self.refresh_callback()

    def previous_ngram(self):
        """Move back to the previous ngram."""
        if self._current_index < 1:
            self.reset_current_ngram()
            return

        prev_ngram = self.all_ngrams[self._current_index - 1]
        self.ordered_ngrams.remove(prev_ngram)
        self._current_index -= 1
        self._start_placing_next_ngram()
        self.refresh_callback()

    @property
    def left_of_current(self) -> KeySeq | None:
        """The ngram next to the current ngram (left side)"""
        return self.placement_state.highest_low

    @property
    def right_of_current(self) -> KeySeq | None:
        """The ngram next to the current ngram (right side)"""
        return self.placement_state.lowest_high

    @property
    def ngrams_left_side_of_current(self) -> list[KeySeq]:
        return self.placement_state.low_effort_sequences

    @property
    def ngrams_right_side_of_current(self) -> list[KeySeq]:
        return self.placement_state.hight_effort_sequences

    def is_finished(self) -> bool:
        return len(self.ordered_ngrams) >= len(self.all_ngrams)

    def current_placement_index(self) -> int | None:
        """When moving the ngram to be placed, the new (current) ngram can be
        thought of as being placed between two ngrams. This method returns the
        index where the ngram would be placed; index of the left side ngram + 1
        which is same as the index of the right side."""
        if self.is_finished():
            return None

        idx_left = (
            self.ordered_ngrams.index(self.left_of_current)
            if self.left_of_current is not None
            else None
        )
        idx_right = (
            self.ordered_ngrams.index(self.right_of_current)
            if self.right_of_current is not None
            else None
        )

        if idx_right is None and idx_left is None:
            return None

        idx = idx_left + 1 if idx_left is not None else idx_right

        return idx

    def ordered_ngrams_area_widths(self) -> tuple[int, int, int, int]:
        """The widths of the areas for the following:
        1) the left side of the current ngram which is NOT anymore part of the
         search area (the placement state)
        2) the left side of the current ngram which is still part of the search area.
        3) the right side of the current ngram which is still part of the search area.
        4) the right side of the current ngram which is NOT anymore part of the
         search area (the placement state)

        if the placement process is finished, returns (0, 50, 50, 0).
        """
        idx = self.current_placement_index()
        if self.is_finished() or idx is None:
            return 0, 50, 50, 0

        n = len(self.ordered_ngrams)

        left_all = idx
        right_all = n - left_all
        left_search_area = len(self.ngrams_left_side_of_current)
        right_search_area = len(self.ngrams_right_side_of_current)
        left_side = left_all - left_search_area
        right_side = right_all - right_search_area
        return left_side, left_search_area, right_search_area, right_side

    def refresh_callback(self):
        if self._callback:
            self._callback()

    def load_state(self, ordered_ngrams: list[KeySeq]):
        n_new = len(ordered_ngrams)
        if set(ordered_ngrams) != set(self.all_ngrams[:n_new]):
            raise ValueError(
                "The data cannot be loaded because it contains ngrams not supported by the configuration."
            )
        self.ordered_ngrams = ordered_ngrams
        self._current_index = n_new
        if not self.is_finished():
            self._start_placing_next_ngram()
        else:
            self.refresh_callback()


def split_ordered_ngrams_into_two_halfs(
    ordered_key_sequences: list[KeySeq],
    larger_size: SideName = "right",
) -> tuple[KeySeq | None, KeySeq | None, list[KeySeq], list[KeySeq]]:
    """Split the ordered key sequences into left and right hand sequences.

    If the split is uneven, the `larger_size` side will have one more sequence.

    Parameters
    ----------
    ordered_key_sequences : list[KeySeq]
        The ordered key sequences.

    Returns
    -------
    highest_low, lowest_high, low_effort_sequences, hight_effort_sequences : tuple[KeySeq | None, KeySeq | None, list[KeySeq], list[KeySeq]]
        left: the last (highest effort) sequence of the left (low effort) side.
        right: the first (lowest effort) sequence of the right (high effort) side.
        low_effort_sequences: the left (low effort) side sequences.
        hight_effort_sequences: the right (high effort) side sequences.
    """
    n = len(ordered_key_sequences)
    midpoint = n // 2

    if n / 2 != midpoint and larger_size == "left":
        midpoint += 1

    low_effort_sequences = ordered_key_sequences[:midpoint]
    hight_effort_sequences = ordered_key_sequences[midpoint:]
    highest_low = low_effort_sequences[-1] if low_effort_sequences else None
    lowest_high = hight_effort_sequences[0] if hight_effort_sequences else None

    return highest_low, lowest_high, low_effort_sequences, hight_effort_sequences
