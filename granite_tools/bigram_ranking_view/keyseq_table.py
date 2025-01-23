from __future__ import annotations

import typing
from enum import Enum
from pathlib import Path

from rich.text import Text
from textual import events, on
from textual.binding import Binding
from textual.containers import Vertical
from textual.coordinate import Coordinate
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import DataTable, Input, Static
from textual.widgets.data_table import CellDoesNotExist

from granite_tools.bigram_ranking_file import load_bigram_rankings
from granite_tools.bigram_ranking_view.twowaydict import change_twowaydct_value_for_key

if typing.TYPE_CHECKING:
    from typing import ClassVar, Iterable

    from textual.binding import BindingType
    from textual.widgets.data_table import ColumnKey

    from granite_tools.hands import Hands

TABLE_COLUMNS = (
    "indices",
    "left",
    "right",
    "fingers",
    "repeats",
    "rowdiff",
    "direction",
)
TABLE_COLUMN_WIDTHS = (10, 5, 5, 7, 7, 9, 8)


class TableMode(str, Enum):
    moving_selection = "moving_selection"
    """The cursor moves the row up/down."""

    moving_cursor = "moving_cursor"
    """The cursor moves, but the rows stay."""


class KeySequenceTable(DataTable, inherit_bindings=True):
    """
    Important unlisted parts (inherited)

    cursor_row: int
        The index of current row
    """

    class Placed(Message):
        """Sent when a key sequence has been placed."""

        def __init__(
            self, keyseq: tuple[int, ...], left: str, right: str, row_label: str
        ) -> None:
            super().__init__()
            self.keyseq = keyseq
            self.left = left
            self.right = right
            self.row_label = row_label

    class DidNotPlace(Message):
        """Tells to the main app that the last key sequence was not placed (
        so that it will be sent with next Next())."""

        def __init__(self) -> None:
            super().__init__()

    class GoToRequested(Message):
        """Sent when the user wants to go to a specific row."""

        def __init__(self) -> None:
            super().__init__()

    class WriteLog(Message):
        """Sent when a message should be shown to user."""

        def __init__(self, text: Text | str) -> None:
            super().__init__()
            self.text = text if isinstance(text, Text) else Text(text)

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "cursor_up", "Cursor up", show=False),
        Binding("down", "cursor_down", "Cursor down", show=False),
        Binding("pageup", "page_up(30)", "Cursor up (30)", show=False),
        Binding("pagedown", "page_down(30)", "Cursor down (30)", show=False),
        Binding("ctrl+home", "page_up(300)", "Cursor up (300)", show=False),
        Binding("ctrl+end", "page_down(300)", "Cursor down (300)", show=False),
        Binding("home", "scroll_home", "Home", show=False),
        Binding("end", "scroll_end", "End", show=False),
        Binding("space", "change_table_mode", "Place/Grab", show=True),
        Binding("g", "goto", "Go to", show=True),
    ]

    def __init__(self, hands: Hands | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.hands = hands
        self.cursor_foreground_priority = "css"
        self.cursor_background_priority = "css"
        self.table_mode = TableMode.moving_cursor
        self.classes = []
        self._col_keys: list[ColumnKey] = []
        for column, width in zip(TABLE_COLUMNS, TABLE_COLUMN_WIDTHS):
            self._col_keys.append(self.add_column(column, width=width))
        self._set_bindings()

        self.loaded_permutations: set[tuple[int, ...]] = set()
        """If data is loaded from a file, this will contain all the permutations
        which have been loaded, so that they are not added again."""

        self.previously_added_row_index: int = -1
        """Keeps track of the index of the row that was added last"""

        self.currently_placing_key_seq: str | None = None

    def add_row_with_autolabel(
        self,
        *contents,
        loc="center",
        select_added_row=True,
        change_to_moving_selection=False,
    ) -> bool:
        """
        loc: str
            Where to add the row. If 'end', the row is added to the end of the table.
            If 'center', the row is added to the center of the table. If 'below_prev',
            the row is added below the previously added row."""
        # prevent polluting the logs with potentially hundreds or thousands
        # instances of the same RowHighlighted message.
        with self.prevent(self.RowHighlighted):
            return self._add_row_with_autolabel(
                *contents,
                loc=loc,
                select_added_row=select_added_row,
                change_to_moving_selection=change_to_moving_selection,
            )

    def _add_row_with_autolabel(
        self,
        *contents,
        loc="center",
        select_added_row=True,
        change_to_moving_selection=False,
    ) -> bool:
        """
        Parameters
        ----------
        loc: str
            Where to add the row. Can be  'end', 'center' or 'below_prev'.

        Returns
        -------
        added: bool
            True if the row was added, False if it was not (e.g. because it was
            already in the table).
        """

        length_before = len(self)
        # Rows are _always_ added first to the end of the table.
        key_seq, left, right, *other_cols = contents

        if key_seq in self.loaded_permutations:
            return False

        self.currently_placing_key_seq = ",".join(str(x) for x in key_seq)

        key_sequence = Text(self.currently_placing_key_seq)
        key_sequence.stylize("italic bright_black")
        left, right = Text(left), Text(right)
        left.stylize("sky_blue1 bold")
        right.stylize("light_pink1 bold")
        fingers = self.hands.get_fingers_str(key_seq) if self.hands else Text("")
        repeats = self.hands.get_repeats_text(key_seq) if self.hands else Text("")
        rowdiff = self.hands.get_rowdiff_text(key_seq) if self.hands else Text("")
        direction = self.hands.get_direction_text(key_seq) if self.hands else Text("")

        self.add_row(
            key_sequence,
            left,
            right,
            fingers,
            repeats,
            rowdiff,
            direction,
            *other_cols,
            label=str(length_before + 1),
        )

        length_new = length_before + 1

        if loc == "center" or loc == "below_prev":
            # move the added row to center (since we don't know if it's a high
            # or low effort key sequence)
            if self.previously_added_row_index < 0:
                loc = "center"

            if loc == "below_prev":
                new = min(self.previously_added_row_index + 1, length_new - 1)
            else:
                new = length_new // 2
            self.bubble_move(length_new - 1, new, select=select_added_row)
        elif loc == "end":
            if select_added_row:
                self.cursor_coordinate = Coordinate(length_new - 1, 0)
        else:
            raise ValueError(f"Invalid location: '{loc}'")

        if change_to_moving_selection:
            self.change_to_moving_selection()
        return True

    def _set_bindings(self):
        # Remove inherited bindings that are not used / needed
        # This removes both: the listing in the help panel and the actual
        # shortcut functionality.
        bindings = self._bindings.key_to_bindings
        bindings.pop("right")
        bindings.pop("left")
        bindings.pop("home")
        bindings.pop("end")
        bindings.pop("enter")
        bindings.pop("ctrl+home")
        bindings.pop("ctrl+end")
        bindings.pop("ctrl+pageup")
        bindings.pop("ctrl+pagedown")
        self._bindings.bind(
            "ctrl+pageup", "page_up(300)", "Cursor down (300)", show=False
        )
        self._bindings.bind(
            "ctrl+pagedown", "page_down(300)", "Cursor down (300)", show=False
        )
        self._bindings.bind("ctrl+up", "page_up(30)", "Cursor up (30)", show=True)
        self._bindings.bind("ctrl+down", "page_down(30)", "Cursor down (30)", show=True)

    def action_change_table_mode(self) -> None:
        if self.table_mode == TableMode.moving_selection:
            self.post_message_placed()
            self.change_to_moving_cursor()
        else:
            self.change_to_moving_selection()

    def change_to_moving_selection(self) -> None:
        # In this mode, the row moves up/down with the cursor
        self.table_mode = TableMode.moving_selection
        self.classes = []

    def change_to_moving_cursor(self) -> None:
        # In this mode, only the cursor moves, not the row
        self.table_mode = TableMode.moving_cursor
        self.classes = ["keyseq_moving-cursor-only"]

    def action_cursor_up(self) -> None:
        if self.table_mode == TableMode.moving_selection:
            self.move_current_row_up()
        else:
            super().action_cursor_up()

    def action_cursor_down(self) -> None:
        if self.table_mode == TableMode.moving_selection:
            self.move_current_row_down()
        else:
            super().action_cursor_down()

    def move_current_row_up(self) -> None:
        self._swap_current_row_with(self.cursor_coordinate.up())

    def move_current_row_down(self) -> None:
        self._swap_current_row_with(self.cursor_coordinate.down())

    def action_page_up(self, amount: int = 100) -> None:
        if self.table_mode == TableMode.moving_selection:
            self._move_multiple_up_or_down(amount, going_up=True)
        else:
            super().action_page_up()

    def action_page_down(self, amount: int = 100) -> None:
        if self.table_mode == TableMode.moving_selection:
            self._move_multiple_up_or_down(amount, going_up=False)
        else:
            super().action_page_down()

    def action_scroll_top(self) -> None:
        if self.table_mode == TableMode.moving_selection:
            current_row_index, _ = self.cursor_coordinate
            self.bubble_move(current_row_index, 0)
            self.action_cursor_down()  # this refreshes the view (no idea why)
            self.action_cursor_up()
        else:
            super().action_scroll_top()

    def action_scroll_bottom(self) -> None:
        if self.table_mode == TableMode.moving_selection:
            current_row_index, _ = self.cursor_coordinate
            self.bubble_move(current_row_index, len(self) - 1)
            self.action_cursor_up()  # This refreshes the view (no idea why)
            self.action_cursor_down()
        else:
            super().action_scroll_bottom()

    def post_message_placed(self):
        self.currently_placing_key_seq = None
        label = str(self.cursor_row + 1)
        self.previously_added_row_index = self.cursor_row
        self.post_message(self.Placed(*self.get_current_row(), row_label=label))

    def action_goto(self) -> None:

        self.post_message(self.GoToRequested())

    def goto_row(self, ngram: str) -> None:
        ngram_lower = ngram.lower()
        for row_idx, (_, left, right) in enumerate(self.iter_rows(plain=True)):
            if ngram_lower == left.lower() or ngram_lower == right.lower():
                break
        else:
            self.post_message(self.WriteLog(f"No matches for '{ngram}' not found."))
            return

        current = self.cursor_coordinate.row
        if self.table_mode == TableMode.moving_selection:
            rows_to_move = row_idx - current
            self._move_multiple_up_or_down(abs(rows_to_move), going_up=rows_to_move < 0)
        elif self.table_mode == TableMode.moving_cursor:
            self.cursor_coordinate = Coordinate(row_idx, 0)

    def _swap_current_row_with(self, new_coordinate: Coordinate) -> None:
        cellkey_current = self.coordinate_to_cell_key(self.cursor_coordinate)

        try:
            cellkey_new = self.coordinate_to_cell_key(new_coordinate)
        except CellDoesNotExist:
            # cannot move to a cell that does not exist
            return

        cells_new = self.get_row(cellkey_new.row_key)
        cells_current = self.get_row(cellkey_current.row_key)

        for content_new, content_current, col_key in zip(
            cells_new, cells_current, self._col_keys
        ):
            self.update_cell(cellkey_current.row_key, col_key, content_new)
            self.update_cell(cellkey_new.row_key, col_key, content_current)

        self.cursor_coordinate = new_coordinate

    def _move_multiple_up_or_down(
        self, amount: int = 100, going_up: bool = True
    ) -> None:
        """Moves the cursor up or down a "page" (multiple rows).

        Parameters
        ----------
        going_up:
            Should pass True if the cursor is moving up, False if moving down.
        """

        # The logic is taken from the DataTable.action_page_down and action_page_up
        # methods. If this breaks when updating Textual, check those methods.

        row_index, _ = self.cursor_coordinate

        if going_up:
            target_row = max(0, row_index - amount)
        else:
            target_row = min(len(self) - 1, row_index + amount)

        self.bubble_move(row_index, target_row)

        # No idea why but this is required for refresh
        if going_up:
            self.action_cursor_down()
            self.action_cursor_up()
        else:
            self.action_cursor_up()
            self.action_cursor_down()

    def bubble_move(self, old: int, new: int, select: bool = True) -> None:
        """ "Moves a row to a new location using the adjacent row swapping method used
        in the bubble sort algorithm.

        Parameters
        ----------
        old : int
            The index of the row to move.
        new : int
            The index to move the row to.
        select : bool
            If True, select the row after moving it.
        """
        self._bubble_move(old, new)
        if select:
            _, col_idx = self.cursor_coordinate
            self.cursor_coordinate = Coordinate(new, col_idx)

    def _bubble_move(self, old: int, new: int) -> None:

        if old == new:
            return
        old_row_key = self._row_locations.get_key(old)
        if old_row_key is None:
            return

        del self._row_locations[old_row_key]

        if new < old:
            # moving single row upwards; others move downwards
            src_idxs = list(range(old - 1, new - 1, -1))
            dest_idxs = [x + 1 for x in src_idxs]
        else:
            # moving single row downwards; others move upwards
            src_idxs = list(range(old + 1, new + 1))
            dest_idxs = [x - 1 for x in src_idxs]

        for src_idx, dest_idx in zip(src_idxs, dest_idxs):
            key = self._row_locations.get_key(src_idx)
            if key is None:
                raise RuntimeError(f"Row key not found for index {src_idx}")
            change_twowaydct_value_for_key(self._row_locations, key, dest_idx)
            row = self.rows[key]
            row.label = Text(str(dest_idx + 1))

        self._row_locations.__setitem__(key=old_row_key, value=new)  # type: ignore
        row = self.rows[old_row_key]
        row.label = Text(str(new + 1))

    def get_left(self, plain: bool = True) -> list[str]:
        return list(x[1] for x in self.iter_rows(plain))

    def get_right(self, plain: bool = True) -> list[str]:
        return list(x[2] for x in self.iter_rows(plain))

    def get_key_indices(self, plain: bool = True) -> list[str]:
        return list(x[0] for x in self.iter_rows(plain))

    def row_indices(self) -> Iterable[int]:
        return range(0, self.row_count)

    def iter_rows(self, plain: bool = False) -> Iterable[tuple[str, str, str]]:
        for i in self.row_indices():
            yield self.get_row_at(i, plain)[:3]

    def _cell_content_to_plain_text(self, content: object) -> str:
        if isinstance(content, Text):
            return content.plain
        return str(content)

    def get_current_left_right(self, plain: bool = True) -> tuple[str, str]:
        return self.get_current_row(plain)[1:3]

    def get_current_row(self, plain: bool = True) -> tuple[str, str, str]:
        return self.get_row_at(self.cursor_row, plain=plain)[:3]

    def get_current_key_seq(self) -> str:
        return self.get_key_seq_at(self.cursor_row)

    def get_key_seq_at(self, index: int) -> str:
        return self.get_row_at(index, plain=True)[0]

    def get_row_at(self, index: int, plain: bool = False) -> tuple[str, str, str]:  # type: ignore
        row = super().get_row_at(index)
        if not plain:
            out = row
        else:
            out = tuple(self._cell_content_to_plain_text(cell) for cell in row)  # type: ignore
        return typing.cast(tuple[str, str, str], out)

    def __len__(self) -> int:
        return self.row_count

    def save(self, path: str):
        """Saves the data table to a file. If a row is selected with the cursor
        to be moved, that row will be excluded (as it has no been placed yet)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        current_row = self.get_current_row(plain=True)

        with open(path, "w") as f:
            for row in self.iter_rows(plain=True):
                if self.table_mode == TableMode.moving_selection and row == current_row:
                    continue
                f.write(f"{row[0]}\n")

    def load(self, path: str, hands: Hands):

        key_seqs = load_bigram_rankings(path)

        for key_seq in key_seqs:
            if key_seq in self.loaded_permutations:
                raise FileHasDuplicatesError(
                    f'The file "{path}" contains duplicates: {key_seq} was found twice.'
                )
            self.add_row_with_autolabel(
                key_seq,
                hands.left.get_symbols_visualization(key_seq),
                hands.right.get_symbols_visualization(key_seq),
                loc="end",
                select_added_row=False,
                change_to_moving_selection=False,
            )
            self.loaded_permutations.add(key_seq)
        self.change_to_moving_cursor()

    async def _on_click(self, event: events.Click) -> None:
        # Disables mouse clicks on the data table
        event.prevent_default()


class FileHasDuplicatesError(RuntimeError):
    pass


class GotoModal(ModalScreen):

    def compose(self):
        with Vertical():
            yield Static("Goto (left or right hand sequence):")
            input_widget = Input(placeholder="Type here", id="goto-input", type="text")
            input_widget.focus()
            input_widget.max_length = 3
            yield input_widget

    @on(Input.Submitted)
    def should_close(self, message: Input.Submitted) -> None:
        self.dismiss(str(message.value))
