import re
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from textual.app import App

from granite_tools.bigram_ranking_file import DuplicateValuesError
from granite_tools.bigram_ranking_view.keyseq_table import KeySequenceTable
from granite_tools.hands import Hands

test_folder = Path(__file__).parent.parent
examples_folder = test_folder.parent / "examples"


@pytest.mark.asyncio
class TestKeySequenceTable:

    class DataTableApp(App):
        def compose(self):
            table = KeySequenceTable()
            yield table

    async def test_rows_are_added_to_center(self):
        app = self.DataTableApp()
        async with app.run_test():
            table = app.query_one(KeySequenceTable)

            table.add_row_with_autolabel((0,), "1", "a")
            table.add_row_with_autolabel((1,), "2", "b")
            table.add_row_with_autolabel((2,), "3", "c")
            table.add_row_with_autolabel((3,), "4", "d")
            table.add_row_with_autolabel((4,), "5", "e")
            table.add_row_with_autolabel((5,), "6", "f")
            table.add_row_with_autolabel((6,), "7", "g")
            table.add_row_with_autolabel((7,), "8", "h")
            table.add_row_with_autolabel((8,), "9", "i")
            table.add_row_with_autolabel((9,), "0", "j")

            # This has previously had a bug which only occurred when the amount
            # of added items is >= 5.
            # fmt: off
            assert table.get_left() == ["1", "3", "5", "7", "9", "0", "8", "6", "4", "2"]
            assert table.get_right() == ["a", "c", "e", "g", "i", "j", "h", "f", "d", "b"]
            # fmt: on

    async def test_moving_with_page_up_and_down(self):
        app = self.DataTableApp()
        async with app.run_test() as pilot:
            table = app.query_one(KeySequenceTable)

            table.add_row_with_autolabel((0,), "1", "a")
            table.add_row_with_autolabel((1,), "2", "b")
            table.add_row_with_autolabel((2,), "3", "c")
            table.add_row_with_autolabel((3,), "4", "d")
            table.add_row_with_autolabel((4,), "5", "e")
            table.add_row_with_autolabel((5,), "6", "f")
            table.add_row_with_autolabel((6,), "7", "g")
            table.add_row_with_autolabel((7,), "8", "h")
            table.add_row_with_autolabel((8,), "9", "i")
            table.add_row_with_autolabel((9,), "0", "j")
            table.change_to_moving_selection()

            # The staring point
            assert table.get_current_left_right() == ("0", "j")
            # fmt: off
            assert table.get_left() == ["1", "3", "5", "7", "9", "0", "8", "6", "4", "2"]
            # fmt: on

            # Now, press Ctrl+up (Page Up)
            await pilot.press("ctrl+up")
            # Current selection does not change
            assert table.get_current_left_right() == ("0", "j")
            # The "0" has just moved to the top, and nothing else changed.
            # fmt: off
            assert table.get_left() == ["0", "1", "3", "5", "7", "9", "8", "6", "4", "2"]
            # fmt: on

            # Now, press ctrl+down (Page Down)
            await pilot.press("ctrl+down")
            assert table.get_current_left_right() == ("0", "j")
            # The "0" has just moved to the bottom, and nothing else changed.
            # fmt: off
            assert table.get_left() == ["1", "3", "5", "7", "9", "8", "6", "4", "2", "0"]
            # fmt: on

            # Now, press go back to top (go top action; not: one page up)
            await pilot.press("ctrl+pageup")
            assert table.get_current_left_right() == ("0", "j")
            # The "0" has just moved to the top, and nothing else changed.
            # fmt: off
            assert table.get_left() == ["0", "1", "3", "5", "7", "9", "8", "6", "4", "2"]
            # fmt: on

            # Now, press go back to bottom (go bottom action; not: one page down)
            await pilot.press("ctrl+down")
            assert table.get_current_left_right() == ("0", "j")
            # The "0" has just moved to the bottom, and nothing else changed.
            # fmt: off
            assert table.get_left() == ["1", "3", "5", "7", "9", "8", "6", "4", "2", "0"]
            # fmt: on

    test_data = """
    0,0
    0,1
    1,0
    0,2
    1,1
    2
    1
    0
    """
    # These use the TEST_CONFIG_MINIMAL_TXT
    test_left = ["11", "12", "21", "13", "22", "3", "2", "1"]
    test_right = ["AA", "AB", "BA", "AC", "BB", "C", "B", "A"]

    async def test_loading_from_file(self, hands_minimal: Hands):
        app = self.DataTableApp()

        async with app.run_test():
            table = app.query_one(KeySequenceTable)
            with patch("builtins.open", mock_open(read_data=self.test_data)):
                table.load("some_file", hands_minimal)

            # The contents from `test_data` in the same order
            assert table.get_key_indices() == [
                "0,0",
                "0,1",
                "1,0",
                "0,2",
                "1,1",
                "2",
                "1",
                "0",
            ]
            # The left and right hand trigams corresponding to key_indices
            # as calculated using the config.
            assert table.get_left() == self.test_left
            assert table.get_right() == self.test_right

    async def test_loading_from_file_with_duplicates(self, hands_minimal: Hands):
        app = self.DataTableApp()

        async with app.run_test():
            table = app.query_one(KeySequenceTable)
            # The file has duplicate  zeroes -> must crash
            with (
                patch("builtins.open", mock_open(read_data="0\n0")),
                pytest.raises(
                    DuplicateValuesError,
                    match=re.escape('Duplicate values for "(0,)" in "some_file".'),
                ),
            ):
                table.load("some_file", hands_minimal)
