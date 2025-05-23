import re
from pathlib import Path
from textwrap import dedent

import pytest

from granite_tools.bigram_ranking_initial.bigram_ranking_initial import (
    KeySequenceSortApp,
)
from granite_tools.bigram_scores.rankings import DuplicateValuesError
from granite_tools.config import Config
from granite_tools.hands import get_hands_data
from granite_tools.permutations import create_bigrams

test_folder = Path(__file__).parent.parent
examples_folder = test_folder.parent / "examples"


@pytest.mark.asyncio
class TestKeySeqApp:

    @pytest.mark.slow
    async def test_move_left_one_step(self, config_minimal: Config):

        app = KeySequenceSortApp("__some_nonexisting_file__", config=config_minimal)
        async with app.run_test() as pilot:

            assert app.ordered_ngrams == [(0, 1)]
            await pilot.press("left")

            pilot.app.action_place_ngram()
            assert pilot.app.ordered_ngrams == [(0, 2), (0, 1)]

    @pytest.mark.slow
    async def test_move_right_one_step(self, config_minimal: Config):

        app = KeySequenceSortApp("__some_nonexisting_file__", config=config_minimal)
        async with app.run_test() as pilot:

            assert app.ordered_ngrams == [(0, 1)]
            await pilot.press("right")

            pilot.app.action_place_ngram()
            assert pilot.app.ordered_ngrams == [(0, 1), (0, 2)]

    testfile_contents = dedent(
        """
    0,1
    0,2
    1,0
    1,2
    2,0
    """.strip(
            "\n"
        )
    )

    @pytest.mark.slow
    async def test_saving_and_loading(self, config_minimal: Config):
        filename = "__test_temp_file_remove_after_tests__"
        Path(filename).unlink(missing_ok=True)
        hands = get_hands_data(config_minimal)
        b = create_bigrams(hands.left, hands.right)

        app = KeySequenceSortApp(filename, config=config_minimal)
        async with app.run_test() as pilot:

            assert app.ordered_ngrams == [(0, 1)]
            await pilot.press("right")
            await pilot.press("enter")
            await pilot.press("right")
            await pilot.press("right")
            await pilot.press("enter")
            await pilot.press("enter")
            await pilot.press("left")
            await pilot.press("enter")
            await pilot.press("ctrl+s")
            assert app.ordered_ngrams == [b[0], b[4], b[3], b[1], b[2]]
            assert app.manager._current_ngram == b[5]
            assert app.manager.ngrams_left_side_of_current == [b[0], b[4]]
            assert app.manager.ngrams_right_side_of_current == [b[3], b[1], b[2]]

        expected_contents = (
            "\n".join(
                ",".join(map(str, bigram)) for bigram in [b[0], b[4], b[3], b[1], b[2]]
            )
            + "\n"
        )

        with open(app.file_out, "r") as f:
            assert f.read() == expected_contents

        # First, starting an app without file will not load it from file
        app = KeySequenceSortApp("__nonexistent__file__", config=config_minimal)
        async with app.run_test() as pilot:
            assert app.ordered_ngrams == [b[0]]

        # Now, load from the file
        app = KeySequenceSortApp(filename, config=config_minimal)
        async with app.run_test() as pilot:

            assert app.ordered_ngrams == [b[0], b[4], b[3], b[1], b[2]]
            assert app.manager._current_ngram == b[5]
            assert app.manager.ngrams_left_side_of_current == [b[0], b[4]]
            assert app.manager.ngrams_right_side_of_current == [b[3], b[1], b[2]]

        Path(filename).unlink()

    @pytest.fixture()
    def file_with_duplicate(self):
        filename = "__test_temp_file2_remove_after_tests__"
        Path(filename).unlink(missing_ok=True)
        contents_with_duplicate = dedent(
            """
        0,1
        0,0
        2,0
        0,1
        """.strip(
                "\n"
            )
        )
        Path(filename).write_text(contents_with_duplicate)
        yield filename
        Path(filename).unlink()

    @pytest.mark.slow
    async def test_loading_duplicates_raises_exception(
        self, config_minimal: Config, file_with_duplicate: str
    ):

        app = KeySequenceSortApp(file_with_duplicate, config=config_minimal)
        with pytest.raises(
            DuplicateValuesError,
            match=re.escape(
                f'Duplicate values for "(0, 1)" in "{file_with_duplicate}"'
            ),
        ):

            async with app.run_test() as pilot:
                pass
