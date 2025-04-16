from pathlib import Path

import pytest

from granite_tools.bigram_ranking_view.viewer_app import NgramTableViewerApp
from granite_tools.config import Config
from granite_tools.hands import get_hands_data
from granite_tools.permutations import create_bigrams

test_folder = Path(__file__).parent.parent
examples_folder = test_folder.parent / "examples"


@pytest.mark.asyncio
class TestNgramTableViewerApp:

    @pytest.mark.slow
    async def test_permutations(self, config: Config):
        hands = get_hands_data(config)
        N = 10  # from config (10 keys per side)
        app = NgramTableViewerApp("__some_nonexisting_file__", config=config)
        async with app.run_test():
            assert len(app.permutations) == N**2 - N  # bigrams without repeats
            # fmt: off
            assert app.permutations == create_bigrams(hands.left, hands.right)

    @pytest.mark.slow
    async def test_loading_file_skips_correct_key_sequences(
        self, test_file1: str, config: Config
    ):
        """When loading files, only those key sequences which were loaded from
        the file are skipped. Skipping should not be based on the number of
        key sequences in the file, but the actual key sequences themselves."""

        app = NgramTableViewerApp(test_file1, config=config)
        async with app.run_test() as pilot:

            # Three items were loaded from the file
            assert len(app.table) == 3
            assert app.permutations[:4] == [(0, 1), (0, 2), (0, 3), (0, 4)]
            # As loaded from the file:
            assert app.table.get_key_indices() == ["2,0", "2,1", "1,2"]
