from pathlib import Path

import pytest

from granite_tools.bigram_ranking_view.viewer_app import NgramTableViewerApp
from granite_tools.config import Config

test_folder = Path(__file__).parent.parent
examples_folder = test_folder.parent / "examples"


@pytest.mark.asyncio
class TestKeySeqApp:

    async def test_permutations(self, config: Config):

        N = 10  # from config (10 keys per side)
        app = NgramTableViewerApp("__some_nonexisting_file__", config=config)
        async with app.run_test():
            assert len(app.permutations) == N**2 + N
            # fmt: off
            assert app.permutations[:21] == [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (1, 0)]
            # fmt: on

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
            assert app.permutations[:4] == [(0,), (1,), (2,), (3,)]
            # As loaded from the file:
            assert app.table.get_key_indices() == ["0,0", "0,1", "0,2"]
