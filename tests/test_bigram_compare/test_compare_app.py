from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from granite_tools.bigram_compare.compare_app import KeySequenceCompareApp
from granite_tools.bigram_compare.scorer import verify_data
from granite_tools.config import Config

this_folder = Path(__file__).parent

KeySeq = tuple[int, ...]


@pytest.fixture
def testfile(permutations_full: list[KeySeq]):
    filename = "__tmp_file_compare_app_tests__"
    file = this_folder / filename
    pickle_file = file.with_suffix(".compare.pickle")
    with open(file, "w") as f:
        for ks in permutations_full:
            f.write(f"{','.join(map(str,ks))}\n")
    if pickle_file.exists():
        pickle_file.unlink()
    yield str(file)
    file.unlink()
    pickle_file.unlink()


@pytest.mark.asyncio
class TestCompareApp:
    async def test_save_and_load(self, testfile, config_full: Config):

        # There was a problem when loading from file and continuing the work. Test that
        # case here.
        app = KeySequenceCompareApp(testfile, config=config_full)
        async with app.run_test() as pilot:
            # Select anything.
            for _ in range(10):
                await pilot.press("left")

            # do the fit.
            await pilot.press("enter")

            # save to pickle
            assert not app.pickle_file.exists()
            await pilot.press("ctrl+s")
            assert app.pickle_file.exists()

        # load the data from file. Not expecting problems here (just a sanity check)
        with open(app.pickle_file, "rb") as f:
            data = pickle.load(f)
        verify_data(data)

        # now load from the pickle
        app2 = KeySequenceCompareApp(app.pickle_file, config=config_full)
        async with app2.run_test() as pilot:
            # Select anything.
            for _ in range(10):
                await pilot.press("left")

            # do the fit.
            await pilot.press("enter")
            # save to pickle. This caused problems earlier.
            await pilot.press("ctrl+s")

        # load the data from file. It should be okay. This had problems earlier.
        with open(app.pickle_file, "rb") as f:
            data2 = pickle.load(f)

        # Data is not saved if there are problems with it when verifying. Therefore, we
        # must check that the saved data is different from the original data.
        assert data != data2, "There should be new data"
        # And the new data should be valid.
        verify_data(data2)
