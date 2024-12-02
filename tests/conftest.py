from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from granite_tools.config import Config, read_config
from granite_tools.effort import Hands, get_hands_data

test_folder = Path(__file__).parent
examples_folder = test_folder.parent / "examples"


TEST_CONFIG_TXT = """
key_indices:
  - [       0,  1,  2,  3,  4,     4,  3,  2,  1,  0       ]
  - [       5,  6,  7,  8,  9,     9,  8,  7,  6,  5,      ]

hands:
  - [      Left, Left, Left, Left, Left,                Right, Right, Right, Right, Right       ]
  - [      Left, Left, Left, Left, Left,                Right, Right, Right, Right, Right       ]

symbols:
  - [       1,  2,  3,  4,  5,     E,  D,  C,  B,  A       ]
  - [       6,  7,  8,  9,  0,     J,  I,  H,  G,  F,      ]
""".strip()

TEST_CONFIG = {
    "key_indices": [[0, 1, 2, 3, 4, 4, 3, 2, 1, 0], [5, 6, 7, 8, 9, 9, 8, 7, 6, 5]],
    "hands": [
        [
            "Left",
            "Left",
            "Left",
            "Left",
            "Left",
            "Right",
            "Right",
            "Right",
            "Right",
            "Right",
        ],
        [
            "Left",
            "Left",
            "Left",
            "Left",
            "Left",
            "Right",
            "Right",
            "Right",
            "Right",
            "Right",
        ],
    ],
    "symbols": [
        ["1", "2", "3", "4", "5", "E", "D", "C", "B", "A"],
        ["6", "7", "8", "9", "0", "J", "I", "H", "G", "F"],
    ],
}

TEST_CONFIG_MINIMAL_TXT = (
    examples_folder / "keyseq_effort_numbers_mini.yml"
).read_text()


@pytest.fixture
def config() -> Config:
    return Config(**TEST_CONFIG)


@pytest.fixture
def config_minimal() -> Config:
    return read_config(examples_folder / "keyseq_effort_numbers_mini.yml")


@pytest.fixture
def config_full() -> Config:
    return read_config(examples_folder / "keyseq_effort.yml")


@patch("builtins.open", mock_open(read_data=TEST_CONFIG_TXT))
def test_read_config(config):

    config_out = read_config("tests/test_config.yaml")

    assert config_out == config


@pytest.fixture
def hands_minimal() -> Hands:
    with patch("builtins.open", mock_open(read_data=TEST_CONFIG_MINIMAL_TXT)):
        config = read_config("foo")
    return get_hands_data(config)


@pytest.fixture
def test_file1():
    # to be paired in tests with TEST_CONFIG_MINIMAL_TXT
    file = test_folder / "test_file1_remove_me"
    file.write_text("0,0\n0,1\n0,2")
    try:
        yield str(file)
    finally:
        file.unlink()
