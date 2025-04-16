import pytest

from granite_tools.config import Config
from granite_tools.hands import get_hands_data
from granite_tools.permutations import create_bigrams


@pytest.fixture
def bigrams_full(config_full: Config):
    hands = get_hands_data(config_full)
    return create_bigrams(hands.left, hands.right)
