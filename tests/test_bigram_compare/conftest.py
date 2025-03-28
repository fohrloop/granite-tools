import pytest

from granite_tools.config import Config
from granite_tools.hands import get_hands_data
from granite_tools.permutations import create_permutations


@pytest.fixture
def permutations_full(config_full: Config):
    hands = get_hands_data(config_full)
    return create_permutations(hands.left, hands.right, sequence_lengths=(1, 2))
