from __future__ import annotations

import typing

import pytest

from granite_tools.trigram_model.params import (
    PARAMETER_DEFAULTS,
    PARAMETER_ORDERING,
    TrigramModelParameters,
    UsingDefaultValueWarning,
)

if typing.TYPE_CHECKING:
    from granite_tools.config import Config


@pytest.fixture
def config_full_random_params(config_full: Config) -> Config:
    config = config_full.model_copy()
    for i, param in enumerate(PARAMETER_ORDERING, start=1):
        # Make a random floating point number so the values won't accidentally match
        setattr(config, param, i + 0.34567812)

    return config


class TestTrigramModelParameters:

    def test_from_config(self, config_full_random_params: Config):

        params = TrigramModelParameters.from_config(config_full_random_params)

        for param in TrigramModelParameters.names:
            assert getattr(params, param) == getattr(config_full_random_params, param)

    def test_from_config_some_vals_from_defaults(
        self, config_full_random_params: Config
    ):
        config = config_full_random_params
        # Set some values to None to test that defaults are used
        config.easy_rolling_coeff = None
        config.alternating_coeff = None

        with pytest.warns(UsingDefaultValueWarning):
            params = TrigramModelParameters.from_config(config)

        for param in TrigramModelParameters.names:
            if param in ("easy_rolling_coeff", "alternating_coeff"):
                assert getattr(params, param) == PARAMETER_DEFAULTS[param]
            else:
                assert getattr(params, param) == getattr(config, param)

    def test_from_tuple_simple(self):

        x = tuple(range(1, len(TrigramModelParameters.names) + 1))
        params = TrigramModelParameters.from_tuple(x)

        for name, value in zip(TrigramModelParameters.names, x):
            assert getattr(params, name) == value

    def test_from_tuple_missing_values_raises_err(self):
        with pytest.raises(
            ValueError, match=r"Tuple must contain \d+ values, but got 2"
        ):
            # Just two param values. All the rest missing.
            TrigramModelParameters.from_tuple((1, 2))

    def test_from_tuple_missing_values(self, config_full_random_params: Config):

        # Situation, where only two values are given.
        # There should be two Nones (missing values) in the configuration
        x = (44, 55)
        config = config_full_random_params

        # These two are missing. The order is taken from PARAMETER_ORDERING
        config.w_ac_one = None  # will be 44
        config.balanced_coeff = None  # will be 55

        params = TrigramModelParameters.from_tuple(x, config)

        # The two given values should be used
        assert params.w_ac_one == 44
        assert params.balanced_coeff == 55

        # The rest of the parameters should be taken from the config
        for param in TrigramModelParameters.names:
            if param not in ("w_ac_one", "balanced_coeff"):
                assert getattr(params, param) == getattr(config, param)

    def test_from_tuple_missing_values_case_2(self, config_full_random_params: Config):

        # Same as test_from_tuple_missing_values but using different None values
        # in the config.
        x = (44, 55)
        config = config_full_random_params

        # These two are missing. The order is taken from PARAMETER_ORDERING
        config.sft_coeff = None  # will be 55 (as later in the ordering)
        config.redir_coeff = None  # will be 44

        params = TrigramModelParameters.from_tuple(x, config)

        # The two given values should be used
        assert params.sft_coeff == 55
        assert params.redir_coeff == 44

        # The rest of the parameters should be taken from the config
        for param in TrigramModelParameters.names:
            if param not in ("sft_coeff", "redir_coeff"):
                assert getattr(params, param) == getattr(config, param)

    def test_from_tuple_too_long_tuple(self, config_full_random_params: Config):

        x = (44, 55, 66)
        config = config_full_random_params

        # Only TWO missing (but THREE values given in x)
        config.sft_coeff = None
        config.redir_coeff = None

        with pytest.raises(
            ValueError, match=r"Tuple must contain \d+ values, but got 3"
        ):
            TrigramModelParameters.from_tuple(x, config)

    def test_from_tuple_too_short(self, config_full_random_params: Config):

        x = (44,)
        config = config_full_random_params

        # TWO missing (but only ONE values given in x)
        config.sft_coeff = None
        config.redir_coeff = None

        with pytest.raises(
            ValueError, match=r"Tuple must contain \d+ values, but got 1"
        ):
            TrigramModelParameters.from_tuple(x, config)

    def test_iter_names_and_values(self):
        params = TrigramModelParameters.from_tuple(
            tuple(range(1, len(TrigramModelParameters.names) + 1))
        )

        iterated_names = []
        for name, value in params.iter_names_and_values():
            assert name in TrigramModelParameters.names
            assert value == getattr(params, name)
            iterated_names.append(name)

        # Check that all names were iterated
        assert iterated_names == TrigramModelParameters.names

    def test_as_tuple(self):

        offset = 5
        input_tuple = tuple(range(offset, len(TrigramModelParameters.names) + offset))
        params = TrigramModelParameters.from_tuple(input_tuple)
        assert params.as_tuple() == input_tuple
