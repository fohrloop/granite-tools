from __future__ import annotations

import typing

import numpy as np
import pytest

from granite_tools.score_ratios import ScoreRatioYamlEntry, create_score_ratio_entries
from granite_tools.trigram_model import (
    TrigramModelParameters,
    create_optimization_target_function,
    get_initial_params,
    get_trigram_score,
)
from granite_tools.trigram_model.optimizer import create_log_m_func

if typing.TYPE_CHECKING:
    from granite_tools.config import Config
    from granite_tools.hands import Hands


@pytest.fixture
def model_params() -> TrigramModelParameters:
    return TrigramModelParameters()


class TestGetInitialParameters:

    def test_simple(self, config_full: Config):
        config = config_full.model_copy()

        # Set all parameters to 1.0, except for two that are None
        for param in TrigramModelParameters.names:
            setattr(config, param, 1.0)

        config.easy_rolling_coeff = None
        config.sftb_coeff = None

        x0 = get_initial_params(config)

        # The order of the parameters is determined by the ordering in
        # TrigramModelParameters.names
        assert x0 == (
            TrigramModelParameters.defaults["easy_rolling_coeff"],
            TrigramModelParameters.defaults["sftb_coeff"],
        )


class TestTrigramParamsErrorFunction:

    def test_simple(self, model_params: TrigramModelParameters, hands_full: Hands):

        target_yaml_ratios = [
            ScoreRatioYamlEntry(ref="SEF", ngram="ELF", score_ratio=1.5)
        ]
        target_ratios = create_score_ratio_entries(target_yaml_ratios, hands_full)
        bigram_scores = {
            (12, 1): 1.0,  # unused. sets the scale min
            (11, 7): 1.6,  # SE
            (7, 5): 1.4,  # EF
            (11, 5): 1.2,  # SF
            (11,): 1.15,  # S, L
            (12, 8): 5.0,  # unused. sets the scale max
        }

        # Form expected score
        score_sef = get_trigram_score("SEF", hands_full, model_params, bigram_scores)[
            "score"
        ]
        score_elf = get_trigram_score("ELF", hands_full, model_params, bigram_scores)[
            "score"
        ]
        estimated_score_ratio = score_elf / score_sef
        target_score_ratio = 1.5

        r_err = target_score_ratio - estimated_score_ratio
        expected_err = abs(r_err)

        scorefunc = create_optimization_target_function(
            target_ratios, hands_full, bigram_scores
        )
        params_tuple = model_params.as_tuple()
        assert scorefunc(params_tuple) == expected_err

    def test_two_trigram_sets(
        self, model_params: TrigramModelParameters, hands_full: Hands
    ):
        target_score_ratio_elf_sef = 3.0
        target_score_ratio_efl_sef = 2.5
        target_score_ratio_xky_dotky = 2.0
        # fmt: off
        target_yaml_ratios = [
            ScoreRatioYamlEntry(ref="SEF", ngram="ELF", score_ratio=target_score_ratio_elf_sef),
            ScoreRatioYamlEntry(ref="SEF", ngram="EFL", score_ratio=target_score_ratio_efl_sef),
            ScoreRatioYamlEntry(ref=".KY", ngram="XKY", score_ratio=target_score_ratio_xky_dotky),
        ]
        # fmt: on
        target_ratios = create_score_ratio_entries(target_yaml_ratios, hands_full)

        bigram_scores = {
            (99, 99): 1.0,  # unused. sets the scale min
            (11, 7): 1.6,  # SE
            (11, 5): 2.22,  # SF
            (7, 5): 1.4,  # EF
            (11,): 1.15,  # S, L
            (12, 8): 1.8,  # XD, .K
            (12, 1): 1.8,  # XY
            (8, 1): 1.2,  # DT, KY
            (12,): 1.1,  # X, .
            (99, 90): 5.0,  # unused. sets the scale max
        }

        # Form expected score
        score_sef = get_trigram_score("SEF", hands_full, model_params, bigram_scores)[
            "score"
        ]
        score_elf = get_trigram_score("ELF", hands_full, model_params, bigram_scores)[
            "score"
        ]
        score_efl = get_trigram_score("EFL", hands_full, model_params, bigram_scores)[
            "score"
        ]
        score_dotky = get_trigram_score(".KY", hands_full, model_params, bigram_scores)[
            "score"
        ]
        score_xky = get_trigram_score("XKY", hands_full, model_params, bigram_scores)[
            "score"
        ]

        estimated_score_elf = score_elf / score_sef
        estimated_score_efl = score_efl / score_sef
        estimated_score_xky = score_xky / score_dotky

        get_log_m = create_log_m_func(hands_full.config.limit_multipliers)

        weight_sum = 0
        sum_ = 0
        for predicted_ratio, actual_ratio in zip(
            (estimated_score_elf, estimated_score_efl, estimated_score_xky),
            (
                target_score_ratio_elf_sef,
                target_score_ratio_efl_sef,
                target_score_ratio_xky_dotky,
            ),
        ):

            residual = actual_ratio - predicted_ratio
            m = np.exp(get_log_m(actual_ratio))

            if residual < 0:
                sigma = m / 2
            else:
                sigma = 1 / (2 * m)
            w = 1 / (sigma**2)
            weight_sum += w
            sum_ += w * (residual**2)

        expected_score = (sum_ / weight_sum) ** 0.5
        scorefunc = create_optimization_target_function(
            target_ratios, hands_full, bigram_scores
        )
        params_tuple = model_params.as_tuple()
        assert scorefunc(params_tuple) == expected_score
