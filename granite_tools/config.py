"""Python model for the granite configuration (ngram scoring model)"""

from __future__ import annotations

import functools
import typing
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from granite_tools.app_types import Vert2uPenaltyConfig

if typing.TYPE_CHECKING:
    from pathlib import Path
    from typing import Self

HandType = Literal["Left", "Right"]


DEFAULT_LIMIT_MULTIPLIERS = {
    0.15: 3,
    0.6: 1.416,
    1: 1.1,
    2.5: 1.35,
    3.0: 1.4,
}

DEFAULT_EASIEST_UNIGRAM_SCORE = 1.0
DEFAULT_MOST_DIFFICULT_UNIGRAM_SCORE = 1.95


class Config(BaseModel):
    key_indices: list[list[int]]
    hands: list[list[HandType]]
    symbols_visualization: list[list[str]]
    symbols_scoring: list[list[str | None]] = Field(
        default_factory=list
    )  # defaults to symbols_visualization
    finger_matrix: list[list[str]] | None = None
    key_category_matrix: list[list[str]] | None = None
    color_matrix: list[list[str]] | None = None
    color_mapping: dict[str, str] | None = None
    matrix_positions: list[list[tuple[int, int]]] | None = None

    vert2u_penalties: Vert2uPenaltyConfig = Field(default_factory=dict)

    # Scoring / Bigram Model Parameters
    easiest_unigram_score: float = DEFAULT_EASIEST_UNIGRAM_SCORE
    most_difficult_unigram_score: float = DEFAULT_MOST_DIFFICULT_UNIGRAM_SCORE

    # Scoring / Trigram Model Parameters
    w_ac_one: float | None = None
    u_weight_balanced: float | None = None
    u_weight_alternating: float | None = None
    easy_rolling_coeff: float | None = None
    balanced_coeff: float | None = None
    alternating_coeff: float | None = None
    redir_coeff: float | None = None
    vert2u_coeff: float | None = None
    sfb_in_onehand_coeff: float | None = None
    sft_coeff: float | None = None
    sftb_coeff: float | None = None
    sfs_coeff: float | None = None
    sfsb_coeff: float | None = None

    # list of trigrams that are considered easy to type
    easy_rolling_trigrams: dict[str, np.ndarray] | None = None

    # Trigram model optimization
    limit_multipliers: dict[float, float] = Field(default=DEFAULT_LIMIT_MULTIPLIERS)

    @model_validator(mode="after")
    def set_default_symbols_scoring(self) -> Self:
        if not self.symbols_scoring:
            vis: list[list[str | None]] = [
                [cell for cell in row] for row in self.symbols_visualization
            ]
            self.symbols_scoring = vis
        return self

    @field_validator("easy_rolling_trigrams", mode="before")
    def convert_to_numpy_arrays(
        cls, value: dict[str, list[list[str]]] | None
    ) -> dict[str, np.ndarray] | None:
        if not value:
            return None
        result = {}

        for key, patterns in value.items():
            processed_patterns = [
                [0 if char == "." else int(char) for char in pattern]
                for pattern in patterns
            ]
            shapearr = np.array(processed_patterns, dtype=int)

            if len(np.argwhere(shapearr == 1)) != 1:
                raise ValueError(
                    f"easy_rolling_trigrams '{key}' must have exactly one 1"
                )
            elif len(np.argwhere(shapearr == 2)) != 1:
                raise ValueError(
                    f"easy_rolling_trigrams '{key}' must have exactly one 2"
                )
            elif not len(np.argwhere(shapearr == 3)) > 0:
                raise ValueError(
                    f"easy_rolling_trigrams '{key}' must have at least one 3"
                )
            result[key] = shapearr
        return result

    model_config = ConfigDict(arbitrary_types_allowed=True)


@functools.lru_cache  # makes tests run a bit faster.
def read_config(file: str | Path) -> Config:
    with open(file, "r") as f:
        contents = f.read()
    return text_to_config(contents)


def text_to_config(text: str) -> Config:
    config = yaml.safe_load(text)
    for item in config["symbols_visualization"]:
        for i, symbol in enumerate(item):
            item[i] = str(symbol)
    return Config(**config)
