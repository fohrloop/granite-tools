"""Python model for the granite configuration (ngram scoring model)"""

from __future__ import annotations

import functools
import typing
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

if typing.TYPE_CHECKING:
    from pathlib import Path

HandType = Literal["Left", "Right"]


DEFAULT_LIMIT_MULTIPLIERS = {
    0.15: 3,
    0.6: 1.416,
    1: 1.1,
    2.5: 1.35,
    3.0: 1.4,
}


class Config(BaseModel):
    key_indices: list[list[int]]
    hands: list[list[HandType]]
    symbols_visualization: list[list[str]]
    finger_matrix: list[list[str]] | None = None
    key_category_matrix: list[list[str]] | None = None
    color_matrix: list[list[str]] | None = None
    color_mapping: dict[str, str] | None = None
    matrix_positions: list[list[tuple[int, int]]] | None = None

    # Scoring
    # Part 1: optimized
    balanced_b_coeff: float | None = None
    unigram_coeff: float | None = None
    skipgram_b_coeff: float | None = None
    vert2u_coeff: float | None = None
    dirchange_coeff: float | None = None

    easy_rolling_coeff: float | None = None

    # Part 2: not optimized with a minimizer (manually set)
    sfb_in_onehand_coeff: float | None = None
    sft_coeff: float | None = None

    # list of trigrams that are considered easy to type
    easy_rolling_trigrams: dict[str, np.ndarray] | None = None

    # Trigram model optimization
    limit_multipliers: dict[float, float] = Field(default=DEFAULT_LIMIT_MULTIPLIERS)

    @field_validator("easy_rolling_trigrams", mode="before")
    def convert_to_numpy_arrays(cls, value: dict[str, list[list[str]]] | None):
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
