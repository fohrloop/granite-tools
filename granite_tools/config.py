from typing import Literal

import yaml
from pydantic import BaseModel

HandType = Literal["Left", "Right"]


class Config(BaseModel):
    key_indices: list[list[int]]
    hands: list[list[HandType]]
    symbols: list[list[str]]
    finger_matrix: list[list[str]] | None = None
    key_category_matrix: list[list[str]] | None = None
    color_matrix: list[list[str]] | None = None
    color_mapping: dict[str, str] | None = None
    matrix_positions: list[list[tuple[int, int]]] | None = None


def read_config(file: str) -> Config:
    with open(file, "r") as f:

        config = yaml.safe_load(f)

    for item in config["symbols"]:
        for i, symbol in enumerate(item):
            item[i] = str(symbol)

    return Config(**config)
