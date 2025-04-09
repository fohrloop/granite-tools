from __future__ import annotations

import typing
import warnings
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

if typing.TYPE_CHECKING:
    from granite_tools.config import Config

# Defines the parameter defaults as well as the order in which they should be when
# converted into a tuple.
PARAMETER_DEFAULTS = {
    "w_ac_one": 0.65,
    "u_weight_balanced": 0.1,
    "u_weight_alternating": 0.1,
    "easy_rolling_coeff": 1.0,
    "balanced_coeff": 1.0,
    "alternating_coeff": 1.0,
    "redir_coeff": 1.3,
    "vert2u_coeff": 0.40,
    "sfb_in_onehand_coeff": 1.03,
    "sft_coeff": 2.0,
    "sftb_coeff": 1.5,
    "sfs_coeff": 1.02,
    "sfsb_coeff": 1.01,
}

PARAMETER_ORDERING = list(PARAMETER_DEFAULTS.keys())


class UsingDefaultValueWarning(UserWarning):
    """Warning for when a parameter is using the default value.
    This is used to help with debugging and testing.
    """

    def __init__(self, param_name: str):
        super().__init__(
            f"Using default value for parameter: {param_name}. This is probably NOT something that you want to do, as combination of custom parameter values with defaults might cause a bad trigram scoring model."
        )
        self.param_name = param_name


class TrigramModelParameters(BaseModel):

    names: ClassVar[list[str]] = PARAMETER_ORDERING
    defaults: ClassVar[dict[str, float]] = PARAMETER_DEFAULTS

    model_config = ConfigDict(extra="forbid")  # helps to spot typos.

    w_ac_one: float = defaults["w_ac_one"]
    u_weight_balanced: float = defaults["u_weight_balanced"]
    u_weight_alternating: float = defaults["u_weight_alternating"]

    easy_rolling_coeff: float = defaults["easy_rolling_coeff"]
    balanced_coeff: float = defaults["balanced_coeff"]
    alternating_coeff: float = defaults["alternating_coeff"]
    redir_coeff: float = defaults["redir_coeff"]
    vert2u_coeff: float = defaults["vert2u_coeff"]

    sfb_in_onehand_coeff: float = defaults["sfb_in_onehand_coeff"]
    sft_coeff: float = defaults["sft_coeff"]
    sftb_coeff: float = defaults["sftb_coeff"]
    sfs_coeff: float = defaults["sfs_coeff"]
    sfsb_coeff: float = defaults["sfsb_coeff"]

    @classmethod
    def from_config(cls, config: Config) -> TrigramModelParameters:
        """Create a TrigramModelParameters object from a Config object.
        If a parameter is None in the config, the default value is used."""
        dct = dict()
        for key, default in cls.defaults.items():
            if getattr(config, key) is None:
                warnings.warn(
                    UsingDefaultValueWarning(key),
                    stacklevel=2,
                )
                dct[key] = default
            else:
                dct[key] = getattr(config, key)
        return cls(**dct)

    class CannotCreateFromTuple(ValueError):
        """Custom error for when the tuple cannot be converted to a TrigramModelParameters object.
        This is used to help with debugging and testing.
        """

        def __init__(self, x: tuple[float, ...]):
            super().__init__(
                f"Tuple must contain {len(TrigramModelParameters.names)} values, but got {len(x)}"
            )

    @classmethod
    def from_tuple(
        cls, x: tuple[float, ...], config: Config | None = None
    ) -> TrigramModelParameters:
        """Create a TrigramModelParameters object from a tuple. If the tuple does not
        contain all parameters, a config object can be supplied to fill in the missing
        parameters. The tuple should be in the order defined by
        TrigramModelParameters.names
        """

        if config is None:
            # CASE: No config is given. The input tuple should contain all parameter values
            dct = cls._create_class_dict_from_full_tuple(x)
        else:
            # CASE: Config is given. The input tuple should contain exactly the number
            # of parameters that are None in the config.
            dct = cls._create_class_dict_from_tuple_and_config(x, config)

        return cls(**dct)

    @classmethod
    def _create_class_dict_from_full_tuple(cls, x: tuple[float, ...]) -> dict:
        dct = dict()
        if len(x) != len(cls.names):
            raise cls.CannotCreateFromTuple(x)
        for i, name in enumerate(cls.names):
            dct[name] = x[i]
        return dct

    @classmethod
    def _create_class_dict_from_tuple_and_config(
        cls, x: tuple[float, ...], config: Config
    ) -> dict:

        dct = dict()
        x_iterator = iter(x)

        for name in cls.names:
            if getattr(config, name) is None:
                try:
                    dct[name] = next(x_iterator)
                except StopIteration:
                    raise cls.CannotCreateFromTuple(x)
            else:
                dct[name] = getattr(config, name)
        # Check if there are any remaining values in the tuple
        try:
            next(x_iterator)
            raise cls.CannotCreateFromTuple(x)
        except StopIteration:
            pass

        return dct

    def iter_names_and_values(self) -> typing.Generator[tuple[str, float]]:
        """Iterate over the names and values of the parameters."""
        for name in self.names:
            yield name, getattr(self, name)

    def as_tuple(self) -> tuple[float, ...]:
        return tuple(getattr(self, name) for name in self.names)
