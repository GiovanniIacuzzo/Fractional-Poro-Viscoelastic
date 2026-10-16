"""
RETECO v1 — Fractional Poro-Viscoelastic Biot-Caputo Framework.
Combines C++20 deal.II FEM core (`reteco._core`) with Python inverse modeling tools.
"""

from __future__ import annotations

__version__ = "1.0.0"

try:
    from . import _core
except ImportError as exc:
    raise ImportError(
        "Il modulo C++ compilato `reteco._core` non è stato trovato. "
        "Esegui `./prepare.sh` oppure `make build` nella root del progetto."
    ) from exc

from .config_schema import (
    CharacteristicScales,
    FixedPhysicsModel,
    ParameterBounds,
    ParameterSpaceConfig,
    RetecoMasterConfig,
    load_master_config,
)

__all__ = [
    "__version__",
    "_core",
    "CharacteristicScales",
    "FixedPhysicsModel",
    "ParameterBounds",
    "ParameterSpaceConfig",
    "RetecoMasterConfig",
    "load_master_config",
]