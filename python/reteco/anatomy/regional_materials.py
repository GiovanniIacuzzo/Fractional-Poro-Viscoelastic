"""Validated, phenomenological BraTS-region material multipliers.

These are dimensionless contrasts around the *same* homogeneous reference model,
NOT MRI-inferred material constants and NOT patient-calibrated coefficients.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping
import math
import yaml

BRATS_IDS = (0, 1, 2, 4)
FIELDS = ("stiffness", "hydraulic_mobility", "tracer_diffusivity", "porosity")


def validate_regional_materials(raw: Mapping, phi0: float) -> dict[int, dict[str, float]]:
    """Require all four material IDs and four finite positive contrast multipliers."""
    if not isinstance(raw, Mapping):
        raise ValueError("regional materials must be a mapping indexed by BraTS material ID")
    table = {int(k): v for k, v in raw.items()}
    if set(table) != set(BRATS_IDS):
        raise ValueError(f"regional material IDs must be exactly {BRATS_IDS}; got {sorted(table)}")
    result = {}
    for region, values in table.items():
        if not isinstance(values, Mapping) or set(values) != set(FIELDS):
            raise ValueError(f"region {region}: required multipliers are {FIELDS}")
        row = {k: float(values[k]) for k in FIELDS}
        if any(not math.isfinite(v) or v <= 0 for v in row.values()):
            raise ValueError(f"region {region}: all multipliers must be finite positive")
        if phi0 * row["porosity"] > 1.0:
            raise ValueError(f"region {region}: phi_0 times porosity multiplier exceeds 1")
        result[region] = row
    return result


def load_regional_materials(path: str | Path, profile: str, phi0: float) -> dict[int, dict[str, float]]:
    p = Path(path)
    source = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or "profiles" not in source:
        raise ValueError("materials configuration must contain 'profiles'")
    profiles = source["profiles"]
    if profile not in profiles:
        raise ValueError(f"unknown region-material profile {profile}; options: {sorted(profiles)}")
    return validate_regional_materials(profiles[profile], phi0)


def to_cpp_region_materials(materials: dict[int, dict[str, float]]) -> dict:
    from .. import _core
    result = {}
    for key, values in materials.items():
        mat = _core.RegionMaterialMultipliers()
        for field in FIELDS:
            setattr(mat, field, float(values[field]))
        result[int(key)] = mat
    return result
