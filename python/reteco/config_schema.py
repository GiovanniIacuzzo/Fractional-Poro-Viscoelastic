"""
python/reteco/config_schema.py
Validazione rigorosa delle scale dimensionali, dei parametri fissati, dei bounds compatti
e della biiezione di normalizzazione psi: Theta -> [0, 1]^4 (Contratto Matematico Sez. 1, 2, 8).
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import math
import numpy as np
import yaml
from pydantic import BaseModel, Field, model_validator

from . import _core


class CharacteristicScales(BaseModel):
    L0_m: float = Field(..., gt=0.0)
    T0_s: float = Field(..., gt=0.0)
    tau_c_s: float = Field(..., gt=0.0)
    E0_Pa: float = Field(..., gt=0.0)
    c_ref_mol_m3: float = Field(..., gt=0.0)

    @model_validator(mode="after")
    def check_tau_c_equals_T0(self) -> "CharacteristicScales":
        if abs(self.T0_s - self.tau_c_s) > 1.0e-12 * self.T0_s:
            raise ValueError(
                f"Violazione del Contratto Matematico (Sez. 1-2): T0_s ({self.T0_s}) "
                f"deve coincidere esattamente con tau_c_s ({self.tau_c_s})."
            )
        return self

    def scale_diffusivity_to_adim(self, D_phys_m2_s: float) -> float:
        """D = D_phys * T_0 / L_0^2 con D_phys in [m^2/s] (Eq. scaled-DE)."""
        return (D_phys_m2_s * self.T0_s) / (self.L0_m ** 2)

    def scale_modulus_to_adim(self, E_phys_Pa: float) -> float:
        """E = E_phys / E_0 (Eq. scaled-DE)."""
        return E_phys_Pa / self.E0_Pa


class FixedPhysicsModel(BaseModel):
    plane_hypothesis: str = Field(default="plane_strain")
    nu: float = Field(..., gt=-1.0, lt=0.5)
    b_biot: float = Field(..., gt=0.0, le=1.0)
    S_p: float = Field(..., gt=0.0)
    K_h: float = Field(..., gt=0.0)
    eta_v: float = Field(..., gt=0.0)
    tau_v: float = Field(..., gt=0.0)
    phi_0: float = Field(..., gt=0.0, le=1.0)

    def to_cpp(self, scales: CharacteristicScales) -> _core.FixedPhysicsConfig:
        cfg = _core.FixedPhysicsConfig()
        cfg.nu = self.nu
        cfg.b = self.b_biot
        cfg.S_p = self.S_p
        cfg.K_h = self.K_h
        cfg.eta_v = self.eta_v
        cfg.tau_v = self.tau_v
        cfg.phi_0 = self.phi_0
        cfg.plane_strain = self.plane_hypothesis == "plane_strain"
        cfg.L_0 = scales.L0_m
        cfg.T_0 = scales.T0_s
        cfg.tau_c = scales.tau_c_s
        cfg.E_0 = scales.E0_Pa
        cfg.c_ref = scales.c_ref_mol_m3
        cfg.validate()
        return cfg


class ParameterBounds(BaseModel):
    alpha: Tuple[float, float]
    beta: Tuple[float, float]
    D: Tuple[float, float]
    E: Tuple[float, float]

    @model_validator(mode="after")
    def validate_compact_bounds(self) -> "ParameterBounds":
        a_min, a_max = self.alpha
        b_min, b_max = self.beta
        d_min, d_max = self.D
        e_min, e_max = self.E
        if not (0.0 < a_min < a_max < 1.0):
            raise ValueError(f"Bounds alpha {self.alpha} devono essere strettamente interni a (0, 1).")
        if not (0.0 < b_min < b_max < 1.0):
            raise ValueError(f"Bounds beta {self.beta} devono essere strettamente interni a (0, 1).")
        if not (0.0 < d_min < d_max):
            raise ValueError(f"Bounds D {self.D} devono essere strettamente positivi.")
        if not (0.0 < e_min < e_max):
            raise ValueError(f"Bounds E {self.E} devono essere strettamente positivi.")
        return self


class ParameterCoordinateTransform:
    """
    Implementa la mappa biiettiva psi: Theta -> [0, 1]^4 e la sua inversa psi^{-1}
    (Sezione 8 del Contratto Matematico):
      alpha_tilde = (alpha - alpha_-) / (alpha_+ - alpha_-)
      beta_tilde  = (beta  - beta_-)  / (beta_+  - beta_-)
      D_tilde     = (log(D) - log(D_-)) / (log(D_+) - log(D_-))
      E_tilde     = (log(E) - log(E_-)) / (log(E_+) - log(E_-))
    """

    def __init__(self, bounds: ParameterBounds):
        self.bounds = bounds
        self.a_min, self.a_max = bounds.alpha
        self.b_min, self.b_max = bounds.beta
        self.d_min, self.d_max = bounds.D
        self.e_min, self.e_max = bounds.E

        self.log_d_min = math.log(self.d_min)
        self.log_d_max = math.log(self.d_max)
        self.log_e_min = math.log(self.e_min)
        self.log_e_max = math.log(self.e_max)

    def to_normalized(self, theta: np.ndarray) -> np.ndarray:
        """Converte theta = [alpha, beta, D, E] in theta_tilde in [0, 1]^4."""
        arr = np.asarray(theta, dtype=np.float64)
        if arr.shape != (4,):
            raise ValueError(f"theta deve avere forma (4,), ricevuto {arr.shape}.")
        alpha, beta, D, E = arr
        if D <= 0.0 or E <= 0.0:
            raise ValueError("D ed E devono essere strettamente positivi per la log-trasformazione.")
        return np.array(
            [
                (alpha - self.a_min) / (self.a_max - self.a_min),
                (beta - self.b_min) / (self.b_max - self.b_min),
                (math.log(D) - self.log_d_min) / (self.log_d_max - self.log_d_min),
                (math.log(E) - self.log_e_min) / (self.log_e_max - self.log_e_min),
            ],
            dtype=np.float64,
        )

    def from_normalized(self, theta_tilde: np.ndarray) -> np.ndarray:
        """Converte theta_tilde in [0, 1]^4 nei parametri adimensionali theta = [alpha, beta, D, E]."""
        t = np.asarray(theta_tilde, dtype=np.float64)
        if t.shape != (4,):
            raise ValueError(f"theta_tilde deve avere forma (4,), ricevuto {t.shape}.")
        alpha = self.a_min + t[0] * (self.a_max - self.a_min)
        beta = self.b_min + t[1] * (self.b_max - self.b_min)
        D = math.exp(self.log_d_min + t[2] * (self.log_d_max - self.log_d_min))
        E = math.exp(self.log_e_min + t[3] * (self.log_e_max - self.log_e_min))
        return np.array([alpha, beta, D, E], dtype=np.float64)

    def dtheta_dtilde_factors(self, theta: np.ndarray) -> np.ndarray:
        """
        Restituisce le derivate analitiche d(theta_j) / d(theta_tilde_j) per la regola della catena
        nel calcolo del Jacobiano normalizzato J^G_{ij} = dG_i / d(theta_tilde_j).
        """
        _, _, D, E = np.asarray(theta, dtype=np.float64)
        return np.array(
            [
                self.a_max - self.a_min,
                self.b_max - self.b_min,
                D * (self.log_d_max - self.log_d_min),
                E * (self.log_e_max - self.log_e_min),
            ],
            dtype=np.float64,
        )

    def is_within_bounds(self, theta: np.ndarray, tol: float = 1.0e-12) -> bool:
        alpha, beta, D, E = np.asarray(theta, dtype=np.float64)
        return (
            (self.a_min - tol <= alpha <= self.a_max + tol)
            and (self.b_min - tol <= beta <= self.b_max + tol)
            and (self.d_min - tol <= D <= self.d_max + tol)
            and (self.e_min - tol <= E <= self.e_max + tol)
        )

    def clip_to_bounds(self, theta: np.ndarray) -> np.ndarray:
        alpha, beta, D, E = np.asarray(theta, dtype=np.float64)
        return np.array(
            [
                np.clip(alpha, self.a_min, self.a_max),
                np.clip(beta, self.b_min, self.b_max),
                np.clip(D, self.d_min, self.d_max),
                np.clip(E, self.e_min, self.e_max),
            ],
            dtype=np.float64,
        )


class ParameterSpaceConfig(BaseModel):
    theta_dagger: Dict[str, float]
    bounds: ParameterBounds
    normalization: Dict[str, str]

    def theta_dagger_array(self) -> np.ndarray:
        return np.array(
            [
                self.theta_dagger["alpha"],
                self.theta_dagger["beta"],
                self.theta_dagger["D"],
                self.theta_dagger["E"],
            ],
            dtype=np.float64,
        )


class RetecoMasterConfig(BaseModel):
    model_version: str
    description: str
    characteristic_scales: CharacteristicScales
    fixed_physics: FixedPhysicsModel
    parameter_space_theta: ParameterSpaceConfig


def load_master_config(yaml_path: str | Path) -> RetecoMasterConfig:
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"File di configurazione non trovato: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return RetecoMasterConfig(**raw)