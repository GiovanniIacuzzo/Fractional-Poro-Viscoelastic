"""
python/reteco/identifiability/sensitivity_jacobian.py
Calcolo del Jacobiano delle osservazioni nelle coordinate normalizzate (Eq. jacobian, Contratto Sez. 8):
    J_{ij}^G(theta_tilde) = d G_i / d theta_tilde_j
con differenze finite centrali del secondo ordine.
"""

from __future__ import annotations
from typing import Callable
import numpy as np

from ..config_schema import ParameterCoordinateTransform


def compute_normalized_jacobian(
    observation_map_G: Callable[[np.ndarray], np.ndarray],
    theta_ref: np.ndarray,
    transform: ParameterCoordinateTransform,
    step_tilde: float = 1.0e-4,
) -> np.ndarray:
    """
    Calcola la matrice J^G di dimensione (N_obs, 4) rispetto a theta_tilde = (alpha~, beta~, D~, E~).
    """
    theta_tilde_0 = transform.to_normalized(theta_ref)
    y0 = observation_map_G(theta_ref)
    n_obs = y0.size
    J = np.zeros((n_obs, 4), dtype=np.float64)

    for j in range(4):
        h = step_tilde
        t_plus = theta_tilde_0.copy()
        t_minus = theta_tilde_0.copy()

        t_plus[j] = min(1.0, theta_tilde_0[j] + h)
        t_minus[j] = max(0.0, theta_tilde_0[j] - h)
        denom = t_plus[j] - t_minus[j]
        if denom <= 0.0:
            raise ValueError(f"Impossibile perturbare la coordinata normalizzata j={j}.")

        theta_plus = transform.from_normalized(t_plus)
        theta_minus = transform.from_normalized(t_minus)

        y_plus = observation_map_G(theta_plus)
        y_minus = observation_map_G(theta_minus)

        J[:, j] = (y_plus - y_minus) / denom

    return J