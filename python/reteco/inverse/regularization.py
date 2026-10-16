"""
python/reteco/inverse/regularization.py
Termine di regolarizzazione di Tikhonov nello spazio normalizzato:
    J_reg(theta) = (lambda / 2) * || L * (theta_tilde - theta_tilde_0) ||_2^2
e calibrazione secondo il principio di discrepanza di Morozov (Contratto Sez. 8).
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from ..config_schema import ParameterCoordinateTransform


class TikhonovRegularizer:
    def __init__(
        self,
        transform: ParameterCoordinateTransform,
        lambda_reg: float = 1.0e-4,
        theta_tilde_0: Optional[np.ndarray] = None,
        L_matrix: Optional[np.ndarray] = None,
    ):
        if lambda_reg < 0.0:
            raise ValueError(f"lambda_reg deve essere >= 0, ricevuto {lambda_reg}.")
        self.transform = transform
        self.lambda_reg = float(lambda_reg)
        self.theta_tilde_0 = (
            np.full(4, 0.5, dtype=np.float64)
            if theta_tilde_0 is None
            else np.asarray(theta_tilde_0, dtype=np.float64)
        )
        self.L_matrix = (
            np.eye(4, dtype=np.float64)
            if L_matrix is None
            else np.asarray(L_matrix, dtype=np.float64)
        )

    def evaluate(self, theta: np.ndarray) -> float:
        if self.lambda_reg == 0.0:
            return 0.0
        theta_tilde = self.transform.to_normalized(theta)
        diff = self.L_matrix @ (theta_tilde - self.theta_tilde_0)
        return 0.5 * self.lambda_reg * float(np.dot(diff, diff))

    @staticmethod
    def morozov_target_discrepancy(tau_safety: float = 1.05) -> float:
        """
        Con pesi W_ii = 1 / (sqrt(N_obs) * sigma_std), il valore atteso del termine
        dati 0.5 * ||W(G(theta_dagger) - y)||_2^2 per rumore gaussiano puro è 0.5.
        """
        return 0.5 * (tau_safety ** 2)