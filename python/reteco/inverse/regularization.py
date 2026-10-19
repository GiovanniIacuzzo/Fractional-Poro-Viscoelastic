"""
python/reteco/inverse/regularization.py
Termine di regolarizzazione di Tikhonov nello spazio normalizzato log-box [0, 1]^4:
    J_reg(theta; lambda) = (lambda / 2) * || L * (theta_tilde - theta_tilde_0) ||_2^2
dove theta_tilde = psi(theta) in [0, 1]^4 è definito secondo la Sezione 8 del
Contratto Matematico:
    alpha_tilde = (alpha - alpha_-) / (alpha_+ - alpha_-),
    beta_tilde  = (beta  - beta_-)  / (beta_+  - beta_-),
    D_tilde     = (log(D) - log(D_-)) / (log(D_+) - log(D_-)),
    E_tilde     = (log(E) - log(E_-)) / (log(E_+) - log(E_-)).

Nota matematica sull'invarianza di base del logaritmo:
Poiché per ogni b > 0, b != 1 vale log_b(x) = ln(x) / ln(b), il fattore 1 / ln(b)
si cancella esattamente tra numeratore e denominatore in D_tilde ed E_tilde.
Pertanto psi(theta) e il prior theta_tilde_0 in [0, 1]^4 sono invarianti rispetto
alla scelta tra logaritmo naturale e logaritmo in base 10.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from ..config_schema import ParameterCoordinateTransform


@dataclass(frozen=True)
class RegularizationBreakdown:
    theta_physical: np.ndarray
    theta_tilde: np.ndarray
    theta_tilde_0: np.ndarray
    theta_0_physical: np.ndarray
    weighted_prior_diff: np.ndarray
    prior_distance_l2_norm: float
    unscaled_penalty_half_norm_sq: float
    lambda_reg: float
    j_reg: float


class TikhonovRegularizer:
    def __init__(
        self,
        transform: ParameterCoordinateTransform,
        lambda_reg: float = 1.0e-4,
        theta_tilde_0: Optional[np.ndarray] = None,
        L_matrix: Optional[np.ndarray] = None,
    ):
        if lambda_reg < 0.0 or not np.isfinite(lambda_reg):
            raise ValueError(
                f"lambda_reg deve essere finito e >= 0, ricevuto {lambda_reg}."
            )
        self.transform = transform
        self.lambda_reg = float(lambda_reg)
        self.theta_tilde_0 = (
            np.full(4, 0.5, dtype=np.float64)
            if theta_tilde_0 is None
            else np.asarray(theta_tilde_0, dtype=np.float64).ravel()
        )
        if self.theta_tilde_0.size != 4 or not np.all(np.isfinite(self.theta_tilde_0)):
            raise ValueError("theta_tilde_0 deve essere un vettore finito di 4 componenti.")
        if np.any((self.theta_tilde_0 < 0.0) | (self.theta_tilde_0 > 1.0)):
            raise ValueError("theta_tilde_0 deve appartenere allo spazio normalizzato [0, 1]^4.")

        self.L_matrix = (
            np.eye(4, dtype=np.float64)
            if L_matrix is None
            else np.asarray(L_matrix, dtype=np.float64)
        )
        if self.L_matrix.shape != (4, 4) or not np.all(np.isfinite(self.L_matrix)):
            raise ValueError("L_matrix deve essere una matrice 4x4 finita.")

    @property
    def prior_physical_theta(self) -> np.ndarray:
        """Restituisce il punto di riferimento del prior nello spazio dei parametri Theta."""
        return np.asarray(
            self.transform.from_normalized(self.theta_tilde_0), dtype=np.float64
        ).ravel()

    def evaluate_unscaled_half_norm_sq(self, theta: np.ndarray) -> float:
        """
        Restituisce 0.5 * || L * (psi(theta) - theta_tilde_0) ||_2^2 (indipendente da lambda).
        """
        theta_tilde = self.transform.to_normalized(theta)
        diff = self.L_matrix @ (theta_tilde - self.theta_tilde_0)
        return 0.5 * float(np.dot(diff, diff))

    def evaluate_with_lambda(self, theta: np.ndarray, lambda_value: float) -> float:
        """
        Valuta J_reg(theta; lambda_value) per un valore arbitrario di lambda_value >= 0
        senza mutare lo stato dell'oggetto.
        """
        if lambda_value < 0.0 or not np.isfinite(lambda_value):
            raise ValueError(
                f"lambda_value deve essere finito e >= 0, ricevuto {lambda_value}."
            )
        if lambda_value == 0.0:
            return 0.0
        return float(lambda_value) * self.evaluate_unscaled_half_norm_sq(theta)

    def evaluate_breakdown(
        self,
        theta: np.ndarray,
        lambda_override: Optional[float] = None,
    ) -> RegularizationBreakdown:
        """
        Restituisce la scomposizione completa del termine di regolarizzazione, inclusi
        il vettore nello spazio normalizzato [0, 1]^4 e la distanza dal prior.
        """
        lam = self.lambda_reg if lambda_override is None else float(lambda_override)
        if lam < 0.0 or not np.isfinite(lam):
            raise ValueError(f"lambda deve essere finito e >= 0, ricevuto {lam}.")

        theta_phys = np.asarray(theta, dtype=np.float64).ravel()
        theta_tilde = np.asarray(
            self.transform.to_normalized(theta_phys), dtype=np.float64
        ).ravel()
        w_diff = self.L_matrix @ (theta_tilde - self.theta_tilde_0)
        dist_l2 = float(np.linalg.norm(w_diff))
        unscaled = 0.5 * float(np.dot(w_diff, w_diff))
        j_reg = 0.0 if lam == 0.0 else lam * unscaled

        return RegularizationBreakdown(
            theta_physical=theta_phys.copy(),
            theta_tilde=theta_tilde,
            theta_tilde_0=self.theta_tilde_0.copy(),
            theta_0_physical=self.prior_physical_theta,
            weighted_prior_diff=w_diff,
            prior_distance_l2_norm=dist_l2,
            unscaled_penalty_half_norm_sq=unscaled,
            lambda_reg=lam,
            j_reg=j_reg,
        )

    def evaluate(self, theta: np.ndarray) -> float:
        if self.lambda_reg == 0.0:
            return 0.0
        return self.evaluate_with_lambda(theta, self.lambda_reg)

    @staticmethod
    def morozov_target_discrepancy(
        tau_safety: float = 1.05,
        n_obs: Optional[int] = None,
        covariance_scaling: str = "mean_normalized",
    ) -> float:
        """
        Restituisce il valore scalare di riferimento teorico per J_data(theta)
        sotto le ipotesi di rumore gaussiano additivo i.i.d. (eta > 0) e assenza
        di errore di discretizzazione (G_{h_inv}(theta_dagger) == y_clean):
        - se `covariance_scaling == "mean_normalized"` (W^T W = Sigma^{-1} / N_obs):
            E[J_data(theta_dagger)] = 0.5  =>  target = 0.5 * tau_safety^2;
        - se `covariance_scaling == "exact_inverse_covariance"` (W^T W = Sigma^{-1}):
            E[J_data(theta_dagger)] = 0.5 * N_obs  =>  target = 0.5 * N_obs * tau_safety^2.

        Avvertenza metodologica:
        Questo metodo fornisce solo un riferimento teorico diagnostico; non seleziona
        automaticamente lambda e non garantisce il principio di Morozov in presenza
        di discrepanza di discretizzazione (h_tar != h_inv).
        """
        if covariance_scaling == "mean_normalized":
            return 0.5 * (tau_safety ** 2)
        if covariance_scaling == "exact_inverse_covariance":
            if n_obs is None or n_obs <= 0:
                raise ValueError(
                    "Per covariance_scaling='exact_inverse_covariance' occorre specificare n_obs > 0."
                )
            return 0.5 * float(n_obs) * (tau_safety ** 2)
        raise ValueError(f"covariance_scaling non valido: '{covariance_scaling}'.")