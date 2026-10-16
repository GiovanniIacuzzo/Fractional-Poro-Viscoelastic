"""
python/reteco/inverse/objective_functional.py
Funzionale obiettivo J_h(theta) unico e condiviso da tutti gli ottimizzatori (Contratto Sez. 8):
    J_h(theta) = 0.5 * || W * (G_{h_inv}(theta) - y) ||_2^2
               + 0.5 * lambda * || L * (theta_tilde - theta_tilde_0) ||_2^2
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import time
import numpy as np

from ..config_schema import ParameterCoordinateTransform
from .regularization import TikhonovRegularizer


class BudgetExhaustedException(Exception):
    """Sollevata quando l'ottimizzatore raggiunge il numero massimo di valutazioni FEM consentite."""


@dataclass
class EvaluationRecord:
    eval_index: int
    elapsed_sec: float
    theta: np.ndarray
    j_total: float
    j_data: float
    j_reg: float
    best_j_so_far: float


class InverseObjectiveFunctional:
    """
    Valuta J_h(theta) contabilizzando ogni singola chiamata a G_{h_inv}(theta).
    Garantisce che CPSO, L-BFGS-B, Powell, Nelder-Mead e DE usino esattamente
    lo stesso operatore diretto, gli stessi pesi W, la stessa regolarizzazione e gli stessi bounds.
    """

    def __init__(
        self,
        forward_obs_map_G_inv: Callable[[np.ndarray], np.ndarray],
        y_obs: np.ndarray,
        weight_diag_W: np.ndarray,
        transform: ParameterCoordinateTransform,
        regularizer: TikhonovRegularizer,
        max_evaluations: Optional[int] = None,
        out_of_bounds_penalty: float = 1.0e12,
    ):
        self.G_inv = forward_obs_map_G_inv
        self.y_obs = np.asarray(y_obs, dtype=np.float64)
        self.W_diag = np.asarray(weight_diag_W, dtype=np.float64)
        self.transform = transform
        self.regularizer = regularizer
        self.max_evaluations = max_evaluations
        self.out_of_bounds_penalty = float(out_of_bounds_penalty)

        if self.y_obs.shape != self.W_diag.shape:
            raise ValueError("Dimensioni incompatibili tra y_obs e weight_diag_W.")

        self.reset_counters()

    def reset_counters(self) -> None:
        self.num_fem_evaluations: int = 0
        self.start_time: float = time.perf_counter()
        self.best_cost: float = float("inf")
        self.best_theta: Optional[np.ndarray] = None
        self.history: List[EvaluationRecord] = []

    def evaluate_detailed(self, theta: np.ndarray) -> tuple[float, float, float]:
        """Restituisce (J_total, J_data, J_reg) incrementando il contatore FEM."""
        arr = np.asarray(theta, dtype=np.float64)
        if not self.transform.is_within_bounds(arr):
            return self.out_of_bounds_penalty, self.out_of_bounds_penalty, 0.0

        if self.max_evaluations is not None and self.num_fem_evaluations >= self.max_evaluations:
            raise BudgetExhaustedException(
                f"Budget di {self.max_evaluations} valutazioni FEM raggiunto."
            )

        self.num_fem_evaluations += 1
        pred = self.G_inv(arr)
        weighted_res = self.W_diag * (pred - self.y_obs)
        j_data = 0.5 * float(np.dot(weighted_res, weighted_res))
        j_reg = self.regularizer.evaluate(arr)
        j_tot = j_data + j_reg

        if j_tot < self.best_cost:
            self.best_cost = j_tot
            self.best_theta = arr.copy()

        self.history.append(
            EvaluationRecord(
                eval_index=self.num_fem_evaluations,
                elapsed_sec=time.perf_counter() - self.start_time,
                theta=arr.copy(),
                j_total=j_tot,
                j_data=j_data,
                j_reg=j_reg,
                best_j_so_far=self.best_cost,
            )
        )
        return j_tot, j_data, j_reg

    def __call__(self, theta: np.ndarray) -> float:
        j_tot, _, _ = self.evaluate_detailed(theta)
        return j_tot

    def call_from_normalized(self, theta_tilde: np.ndarray) -> float:
        """Valuta J_h a partire dalle coordinate normalizzate theta_tilde in [0, 1]^4."""
        t = np.asarray(theta_tilde, dtype=np.float64)
        if np.any(t < -1.0e-12) or np.any(t > 1.0 + 1.0e-12):
            return self.out_of_bounds_penalty
        t_clipped = np.clip(t, 0.0, 1.0)
        theta = self.transform.from_normalized(t_clipped)
        return self(theta)