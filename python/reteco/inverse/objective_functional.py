"""
python/reteco/inverse/objective_functional.py
Funzionale obiettivo inverso J_h(theta) = J_data(theta) + J_reg(theta)
con decomposizione esplicita dei tre contributi, diagnostiche della scala dei residui,
supporto per Modalità A (deterministica normalizzata) e Modalità B (pesata sul rumore),
e contatore delle valutazioni del solutore diretto FEM.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from ..config_schema import ParameterCoordinateTransform
from ..observation.noise_model import (
    CovarianceScalingType,
    EtaZeroPolicy,
    NoisyObservationSample,
    ReferenceScaleType,
    build_mode_a_deterministic_weights,
    build_mode_b_noise_weights,
)
from .regularization import TikhonovRegularizer


class BudgetExhaustedException(RuntimeError):
    """Sollevata quando il numero di valutazioni FEM raggiunge `max_evaluations`."""


@dataclass(frozen=True)
class ObjectiveEvaluationBreakdown:
    """
    Record immutabile di una singola valutazione di J_h(theta), compatibile sia con
    l'accesso tramite attributi (`rec.j_total`) sia tramite chiavi dizionario (`rec["j_total"]`).
    """
    eval_index: int
    theta: np.ndarray
    theta_tilde: np.ndarray
    j_data: float
    j_reg: float
    j_total: float
    best_j_so_far: float
    unweighted_residual_l2_norm: float
    unweighted_residual_linf_norm: float
    relative_residual_l2: float
    weighted_residual_l2_norm: float
    prior_distance_l2_norm: float
    data_share_of_total: float
    reg_share_of_total: float
    residual_reference_scale: float
    objective_mode: str

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["theta"] = self.theta.tolist()
        d["theta_tilde"] = self.theta_tilde.tolist()
        return d


class InverseObjectiveFunctional:
    def __init__(
        self,
        forward_operator: Callable[[np.ndarray], np.ndarray],
        y_observed: np.ndarray,
        weight_matrix_diag: np.ndarray,
        transform: ParameterCoordinateTransform,
        regularizer: TikhonovRegularizer,
        max_evaluations: int = 240,
        *,
        objective_mode: str = "legacy_compatible",
        residual_reference_scale: Optional[float] = None,
    ):
        self.forward_operator = forward_operator
        self.y_observed = np.asarray(y_observed, dtype=np.float64).ravel()
        self.weight_matrix_diag = np.asarray(weight_matrix_diag, dtype=np.float64).ravel()
        if self.y_observed.size == 0:
            raise ValueError("y_observed non può essere vuoto.")
        if self.weight_matrix_diag.shape != self.y_observed.shape:
            raise ValueError(
                f"Dimensioni incompatibili tra weight_matrix_diag {self.weight_matrix_diag.shape} "
                f"e y_observed {self.y_observed.shape}."
            )
        if not np.all(np.isfinite(self.weight_matrix_diag)) or np.any(self.weight_matrix_diag <= 0.0):
            raise ValueError("Tutti gli elementi di weight_matrix_diag devono essere finiti e > 0.")

        self.transform = transform
        self.regularizer = regularizer
        self.max_evaluations = int(max_evaluations)
        self.objective_mode = str(objective_mode)

        obs_l2 = float(np.linalg.norm(self.y_observed))
        self._y_obs_l2_denom = obs_l2 if obs_l2 > 0.0 else 1.0
        if residual_reference_scale is not None:
            self.residual_reference_scale = float(residual_reference_scale)
        else:
            mean_w = float(np.mean(self.weight_matrix_diag))
            self.residual_reference_scale = 1.0 / (np.sqrt(self.y_observed.size) * mean_w)

        self.num_fem_evaluations: int = 0
        self.num_evaluations: int = 0
        self.eval_count: int = 0
        self.best_cost: float = float("inf")
        self.best_cost_J: float = float("inf")
        self.best_theta: Optional[np.ndarray] = None
        self.best_theta_tilde: Optional[np.ndarray] = None
        self.history: List[ObjectiveEvaluationBreakdown] = []

    def reset_counters(self) -> None:
        """Azzera il contatore delle valutazioni FEM, il miglior costo e la storia."""
        self.num_fem_evaluations = 0
        self.num_evaluations = 0
        self.eval_count = 0
        self.best_cost = float("inf")
        self.best_cost_J = float("inf")
        self.best_theta = None
        self.best_theta_tilde = None
        self.history.clear()

    @property
    def cost_history(self) -> List[float]:
        """Restituisce la sequenza di `best_j_so_far` registrata in `history`."""
        return [float(h.best_j_so_far) for h in self.history]

    @classmethod
    def from_mode_a_deterministic(
        cls,
        forward_operator: Callable[[np.ndarray], np.ndarray],
        y_observed: np.ndarray,
        transform: ParameterCoordinateTransform,
        regularizer: TikhonovRegularizer,
        *,
        y_reference: Optional[np.ndarray] = None,
        scale_type: ReferenceScaleType = "linf",
        max_evaluations: int = 240,
    ) -> "InverseObjectiveFunctional":
        """
        Costruisce il funzionale nella Modalità A (Obiettivo deterministico normalizzato).
        Usa una scala di riferimento strettamente positiva calcolata su `y_reference`
        (o `y_observed` se non specificato), identica per qualsiasi livello di rumore.
        """
        ref_vec = y_observed if y_reference is None else y_reference
        w_diag, s_ref = build_mode_a_deterministic_weights(ref_vec, scale_type=scale_type)
        return cls(
            forward_operator=forward_operator,
            y_observed=y_observed,
            weight_matrix_diag=w_diag,
            transform=transform,
            regularizer=regularizer,
            max_evaluations=max_evaluations,
            objective_mode=f"mode_a_deterministic_normalized_{scale_type}",
            residual_reference_scale=s_ref,
        )

    @classmethod
    def from_mode_b_noise_model(
        cls,
        forward_operator: Callable[[np.ndarray], np.ndarray],
        sample: NoisyObservationSample,
        transform: ParameterCoordinateTransform,
        regularizer: TikhonovRegularizer,
        *,
        covariance_scaling: CovarianceScalingType = "mean_normalized",
        eta_zero_policy: EtaZeroPolicy = "fallback_mode_a",
        reference_scale_type: ReferenceScaleType = "linf",
        max_evaluations: int = 240,
    ) -> "InverseObjectiveFunctional":
        """
        Costruisce il funzionale nella Modalità B (Obiettivo pesato secondo il modello di rumore).
        """
        w_diag, sigma_nom, s_ref, _ = build_mode_b_noise_weights(
            sample.y_clean,
            eta_level=sample.eta_level,
            covariance_scaling=covariance_scaling,
            eta_zero_policy=eta_zero_policy,
            reference_scale_type=reference_scale_type,
        )
        eff_scale = sigma_nom if sigma_nom > 0.0 else s_ref
        mode_label = (
            f"mode_b_noise_weighted_{covariance_scaling}"
            if sample.eta_level > 0.0
            else f"mode_b_eta0_fallback_{reference_scale_type}"
        )
        return cls(
            forward_operator=forward_operator,
            y_observed=sample.y_noisy,
            weight_matrix_diag=w_diag,
            transform=transform,
            regularizer=regularizer,
            max_evaluations=max_evaluations,
            objective_mode=mode_label,
            residual_reference_scale=eff_scale,
        )

    def evaluate_from_prediction(
        self,
        theta: np.ndarray,
        y_predicted: np.ndarray,
        *,
        lambda_override: Optional[float] = None,
        record_in_history: bool = False,
        eval_index_override: Optional[int] = None,
    ) -> ObjectiveEvaluationBreakdown:
        """
        Calcola separatamente J_data(theta), J_reg(theta), J_total(theta) e tutte le
        diagnostiche dei residui a partire da una predizione `y_predicted` già disponibile.
        """
        theta_arr = np.asarray(theta, dtype=np.float64).ravel()
        y_pred = np.asarray(y_predicted, dtype=np.float64).ravel()
        if y_pred.shape != self.y_observed.shape:
            raise ValueError(
                f"Dimensione y_predicted {y_pred.shape} incompatibile con y_observed {self.y_observed.shape}."
            )
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("La predizione del modello diretto contiene valori non finiti.")

        residual = y_pred - self.y_observed
        unweighted_l2 = float(np.linalg.norm(residual))
        unweighted_linf = float(np.max(np.abs(residual)))
        rel_res_l2 = unweighted_l2 / self._y_obs_l2_denom

        weighted_residual = self.weight_matrix_diag * residual
        weighted_l2 = float(np.linalg.norm(weighted_residual))
        j_data = 0.5 * float(np.dot(weighted_residual, weighted_residual))

        reg_bd = self.regularizer.evaluate_breakdown(
            theta_arr, lambda_override=lambda_override
        )
        j_reg = float(reg_bd.j_reg)
        j_total = float(j_data + j_reg)

        if j_total > 0.0:
            data_share = float(j_data / j_total)
            reg_share = float(j_reg / j_total)
        else:
            data_share = 0.0
            reg_share = 0.0

        running_best = min(self.best_cost, j_total)
        ev_idx = (
            int(eval_index_override)
            if eval_index_override is not None
            else int(self.num_fem_evaluations)
        )

        breakdown = ObjectiveEvaluationBreakdown(
            eval_index=ev_idx,
            theta=theta_arr.copy(),
            theta_tilde=reg_bd.theta_tilde.copy(),
            j_data=j_data,
            j_reg=j_reg,
            j_total=j_total,
            best_j_so_far=running_best,
            unweighted_residual_l2_norm=unweighted_l2,
            unweighted_residual_linf_norm=unweighted_linf,
            relative_residual_l2=rel_res_l2,
            weighted_residual_l2_norm=weighted_l2,
            prior_distance_l2_norm=reg_bd.prior_distance_l2_norm,
            data_share_of_total=data_share,
            reg_share_of_total=reg_share,
            residual_reference_scale=self.residual_reference_scale,
            objective_mode=self.objective_mode,
        )

        if record_in_history:
            if j_total < self.best_cost:
                self.best_cost = j_total
                self.best_cost_J = j_total
                self.best_theta = theta_arr.copy()
                self.best_theta_tilde = reg_bd.theta_tilde.copy()
            self.history.append(breakdown)

        return breakdown

    def evaluate_breakdown(self, theta: np.ndarray) -> ObjectiveEvaluationBreakdown:
        """
        Esegue una valutazione del solutore diretto G_h(theta), aggiorna il contatore
        delle valutazioni FEM e restituisce il record completo ObjectiveEvaluationBreakdown.
        """
        if self.num_fem_evaluations >= self.max_evaluations:
            raise BudgetExhaustedException(
                f"Budget di {self.max_evaluations} valutazioni FEM esaurito."
            )
        theta_arr = np.asarray(theta, dtype=np.float64).ravel()
        if not self.transform.is_within_bounds(theta_arr):
            theta_arr = self.transform.clip_to_bounds(theta_arr)

        y_pred = self.forward_operator(theta_arr)
        self.num_fem_evaluations += 1
        self.num_evaluations = self.num_fem_evaluations
        self.eval_count = self.num_fem_evaluations

        return self.evaluate_from_prediction(
            theta_arr,
            y_pred,
            record_in_history=True,
            eval_index_override=self.num_fem_evaluations,
        )

    def __call__(self, theta: np.ndarray) -> float:
        """Restituisce il costo scalare J_total(theta) per compatibilità con gli ottimizzatori."""
        bd = self.evaluate_breakdown(theta)
        return float(bd.j_total)

    def call_from_normalized(self, theta_tilde: np.ndarray) -> float:
        """Valuta J_total(theta) a partire dalle coordinate normalizzate theta_tilde in [0, 1]^4."""
        theta_tilde_clipped = np.clip(
            np.asarray(theta_tilde, dtype=np.float64).ravel(), 0.0, 1.0
        )
        theta_phys = self.transform.from_normalized(theta_tilde_clipped)
        return self.__call__(theta_phys)