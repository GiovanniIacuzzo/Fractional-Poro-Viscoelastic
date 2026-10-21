"""
python/reteco/inverse/objective_functional.py
Funzionale obiettivo inverso J_h(theta) = J_data(theta) + J_reg(theta)
con decomposizione esplicita dei contributi, controlli di coerenza della cronologia,
isolamento rigoroso del vettore pulito in Modalità B (eta > 0), e contatore FEM.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from ..config_schema import ParameterCoordinateTransform
from ..observation.noise_model import (
    CovarianceScalingType,
    EtaZeroPolicy,
    NoisyObservationSample,
    ReferenceScaleType,
    build_mode_a_deterministic_weights,
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
        nominal_noise_sigma: Optional[float] = None,
        uses_clean_target_vector_in_evaluation: bool = False,
    ):
        self.forward_operator = forward_operator
        self.y_observed = np.asarray(y_observed, dtype=np.float64).ravel().copy()
        self.weight_matrix_diag = np.asarray(weight_matrix_diag, dtype=np.float64).ravel().copy()
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
        if self.max_evaluations <= 0:
            raise ValueError(f"max_evaluations deve essere > 0, ricevuto {self.max_evaluations}.")
        self.objective_mode = str(objective_mode)
        self.nominal_noise_sigma: Optional[float] = (
            float(nominal_noise_sigma) if nominal_noise_sigma is not None else None
        )
        self.uses_clean_target_vector_in_evaluation: bool = bool(
            uses_clean_target_vector_in_evaluation
        )

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
        self.best_eval_index: int = 0
        self.best_theta: Optional[np.ndarray] = None
        self.best_theta_tilde: Optional[np.ndarray] = None
        self.history: List[ObjectiveEvaluationBreakdown] = []

    @property
    def is_covariance_invertible(self) -> bool:
        return bool(
            self.objective_mode.startswith("mode_b_noise_weighted_")
            and self.nominal_noise_sigma is not None
            and np.isfinite(self.nominal_noise_sigma)
            and self.nominal_noise_sigma > 0.0
            and np.all(np.isfinite(self.weight_matrix_diag))
            and np.all(self.weight_matrix_diag > 0.0)
        )

    @property
    def cost_history(self) -> List[float]:
        """Restituisce la sequenza monotona non crescente di `best_j_so_far` registrata in `history`."""
        return [float(h.best_j_so_far) for h in self.history]

    def reset_counters(self) -> None:
        """Azzera il contatore delle valutazioni FEM, il miglior costo e la storia."""
        self.num_fem_evaluations = 0
        self.num_evaluations = 0
        self.eval_count = 0
        self.best_cost = float("inf")
        self.best_cost_J = float("inf")
        self.best_eval_index = 0
        self.best_theta = None
        self.best_theta_tilde = None
        self.history.clear()

    @classmethod
    def from_mode_a_deterministic(
        cls,
        forward_operator: Callable[[np.ndarray], np.ndarray],
        y_observed: np.ndarray,
        transform: ParameterCoordinateTransform,
        regularizer: TikhonovRegularizer,
        *,
        y_reference: Optional[np.ndarray] = None,
        allow_observed_as_reference: bool = False,
        scale_type: ReferenceScaleType = "linf",
        max_evaluations: int = 240,
    ) -> "InverseObjectiveFunctional":
        if y_reference is None:
            if not allow_observed_as_reference:
                raise ValueError(
                    "y_reference è None ma allow_observed_as_reference=False."
                )
            ref_vec = y_observed
        else:
            ref_vec = y_reference
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
            nominal_noise_sigma=None,
            uses_clean_target_vector_in_evaluation=False,
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
        y_noisy_arr = np.asarray(sample.y_noisy, dtype=np.float64).ravel()
        n_obs = int(y_noisy_arr.size)
        if n_obs == 0:
            raise ValueError("sample.y_noisy non può essere vuoto.")

        eta_val = float(sample.eta_level)
        if eta_val > 0.0:
            sigma_nom = float(sample.sigma_std)
            if not np.isfinite(sigma_nom) or sigma_nom <= 0.0:
                raise ValueError(
                    f"In Modalità B con eta > 0, sample.sigma_std deve essere finito e > 0 (ricevuto {sigma_nom})."
                )
            if covariance_scaling == "exact_inverse_covariance":
                w_diag = np.full(n_obs, 1.0 / sigma_nom, dtype=np.float64)
            elif covariance_scaling == "mean_normalized":
                w_diag = np.full(
                    n_obs, 1.0 / (np.sqrt(float(n_obs)) * sigma_nom), dtype=np.float64
                )
            else:
                raise ValueError(f"covariance_scaling non supportato: {covariance_scaling}")
            eff_scale = sigma_nom
            mode_label = f"mode_b_noise_weighted_{covariance_scaling}"
            nom_sigma: Optional[float] = sigma_nom
        else:
            if eta_zero_policy == "raise":
                raise ValueError(
                    "Costruzione Modalità B non consentita per eta_level == 0 con eta_zero_policy='raise'."
                )
            w_diag, s_ref = build_mode_a_deterministic_weights(
                y_noisy_arr, scale_type=reference_scale_type
            )
            eff_scale = s_ref
            mode_label = f"mode_b_eta0_fallback_{reference_scale_type}_non_statistical"
            nom_sigma = None

        return cls(
            forward_operator=forward_operator,
            y_observed=y_noisy_arr,
            weight_matrix_diag=w_diag,
            transform=transform,
            regularizer=regularizer,
            max_evaluations=max_evaluations,
            objective_mode=mode_label,
            residual_reference_scale=eff_scale,
            nominal_noise_sigma=nom_sigma,
            uses_clean_target_vector_in_evaluation=False,
        )

    def verify_exact_inverse_covariance_identity(
        self, rtol: float = 1.0e-12, atol: float = 1.0e-15
    ) -> bool:
        if (
            not self.is_covariance_invertible
            or self.nominal_noise_sigma is None
            or "exact_inverse_covariance" not in self.objective_mode
        ):
            return False

        wt_w_diag = np.square(self.weight_matrix_diag)
        sigma_inv_diag = np.full_like(
            wt_w_diag, 1.0 / (self.nominal_noise_sigma ** 2)
        )
        return bool(np.allclose(wt_w_diag, sigma_inv_diag, rtol=rtol, atol=atol))

    @classmethod
    def verify_clean_target_isolation_mode_b(
        cls,
        sample: NoisyObservationSample,
        transform: ParameterCoordinateTransform,
        regularizer: TikhonovRegularizer,
        *,
        covariance_scaling: CovarianceScalingType = "exact_inverse_covariance",
    ) -> Dict[str, Any]:
        """
        Verifica concretamente che due campioni con identico `y_noisy` e identica `sigma_std`
        ma vettori `y_clean` differenti producano lo stesso identico funzionale in Modalità B.
        """
        if float(sample.eta_level) <= 0.0:
            raise ValueError("Il test di isolamento in Modalità B richiede eta_level > 0.")

        alt_y_clean = -3.75 * np.asarray(sample.y_clean, dtype=np.float64) + 42.125
        sample_alt = replace(
            sample,
            y_clean=alt_y_clean,
            y_noisy=np.asarray(sample.y_noisy, dtype=np.float64).copy(),
            noise_vector=np.asarray(sample.noise_vector, dtype=np.float64).copy(),
            weight_matrix_diag=np.asarray(sample.weight_matrix_diag, dtype=np.float64).copy(),
            inverse_covariance_diag=(
                None
                if sample.inverse_covariance_diag is None
                else np.asarray(sample.inverse_covariance_diag, dtype=np.float64).copy()
            ),
        )

        dummy_fwd = lambda th: np.zeros_like(sample.y_noisy, dtype=np.float64)
        obj_a = cls.from_mode_b_noise_model(
            dummy_fwd,
            sample,
            transform,
            regularizer,
            covariance_scaling=covariance_scaling,
        )
        obj_b = cls.from_mode_b_noise_model(
            dummy_fwd,
            sample_alt,
            transform,
            regularizer,
            covariance_scaling=covariance_scaling,
        )

        theta_probe = transform.from_normalized(np.array([0.35, 0.65, 0.25, 0.75], dtype=np.float64))
        y_pred_probe = 0.92 * np.asarray(sample.y_noisy, dtype=np.float64) + 0.01
        bd_a = obj_a.evaluate_from_prediction(theta_probe, y_pred_probe, record_in_history=False)
        bd_b = obj_b.evaluate_from_prediction(theta_probe, y_pred_probe, record_in_history=False)

        weights_equal = bool(np.array_equal(obj_a.weight_matrix_diag, obj_b.weight_matrix_diag))
        obs_equal = bool(np.array_equal(obj_a.y_observed, obj_b.y_observed))
        scale_equal = bool(obj_a.residual_reference_scale == obj_b.residual_reference_scale)
        j_data_equal = bool(bd_a.j_data == bd_b.j_data)
        j_reg_equal = bool(bd_a.j_reg == bd_b.j_reg)
        j_total_equal = bool(bd_a.j_total == bd_b.j_total)
        no_clean_flag = bool(
            (not obj_a.uses_clean_target_vector_in_evaluation)
            and (not obj_b.uses_clean_target_vector_in_evaluation)
        )
        clean_vectors_distinct = bool(not np.allclose(sample.y_clean, sample_alt.y_clean))

        passed = (
            weights_equal
            and obs_equal
            and scale_equal
            and j_data_equal
            and j_reg_equal
            and j_total_equal
            and no_clean_flag
            and clean_vectors_distinct
        )
        return {
            "clean_target_isolation_verified": passed,
            "weights_identical": weights_equal,
            "y_observed_identical": obs_equal,
            "reference_scale_identical": scale_equal,
            "j_data_identical": j_data_equal,
            "j_reg_identical": j_reg_equal,
            "j_total_identical": j_total_equal,
            "clean_vectors_distinct": clean_vectors_distinct,
        }

    def evaluate_from_prediction(
        self,
        theta: np.ndarray,
        y_predicted: np.ndarray,
        *,
        lambda_override: Optional[float] = None,
        record_in_history: bool = False,
        eval_index_override: Optional[int] = None,
    ) -> ObjectiveEvaluationBreakdown:
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
                self.best_eval_index = ev_idx
                self.best_theta = theta_arr.copy()
                self.best_theta_tilde = reg_bd.theta_tilde.copy()
            self.history.append(breakdown)

        return breakdown

    def evaluate_breakdown(self, theta: np.ndarray) -> ObjectiveEvaluationBreakdown:
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
        bd = self.evaluate_breakdown(theta)
        return float(bd.j_total)

    def call_from_normalized(self, theta_tilde: np.ndarray) -> float:
        theta_tilde_clipped = np.clip(
            np.asarray(theta_tilde, dtype=np.float64).ravel(), 0.0, 1.0
        )
        theta_phys = self.transform.from_normalized(theta_tilde_clipped)
        return self.__call__(theta_phys)

    def verify_history_and_best_candidate_consistency(
        self,
        opt_result: Optional[Any] = None,
        *,
        rtol: float = 1.0e-12,
        atol: float = 1.0e-14,
        raise_on_failure: bool = True,
    ) -> Dict[str, Any]:
        """
        Verifica la coerenza interna della cronologia delle valutazioni e del miglior candidato.
        """
        n_hist = len(self.history)
        if n_hist == 0:
            if raise_on_failure:
                raise RuntimeError("Cronologia delle valutazioni vuota: impossibile verificare la coerenza.")
            return {"all_checks_passed": False, "reason": "empty_history"}

        counters_match = (
            self.num_fem_evaluations == n_hist
            and self.num_evaluations == n_hist
            and self.eval_count == n_hist
        )
        indices_sequential = all(
            int(rec.eval_index) == idx + 1 for idx, rec in enumerate(self.history)
        )
        opt_eval_match = True
        optimizer_trace = None
        if opt_result is not None:
            optimizer_trace = getattr(opt_result, "convergence_eval_trace", None)
            if optimizer_trace is None:
                optimizer_trace = getattr(opt_result, "convergence_history_J", None)
            if hasattr(opt_result, "num_fem_evaluations"):
                opt_eval_match = int(opt_result.num_fem_evaluations) == n_hist
            else:
                opt_eval_match = False
            opt_eval_match = opt_eval_match and optimizer_trace is not None
            if optimizer_trace is not None:
                opt_eval_match = opt_eval_match and len(optimizer_trace) == n_hist
        eval_count_matches_history_length = bool(
            counters_match and indices_sequential and opt_eval_match
        )

        decomp_errors = [
            abs(float(rec.j_total) - (float(rec.j_data) + float(rec.j_reg)))
            for rec in self.history
        ]
        max_abs_decomp_error = float(max(decomp_errors))
        j_total_equals_j_data_plus_j_reg = all(
            np.isclose(
                float(rec.j_total),
                float(rec.j_data) + float(rec.j_reg),
                rtol=rtol,
                atol=atol,
            )
            for rec in self.history
        )

        j_totals = [float(rec.j_total) for rec in self.history]
        best_so_far_seq = [float(rec.best_j_so_far) for rec in self.history]
        monotone_non_increasing = all(
            best_so_far_seq[k] <= best_so_far_seq[k - 1] + atol
            for k in range(1, n_hist)
        )
        running_min_exact = all(
            np.isclose(best_so_far_seq[k], min(j_totals[: k + 1]), rtol=rtol, atol=atol)
            for k in range(n_hist)
        )
        opt_hist_match = True
        if opt_result is not None:
            if optimizer_trace is None:
                opt_hist_match = False
            else:
                opt_hist_match = bool(
                    np.allclose(
                        np.asarray(optimizer_trace, dtype=np.float64),
                        np.asarray(best_so_far_seq, dtype=np.float64),
                        rtol=rtol,
                        atol=atol,
                    )
                )
        best_cost_history_monotone = bool(
            monotone_non_increasing and running_min_exact and opt_hist_match
        )

        min_pos = int(np.argmin(j_totals))
        min_rec = self.history[min_pos]
        expected_best_eval_index = min_pos + 1

        cost_matches = bool(
            np.isclose(min_rec.j_total, self.best_cost, rtol=rtol, atol=atol)
            and np.isclose(min_rec.j_total, self.best_cost_J, rtol=rtol, atol=atol)
            and np.isclose(best_so_far_seq[-1], self.best_cost, rtol=rtol, atol=atol)
        )
        eval_index_matches = bool(
            int(min_rec.eval_index) == expected_best_eval_index
            and int(self.best_eval_index) == expected_best_eval_index
        )
        theta_matches_obj = bool(
            self.best_theta is not None
            and np.allclose(min_rec.theta, self.best_theta, rtol=rtol, atol=atol)
        )
        opt_best_matches = True
        if opt_result is not None:
            if hasattr(opt_result, "best_cost_J"):
                opt_best_matches = opt_best_matches and bool(
                    np.isclose(
                        float(opt_result.best_cost_J),
                        float(min_rec.j_total),
                        rtol=rtol,
                        atol=atol,
                    )
                )
            if hasattr(opt_result, "theta_hat"):
                opt_best_matches = opt_best_matches and bool(
                    np.allclose(
                        np.asarray(opt_result.theta_hat, dtype=np.float64),
                        min_rec.theta,
                        rtol=rtol,
                        atol=atol,
                    )
                )
        best_candidate_consistent = bool(
            cost_matches and eval_index_matches and theta_matches_obj and opt_best_matches
        )

        coord_errors_tilde: List[float] = []
        coord_errors_phys: List[float] = []
        bounds_ok = True
        for rec in self.history:
            tilde_calc = self.transform.to_normalized(rec.theta)
            phys_calc = self.transform.from_normalized(rec.theta_tilde)
            coord_errors_tilde.append(float(np.max(np.abs(rec.theta_tilde - tilde_calc))))
            coord_errors_phys.append(float(np.max(np.abs(rec.theta - phys_calc))))
            if not (
                np.all(rec.theta_tilde >= -atol)
                and np.all(rec.theta_tilde <= 1.0 + atol)
                and self.transform.is_within_bounds(rec.theta, tol=1.0e-12)
            ):
                bounds_ok = False

        max_tilde_err = float(max(coord_errors_tilde))
        max_phys_err = float(max(coord_errors_phys))
        opt_coord_ok = True
        if opt_result is not None and hasattr(opt_result, "theta_tilde_hat") and hasattr(opt_result, "theta_hat"):
            opt_coord_ok = bool(
                np.allclose(
                    np.asarray(opt_result.theta_tilde_hat, dtype=np.float64),
                    self.transform.to_normalized(np.asarray(opt_result.theta_hat, dtype=np.float64)),
                    rtol=rtol,
                    atol=atol,
                )
            )
        coordinates_consistent = bool(
            bounds_ok
            and max_tilde_err <= atol + rtol
            and max_phys_err <= atol + rtol
            and opt_coord_ok
        )

        all_passed = bool(
            eval_count_matches_history_length
            and j_total_equals_j_data_plus_j_reg
            and best_cost_history_monotone
            and best_candidate_consistent
            and coordinates_consistent
        )

        report: Dict[str, Any] = {
            "all_checks_passed": all_passed,
            "eval_count_matches_history_length": eval_count_matches_history_length,
            "num_fem_evaluations": int(self.num_fem_evaluations),
            "history_length": int(n_hist),
            "j_total_equals_j_data_plus_j_reg_all_records": j_total_equals_j_data_plus_j_reg,
            "max_abs_j_decomposition_error": max_abs_decomp_error,
            "best_cost_history_monotone_non_increasing": best_cost_history_monotone,
            "historical_min_matches_best_cost_eval_index_and_theta_hat": best_candidate_consistent,
            "best_eval_index": int(expected_best_eval_index),
            "best_cost_J_total": float(min_rec.j_total),
            "normalized_and_physical_coordinates_consistent": coordinates_consistent,
            "max_abs_coordinate_tilde_error": max_tilde_err,
            "max_abs_coordinate_physical_error": max_phys_err,
        }

        if raise_on_failure and not all_passed:
            raise RuntimeError(
                f"Controllo di coerenza della cronologia fallito: {report}"
            )

        return report