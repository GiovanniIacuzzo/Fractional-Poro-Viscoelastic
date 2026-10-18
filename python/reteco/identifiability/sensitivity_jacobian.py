"""
python/reteco/identifiability/sensitivity_jacobian.py
Calcolo verificato del Jacobiano di sensibilità nelle coordinate normalizzate
    J_{ij}^G(theta_tilde) = d G_i / d theta_tilde_j,   theta_tilde in [0, 1]^4
con:
- differenze finite centrali del 2° ordine nei punti interni e schemi
  unidirezionali del 2° ordine vicino ai bordi di [0, 1]^4;
- audit delle scale osservazionali e pesatura opzionale J_w = W^{1/2} J
  esclusivamente da pesi/incertezze dichiarate;
- studio opzionale di stabilità rispetto al passo delle differenze finite.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

from ..config_schema import ParameterCoordinateTransform

PARAMETER_NAMES: Tuple[str, ...] = ("alpha", "beta", "D", "E")


@dataclass(frozen=True)
class ObservationWeightingSpec:
    """
    Specifica esplicita della normalizzazione/pesatura delle osservazioni per J_w = W^{1/2} J.
    Se non sono dichiarati pesi o incertezze statistiche nella configurazione,
    usa l'identità (W^{1/2} = I) sulle osservazioni adimensionali senza inventare rumore.
    """
    weighting_mode: str = "unweighted_dimensionless_state_identity"
    weights_sqrt_diag: Optional[np.ndarray] = None
    is_statistically_calibrated: bool = False
    observed_fields: Tuple[str, ...] = ("c",)
    provenance_description: str = (
        "Unweighted dimensionless observations from DiscreteObservationOperator "
        "(W^{1/2} = I). Quantities represent geometric sensitivity in dimensionless "
        "state units, not noise-calibrated statistical estimators."
    )

    def apply_to_jacobian(self, J_raw: np.ndarray) -> np.ndarray:
        J = np.asarray(J_raw, dtype=np.float64)
        if self.weights_sqrt_diag is None:
            return J.copy()
        w = np.asarray(self.weights_sqrt_diag, dtype=np.float64)
        if w.ndim != 1 or w.shape[0] != J.shape[0]:
            raise ValueError(
                f"Dimensione di weights_sqrt_diag ({w.shape}) incompatibile con J_raw ({J.shape})."
            )
        if not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("weights_sqrt_diag deve contenere valori finiti e non negativi.")
        return w[:, None] * J

    def apply_to_residual(self, residual: np.ndarray) -> np.ndarray:
        r = np.asarray(residual, dtype=np.float64)
        if self.weights_sqrt_diag is None:
            return r
        return np.asarray(self.weights_sqrt_diag, dtype=np.float64) * r

    def to_metadata_dict(self, n_obs: int) -> Dict[str, Any]:
        w_summary: Dict[str, Any] = {"type": "identity", "n_obs": int(n_obs)}
        if self.weights_sqrt_diag is not None:
            w = np.asarray(self.weights_sqrt_diag, dtype=np.float64)
            w_summary = {
                "type": "diagonal_W_sqrt",
                "n_obs": int(n_obs),
                "min_weight_sqrt": float(np.min(w)),
                "max_weight_sqrt": float(np.max(w)),
                "mean_weight_sqrt": float(np.mean(w)),
            }
        return {
            "weighting_mode": self.weighting_mode,
            "is_statistically_calibrated": bool(self.is_statistically_calibrated),
            "observed_fields": list(self.observed_fields),
            "provenance_description": self.provenance_description,
            "weights_summary": w_summary,
        }


@dataclass(frozen=True)
class JacobianAuditResult:
    jacobian_raw: np.ndarray                   # (N_obs, 4) Jacobiano non pesato J
    jacobian_weighted: np.ndarray              # (N_obs, 4) Jacobiano effettivo J_w = W^{1/2} J
    baseline_observation: np.ndarray           # (N_obs,) G(theta_ref)
    theta_ref_physical: np.ndarray             # (4,)
    theta_ref_normalized: np.ndarray           # (4,) in [0, 1]^4
    eps_step_requested: float
    steps_used_per_parameter: Dict[str, float]
    scheme_per_parameter: Dict[str, str]
    near_boundary_flags: Dict[str, bool]
    num_forward_evaluations: int
    weighting_spec: ObservationWeightingSpec

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "num_observations": int(self.baseline_observation.shape[0]),
            "num_parameters": 4,
            "parameter_names": list(PARAMETER_NAMES),
            "theta_ref_physical": self.theta_ref_physical.tolist(),
            "theta_ref_normalized": self.theta_ref_normalized.tolist(),
            "eps_step_requested": float(self.eps_step_requested),
            "steps_used_per_parameter": dict(self.steps_used_per_parameter),
            "scheme_per_parameter": dict(self.scheme_per_parameter),
            "near_boundary_flags": dict(self.near_boundary_flags),
            "num_forward_evaluations": int(self.num_forward_evaluations),
            "baseline_observation_summary": {
                "min": float(np.min(self.baseline_observation)),
                "max": float(np.max(self.baseline_observation)),
                "l2_norm": float(np.linalg.norm(self.baseline_observation)),
            },
            "weighting": self.weighting_spec.to_metadata_dict(
                int(self.baseline_observation.shape[0])
            ),
        }


@dataclass(frozen=True)
class FDStepStabilityReport:
    reference_step: float
    tested_steps: List[float]
    frobenius_rel_diff_vs_reference: Dict[str, float]
    column_rel_diffs_vs_reference: Dict[str, Dict[str, float]]
    singular_values_by_step: Dict[str, List[float]]
    column_norms_by_step: Dict[str, List[float]]
    total_forward_evaluations: int

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "reference_step": float(self.reference_step),
            "tested_steps": [float(s) for s in self.tested_steps],
            "frobenius_rel_diff_vs_reference": dict(self.frobenius_rel_diff_vs_reference),
            "column_rel_diffs_vs_reference": dict(self.column_rel_diffs_vs_reference),
            "singular_values_by_step": dict(self.singular_values_by_step),
            "column_norms_by_step": dict(self.column_norms_by_step),
            "total_forward_evaluations": int(self.total_forward_evaluations),
        }


def _evaluate_and_validate_obs(
    forward_obs_callable: Callable[[np.ndarray], np.ndarray],
    theta_phys: np.ndarray,
    expected_size: Optional[int] = None,
) -> np.ndarray:
    y = np.asarray(forward_obs_callable(theta_phys), dtype=np.float64)
    if y.ndim != 1 or y.size == 0:
        raise ValueError(
            f"L'operatore osservativo deve restituire un vettore 1D non vuoto, ricevuto shape={y.shape}."
        )
    if not np.all(np.isfinite(y)):
        raise ValueError("L'operatore osservativo ha restituito valori NaN o Inf.")
    if expected_size is not None and y.size != expected_size:
        raise ValueError(
            f"Dimensione osservazione incoerente: atteso {expected_size}, ottenuto {y.size}."
        )
    return y


def compute_normalized_jacobian_with_audit(
    forward_obs_callable: Callable[[np.ndarray], np.ndarray],
    theta_ref: np.ndarray,
    transform: ParameterCoordinateTransform,
    eps_step: float = 1.0e-4,
    weighting_spec: Optional[ObservationWeightingSpec] = None,
) -> JacobianAuditResult:
    """
    Calcola il Jacobiano normalizzato J_{ij} = d G_i / d theta_tilde_j con audit completo:
    - Punti interni (h <= theta_tilde_j <= 1 - h): differenze finite centrali del 2° ordine;
    - Punti vicini al bordo inferiore (theta_tilde_j < h): schema in avanti del 2° ordine a 3 punti;
    - Punti vicini al bordo superiore (theta_tilde_j > 1 - h): schema all'indietro del 2° ordine a 3 punti.
    """
    if not np.isfinite(eps_step) or eps_step <= 0.0 or eps_step >= 0.25:
        raise ValueError(f"eps_step deve essere in (0, 0.25), ricevuto {eps_step}.")

    theta_phys_0 = np.asarray(theta_ref, dtype=np.float64)
    if theta_phys_0.shape != (4,) or not np.all(np.isfinite(theta_phys_0)):
        raise ValueError(f"theta_ref deve essere un vettore finito di 4 componenti, ricevuto {theta_phys_0}.")

    theta_tilde_0 = np.asarray(transform.to_normalized(theta_phys_0), dtype=np.float64)
    if np.any(theta_tilde_0 < 0.0) or np.any(theta_tilde_0 > 1.0):
        raise ValueError(
            f"theta_ref fuori dai bounds ammissibili: theta_tilde={theta_tilde_0.tolist()}."
        )

    y0 = _evaluate_and_validate_obs(forward_obs_callable, theta_phys_0)
    n_obs = int(y0.size)
    n_evals = 1

    J_raw = np.zeros((n_obs, 4), dtype=np.float64)
    steps_used: Dict[str, float] = {}
    schemes_used: Dict[str, str] = {}
    near_bdy: Dict[str, bool] = {}

    for j, pname in enumerate(PARAMETER_NAMES):
        tj = float(theta_tilde_0[j])
        h = float(eps_step)

        if tj - h >= 0.0 and tj + h <= 1.0:
            # Schema centrale del secondo ordine O(h^2)
            t_plus = theta_tilde_0.copy()
            t_minus = theta_tilde_0.copy()
            t_plus[j] = tj + h
            t_minus[j] = tj - h

            y_plus = _evaluate_and_validate_obs(
                forward_obs_callable, transform.from_normalized(t_plus), expected_size=n_obs
            )
            y_minus = _evaluate_and_validate_obs(
                forward_obs_callable, transform.from_normalized(t_minus), expected_size=n_obs
            )
            n_evals += 2

            J_raw[:, j] = (y_plus - y_minus) / (2.0 * h)
            steps_used[pname] = h
            schemes_used[pname] = "central_2nd_order"
            near_bdy[pname] = False

        elif tj - h < 0.0:
            # Vicino al limite inferiore: schema unidirezionale in avanti del 2° ordine a 3 punti
            h_eff = min(h, 0.5 * (1.0 - tj))
            t_p1 = theta_tilde_0.copy()
            t_p2 = theta_tilde_0.copy()
            t_p1[j] = tj + h_eff
            t_p2[j] = tj + 2.0 * h_eff

            y_p1 = _evaluate_and_validate_obs(
                forward_obs_callable, transform.from_normalized(t_p1), expected_size=n_obs
            )
            y_p2 = _evaluate_and_validate_obs(
                forward_obs_callable, transform.from_normalized(t_p2), expected_size=n_obs
            )
            n_evals += 2

            J_raw[:, j] = (-3.0 * y0 + 4.0 * y_p1 - y_p2) / (2.0 * h_eff)
            steps_used[pname] = h_eff
            schemes_used[pname] = "forward_2nd_order_boundary"
            near_bdy[pname] = True

        else:
            # Vicino al limite superiore: schema unidirezionale all'indietro del 2° ordine a 3 punti
            h_eff = min(h, 0.5 * tj)
            t_m1 = theta_tilde_0.copy()
            t_m2 = theta_tilde_0.copy()
            t_m1[j] = tj - h_eff
            t_m2[j] = tj - 2.0 * h_eff

            y_m1 = _evaluate_and_validate_obs(
                forward_obs_callable, transform.from_normalized(t_m1), expected_size=n_obs
            )
            y_m2 = _evaluate_and_validate_obs(
                forward_obs_callable, transform.from_normalized(t_m2), expected_size=n_obs
            )
            n_evals += 2

            J_raw[:, j] = (3.0 * y0 - 4.0 * y_m1 + y_m2) / (2.0 * h_eff)
            steps_used[pname] = h_eff
            schemes_used[pname] = "backward_2nd_order_boundary"
            near_bdy[pname] = True

    if not np.all(np.isfinite(J_raw)):
        raise RuntimeError("Il Jacobiano calcolato contiene valori non finiti (NaN o Inf).")

    w_spec = weighting_spec or ObservationWeightingSpec()
    J_weighted = w_spec.apply_to_jacobian(J_raw)

    return JacobianAuditResult(
        jacobian_raw=J_raw,
        jacobian_weighted=J_weighted,
        baseline_observation=y0,
        theta_ref_physical=theta_phys_0,
        theta_ref_normalized=theta_tilde_0,
        eps_step_requested=float(eps_step),
        steps_used_per_parameter=steps_used,
        scheme_per_parameter=schemes_used,
        near_boundary_flags=near_bdy,
        num_forward_evaluations=n_evals,
        weighting_spec=w_spec,
    )


def compute_normalized_jacobian(
    forward_obs_callable: Callable[[np.ndarray], np.ndarray],
    theta_ref: np.ndarray,
    transform: ParameterCoordinateTransform,
    eps_step: float = 1.0e-4,
    weighting_spec: Optional[ObservationWeightingSpec] = None,
    return_audit: bool = False,
) -> np.ndarray | JacobianAuditResult:
    """
    Funzione pubblica compatibile con i test preesistenti:
    - se `return_audit=False` (default), restituisce `jacobian_weighted` come `np.ndarray` (N_obs, 4);
    - se `return_audit=True`, restituisce l'oggetto completo `JacobianAuditResult`.
    """
    audit = compute_normalized_jacobian_with_audit(
        forward_obs_callable=forward_obs_callable,
        theta_ref=theta_ref,
        transform=transform,
        eps_step=eps_step,
        weighting_spec=weighting_spec,
    )
    if return_audit:
        return audit
    return audit.jacobian_weighted


def verify_fd_step_stability(
    forward_obs_callable: Callable[[np.ndarray], np.ndarray],
    theta_ref: np.ndarray,
    transform: ParameterCoordinateTransform,
    steps: Sequence[float] = (1.0e-3, 1.0e-4, 1.0e-5),
    reference_step: float = 1.0e-4,
    weighting_spec: Optional[ObservationWeightingSpec] = None,
) -> FDStepStabilityReport:
    """
    Verifica la stabilità numerica del Jacobiano rispetto al passo h delle differenze finite.
    Riporta le differenze relative rispetto a `reference_step` senza imporre soglie artificiali.
    """
    step_list = [float(s) for s in steps]
    if reference_step not in step_list:
        step_list.append(float(reference_step))
        step_list.sort(reverse=True)

    audits_by_step: Dict[float, JacobianAuditResult] = {}
    total_evals = 0
    for s in step_list:
        aud = compute_normalized_jacobian_with_audit(
            forward_obs_callable=forward_obs_callable,
            theta_ref=theta_ref,
            transform=transform,
            eps_step=s,
            weighting_spec=weighting_spec,
        )
        audits_by_step[s] = aud
        total_evals += aud.num_forward_evaluations

    J_ref = audits_by_step[float(reference_step)].jacobian_weighted
    frob_ref = float(np.linalg.norm(J_ref, ord="fro"))

    frob_rel_diffs: Dict[str, float] = {}
    col_rel_diffs: Dict[str, Dict[str, float]] = {}
    sv_by_step: Dict[str, List[float]] = {}
    cn_by_step: Dict[str, List[float]] = {}

    for s in step_list:
        key = f"{s:.1e}"
        J_s = audits_by_step[s].jacobian_weighted
        diff_frob = float(np.linalg.norm(J_s - J_ref, ord="fro"))
        frob_rel_diffs[key] = diff_frob / max(frob_ref, 1.0e-15)

        c_norms = np.linalg.norm(J_s, axis=0)
        s_vals = np.linalg.svd(J_s, compute_uv=False)
        cn_by_step[key] = [float(x) for x in c_norms]
        sv_by_step[key] = [float(x) for x in s_vals]

        per_col: Dict[str, float] = {}
        for j, pname in enumerate(PARAMETER_NAMES):
            norm_ref_j = float(np.linalg.norm(J_ref[:, j]))
            norm_s_j = float(np.linalg.norm(J_s[:, j]))
            if norm_ref_j < 1.0e-14 and norm_s_j < 1.0e-14:
                per_col[pname] = 0.0
            else:
                per_col[pname] = float(
                    np.linalg.norm(J_s[:, j] - J_ref[:, j]) / max(norm_ref_j, 1.0e-15)
                )
        col_rel_diffs[key] = per_col

    return FDStepStabilityReport(
        reference_step=float(reference_step),
        tested_steps=step_list,
        frobenius_rel_diff_vs_reference=frob_rel_diffs,
        column_rel_diffs_vs_reference=col_rel_diffs,
        singular_values_by_step=sv_by_step,
        column_norms_by_step=cn_by_step,
        total_forward_evaluations=total_evals,
    )