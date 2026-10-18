"""
python/reteco/identifiability/svd_conditioning.py
Analisi spettrale SVD rigorosa del Jacobiano di sensibilità (Contratto Sez. 8):
- Gestione trasparente dei casi a rango deficiente (nessuna correlazione fittizia da
  pseudoinversa di Fisher per parametri non identificabili);
- Distinzione esplicita tra diagnostica geometrica di sensibilità (F = J^T J non pesata)
  e correlazione statistica locale calibrata sul rumore (F = J_w^T J_w);
- Identificazione del modo singolare meno informativo e dell'indice di collinearità di Brun.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .sensitivity_jacobian import PARAMETER_NAMES, ObservationWeightingSpec


def sanitize_for_json(obj: Any) -> Any:
    """
    Converte ricorsivamente array NumPy, float NaN e Inf in strutture serializzabili
    JSON standard (RFC 8259), sostituendo NaN/Inf con None (`null` in JSON).
    """
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


@dataclass(frozen=True)
class LocalIdentifiabilityReport:
    jacobian_tilde: np.ndarray                 # (N_obs, 4) Jacobiano analizzato
    column_norms: np.ndarray                   # (4,) norme L2 delle colonne [alpha~, beta~, D~, E~]
    singular_values: np.ndarray                # (4,) sigma_1 >= sigma_2 >= sigma_3 >= sigma_4
    right_singular_vectors_V: np.ndarray       # (4, 4) colonne = modi singolari destri v_k
    numerical_rank: int
    condition_number: float
    collinearity_index: float                  # 1 / sigma_min(J_col_normalized)
    fisher_correlation_matrix: np.ndarray      # (4, 4) definita solo se numerical_rank == 4, altrimenti NaN
    # Campi diagnostici estesi per la revisione scientifica
    parameter_names: Tuple[str, ...] = PARAMETER_NAMES
    rank_threshold_relative: float = 1.0e-8
    rank_threshold_absolute: float = 0.0
    effective_condition_number_on_rank: float = float("inf")
    active_subset_collinearity_index: float = float("inf")
    collinearity_definition: str = (
        "Brun et al. (2001) collinearity index gamma = 1 / sigma_min(J_unit), where "
        "J_unit is the column-normalized Jacobian (each active column scaled to unit L2 norm). "
        "Undefined (inf) on the full 4-parameter set when any column is zero or rank < 4."
    )
    identifiable_columns_mask: Tuple[bool, ...] = (True, True, True, True)
    unidentifiable_parameters: Tuple[str, ...] = ()
    correlation_matrix_defined: bool = True
    correlation_interpretation: str = "geometric_sensitivity_correlation_unweighted"
    correlation_status_note: str = ""
    column_cosine_similarity_matrix: np.ndarray = field(
        default_factory=lambda: np.full((4, 4), np.nan, dtype=np.float64)
    )
    least_informative_mode: Dict[str, Any] = field(default_factory=dict)
    weighting_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        return sanitize_for_json(
            {
                "parameter_names": list(self.parameter_names),
                "num_observations": int(self.jacobian_tilde.shape[0]),
                "rank_threshold_relative": self.rank_threshold_relative,
                "rank_threshold_absolute": self.rank_threshold_absolute,
                "numerical_rank": self.numerical_rank,
                "is_full_rank": bool(self.numerical_rank == 4),
                "column_norms_alpha_beta_D_E": self.column_norms.tolist(),
                "identifiable_columns_mask": list(self.identifiable_columns_mask),
                "unidentifiable_parameters": list(self.unidentifiable_parameters),
                "singular_values": self.singular_values.tolist(),
                "condition_number_full_4x4": self.condition_number,
                "condition_number_is_finite": bool(math.isfinite(self.condition_number)),
                "effective_condition_number_on_detected_rank": self.effective_condition_number_on_rank,
                "collinearity_index_full_4x4": self.collinearity_index,
                "collinearity_index_is_finite": bool(math.isfinite(self.collinearity_index)),
                "active_subset_collinearity_index": self.active_subset_collinearity_index,
                "collinearity_definition": self.collinearity_definition,
                "right_singular_vectors_V_columns": {
                    f"mode_{k + 1}_sigma_{self.singular_values[k]:.3e}": {
                        "singular_value": float(self.singular_values[k]),
                        "vector_components": {
                            self.parameter_names[j]: float(self.right_singular_vectors_V[j, k])
                            for j in range(4)
                        },
                    }
                    for k in range(4)
                },
                "least_informative_mode": self.least_informative_mode,
                "local_correlation_analysis": {
                    "correlation_matrix_defined": self.correlation_matrix_defined,
                    "correlation_interpretation": self.correlation_interpretation,
                    "correlation_status_note": self.correlation_status_note,
                    "fisher_correlation_matrix": (
                        self.fisher_correlation_matrix.tolist()
                        if self.correlation_matrix_defined
                        else None
                    ),
                    "column_cosine_similarity_matrix": self.column_cosine_similarity_matrix.tolist(),
                },
                "weighting_metadata": self.weighting_metadata,
            }
        )


def analyze_jacobian_svd(
    J_tilde: np.ndarray,
    rank_threshold: float = 1.0e-8,
    weighting_spec: Optional[ObservationWeightingSpec] = None,
) -> LocalIdentifiabilityReport:
    """
    Esegue la SVD di J_w(theta_tilde) e quantifica rango numerico, condizionamento,
    indice di collinearità, modo meno informativo e correlazione locale.
    Se il rango numerico è < 4, NON utilizza la pseudoinversa di Fisher per produrre
    correlazioni apparenti sui parametri non identificabili, ma dichiara la matrice
    di correlazione non definita (`correlation_matrix_defined=False`, valori `NaN`).
    """
    J = np.asarray(J_tilde, dtype=np.float64)
    if J.ndim != 2 or J.shape[1] != 4:
        raise ValueError(f"Atteso Jacobiano di forma (N_obs, 4), ricevuto {J.shape}.")
    if not np.all(np.isfinite(J)):
        raise ValueError("Il Jacobiano passato ad analyze_jacobian_svd contiene NaN o Inf.")
    if not np.isfinite(rank_threshold) or rank_threshold <= 0.0 or rank_threshold >= 1.0:
        raise ValueError(f"rank_threshold deve essere in (0, 1), ricevuto {rank_threshold}.")

    col_norms = np.linalg.norm(J, axis=0)
    _, s_vals, Vt = np.linalg.svd(J, full_matrices=False)
    V = Vt.T

    sigma_max = float(s_vals[0]) if s_vals.size > 0 else 0.0
    abs_thresh = float(rank_threshold * sigma_max)

    if sigma_max <= 1.0e-15:
        num_rank = 0
        cond_num = float("inf")
        eff_cond = float("inf")
    else:
        num_rank = int(np.sum(s_vals > abs_thresh))
        sigma_min = float(s_vals[-1])
        cond_num = (
            float(sigma_max / sigma_min) if (num_rank == 4 and sigma_min > abs_thresh) else float("inf")
        )
        eff_cond = float(sigma_max / float(s_vals[num_rank - 1])) if num_rank >= 1 else float("inf")

    col_tol = max(abs_thresh, 1.0e-14)
    identifiable_mask = tuple(bool(col_norms[j] > col_tol) for j in range(4))
    unidentifiable_params: List[str] = [
        PARAMETER_NAMES[j] for j in range(4) if not identifiable_mask[j]
    ]
    if num_rank < 4 and not unidentifiable_params:
        # Caso di collinearità esatta tra colonne non nulle: individua i parametri nel nucleo
        v_last = np.abs(V[:, -1])
        unidentifiable_params = [
            PARAMETER_NAMES[j] for j in range(4) if v_last[j] > 0.25
        ]

    # Cosini direttori tra colonne attive (diagnostica geometrica al primo ordine)
    cos_mat = np.full((4, 4), np.nan, dtype=np.float64)
    for i in range(4):
        for j in range(4):
            if identifiable_mask[i] and identifiable_mask[j]:
                if i == j:
                    cos_mat[i, j] = 1.0
                else:
                    cos_mat[i, j] = float(
                        np.clip(
                            np.dot(J[:, i], J[:, j]) / (col_norms[i] * col_norms[j]),
                            -1.0,
                            1.0,
                        )
                    )

    # Indice di collinearità di Brun et al. (2001)
    active_indices = [j for j in range(4) if identifiable_mask[j]]
    if len(active_indices) >= 1:
        J_active_unit = J[:, active_indices] / col_norms[active_indices][None, :]
        s_active_unit = np.linalg.svd(J_active_unit, compute_uv=False)
        active_collinearity = (
            float(1.0 / s_active_unit[-1])
            if s_active_unit[-1] > rank_threshold
            else float("inf")
        )
    else:
        active_collinearity = float("inf")

    if num_rank == 4 and all(identifiable_mask):
        collinearity = active_collinearity
    else:
        collinearity = float("inf")

    # Modo singolare meno informativo (k = 4, corrispondente a sigma_4 = sigma_min)
    v4 = V[:, 3]
    v4_sq = v4 ** 2
    dom_idx = int(np.argmax(np.abs(v4)))
    least_info_dict = {
        "mode_index_1based": 4,
        "singular_value": float(s_vals[3]),
        "relative_to_sigma_max": float(s_vals[3] / sigma_max) if sigma_max > 0.0 else 0.0,
        "is_in_numerical_nullspace": bool(num_rank < 4),
        "nullspace_dimension": int(4 - num_rank),
        "right_singular_vector": {PARAMETER_NAMES[j]: float(v4[j]) for j in range(4)},
        "parameter_participation_squared": {PARAMETER_NAMES[j]: float(v4_sq[j]) for j in range(4)},
        "dominant_parameter": PARAMETER_NAMES[dom_idx],
    }

    w_spec = weighting_spec or ObservationWeightingSpec()
    w_meta = w_spec.to_metadata_dict(int(J.shape[0]))

    # Calcolo rigoroso della matrice di correlazione locale: SOLO per rango pieno (num_rank == 4)
    if num_rank < 4:
        corr_defined = False
        corr_mat = np.full((4, 4), np.nan, dtype=np.float64)
        corr_interp = "undefined_rank_deficient_singular_fisher_matrix"
        corr_note = (
            f"RANK_DEFICIENT (numerical_rank={num_rank} < 4): The sensitivity matrix F = J_w^T J_w "
            f"is singular. Unidentifiable or collinear parameters: {unidentifiable_params}. "
            "A Moore-Penrose pseudoinverse cannot be interpreted as a valid covariance or correlation "
            "matrix for non-identifiable parameters; correlations involving the nullspace are undefined."
        )
    else:
        corr_defined = True
        # Calcolo diretto F^{-1} = V * diag(sigma^{-2}) * V^T tramite SVD senza formare J^T J
        inv_sigma2 = 1.0 / (s_vals ** 2)
        F_inv = (V * inv_sigma2[None, :]) @ V.T
        diag_std = np.sqrt(np.maximum(np.diag(F_inv), 0.0))
        corr_mat = np.eye(4, dtype=np.float64)
        for i in range(4):
            for j in range(4):
                if i != j:
                    if diag_std[i] > 0.0 and diag_std[j] > 0.0:
                        corr_mat[i, j] = float(
                            np.clip(F_inv[i, j] / (diag_std[i] * diag_std[j]), -1.0, 1.0)
                        )
                    else:
                        corr_mat[i, j] = np.nan

        if w_spec.is_statistically_calibrated:
            corr_interp = "asymptotic_statistical_correlation_under_declared_noise_model"
            corr_note = (
                "Full-rank Jacobian weighted by declared observation noise inverse covariance "
                "(F = J^T W J). Represents local Gauss-Newton / Cramer-Rao parameter correlation."
            )
        else:
            corr_interp = "geometric_sensitivity_correlation_unweighted"
            corr_note = (
                "GEOMETRIC_DIAGNOSTIC_ONLY: Computed from unweighted dimensionless Jacobian "
                "F = J^T J (equivalent to uniform i.i.d. homoscedastic scaling up to a constant factor). "
                "Describes the local geometric coupling of the quadratic objective contours around "
                "theta_dagger; do NOT interpret as calibrated statistical confidence intervals."
            )

    return LocalIdentifiabilityReport(
        jacobian_tilde=J,
        column_norms=col_norms,
        singular_values=s_vals,
        right_singular_vectors_V=V,
        numerical_rank=num_rank,
        condition_number=cond_num,
        collinearity_index=collinearity,
        fisher_correlation_matrix=corr_mat,
        parameter_names=PARAMETER_NAMES,
        rank_threshold_relative=float(rank_threshold),
        rank_threshold_absolute=abs_thresh,
        effective_condition_number_on_rank=eff_cond,
        active_subset_collinearity_index=active_collinearity,
        identifiable_columns_mask=identifiable_mask,
        unidentifiable_parameters=tuple(unidentifiable_params),
        correlation_matrix_defined=corr_defined,
        correlation_interpretation=corr_interp,
        correlation_status_note=corr_note,
        column_cosine_similarity_matrix=cos_mat,
        least_informative_mode=least_info_dict,
        weighting_metadata=w_meta,
    )