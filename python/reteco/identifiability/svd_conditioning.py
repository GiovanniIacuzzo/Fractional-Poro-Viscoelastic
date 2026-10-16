"""
python/reteco/identifiability/svd_conditioning.py
Analisi spettrale SVD del Jacobiano di sensibilità, calcolo del rango numerico,
indice di collinearità e matrice di correlazione locale di Fisher (Contratto Sez. 8).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LocalIdentifiabilityReport:
    jacobian_tilde: np.ndarray         # (N_obs, 4)
    column_norms: np.ndarray           # (4,) norme L2 delle colonne [alpha~, beta~, D~, E~]
    singular_values: np.ndarray        # (4,) sigma_1 >= sigma_2 >= sigma_3 >= sigma_4
    right_singular_vectors_V: np.ndarray  # (4, 4) colonne = modi destri v_k
    numerical_rank: int
    condition_number: float
    collinearity_index: float          # 1 / sigma_min(J_col_normalized)
    fisher_correlation_matrix: np.ndarray  # (4, 4)


def analyze_jacobian_svd(
    J_tilde: np.ndarray,
    rank_threshold: float = 1.0e-8,
) -> LocalIdentifiabilityReport:
    """
    Esegue la SVD di J^G(theta_tilde) e quantifica il rango effettivo e il condizionamento.
    Gestisce in modo sicuro il caso di colonne nulle (es. Proposizione flusso nullo q=0).
    """
    J = np.asarray(J_tilde, dtype=np.float64)
    if J.ndim != 2 or J.shape[1] != 4:
        raise ValueError(f"Atteso Jacobiano di forma (N_obs, 4), ricevuto {J.shape}.")

    col_norms = np.linalg.norm(J, axis=0)
    _, s_vals, Vt = np.linalg.svd(J, full_matrices=False)
    V = Vt.T

    sigma_max = float(s_vals[0]) if s_vals.size > 0 else 0.0
    if sigma_max <= 0.0:
        num_rank = 0
        cond_num = float("inf")
    else:
        num_rank = int(np.sum(s_vals > rank_threshold * sigma_max))
        sigma_min = float(s_vals[-1])
        cond_num = float(sigma_max / sigma_min) if sigma_min > 1.0e-15 * sigma_max else float("inf")

    # Indice di collinearità di Brun et al. (sulle colonne normalizzate a norma unitaria)
    if np.all(col_norms > 1.0e-14):
        J_unit = J / col_norms[None, :]
        s_unit = np.linalg.svd(J_unit, compute_uv=False)
        collinearity = float(1.0 / s_unit[-1]) if s_unit[-1] > 1.0e-14 else float("inf")
    else:
        collinearity = float("inf")

    # Matrice di correlazione locale asintotica dalla pseudoinversa di Fisher F = J^T J
    F = J.T @ J
    F_pinv = np.linalg.pinv(F, rcond=rank_threshold)
    diag_std = np.sqrt(np.maximum(np.diag(F_pinv), 0.0))
    corr = np.eye(4, dtype=np.float64)
    for i in range(4):
        for j in range(4):
            if diag_std[i] > 1.0e-14 and diag_std[j] > 1.0e-14:
                corr[i, j] = np.clip(F_pinv[i, j] / (diag_std[i] * diag_std[j]), -1.0, 1.0)
            elif i != j:
                corr[i, j] = 0.0

    return LocalIdentifiabilityReport(
        jacobian_tilde=J,
        column_norms=col_norms,
        singular_values=s_vals,
        right_singular_vectors_V=V,
        numerical_rank=num_rank,
        condition_number=cond_num,
        collinearity_index=collinearity,
        fisher_correlation_matrix=corr,
    )