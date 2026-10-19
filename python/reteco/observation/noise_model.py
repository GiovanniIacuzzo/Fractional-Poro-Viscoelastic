"""
python/reteco/observation/noise_model.py
Generazione riproducibile del rumore osservativo gaussiano e costruzione esplicita
dei pesi osservativi W per le due modalità del funzionale obiettivo (Fase 2):

1. Modalità A — Obiettivo deterministico normalizzato (`mode_a_deterministic_normalized`):
   I residui sono normalizzati tramite una scala positiva di riferimento s_ref > 0
   calcolata sui dati di riferimento e indipendente sia dal candidato theta sia da eta:
   - con `reference_scale_type = "linf"` (default):
       s_ref = max(||y_ref||_infty, 1.0_se_nullo),
       W_ii^(A) = 1 / (sqrt(N_obs) * s_ref),
       J_data^(A)(theta) = (1 / (2 * N_obs * s_ref^2)) * ||G_h(theta) - y_obs||_2^2;
   - con `reference_scale_type = "rms"`:
       s_ref = ||y_ref||_2 / sqrt(N_obs),
       W_ii^(A) = 1 / ||y_ref||_2 = 1 / (sqrt(N_obs) * s_ref),
       J_data^(A)(theta) = (1 / (2 * ||y_ref||_2^2)) * ||G_h(theta) - y_obs||_2^2.
   Questa modalità applica la medesima regola a tutti i livelli di rumore eta >= 0
   e non introduce mai divisioni per zero per eta = 0.

2. Modalità B — Obiettivo pesato secondo il modello di rumore (`mode_b_noise_weighted`):
   Per eta > 0, il rumore additivo è campionato come epsilon ~ N(0, sigma_nom^2 I_{N_obs}),
   con deviazione standard nominale:
       sigma_nom = eta * ||y_clean||_infty.
   - con `covariance_scaling = "mean_normalized"` (default compatibile con il codice storico):
       W_ii^(B,mean) = 1 / (sqrt(N_obs) * sigma_nom)  =>  W^T W = Sigma^{-1} / N_obs;
   - con `covariance_scaling = "exact_inverse_covariance"` (forma letterale W^T W = Sigma^{-1}):
       W_ii^(B,cov)  = 1 / sigma_nom                  =>  W^T W = Sigma^{-1}.
   Per eta = 0, la covarianza Sigma = 0 non è invertibile: la funzione gestisce il caso
   separatamente tramite `eta_zero_policy = "fallback_mode_a"` (default retrocompatibile,
   con flag `is_covariance_invertible = False`) oppure `"raise"`.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np

ObjectiveWeightingMode = Literal[
    "mode_a_deterministic_normalized",
    "mode_b_noise_weighted",
]
ReferenceScaleType = Literal["linf", "rms"]
CovarianceScalingType = Literal["mean_normalized", "exact_inverse_covariance"]
EtaZeroPolicy = Literal["fallback_mode_a", "raise"]


@dataclass(frozen=True)
class NoisyObservationSample:
    y_clean: np.ndarray
    y_noisy: np.ndarray
    noise_vector: np.ndarray
    sigma_std: float
    eta_level: float
    seed: int
    weight_matrix_diag: np.ndarray
    # Campi diagnostici espliciti per la Fase 2 (con default per piena retrocompatibilità)
    signal_reference_scale: float = 1.0
    effective_noise_l2_norm: float = 0.0
    effective_relative_noise_norm: float = 0.0
    empirical_noise_std: float = 0.0
    weighting_mode: str = "mode_b_noise_weighted"
    reference_scale_type: str = "linf"
    covariance_scaling: str = "mean_normalized"
    is_covariance_invertible: bool = False


def compute_positive_reference_scale(
    y_reference: np.ndarray,
    scale_type: ReferenceScaleType = "linf",
) -> float:
    """
    Calcola una scala di riferimento strettamente positiva s_ref > 0 a partire dal
    vettore di riferimento `y_reference`, evitando qualsiasi divisione per zero.
    """
    arr = np.asarray(y_reference, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("Il vettore y_reference non può essere vuoto.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Il vettore y_reference contiene valori non finiti.")

    if scale_type == "linf":
        scale = float(np.max(np.abs(arr)))
    elif scale_type == "rms":
        scale = float(np.linalg.norm(arr) / np.sqrt(arr.size))
    else:
        raise ValueError(
            f"scale_type non valido: '{scale_type}'. Valori ammessi: 'linf', 'rms'."
        )

    if scale <= 0.0:
        scale = 1.0
    return scale


def build_mode_a_deterministic_weights(
    y_reference: np.ndarray,
    scale_type: ReferenceScaleType = "linf",
) -> tuple[np.ndarray, float]:
    """
    Costruisce i pesi diagonali della Modalità A (obiettivo deterministico normalizzato):
        W_ii^(A) = 1 / (sqrt(N_obs) * s_ref),
    dove s_ref > 0 dipende solo da `y_reference` e da `scale_type`, ed è indipendente
    sia dal candidato theta sia dal livello di rumore eta.
    Restituisce (weight_matrix_diag, s_ref).
    """
    arr = np.asarray(y_reference, dtype=np.float64).ravel()
    n_obs = arr.size
    s_ref = compute_positive_reference_scale(arr, scale_type=scale_type)
    w_diag = np.full(n_obs, 1.0 / (np.sqrt(n_obs) * s_ref), dtype=np.float64)
    return w_diag, s_ref


def build_mode_b_noise_weights(
    y_clean: np.ndarray,
    eta_level: float,
    covariance_scaling: CovarianceScalingType = "mean_normalized",
    eta_zero_policy: EtaZeroPolicy = "fallback_mode_a",
    reference_scale_type: ReferenceScaleType = "linf",
) -> tuple[np.ndarray, float, float, bool]:
    """
    Costruisce i pesi diagonali della Modalità B (obiettivo pesato secondo il modello di rumore).
    Restituisce (weight_matrix_diag, sigma_nom, s_ref, is_covariance_invertible).
    """
    clean = np.asarray(y_clean, dtype=np.float64).ravel()
    if eta_level < 0.0 or not np.isfinite(eta_level):
        raise ValueError(
            f"Il livello di rumore eta_level deve essere finito e >= 0, ricevuto {eta_level}."
        )
    if covariance_scaling not in ("mean_normalized", "exact_inverse_covariance"):
        raise ValueError(
            f"covariance_scaling non valido: '{covariance_scaling}'. "
            "Valori ammessi: 'mean_normalized', 'exact_inverse_covariance'."
        )

    n_obs = clean.size
    s_ref = compute_positive_reference_scale(clean, scale_type=reference_scale_type)

    if eta_level == 0.0:
        if eta_zero_policy == "raise":
            raise ValueError(
                "Modalità B con eta_level == 0.0: la matrice di covarianza del rumore "
                "Sigma = 0 non è invertibile. Usa la Modalità A oppure imposta "
                "eta_zero_policy='fallback_mode_a'."
            )
        w_diag, _ = build_mode_a_deterministic_weights(
            clean, scale_type=reference_scale_type
        )
        return w_diag, 0.0, s_ref, False

    sigma_nom = float(eta_level * s_ref)
    if covariance_scaling == "mean_normalized":
        w_val = 1.0 / (np.sqrt(n_obs) * sigma_nom)
    else:
        w_val = 1.0 / sigma_nom

    w_diag = np.full(n_obs, w_val, dtype=np.float64)
    return w_diag, sigma_nom, s_ref, True


def generate_gaussian_noisy_observations(
    y_clean: np.ndarray,
    eta_level: float,
    seed: int,
    *,
    weighting_mode: ObjectiveWeightingMode = "mode_b_noise_weighted",
    reference_scale_type: ReferenceScaleType = "linf",
    covariance_scaling: CovarianceScalingType = "mean_normalized",
    eta_zero_policy: EtaZeroPolicy = "fallback_mode_a",
) -> NoisyObservationSample:
    """
    Genera osservazioni sintetiche y_noisy = y_clean + epsilon e costruisce i pesi
    osservativi secondo la modalità selezionata (`weighting_mode`).

    Con i parametri di default (`weighting_mode="mode_b_noise_weighted"`,
    `reference_scale_type="linf"`, `covariance_scaling="mean_normalized"`,
    `eta_zero_policy="fallback_mode_a"`), il comportamento numerico è bit-a-bit
    identico all'implementazione preesistente.
    """
    clean = np.asarray(y_clean, dtype=np.float64).ravel()
    if eta_level < 0.0 or not np.isfinite(eta_level):
        raise ValueError(
            f"Il livello di rumore eta_level deve essere finito e >= 0, ricevuto {eta_level}."
        )

    n_obs = clean.size
    s_ref = compute_positive_reference_scale(clean, scale_type=reference_scale_type)

    if eta_level == 0.0:
        sigma_std = 0.0
        noise = np.zeros_like(clean)
    else:
        rng = np.random.default_rng(int(seed))
        sigma_std = float(eta_level * s_ref)
        noise = rng.normal(loc=0.0, scale=sigma_std, size=n_obs)

    if weighting_mode == "mode_a_deterministic_normalized":
        w_diag, _ = build_mode_a_deterministic_weights(
            clean, scale_type=reference_scale_type
        )
        is_cov_inv = False
    elif weighting_mode == "mode_b_noise_weighted":
        w_diag, _, _, is_cov_inv = build_mode_b_noise_weights(
            clean,
            eta_level=eta_level,
            covariance_scaling=covariance_scaling,
            eta_zero_policy=eta_zero_policy,
            reference_scale_type=reference_scale_type,
        )
    else:
        raise ValueError(
            f"weighting_mode non riconosciuto: '{weighting_mode}'. "
            "Valori ammessi: 'mode_a_deterministic_normalized', 'mode_b_noise_weighted'."
        )

    clean_l2 = float(np.linalg.norm(clean))
    if clean_l2 <= 0.0:
        clean_l2 = 1.0
    noise_l2 = float(np.linalg.norm(noise))
    eff_rel_noise = noise_l2 / clean_l2
    emp_std = float(np.sqrt(np.mean(noise ** 2))) if n_obs > 0 else 0.0

    return NoisyObservationSample(
        y_clean=clean.copy(),
        y_noisy=clean + noise,
        noise_vector=noise,
        sigma_std=sigma_std,
        eta_level=float(eta_level),
        seed=int(seed),
        weight_matrix_diag=w_diag,
        signal_reference_scale=s_ref,
        effective_noise_l2_norm=noise_l2,
        effective_relative_noise_norm=eff_rel_noise,
        empirical_noise_std=emp_std,
        weighting_mode=str(weighting_mode),
        reference_scale_type=str(reference_scale_type),
        covariance_scaling=str(covariance_scaling),
        is_covariance_invertible=bool(is_cov_inv),
    )