"""
python/reteco/observation/noise_model.py
Generazione riproducibile del rumore osservativo gaussiano e costruzione esplicita
dei pesi osservativi W per le due modalità del funzionale obiettivo (Fase 2.1):

1. Modalità A — Obiettivo deterministico normalizzato (`mode_a_deterministic_normalized`):
   I residui r(theta) = G_h(theta) - y_obs sono normalizzati mediante una scala di
   riferimento strettamente positiva s_ref > 0 calcolata su un vettore di riferimento
   esplicito `y_reference` (indipendente sia da theta sia dal livello di rumore eta):
       W_ii^(A) = 1 / (sqrt(N_obs) * s_ref),
       J_data^(A)(theta) = (1 / (2 * N_obs * s_ref^2)) * ||r(theta)||_2^2.
   Se `y_reference` è identicamente nullo, viene sollevato ValueError a meno che non
   sia specificata esplicitamente `zero_reference_policy="explicit_fallback"` con
   `zero_reference_fallback_scale > 0`.

2. Modalità B — Obiettivo pesato secondo il modello di rumore (`mode_b_noise_weighted`):
   Per eta > 0, il rumore additivo è campionato come epsilon ~ N(0, Sigma), con
   Sigma = sigma_nom^2 * I_{N_obs} e deviazione standard nominale:
       sigma_nom = eta * s_ref(y_clean).
   Sono supportate due convenzioni esplicite di scala:
   - `covariance_scaling = "exact_inverse_covariance"` (formulazione canonica del
     Contratto Matematico Sez. 8, W^T W = Sigma^{-1}):
       W_ii^(B,cov) = 1 / sigma_nom  =>  (W^(B,cov))^T W^(B,cov) = Sigma^{-1},
       J_data^(B,cov)(theta) = 0.5 * r(theta)^T Sigma^{-1} r(theta)
                             = (N_obs / eta^2) * J_data^(A)(theta);
   - `covariance_scaling = "mean_normalized"` (formulazione storica normalizzata per N_obs):
       W_ii^(B,mean) = 1 / (sqrt(N_obs) * sigma_nom)  =>  W^T W = Sigma^{-1} / N_obs,
       J_data^(B,mean)(theta) = (1 / eta^2) * J_data^(A)(theta).

   Per eta = 0, la covarianza Sigma = 0 non è invertibile (`is_covariance_invertible = False`,
   `inverse_covariance_diag = None`). Il problema deterministico è gestito separatamente
   senza costruire una falsa covarianza invertibile.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

ObjectiveWeightingMode = Literal[
    "mode_a_deterministic_normalized",
    "mode_b_noise_weighted",
]
ReferenceScaleType = Literal["linf", "rms"]
CovarianceScalingType = Literal["mean_normalized", "exact_inverse_covariance"]
EtaZeroPolicy = Literal["fallback_mode_a", "raise"]
ZeroReferencePolicy = Literal["raise", "explicit_fallback"]


@dataclass(frozen=True)
class NoisyObservationSample:
    y_clean: np.ndarray
    y_noisy: np.ndarray
    noise_vector: np.ndarray
    sigma_std: float
    eta_level: float
    seed: int
    weight_matrix_diag: np.ndarray
    signal_reference_scale: float = 1.0
    effective_noise_l2_norm: float = 0.0
    effective_relative_noise_norm: float = 0.0
    empirical_noise_std: float = 0.0
    weighting_mode: str = "mode_b_noise_weighted"
    reference_scale_type: str = "linf"
    covariance_scaling: str = "mean_normalized"
    is_covariance_invertible: bool = False
    inverse_covariance_diag: Optional[np.ndarray] = None


def compute_positive_reference_scale(
    y_reference: np.ndarray,
    scale_type: ReferenceScaleType = "linf",
    *,
    zero_reference_policy: ZeroReferencePolicy = "raise",
    zero_reference_fallback_scale: Optional[float] = None,
) -> float:
    """
    Calcola una scala di riferimento strettamente positiva s_ref > 0 a partire da
    `y_reference`.

    Se `y_reference` è identicamente nullo:
    - con `zero_reference_policy == "raise"` (default): solleva ValueError per impedire
      normalizzazioni silenziosamente arbitrarie;
    - con `zero_reference_policy == "explicit_fallback"`: richiede un valore finito e
      strettamente positivo `zero_reference_fallback_scale > 0`.
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
        if zero_reference_policy == "raise":
            raise ValueError(
                "Il vettore di riferimento ha norma nulla (scale == 0.0): impossibile "
                "definire una scala di normalizzazione senza introdurre una costante "
                "arbitraria. Se si desidera un fallback esplicito, impostare "
                "zero_reference_policy='explicit_fallback' e specificare "
                "zero_reference_fallback_scale > 0."
            )
        if zero_reference_policy == "explicit_fallback":
            if (
                zero_reference_fallback_scale is None
                or isinstance(zero_reference_fallback_scale, bool)
                or not np.isfinite(zero_reference_fallback_scale)
                or float(zero_reference_fallback_scale) <= 0.0
            ):
                raise ValueError(
                    "Con zero_reference_policy='explicit_fallback' occorre fornire "
                    "zero_reference_fallback_scale finito e > 0."
                )
            return float(zero_reference_fallback_scale)
        raise ValueError(f"zero_reference_policy non valido: '{zero_reference_policy}'.")

    return scale


def build_mode_a_deterministic_weights(
    y_reference: np.ndarray,
    scale_type: ReferenceScaleType = "linf",
    *,
    zero_reference_policy: ZeroReferencePolicy = "raise",
    zero_reference_fallback_scale: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """
    Costruisce i pesi diagonali della Modalità A (obiettivo deterministico normalizzato):
        W_ii^(A) = 1 / (sqrt(N_obs) * s_ref),
    dove s_ref > 0 dipende esclusivamente da `y_reference` e da `scale_type`, ed è
    indipendente sia dal candidato theta sia dal livello di rumore eta.
    Restituisce (weight_matrix_diag, s_ref).
    """
    arr = np.asarray(y_reference, dtype=np.float64).ravel()
    n_obs = arr.size
    s_ref = compute_positive_reference_scale(
        arr,
        scale_type=scale_type,
        zero_reference_policy=zero_reference_policy,
        zero_reference_fallback_scale=zero_reference_fallback_scale,
    )
    w_diag = np.full(n_obs, 1.0 / (np.sqrt(n_obs) * s_ref), dtype=np.float64)
    return w_diag, s_ref


def build_mode_b_noise_weights(
    y_clean: np.ndarray,
    eta_level: float,
    covariance_scaling: CovarianceScalingType = "mean_normalized",
    eta_zero_policy: EtaZeroPolicy = "fallback_mode_a",
    reference_scale_type: ReferenceScaleType = "linf",
    *,
    zero_reference_policy: ZeroReferencePolicy = "raise",
    zero_reference_fallback_scale: Optional[float] = None,
) -> tuple[np.ndarray, float, float, bool, Optional[np.ndarray]]:
    """
    Costruisce i pesi diagonali della Modalità B (obiettivo pesato secondo il modello di rumore).
    Restituisce:
        (weight_matrix_diag, sigma_nom, s_ref, is_covariance_invertible, inverse_covariance_diag).
    Per eta_level > 0:
        Sigma = sigma_nom^2 * I_{N_obs}, inverse_covariance_diag = (1 / sigma_nom^2) * 1_{N_obs}.
        - Se covariance_scaling == "exact_inverse_covariance": W_ii = 1 / sigma_nom e W^T W = Sigma^{-1}.
        - Se covariance_scaling == "mean_normalized": W_ii = 1 / (sqrt(N_obs) * sigma_nom) e W^T W = Sigma^{-1} / N_obs.
    Per eta_level == 0:
        Sigma = 0 non è invertibile (`is_covariance_invertible = False`, `inverse_covariance_diag = None`).
    """
    clean = np.asarray(y_clean, dtype=np.float64).ravel()
    if isinstance(eta_level, bool) or not np.isfinite(eta_level) or eta_level < 0.0:
        raise ValueError(
            f"Il livello di rumore eta_level deve essere finito e >= 0, ricevuto {eta_level}."
        )
    if covariance_scaling not in ("mean_normalized", "exact_inverse_covariance"):
        raise ValueError(
            f"covariance_scaling non valido: '{covariance_scaling}'. "
            "Valori ammessi: 'mean_normalized', 'exact_inverse_covariance'."
        )

    n_obs = clean.size
    s_ref = compute_positive_reference_scale(
        clean,
        scale_type=reference_scale_type,
        zero_reference_policy=zero_reference_policy,
        zero_reference_fallback_scale=zero_reference_fallback_scale,
    )

    if eta_level == 0.0:
        if eta_zero_policy == "raise":
            raise ValueError(
                "Modalità B con eta_level == 0.0: la matrice di covarianza del rumore "
                "Sigma = 0 non è invertibile. Usa la Modalità A per il problema "
                "deterministico oppure imposta eta_zero_policy='fallback_mode_a'."
            )
        if eta_zero_policy != "fallback_mode_a":
            raise ValueError(f"eta_zero_policy non valido: '{eta_zero_policy}'.")
        w_diag, _ = build_mode_a_deterministic_weights(
            clean,
            scale_type=reference_scale_type,
            zero_reference_policy=zero_reference_policy,
            zero_reference_fallback_scale=zero_reference_fallback_scale,
        )
        return w_diag, 0.0, s_ref, False, None

    sigma_nom = float(eta_level * s_ref)
    inv_cov_diag = np.full(n_obs, 1.0 / (sigma_nom ** 2), dtype=np.float64)

    if covariance_scaling == "exact_inverse_covariance":
        w_val = 1.0 / sigma_nom
    else:
        w_val = 1.0 / (np.sqrt(n_obs) * sigma_nom)

    w_diag = np.full(n_obs, w_val, dtype=np.float64)
    return w_diag, sigma_nom, s_ref, True, inv_cov_diag


def generate_gaussian_noisy_observations(
    y_clean: np.ndarray,
    eta_level: float,
    seed: int,
    *,
    weighting_mode: ObjectiveWeightingMode = "mode_b_noise_weighted",
    reference_scale_type: ReferenceScaleType = "linf",
    covariance_scaling: CovarianceScalingType = "mean_normalized",
    eta_zero_policy: EtaZeroPolicy = "fallback_mode_a",
    zero_reference_policy: ZeroReferencePolicy = "raise",
    zero_reference_fallback_scale: Optional[float] = None,
) -> NoisyObservationSample:
    """
    Genera osservazioni sintetiche y_noisy = y_clean + epsilon e costruisce i pesi
    osservativi secondo `weighting_mode`.

    Proprietà garantita:
    La realizzazione del vettore di rumore `noise_vector` e dei dati osservati `y_noisy`
    dipende esclusivamente da `(y_clean, eta_level, seed, reference_scale_type)` ed è
    bit-a-bit identica qualunque sia la modalità di pesatura (`weighting_mode` o
    `covariance_scaling`) selezionata.
    """
    clean = np.asarray(y_clean, dtype=np.float64).ravel()
    if isinstance(eta_level, bool) or not np.isfinite(eta_level) or eta_level < 0.0:
        raise ValueError(
            f"Il livello di rumore eta_level deve essere finito e >= 0, ricevuto {eta_level}."
        )

    n_obs = clean.size
    s_ref = compute_positive_reference_scale(
        clean,
        scale_type=reference_scale_type,
        zero_reference_policy=zero_reference_policy,
        zero_reference_fallback_scale=zero_reference_fallback_scale,
    )

    if eta_level == 0.0:
        sigma_std = 0.0
        noise = np.zeros_like(clean)
    else:
        rng = np.random.default_rng(int(seed))
        sigma_std = float(eta_level * s_ref)
        noise = rng.normal(loc=0.0, scale=sigma_std, size=n_obs)

    if weighting_mode == "mode_a_deterministic_normalized":
        w_diag, _ = build_mode_a_deterministic_weights(
            clean,
            scale_type=reference_scale_type,
            zero_reference_policy=zero_reference_policy,
            zero_reference_fallback_scale=zero_reference_fallback_scale,
        )
        is_cov_inv = False
        inv_cov_diag = None
    elif weighting_mode == "mode_b_noise_weighted":
        w_diag, _, _, is_cov_inv, inv_cov_diag = build_mode_b_noise_weights(
            clean,
            eta_level=eta_level,
            covariance_scaling=covariance_scaling,
            eta_zero_policy=eta_zero_policy,
            reference_scale_type=reference_scale_type,
            zero_reference_policy=zero_reference_policy,
            zero_reference_fallback_scale=zero_reference_fallback_scale,
        )
    else:
        raise ValueError(
            f"weighting_mode non riconosciuto: '{weighting_mode}'. "
            "Valori ammessi: 'mode_a_deterministic_normalized', 'mode_b_noise_weighted'."
        )

    clean_l2 = float(np.linalg.norm(clean))
    noise_l2 = float(np.linalg.norm(noise))
    eff_rel_noise = noise_l2 / clean_l2 if clean_l2 > 0.0 else 0.0
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
        inverse_covariance_diag=inv_cov_diag,
    )