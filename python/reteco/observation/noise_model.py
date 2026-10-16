"""
python/reteco/observation/noise_model.py
Generazione riproducibile del rumore osservativo gaussiano e costruzione della matrice di peso W
tale che W^T W = Sigma^{-1} (Contratto Matematico Sez. 8).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NoisyObservationSample:
    y_clean: np.ndarray
    y_noisy: np.ndarray
    noise_vector: np.ndarray
    sigma_std: float
    eta_level: float
    seed: int
    weight_matrix_diag: np.ndarray  # Diagonale di W tale che W^T W = Sigma^{-1}


def generate_gaussian_noisy_observations(
    y_clean: np.ndarray,
    eta_level: float,
    seed: int,
) -> NoisyObservationSample:
    """
    Genera y = y_clean + epsilon, con epsilon ~ N(0, sigma^2 I) e
    sigma = eta_level * ||y_clean||_{infty} (se eta_level > 0).
    Costruisce i pesi W coerenti con la covarianza Sigma = sigma^2 I.
    """
    clean = np.asarray(y_clean, dtype=np.float64)
    if eta_level < 0.0:
        raise ValueError(f"Il livello di rumore eta_level deve essere >= 0, ricevuto {eta_level}.")

    n_obs = clean.size
    signal_scale = float(np.max(np.abs(clean)))
    if signal_scale <= 0.0:
        signal_scale = 1.0

    if eta_level == 0.0:
        sigma_std = 0.0
        noise = np.zeros_like(clean)
        # Nel caso noise-free usiamo una normalizzazione euclidea uniforme 1 / (sqrt(N_obs) * signal_scale)
        w_diag = np.full(n_obs, 1.0 / ( np.sqrt(n_obs) * signal_scale ), dtype=np.float64)
    else:
        rng = np.random.default_rng(int(seed))
        sigma_std = float(eta_level * signal_scale)
        noise = rng.normal(loc=0.0, scale=sigma_std, size=n_obs)
        # Contratto Sez. 8: W^T W = Sigma^{-1} => W = (1 / sigma_std) * I (scalato per 1/sqrt(N_obs))
        w_diag = np.full(n_obs, 1.0 / (np.sqrt(n_obs) * sigma_std), dtype=np.float64)

    return NoisyObservationSample(
        y_clean=clean.copy(),
        y_noisy=clean + noise,
        noise_vector=noise,
        sigma_std=sigma_std,
        eta_level=float(eta_level),
        seed=int(seed),
        weight_matrix_diag=w_diag,
    )