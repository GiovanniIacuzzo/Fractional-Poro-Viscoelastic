"""
python/reteco/identifiability/profile_likelihood.py
Calcolo di sezioni 1D e superfici 2D del funzionale di costo J_h(theta)
per l'analisi di identificabilità pratica (Contratto Sez. 8 e 9).
"""

from __future__ import annotations
from typing import Callable, Dict
import numpy as np


def compute_1d_cost_profiles(
    cost_func: Callable[[np.ndarray], float],
    theta_center: np.ndarray,
    bounds_tuple: list[tuple[float, float]],
    num_points: int = 21,
) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Valuta i profili 1D di J_h(theta) variando un parametro alla volta attorno a theta_center.
    """
    names = ["alpha", "beta", "D", "E"]
    results: Dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for idx, name in enumerate(names):
        low, high = bounds_tuple[idx]
        grid_vals = np.linspace(low, high, num_points, dtype=np.float64)
        costs = np.zeros(num_points, dtype=np.float64)
        for k, val in enumerate(grid_vals):
            theta_probe = theta_center.copy()
            theta_probe[idx] = val
            costs[k] = float(cost_func(theta_probe))
        results[name] = (grid_vals, costs)

    return results


def compute_2d_cost_surface(
    cost_func: Callable[[np.ndarray], float],
    theta_center: np.ndarray,
    idx_x: int,
    idx_y: int,
    range_x: tuple[float, float],
    range_y: tuple[float, float],
    num_x: int = 15,
    num_y: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Valuta la superficie 2D J_h(theta) nel piano (theta[idx_x], theta[idx_y])."""
    xs = np.linspace(range_x[0], range_x[1], num_x, dtype=np.float64)
    ys = np.linspace(range_y[0], range_y[1], num_y, dtype=np.float64)
    Z = np.zeros((num_y, num_x), dtype=np.float64)

    for j, y_val in enumerate(ys):
        for i, x_val in enumerate(xs):
            probe = theta_center.copy()
            probe[idx_x] = x_val
            probe[idx_y] = y_val
            Z[j, i] = float(cost_func(probe))

    return xs, ys, Z