"""
python/reteco/observation/operator.py
Operatore osservativo esplicito H_h(X_theta) che campiona i campi selezionati
su una griglia fissa di sensori spaziali e istanti temporali (Contratto Sez. 8).
"""

from __future__ import annotations
from typing import List, Sequence
import numpy as np
from scipy.interpolate import griddata

from ..forward.wrapper import RetecoForwardProblem2D, ForwardTrajectoryResult


class DiscreteObservationOperator:
    """
    Definisce la mappa osservativa G_h(theta) = H_h(X_h(theta)).
    Se observed_fields == ["c"], seleziona esclusivamente la concentrazione del tracciante
    senza assumere che spostamento o pressione siano misurati (Contratto Sez. 8).
    """

    def __init__(
        self,
        sensor_coords: np.ndarray,
        observation_times: Sequence[float],
        observed_fields: Sequence[str] = ("c",),
    ):
        pts = np.asarray(sensor_coords, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("sensor_coords deve avere forma (N_sensors, 2).")
        self.sensor_coords = pts
        self.observation_times = np.asarray(observation_times, dtype=np.float64)
        self.observed_fields = list(observed_fields)

        valid_fields = {"c", "p"}
        for f in self.observed_fields:
            if f not in valid_fields:
                raise ValueError(f"Campo osservato '{f}' non supportato in {valid_fields}.")

    @classmethod
    def from_regular_grid(
        cls,
        nx: int = 6,
        ny: int = 6,
        margin: float = 0.15,
        observation_times: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
        observed_fields: Sequence[str] = ("c",),
    ) -> "DiscreteObservationOperator":
        xs = np.linspace(margin, 1.0 - margin, nx)
        ys = np.linspace(margin, 1.0 - margin, ny)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        return cls(coords, observation_times, observed_fields)

    @property
    def num_observations(self) -> int:
        return len(self.observed_fields) * len(self.observation_times) * self.sensor_coords.shape[0]

    def apply(
        self,
        problem: RetecoForwardProblem2D,
        trajectory: ForwardTrajectoryResult,
    ) -> np.ndarray:
        """
        Valuta H_h(X_h) interpolando spazialmente i campi richiesti sui sensori
        e linearmente nel tempo sugli istanti di osservazione.
        """
        t_grid = trajectory.times
        obs_blocks: List[np.ndarray] = []

        for field_name in self.observed_fields:
            if field_name == "c":
                support_pts = problem.tracer_pts
                field_hist = trajectory.c_hist
            elif field_name == "p":
                support_pts = problem.press_pts
                field_hist = trajectory.p_hist
            else:
                raise ValueError(f"Campo non gestito: {field_name}")

            for t_obs in self.observation_times:
                if t_obs < t_grid[0] - 1e-12 or t_obs > t_grid[-1] + 1e-12:
                    raise ValueError(
                        f"Istante osservato t={t_obs} fuori dall'intervallo simulato [{t_grid[0]}, {t_grid[-1]}]."
                    )
                idx_right = int(np.searchsorted(t_grid, t_obs))
                if idx_right == 0:
                    snap = field_hist[0]
                elif abs(t_grid[idx_right] - t_obs) < 1e-12:
                    snap = field_hist[idx_right]
                else:
                    idx_left = idx_right - 1
                    t0, t1 = t_grid[idx_left], t_grid[idx_right]
                    w = (t_obs - t0) / (t1 - t0)
                    snap = (1.0 - w) * field_hist[idx_left] + w * field_hist[idx_right]

                vals = griddata(support_pts, snap, self.sensor_coords, method="linear")
                if np.any(np.isnan(vals)):
                    vals_near = griddata(support_pts, snap, self.sensor_coords, method="nearest")
                    vals = np.where(np.isnan(vals), vals_near, vals)
                obs_blocks.append(vals)

        return np.concatenate(obs_blocks).astype(np.float64)