"""Exact isoparametric Q1 evaluation of tracer observations on quadrilateral FEM meshes.

This is the same operator previously defined in experiment 13.  It is now a
reusable library component, and no default is changed in legacy experiments.
"""
from __future__ import annotations
from typing import Any, Dict, Sequence, Tuple
import numpy as np
from matplotlib.path import Path as MplPath
from ..forward.wrapper import RetecoForwardProblem2D, ForwardTrajectoryResult

class ExactQ1TracerObservationOperator:
    """
    Valuta il campo finito-elementare c_h in Q1 esattamente tramite la mappa
    isoparametrica bilineare su ciascuna cella quadrilatera della mesh deal.II,
    con interpolazione temporale lineare sugli istanti di osservazione.
    """

    def __init__(
        self,
        problem: RetecoForwardProblem2D,
        sensor_points: np.ndarray,
        observation_times: Sequence[float],
        newton_tol: float = 1.0e-13,
        max_newton_iter: int = 25,
    ):
        self.sensor_points = np.asarray(sensor_points, dtype=np.float64)
        if self.sensor_points.ndim != 2 or self.sensor_points.shape[1] != 2 or not np.all(np.isfinite(self.sensor_points)):
            raise ValueError("sensor_points must be a finite array of shape (N,2).")
        self.observation_times = np.asarray(observation_times, dtype=np.float64)
        if self.observation_times.ndim != 1 or self.observation_times.size == 0 or not np.all(np.isfinite(self.observation_times)):
            raise ValueError("observation_times must be a nonempty, finite 1D array.")
        self.num_sensors = int(self.sensor_points.shape[0])
        self.num_times = int(self.observation_times.size)

        self.sensor_cell_indices = np.full(self.num_sensors, -1, dtype=np.int64)
        self.sensor_dofs = np.zeros((self.num_sensors, 4), dtype=np.int64)
        self.sensor_weights = np.zeros((self.num_sensors, 4), dtype=np.float64)

        tracer_pts = problem.tracer_pts
        cell_dofs_all = problem.cell_tracer_dof_connectivity
        if cell_dofs_all.ndim != 2 or cell_dofs_all.shape[1] != 4:
            raise ValueError('ExactQ1TracerObservationOperator requires 4-node Q1 tracer cells.')
        n_cells = cell_dofs_all.shape[0]

        ordered_cell_dofs = np.zeros_like(cell_dofs_all)
        ordered_cell_coords = np.zeros((n_cells, 4, 2), dtype=np.float64)
        cell_bboxes = np.zeros((n_cells, 4), dtype=np.float64)

        for q in range(n_cells):
            dofs = cell_dofs_all[q]
            pts = tracer_pts[dofs]
            center = np.mean(pts, axis=0)
            angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
            ccw_order = np.argsort(angles)
            dofs_ccw = dofs[ccw_order]
            pts_ccw = pts[ccw_order]
            ordered_cell_dofs[q] = dofs_ccw
            ordered_cell_coords[q] = pts_ccw
            cell_bboxes[q, 0] = np.min(pts_ccw[:, 0]) - 1.0e-7
            cell_bboxes[q, 1] = np.max(pts_ccw[:, 0]) + 1.0e-7
            cell_bboxes[q, 2] = np.min(pts_ccw[:, 1]) - 1.0e-7
            cell_bboxes[q, 3] = np.max(pts_ccw[:, 1]) + 1.0e-7

        for s_idx, pt in enumerate(self.sensor_points):
            px, py = float(pt[0]), float(pt[1])
            cand_mask = (
                (px >= cell_bboxes[:, 0])
                & (px <= cell_bboxes[:, 1])
                & (py >= cell_bboxes[:, 2])
                & (py <= cell_bboxes[:, 3])
            )
            cand_cells = np.where(cand_mask)[0]

            found = False
            for q in cand_cells:
                poly = ordered_cell_coords[q]
                if MplPath(poly).contains_point(pt, radius=1.0e-7):
                    xi, eta, ok = self._invert_bilinear_map(
                        pt, poly, tol=newton_tol, max_iter=max_newton_iter
                    )
                    if ok and (-1.0e-6 <= xi <= 1.0 + 1.0e-6) and (-1.0e-6 <= eta <= 1.0 + 1.0e-6):
                        xi_c = float(np.clip(xi, 0.0, 1.0))
                        eta_c = float(np.clip(eta, 0.0, 1.0))
                        w = np.array(
                            [
                                (1.0 - xi_c) * (1.0 - eta_c),
                                xi_c * (1.0 - eta_c),
                                xi_c * eta_c,
                                (1.0 - xi_c) * eta_c,
                            ],
                            dtype=np.float64,
                        )
                        self.sensor_cell_indices[s_idx] = q
                        self.sensor_dofs[s_idx] = ordered_cell_dofs[q]
                        self.sensor_weights[s_idx] = w
                        found = True
                        break

            if not found:
                raise RuntimeError(
                    f"Sensore #{s_idx} in {pt.tolist()} non localizzato in alcuna cella Q1."
                )

    @staticmethod
    def _invert_bilinear_map(
        target_xy: np.ndarray,
        quad_ccw: np.ndarray,
        tol: float = 1.0e-13,
        max_iter: int = 25,
    ) -> Tuple[float, float, bool]:
        p00, p10, p11, p01 = quad_ccw[0], quad_ccw[1], quad_ccw[2], quad_ccw[3]
        xi, eta = 0.5, 0.5
        for _ in range(max_iter):
            n00 = (1.0 - xi) * (1.0 - eta)
            n10 = xi * (1.0 - eta)
            n11 = xi * eta
            n01 = (1.0 - xi) * eta
            x_cur = n00 * p00 + n10 * p10 + n11 * p11 + n01 * p01
            res = x_cur - target_xy
            if float(np.linalg.norm(res, ord=np.inf)) <= tol:
                return xi, eta, True

            dx_dxi = (1.0 - eta) * (p10 - p00) + eta * (p11 - p01)
            dx_deta = (1.0 - xi) * (p01 - p00) + xi * (p11 - p10)
            J = np.column_stack((dx_dxi, dx_deta))
            det = float(np.linalg.det(J))
            if abs(det) < 1.0e-15:
                return xi, eta, False
            delta = np.linalg.solve(J, res)
            xi -= float(delta[0])
            eta -= float(delta[1])

        # Recompute the residual after the last Newton update.
        n00 = (1.0 - xi) * (1.0 - eta)
        n10 = xi * (1.0 - eta)
        n11 = xi * eta
        n01 = (1.0 - xi) * eta
        residual = n00 * p00 + n10 * p10 + n11 * p11 + n01 * p01 - target_xy
        return xi, eta, (float(np.linalg.norm(residual, ord=np.inf)) <= 1.0e-9)

    def evaluate_spatial_field(self, c_nodal: np.ndarray) -> np.ndarray:
        vals_4 = c_nodal[self.sensor_dofs]
        return np.sum(self.sensor_weights * vals_4, axis=1)

    def apply(self, problem: RetecoForwardProblem2D, traj: ForwardTrajectoryResult) -> np.ndarray:
        times = np.asarray(traj.times, dtype=np.float64)
        c_hist = np.asarray(traj.c_hist, dtype=np.float64)
        if float(np.min(self.observation_times)) < times[0] - 1e-12 or float(np.max(self.observation_times)) > times[-1] + 1e-12:
            raise ValueError("Observation time outside simulated interval.")
        out = np.zeros((self.num_times, self.num_sensors), dtype=np.float64)

        for k, t_obs in enumerate(self.observation_times):
            if t_obs <= times[0]:
                c_interp = c_hist[0]
            elif t_obs >= times[-1]:
                c_interp = c_hist[-1]
            else:
                idx_hi = int(np.searchsorted(times, t_obs, side="right"))
                idx_lo = idx_hi - 1
                t0, t1 = float(times[idx_lo]), float(times[idx_hi])
                w = (float(t_obs) - t0) / max(t1 - t0, 1.0e-15)
                c_interp = (1.0 - w) * c_hist[idx_lo] + w * c_hist[idx_hi]
            out[k, :] = self.evaluate_spatial_field(c_interp)

        return out.ravel()

    def verify_linear_and_bilinear_reproduction(
        self, problem: RetecoForwardProblem2D, is_cartesian: bool = False
    ) -> Dict[str, float]:
        x_nodes = problem.tracer_pts[:, 0]
        y_nodes = problem.tracer_pts[:, 1]
        xs = self.sensor_points[:, 0]
        ys = self.sensor_points[:, 1]

        f_lin_nodes = 1.25 + 2.30 * x_nodes - 0.85 * y_nodes
        f_lin_exact = 1.25 + 2.30 * xs - 0.85 * ys
        err_lin = float(np.max(np.abs(self.evaluate_spatial_field(f_lin_nodes) - f_lin_exact)))

        res = {"max_abs_error_linear_field": err_lin}
        if is_cartesian:
            f_bilin_nodes = 0.50 + 1.10 * x_nodes - 0.40 * y_nodes + 0.75 * x_nodes * y_nodes
            f_bilin_exact = 0.50 + 1.10 * xs - 0.40 * ys + 0.75 * xs * ys
            err_bilin = float(
                np.max(np.abs(self.evaluate_spatial_field(f_bilin_nodes) - f_bilin_exact))
            )
            res["max_abs_error_bilinear_field_cartesian"] = err_bilin
        return res


