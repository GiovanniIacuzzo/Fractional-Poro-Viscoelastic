"""
python/reteco/forward/target_generator.py
Costruzione del protocollo di forzante sintetica e generazione del target Ground Truth
su discretizzazione indipendente/raffinata h_tar (Contratto Matematico Sez. 8 e 9.8).
"""

from __future__ import annotations
from typing import Callable, Dict, Tuple
import numpy as np

from .wrapper import RetecoForwardProblem2D, ForwardTrajectoryResult


def build_standard_driving_Callbacks(
    problem: RetecoForwardProblem2D,
    driving_cfg: Dict,
) -> Tuple[
    np.ndarray,
    Callable[[float], np.ndarray],
    Callable[[float], np.ndarray],
]:
    """
    Costruisce:
    1) Condizione iniziale del tracciante c_0(x) Gaussiana;
    2) Forzante meccanica F_mech(t) che si annulla a t=0 (sin(omega * t)) per compatibilità con d0=0;
    3) Sorgente idraulica S_fluid(t) che genera un flusso di Darcy q != 0 sensibile a (beta, E).
    """
    ic_cfg = driving_cfg["initial_tracer_gaussian"]
    c0 = problem.build_gaussian_initial_tracer(
        center=tuple(ic_cfg["center"]),
        sigma=float(ic_cfg["sigma"]),
        amplitude=float(ic_cfg["amplitude"]),
        baseline=float(ic_cfg["baseline"]),
    )

    # Costruzione del profilo spaziale della sorgente idraulica integrato con la matrice di massa M_p
    sf_cfg = driving_cfg["fluid_injection_source"]
    px = problem.press_pts[:, 0]
    py = problem.press_pts[:, 1]
    r2_p = (px - sf_cfg["center"][0]) ** 2 + (py - sf_cfg["center"][1]) ** 2
    # Sorgente a media nulla (iniezione al centro, drenaggio diffuso) per enfatizzare il gradiente di p
    raw_sf = float(sf_cfg["amplitude"]) * (
        np.exp(-r2_p / (2.0 * (float(sf_cfg["sigma"]) ** 2))) - 0.5
    )
    Sf_spatial_rhs = problem.M_p_csr @ raw_sf

    # Costruzione del carico meccanico integrato
    mech_cfg = driving_cfg["mechanical_body_force"]
    omega = float(mech_cfg["omega_t"])
    dx_pts = problem.disp_pts[:, 0]
    dy_pts = problem.disp_pts[:, 1]

    # In deal.II FESystem(FE_Q, 2), usiamo un profilo spaziale regolare
    spatial_profile = np.sin(np.pi * dx_pts) * np.sin(np.pi * dy_pts)
    F_spatial_base = problem.K0_csr @ (0.01 * spatial_profile)
    # Neutralizziamo i DOFs di Dirichlet
    is_dir = np.asarray(problem.fem.is_dirichlet_disp(), dtype=bool)
    F_spatial_base[is_dir] = 0.0
    amp_mech = math.hypot(float(mech_cfg["amplitude_x"]), float(mech_cfg["amplitude_y"]))

    def F_mech_func(t: float) -> np.ndarray:
        if t <= 0.0:
            return np.zeros(problem.num_disp_dofs, dtype=np.float64)
        return (amp_mech * math.sin(omega * t)) * F_spatial_base

    def S_fluid_func(t: float) -> np.ndarray:
        if t <= 0.0:
            return np.zeros(problem.num_press_dofs, dtype=np.float64)
        return math.sin(omega * t) * Sf_spatial_rhs

    return c0, F_mech_func, S_fluid_func


import math