"""
python/reteco/forward/target_generator.py
Costruzione delle forzanti per il problema diretto RETECO v1:
- Protocollo sintetico standard sul quadrato unitario [0,1]^2;
- Protocollo di forzante sintetica guidata dall'anatomia (Requisito 6), con trasformazione
  verificata del centroide della lesione da coordinate fisiche (mm) al riferimento della mesh (L0)
  e conversione esplicita della larghezza spaziale sigma_mm mediante L0.
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

from .wrapper import ForwardTrajectoryResult, RetecoForwardProblem2D
from .. import _core
from ..config_schema import RetecoMasterConfig


def _evaluate_temporal_modulation(t: float, profile_cfg: Dict[str, Any]) -> float:
    """
    Valuta il profilo temporale esplicito specificato in configurazione:
    - 'sinusoidal': sin(omega_t * t)
    - 'exponential_decay': exp(-decay_rate * t)
    - 'constant': 1.0
    - 'ramp_plateau': min(1.0, t / ramp_time)
    """
    kind = str(profile_cfg.get("temporal_profile", "sinusoidal")).lower()
    if kind == "constant":
        return 1.0
    if kind == "sinusoidal":
        omega = float(profile_cfg.get("omega_t", math.pi))
        return math.sin(omega * t)
    if kind == "exponential_decay":
        rate = float(profile_cfg.get("decay_rate", 1.0))
        return math.exp(-rate * t)
    if kind == "ramp_plateau":
        t_ramp = max(float(profile_cfg.get("ramp_time", 0.1)), 1.0e-12)
        return min(1.0, t / t_ramp)
    raise ValueError(f"Profilo temporale non riconosciuto in configurazione: '{kind}'.")


def _resolve_center_and_sigma_adim(
    item_cfg: Dict[str, Any],
    L0_mm: float,
    origin_xy_mm: Optional[Tuple[float, float]] = None,
    lesion_center_mm_from_report: Optional[Tuple[float, float]] = None,
    lesion_center_adim_from_report: Optional[Tuple[float, float]] = None,
) -> Tuple[Tuple[float, float], float, Dict[str, Any]]:
    """
    Risolve il centro e la larghezza spaziale (sigma) di una sorgente Gaussiana
    nel sistema di coordinate adimensionali della mesh:
        x_adim = (x_mm - origin_xy_mm) / L0_mm,   sigma_adim = sigma_mm / L0_mm.
    """
    if item_cfg.get("use_preprocessed_lesion_center", False):
        if lesion_center_mm_from_report is None or origin_xy_mm is None:
            raise ValueError(
                "use_preprocessed_lesion_center=True richiede origin_xy_mm e center_physical_mm dal report di Fase 1."
            )
        cx_mm, cy_mm = float(lesion_center_mm_from_report[0]), float(lesion_center_mm_from_report[1])
        cx_adim = (cx_mm - float(origin_xy_mm[0])) / L0_mm
        cy_adim = (cy_mm - float(origin_xy_mm[1])) / L0_mm
        if lesion_center_adim_from_report is not None:
            if (
                abs(cx_adim - float(lesion_center_adim_from_report[0])) > 1.0e-4
                or abs(cy_adim - float(lesion_center_adim_from_report[1])) > 1.0e-4
            ):
                raise ValueError(
                    "Incoerenza tra la trasformazione del centro lesione (x_mm - x0_mm)/L0_mm "
                    "e center_mesh_adim salvato nel report di Fase 1."
                )
        center_source = "preprocessed_lesion_centroid_transformed_via_L0"
    elif "center_mm" in item_cfg:
        if origin_xy_mm is None:
            raise ValueError("Specificato 'center_mm' senza 'origin_xy_mm' per la conversione in coordinate mesh.")
        c_mm = item_cfg["center_mm"]
        cx_mm, cy_mm = float(c_mm[0]), float(c_mm[1])
        cx_adim = (cx_mm - float(origin_xy_mm[0])) / L0_mm
        cy_adim = (cy_mm - float(origin_xy_mm[1])) / L0_mm
        center_source = "explicit_physical_mm_transformed_via_L0"
    elif "center" in item_cfg or "center_adim" in item_cfg:
        c_adim = item_cfg.get("center_adim", item_cfg.get("center"))
        cx_adim, cy_adim = float(c_adim[0]), float(c_adim[1])
        if origin_xy_mm is not None:
            cx_mm = cx_adim * L0_mm + float(origin_xy_mm[0])
            cy_mm = cy_adim * L0_mm + float(origin_xy_mm[1])
        else:
            cx_mm = cx_adim * L0_mm
            cy_mm = cy_adim * L0_mm
        center_source = "explicit_dimensionless_mesh_coordinates"
    else:
        raise KeyError("La configurazione della sorgente deve specificare center, center_mm o use_preprocessed_lesion_center.")

    if "sigma_mm" in item_cfg:
        sigma_mm = float(item_cfg["sigma_mm"])
        sigma_adim = sigma_mm / L0_mm
        sigma_source = "physical_mm_divided_by_L0_mm"
    elif "sigma" in item_cfg or "sigma_adim" in item_cfg:
        sigma_adim = float(item_cfg.get("sigma_adim", item_cfg.get("sigma")))
        sigma_mm = sigma_adim * L0_mm
        sigma_source = "explicit_dimensionless_converted_with_L0_mm"
    else:
        raise KeyError("La configurazione della sorgente deve specificare 'sigma_mm' oppure 'sigma'.")

    if sigma_adim <= 0.0 or not np.isfinite(sigma_adim):
        raise ValueError(f"Larghezza spaziale della sorgente non valida: sigma_adim={sigma_adim}")

    meta = {
        "center_physical_mm": [cx_mm, cy_mm],
        "center_mesh_adim": [cx_adim, cy_adim],
        "sigma_physical_mm": sigma_mm,
        "sigma_mesh_adim": sigma_adim,
        "L0_mm_used": L0_mm,
        "center_resolution_mode": center_source,
        "sigma_resolution_mode": sigma_source,
    }
    return (cx_adim, cy_adim), sigma_adim, meta


def build_standard_driving_Callbacks(
    problem: RetecoForwardProblem2D,
    driving_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Callable[[float], np.ndarray], Callable[[float], np.ndarray]]:
    """
    Costruisce (c0, F_mech_func, S_fluid_func) per il problema diretto.
    Compatibile al 100% con la configurazione MMS/Twin sul quadrato unitario e
    supporta opzionalmente la conversione esplicita tramite L0.
    """
    c0, F_mech_func, S_fluid_func, _ = build_anatomical_driving_Callbacks(
        problem=problem,
        driving_cfg=driving_cfg,
        geometry_report_path=driving_cfg.get("geometry_report_path", None),
    )
    return c0, F_mech_func, S_fluid_func


def build_anatomical_driving_Callbacks(
    problem: RetecoForwardProblem2D,
    driving_cfg: Dict[str, Any],
    geometry_report_path: Optional[str | Path] = None,
) -> Tuple[
    np.ndarray,
    Callable[[float], np.ndarray],
    Callable[[float], np.ndarray],
    Dict[str, Any],
]:
    """
    Costruisce (c0, F_mech_func, S_fluid_func, forcing_metadata) trasformando la posizione
    della lesione dal report di preprocessing al riferimento della mesh tramite L0
    e registrando le scelte come forzanti sintetiche guidate dall'anatomia (Requisito 6).
    """
    L0_mm = float(problem.scales.L0_m) * 1000.0
    origin_xy_mm: Optional[Tuple[float, float]] = None
    lesion_mm: Optional[Tuple[float, float]] = None
    lesion_adim: Optional[Tuple[float, float]] = None

    if geometry_report_path is not None:
        rep_path = Path(geometry_report_path)
        if not rep_path.exists():
            raise FileNotFoundError(f"Report geometrico di Fase 1 non trovato: {rep_path}")
        rep_data = json.loads(rep_path.read_text(encoding="utf-8"))
        rep_l0_mm = float(rep_data["L0_mm"])
        if abs(rep_l0_mm - L0_mm) > 1.0e-6:
            raise ValueError(
                f"Incoerenza di scala L0 tra solver ({L0_mm} mm) e report geometrico ({rep_l0_mm} mm)."
            )
        orig_list = rep_data["coordinate_transform"]["origin_xy_mm"]
        origin_xy_mm = (float(orig_list[0]), float(orig_list[1]))
        if rep_data["lesion_centroids"]["has_lesion_in_slice"]:
            c_mm = rep_data["lesion_centroids"]["center_physical_mm"]
            c_ad = rep_data["lesion_centroids"]["center_mesh_adim"]
            lesion_mm = (float(c_mm[0]), float(c_mm[1]))
            lesion_adim = (float(c_ad[0]), float(c_ad[1]))

    # 1. Concentrazione iniziale del tracciante c0(x)
    g_cfg = driving_cfg["initial_tracer_gaussian"]
    c_center_adim, c_sigma_adim, c_meta = _resolve_center_and_sigma_adim(
        g_cfg, L0_mm, origin_xy_mm, lesion_mm, lesion_adim
    )
    c_amp = float(g_cfg["amplitude"])
    c_base = float(g_cfg.get("baseline", 0.1))
    c0 = problem.build_gaussian_initial_tracer(
        center=c_center_adim,
        sigma=c_sigma_adim,
        amplitude=c_amp,
        baseline=c_base,
    )

    # 2. Sorgente idraulica transitoria s_f(x, t) centrata nella regione specificata / lesione
    sf_cfg = driving_cfg["fluid_injection_source"]
    sf_center_adim, sf_sigma_adim, sf_meta = _resolve_center_and_sigma_adim(
        sf_cfg, L0_mm, origin_xy_mm, lesion_mm, lesion_adim
    )
    sf_amp = float(sf_cfg["amplitude"])

    px = problem.press_pts[:, 0]
    py = problem.press_pts[:, 1]
    r2_p = (px - sf_center_adim[0]) ** 2 + (py - sf_center_adim[1]) ** 2
    sf_spatial_nodal = sf_amp * np.exp(-r2_p / (2.0 * (sf_sigma_adim ** 2)))
    S_fluid_base = problem.M_p_csr.dot(sf_spatial_nodal)

    def S_fluid_func(t: float) -> np.ndarray:
        mod = _evaluate_temporal_modulation(t, sf_cfg)
        return S_fluid_base * mod

    # 3. Forzante meccanica di volume f_s(x, t)
    fm_cfg = driving_cfg["mechanical_body_force"]
    amp_x = float(fm_cfg["amplitude_x"])
    amp_y = float(fm_cfg["amplitude_y"])
    is_dirichlet = np.asarray(problem.fem.is_dirichlet_disp(), dtype=bool)

    if fm_cfg.get("localized_on_lesion", False):
        fm_center_adim, fm_sigma_adim, fm_meta = _resolve_center_and_sigma_adim(
            fm_cfg, L0_mm, origin_xy_mm, lesion_mm, lesion_adim
        )
    else:
        fm_center_adim = sf_center_adim
        fm_sigma_adim = sf_sigma_adim
        fm_meta = {"mode": "sinusoidal_domain_excitation"}

    # Vettore di carico meccanico spaziale integrato sulle componenti libere (non-Dirichlet)
    dx_pts = problem.disp_pts[:, 0]
    dy_pts = problem.disp_pts[:, 1]
    F_base = np.zeros(problem.num_disp_dofs, dtype=np.float64)
    if fm_cfg.get("localized_on_lesion", False):
        r2_d = (dx_pts - fm_center_adim[0]) ** 2 + (dy_pts - fm_center_adim[1]) ** 2
        env_d = np.exp(-r2_d / (2.0 * (fm_sigma_adim ** 2)))
    else:
        env_d = np.sin(np.pi * dx_pts) * np.sin(np.pi * dy_pts)

    F_base[0::2] = amp_x * env_d[0::2]
    F_base[1::2] = amp_y * env_d[1::2]
    # Riscalamento con la diagonale della matrice elastica / scala cella e azzeramento su Gamma_D
    diag_scale = np.asarray(problem.K0_csr.diagonal(), dtype=np.float64)
    mean_diag = float(np.mean(diag_scale)) if diag_scale.size > 0 else 1.0
    F_base = (F_base / max(mean_diag, 1.0e-12)) * (1.0 / float(max(problem.num_active_cells, 1)))
    F_base[is_dirichlet] = 0.0

    def F_mech_func(t: float) -> np.ndarray:
        return F_base * _evaluate_temporal_modulation(t, fm_cfg)

    forcing_metadata = {
        "forcing_classification": "anatomy_guided_synthetic_forcing",
        "clinical_disclaimer": (
            "SYNTHETIC_FORCING_ONLY: Source locations are anchored to the preprocessed "
            "BraTS lesion centroid transformed via L0, while spatial widths, amplitudes, "
            "and time profiles are explicit synthetic model inputs."
        ),
        "initial_tracer_gaussian": {
            **c_meta,
            "amplitude": c_amp,
            "baseline": c_base,
        },
        "fluid_injection_source": {
            **sf_meta,
            "amplitude": sf_amp,
            "temporal_profile": str(sf_cfg.get("temporal_profile", "sinusoidal")),
        },
        "mechanical_body_force": {
            **fm_meta,
            "amplitude_x": amp_x,
            "amplitude_y": amp_y,
            "temporal_profile": str(fm_cfg.get("temporal_profile", "sinusoidal")),
            "omega_t": float(fm_cfg.get("omega_t", math.pi)),
        },
    }

    return c0, F_mech_func, S_fluid_func, forcing_metadata


def generate_fine_target_trajectory(
    master_cfg: RetecoMasterConfig,
    twin_cfg: Dict[str, Any],
) -> Tuple[RetecoForwardProblem2D, ForwardTrajectoryResult]:
    t_disc = twin_cfg["target_discretization"]
    grid = _core.TimeGrid.make_graded(
        float(t_disc["final_time_T"]),
        int(t_disc["time_steps_M"]),
        float(t_disc["grading_r"]),
    )
    problem = RetecoForwardProblem2D(
        subdivisions_N=int(t_disc.get("subdivisions_N", 32)),
        physics_model=master_cfg.fixed_physics,
        scales=master_cfg.characteristic_scales,
        time_grid=grid,
        fe_deg_disp=int(t_disc.get("fe_deg_disp", 2)),
        fe_deg_press=int(t_disc.get("fe_deg_press", 1)),
        fe_deg_tracer=int(t_disc.get("fe_deg_tracer", 1)),
        mesh_file_path=t_disc.get("mesh_file_path", None),
        dirichlet_disp_boundary_ids=t_disc.get("dirichlet_disp_boundary_ids", None),
    )

    c0, F_mech_func, S_fluid_func = build_standard_driving_Callbacks(
        problem, twin_cfg["driving_protocol"]
    )
    theta_dagger = master_cfg.parameter_space_theta.theta_dagger_array()

    traj = problem.solve_trajectory(
        theta_vec=theta_dagger,
        c0=c0,
        p0=None,
        d0=None,
        F_mech_func=F_mech_func,
        S_fluid_func=S_fluid_func,
        S_tracer_func=None,
        enforce_initial_compatibility=True,
        compute_diagnostics=True,
    )
    return problem, traj