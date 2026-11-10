#!/usr/bin/env python3
"""
experiments/13_run_reviewer_identifiability_extensions.py

Esegue i tre esperimenti integrativi sul problema di identificabilità senza modificare
alcun artefatto congelato del manoscritto:
1. Audit rigoroso dell'operatore osservativo: valutazione bilineare isoparametrica Q1
   esatta sugli elementi finiti vs interpolazione lineare `griddata` (su y_clean e sul Jacobiano).
2. Studio parametrico di transizione dal regime a flusso nullo (q == 0) al regime a flusso
   attivo al variare del fattore di forzamento kappa_q (e del numero di Péclet).
3. Analisi locale di identificabilità (SVD del Jacobiano a flusso nullo e a flusso attivo)
   direttamente sulla mesh quadrilatera anatomica BraTS 2020 (h = 3.0 mm).
"""

from __future__ import annotations
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT_DIR / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from reteco import _core, load_master_config
from reteco.config_schema import ParameterCoordinateTransform
from reteco.forward.target_generator import (
    build_anatomical_driving_Callbacks,
    build_standard_driving_Callbacks,
)
from reteco.forward.wrapper import ForwardTrajectoryResult, RetecoForwardProblem2D
from reteco.identifiability.sensitivity_jacobian import (
    PARAMETER_NAMES,
    ObservationWeightingSpec,
    compute_normalized_jacobian_with_audit,
)
from reteco.identifiability.svd_conditioning import (
    analyze_jacobian_svd,
    sanitize_for_json,
)
from reteco.observation.operator import DiscreteObservationOperator
from reteco.observation.q1_exact import ExactQ1TracerObservationOperator

EXPECTED_TARGET_SHA256 = (
    "1bca9e29c60d77c1450fd2fcea845d2320c9e44945dd7d3ccda52aa33ef261af"
)


def _get_cond_4x4(rep: Any) -> Optional[float]:
    """Estrae in modo sicuro il numero di condizionamento 4x4 da LocalIdentifiabilityReport."""
    if hasattr(rep, "to_json_dict"):
        d = rep.to_json_dict()
        val = d.get("condition_number_full_4x4", d.get("condition_number"))
        if val is not None and np.isfinite(float(val)):
            return float(val)
    if hasattr(rep, "condition_number") and rep.condition_number is not None:
        if np.isfinite(float(rep.condition_number)):
            return float(rep.condition_number)
    sv = np.asarray(rep.singular_values, dtype=np.float64)
    if sv.size >= 4 and sv[0] > 0.0 and sv[-1] > 1.0e-15 * sv[0]:
        return float(sv[0] / sv[-1])
    return None


# =============================================================================
# 1. Valutatore Bilineare Isoparametrico Q1 Esatto (su mesh cartesiana e BraTS)
# =============================================================================
# =============================================================================
# 2. Audit Q1 vs Griddata sul Quadrato Unitario (Target e Jacobiano)
# =============================================================================
def run_q1_vs_griddata_audit(
    master: Any,
    twin_cfg: Dict[str, Any],
    transform: ParameterCoordinateTransform,
    theta_dagger: np.ndarray,
    eps_step: float = 1.0e-4,
    rank_threshold: float = 1.0e-8,
) -> Dict[str, Any]:
    obs_cfg = twin_cfg["observation_protocol"]
    observed_fields = tuple(str(f) for f in obs_cfg.get("observed_fields", ["c"]))
    obs_op_griddata = DiscreteObservationOperator.from_regular_grid(
        nx=int(obs_cfg["sensor_grid_nx"]),
        ny=int(obs_cfg["sensor_grid_ny"]),
        margin=float(obs_cfg["sensor_margin"]),
        observation_times=obs_cfg["observation_times"],
        observed_fields=list(observed_fields),
    )

    nx = int(obs_cfg["sensor_grid_nx"])
    ny = int(obs_cfg["sensor_grid_ny"])
    margin = float(obs_cfg["sensor_margin"])
    gx = np.linspace(margin, 1.0 - margin, nx, dtype=np.float64)
    gy = np.linspace(margin, 1.0 - margin, ny, dtype=np.float64)
    xx, yy = np.meshgrid(gx, gy, indexing="xy")
    sensor_pts = np.column_stack([xx.ravel(), yy.ravel()])

    # 2a. Confronto sul target fine (N=32, M=160)
    tar_cfg = twin_cfg["target_discretization"]
    grid_tar = _core.TimeGrid.make_graded(
        float(tar_cfg["final_time_T"]),
        int(tar_cfg["time_steps_M"]),
        float(tar_cfg["grading_r"]),
    )
    prob_tar = RetecoForwardProblem2D(
        subdivisions_N=int(tar_cfg["subdivisions_N"]),
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid_tar,
    )
    c0_tar, F_tar, Sf_tar = build_standard_driving_Callbacks(prob_tar, twin_cfg["driving_protocol"])
    traj_tar = prob_tar.solve_trajectory(
        theta_dagger, c0=c0_tar, F_mech_func=F_tar, S_fluid_func=Sf_tar
    )

    obs_op_q1_tar = ExactQ1TracerObservationOperator(
        prob_tar, sensor_pts, obs_cfg["observation_times"]
    )
    y_tar_griddata = np.asarray(obs_op_griddata.apply(prob_tar, traj_tar), dtype=np.float64).ravel()
    y_tar_q1 = obs_op_q1_tar.apply(prob_tar, traj_tar)

    sha_griddata = hashlib.sha256(
        np.ascontiguousarray(y_tar_griddata, dtype="<f8").tobytes()
    ).hexdigest()

    # 2b. Confronto sulla griglia inversa (N=16, M=80) e sul Jacobiano
    inv_cfg = twin_cfg["inverse_discretization"]
    grid_inv = _core.TimeGrid.make_graded(
        float(inv_cfg["final_time_T"]),
        int(inv_cfg["time_steps_M"]),
        float(inv_cfg["grading_r"]),
    )
    prob_inv = RetecoForwardProblem2D(
        subdivisions_N=int(inv_cfg["subdivisions_N"]),
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid_inv,
    )
    c0_inv, F_inv, Sf_inv = build_standard_driving_Callbacks(prob_inv, twin_cfg["driving_protocol"])
    obs_op_q1_inv = ExactQ1TracerObservationOperator(
        prob_inv, sensor_pts, obs_cfg["observation_times"]
    )

    weighting_spec = ObservationWeightingSpec(
        weighting_mode="unweighted_dimensionless_state_identity",
        weights_sqrt_diag=None,
        is_statistically_calibrated=False,
        observed_fields=observed_fields,
        provenance_description="Dimensionless identity weights.",
    )

    def G_inv_griddata(theta: np.ndarray) -> np.ndarray:
        tr = prob_inv.solve_trajectory(theta, c0=c0_inv, F_mech_func=F_inv, S_fluid_func=Sf_inv)
        return np.asarray(obs_op_griddata.apply(prob_inv, tr), dtype=np.float64).ravel()

    def G_inv_q1(theta: np.ndarray) -> np.ndarray:
        tr = prob_inv.solve_trajectory(theta, c0=c0_inv, F_mech_func=F_inv, S_fluid_func=Sf_inv)
        return obs_op_q1_inv.apply(prob_inv, tr)

    audit_griddata = compute_normalized_jacobian_with_audit(
        forward_obs_callable=G_inv_griddata,
        theta_ref=theta_dagger,
        transform=transform,
        eps_step=eps_step,
        weighting_spec=weighting_spec,
    )
    audit_q1 = compute_normalized_jacobian_with_audit(
        forward_obs_callable=G_inv_q1,
        theta_ref=theta_dagger,
        transform=transform,
        eps_step=eps_step,
        weighting_spec=weighting_spec,
    )

    svd_griddata = analyze_jacobian_svd(
        audit_griddata.jacobian_weighted, rank_threshold=rank_threshold, weighting_spec=weighting_spec
    )
    svd_q1 = analyze_jacobian_svd(
        audit_q1.jacobian_weighted, rank_threshold=rank_threshold, weighting_spec=weighting_spec
    )

    y_inv_griddata = audit_griddata.baseline_observation
    y_inv_q1 = audit_q1.baseline_observation

    discretization_gap_l2 = float(
        np.linalg.norm(y_inv_griddata - y_tar_griddata) / np.linalg.norm(y_tar_griddata)
    )
    rel_diff_target_l2 = float(
        np.linalg.norm(y_tar_q1 - y_tar_griddata) / np.linalg.norm(y_tar_griddata)
    )
    rel_diff_inv_l2 = float(
        np.linalg.norm(y_inv_q1 - y_inv_griddata) / np.linalg.norm(y_inv_griddata)
    )
    rel_frob_jacobian = float(
        np.linalg.norm(audit_q1.jacobian_weighted - audit_griddata.jacobian_weighted, ord="fro")
        / np.linalg.norm(audit_griddata.jacobian_weighted, ord="fro")
    )

    return {
        "exact_polynomial_reproduction_checks": {
            "target_mesh_N32": obs_op_q1_tar.verify_linear_and_bilinear_reproduction(
                prob_tar, is_cartesian=True
            ),
            "inverse_mesh_N16": obs_op_q1_inv.verify_linear_and_bilinear_reproduction(
                prob_inv, is_cartesian=True
            ),
        },
        "frozen_target_sha256_verified": bool(sha_griddata == EXPECTED_TARGET_SHA256),
        "target_N32_M160_comparison": {
            "rel_l2_diff_q1_vs_griddata": rel_diff_target_l2,
            "max_abs_diff_q1_vs_griddata": float(np.max(np.abs(y_tar_q1 - y_tar_griddata))),
        },
        "inverse_N16_M80_comparison": {
            "rel_l2_diff_q1_vs_griddata": rel_diff_inv_l2,
            "max_abs_diff_q1_vs_griddata": float(np.max(np.abs(y_inv_q1 - y_inv_griddata))),
            "discretization_gap_rel_l2_N16_vs_N32": discretization_gap_l2,
            "ratio_interpolation_diff_over_discretization_gap": float(
                rel_diff_inv_l2 / max(discretization_gap_l2, 1.0e-15)
            ),
        },
        "jacobian_and_svd_comparison": {
            "relative_frobenius_difference": rel_frob_jacobian,
            "singular_values_griddata": [float(x) for x in svd_griddata.singular_values],
            "singular_values_exact_q1": [float(x) for x in svd_q1.singular_values],
            "max_rel_diff_singular_values": float(
                np.max(
                    np.abs(svd_q1.singular_values - svd_griddata.singular_values)
                    / np.maximum(svd_griddata.singular_values, 1.0e-15)
                )
            ),
            "condition_number_griddata": _get_cond_4x4(svd_griddata),
            "condition_number_exact_q1": _get_cond_4x4(svd_q1),
        },
    }


# =============================================================================
# 3. Studio Parametrico della Transizione di Flusso (kappa_q / Péclet Sweep)
# =============================================================================
def run_flow_transition_peclet_sweep(
    master: Any,
    twin_cfg: Dict[str, Any],
    transform: ParameterCoordinateTransform,
    theta_dagger: np.ndarray,
    kappa_values: Sequence[float] = (0.0, 1.0e-3, 1.0e-2, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00),
    eps_step: float = 1.0e-4,
    rank_threshold: float = 1.0e-8,
) -> List[Dict[str, Any]]:
    inv_cfg = twin_cfg["inverse_discretization"]
    grid = _core.TimeGrid.make_graded(
        float(inv_cfg["final_time_T"]),
        int(inv_cfg["time_steps_M"]),
        float(inv_cfg["grading_r"]),
    )
    problem = RetecoForwardProblem2D(
        subdivisions_N=int(inv_cfg["subdivisions_N"]),
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid,
    )
    c0, F_base, Sf_base = build_standard_driving_Callbacks(problem, twin_cfg["driving_protocol"])

    obs_cfg = twin_cfg["observation_protocol"]
    observed_fields = tuple(str(f) for f in obs_cfg.get("observed_fields", ["c"]))
    obs_op = DiscreteObservationOperator.from_regular_grid(
        nx=int(obs_cfg["sensor_grid_nx"]),
        ny=int(obs_cfg["sensor_grid_ny"]),
        margin=float(obs_cfg["sensor_margin"]),
        observation_times=obs_cfg["observation_times"],
        observed_fields=list(observed_fields),
    )
    weighting_spec = ObservationWeightingSpec(
        weighting_mode="unweighted_dimensionless_state_identity",
        weights_sqrt_diag=None,
        is_statistically_calibrated=False,
        observed_fields=observed_fields,
        provenance_description="Dimensionless identity weights.",
    )

    phi_0 = float(master.fixed_physics.phi_0)
    D_ref = float(theta_dagger[2])
    L_char_adim = 1.0

    records: List[Dict[str, Any]] = []
    for kappa in kappa_values:
        k_val = float(kappa)
        q_ref_holder: List[float] = []

        if k_val == 0.0:
            F_scaled = None
            Sf_scaled = None
        else:
            F_scaled = lambda t, k=k_val: k * F_base(t)
            Sf_scaled = lambda t, k=k_val: k * Sf_base(t)

        def G_kappa(theta: np.ndarray) -> np.ndarray:
            tr = problem.solve_trajectory(
                theta, c0=c0, F_mech_func=F_scaled, S_fluid_func=Sf_scaled
            )
            q_ref_holder.append(float(tr.max_darcy_velocity))
            return np.asarray(obs_op.apply(problem, tr), dtype=np.float64).ravel()

        audit_k = compute_normalized_jacobian_with_audit(
            forward_obs_callable=G_kappa,
            theta_ref=theta_dagger,
            transform=transform,
            eps_step=eps_step,
            weighting_spec=weighting_spec,
        )
        rep_k = analyze_jacobian_svd(
            audit_k.jacobian_weighted, rank_threshold=rank_threshold, weighting_spec=weighting_spec
        )

        q_max_ref = float(q_ref_holder[0])
        peclet_global = float(q_max_ref * L_char_adim / max(phi_0 * D_ref, 1.0e-15))
        col_norms = np.linalg.norm(audit_k.jacobian_weighted, axis=0)
        sv = rep_k.singular_values

        records.append(
            {
                "kappa_q": k_val,
                "max_darcy_velocity_adim": q_max_ref,
                "peclet_number": peclet_global,
                "numerical_rank": int(rep_k.numerical_rank),
                "col_norm_alpha": float(col_norms[0]),
                "col_norm_beta": float(col_norms[1]),
                "col_norm_D": float(col_norms[2]),
                "col_norm_E": float(col_norms[3]),
                "sigma_1": float(sv[0]),
                "sigma_2": float(sv[1]),
                "sigma_3": float(sv[2]),
                "sigma_4": float(sv[3]),
                "condition_number_4x4": _get_cond_4x4(rep_k),
            }
        )

    return records


# =============================================================================
# 4. Identificabilità Locale (SVD) sul Dominio Anatomico BraTS 2020 (h = 3 mm)
# =============================================================================
def _build_anatomical_sensor_grid_around_lesion(
    lesion_center_adim: np.ndarray,
    nx: int = 6,
    ny: int = 6,
    half_span_adim: float = 2.5,
) -> np.ndarray:
    gx = np.linspace(
        float(lesion_center_adim[0]) - half_span_adim,
        float(lesion_center_adim[0]) + half_span_adim,
        nx,
        dtype=np.float64,
    )
    gy = np.linspace(
        float(lesion_center_adim[1]) - half_span_adim,
        float(lesion_center_adim[1]) + half_span_adim,
        ny,
        dtype=np.float64,
    )
    xx, yy = np.meshgrid(gx, gy, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()])


def run_anatomical_brats_identifiability(
    master: Any,
    transform: ParameterCoordinateTransform,
    theta_dagger: np.ndarray,
    anatomical_config_path: str | Path = "configs/brats_anatomical_forward.yaml",
    eps_step: float = 1.0e-4,
    rank_threshold: float = 1.0e-8,
) -> Dict[str, Any]:
    with Path(anatomical_config_path).open("r", encoding="utf-8") as fh:
        raw_cfg = yaml.safe_load(fh)

    anat_out_dir = Path(
        raw_cfg.get("experiment_output_dir", "artifacts/02_anatomical_forward_clinical")
    )
    geom_report_path = anat_out_dir / "geometry_quality_report.json"

    if not geom_report_path.exists():
        raise FileNotFoundError(
            f"Report geometrico non trovato in {geom_report_path}. "
            "Esegui prima `python experiments/02_run_anatomical_forward.py`."
        )

    geom_report_dict = json.loads(geom_report_path.read_text(encoding="utf-8"))
    msh_path = Path(geom_report_dict["exported_files"]["gmsh_msh"])
    if not msh_path.exists():
        raise FileNotFoundError(f"Mesh anatomica Gmsh non trovata: {msh_path}")

    fwd_cfg = raw_cfg["forward_simulation"]
    T_anat = float(fwd_cfg["final_time_T"])
    M_anat = int(fwd_cfg["time_steps_M"])
    r_anat = float(fwd_cfg["grading_r"])
    grid_anat = _core.TimeGrid.make_graded(T_anat, M_anat, r_anat)

    problem_anat = RetecoForwardProblem2D(
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid_anat,
        fe_deg_disp=int(fwd_cfg.get("fe_deg_disp", 2)),
        fe_deg_press=int(fwd_cfg.get("fe_deg_press", 1)),
        fe_deg_tracer=int(fwd_cfg.get("fe_deg_tracer", 1)),
        use_supg=bool(fwd_cfg.get("use_supg", False)),
        mesh_file_path=msh_path,
        dirichlet_disp_boundary_ids=fwd_cfg.get("dirichlet_disp_boundary_ids", [1]),
    )

    c0_anat, F_mech_anat, S_fluid_anat, _ = build_anatomical_driving_Callbacks(
        problem=problem_anat,
        driving_cfg=fwd_cfg["driving_protocol"],
        geometry_report_path=geom_report_path,
    )

    lesion_center_adim = np.asarray(
        geom_report_dict["lesion_centroids"]["center_mesh_adim"], dtype=np.float64
    )
    sensor_pts_anat = _build_anatomical_sensor_grid_around_lesion(
        lesion_center_adim, nx=6, ny=6, half_span_adim=2.5
    )
    obs_times_anat = [round(frac * T_anat, 6) for frac in (0.15, 0.30, 0.50, 0.75, 1.00)]

    obs_op_anat_q1 = ExactQ1TracerObservationOperator(
        problem_anat, sensor_pts_anat, obs_times_anat
    )
    q1_reprod_anat = obs_op_anat_q1.verify_linear_and_bilinear_reproduction(
        problem_anat, is_cartesian=False
    )

    weighting_spec = ObservationWeightingSpec(
        weighting_mode="unweighted_dimensionless_state_identity",
        weights_sqrt_diag=None,
        is_statistically_calibrated=False,
        observed_fields=("c",),
        provenance_description="Dimensionless identity weights on 36 focal sensors x 5 times (N_obs=180).",
    )

    # 4a. Flusso Nullo su dominio anatomico BraTS
    q_zero_anat: List[float] = []

    def G_anat_zero(theta: np.ndarray) -> np.ndarray:
        tr = problem_anat.solve_trajectory(
            theta_vec=theta,
            c0=c0_anat,
            F_mech_func=None,
            S_fluid_func=None,
            enforce_initial_compatibility=True,
        )
        q_zero_anat.append(float(tr.max_darcy_velocity))
        return obs_op_anat_q1.apply(problem_anat, tr)

    audit_anat_zero = compute_normalized_jacobian_with_audit(
        forward_obs_callable=G_anat_zero,
        theta_ref=theta_dagger,
        transform=transform,
        eps_step=eps_step,
        weighting_spec=weighting_spec,
    )
    svd_anat_zero = analyze_jacobian_svd(
        audit_anat_zero.jacobian_weighted, rank_threshold=rank_threshold, weighting_spec=weighting_spec
    )

    # 4b. Flusso Attivo su dominio anatomico BraTS
    q_active_anat: List[float] = []

    def G_anat_active(theta: np.ndarray) -> np.ndarray:
        tr = problem_anat.solve_trajectory(
            theta_vec=theta,
            c0=c0_anat,
            F_mech_func=F_mech_anat,
            S_fluid_func=S_fluid_anat,
            enforce_initial_compatibility=True,
        )
        q_active_anat.append(float(tr.max_darcy_velocity))
        return obs_op_anat_q1.apply(problem_anat, tr)

    audit_anat_active = compute_normalized_jacobian_with_audit(
        forward_obs_callable=G_anat_active,
        theta_ref=theta_dagger,
        transform=transform,
        eps_step=eps_step,
        weighting_spec=weighting_spec,
    )
    svd_anat_active = analyze_jacobian_svd(
        audit_anat_active.jacobian_weighted, rank_threshold=rank_threshold, weighting_spec=weighting_spec
    )

    col_norms_zero = np.linalg.norm(audit_anat_zero.jacobian_weighted, axis=0)
    col_norms_active = np.linalg.norm(audit_anat_active.jacobian_weighted, axis=0)

    return {
        "mesh_file": str(msh_path),
        "num_quad_cells": int(problem_anat.num_active_cells),
        "lesion_center_adim": lesion_center_adim.tolist(),
        "sensor_grid_specification": {
            "nx": 6,
            "ny": 6,
            "num_sensors": 36,
            "half_span_adim": 2.5,
            "half_span_mm": 25.0,
            "observation_times_adim": obs_times_anat,
            "total_observations_N_obs": 180,
            "q1_linear_reproduction_error": q1_reprod_anat["max_abs_error_linear_field"],
        },
        "zero_flow": {
            "max_darcy_velocity_across_perturbations": float(max(q_zero_anat)),
            "numerical_rank": int(svd_anat_zero.numerical_rank),
            "singular_values": [float(x) for x in svd_anat_zero.singular_values],
            "column_norms": {
                PARAMETER_NAMES[j]: float(col_norms_zero[j]) for j in range(4)
            },
        },
        "active_flow": {
            "max_darcy_velocity_at_theta_ref": float(q_active_anat[0]),
            "numerical_rank": int(svd_anat_active.numerical_rank),
            "singular_values": [float(x) for x in svd_anat_active.singular_values],
            "condition_number_4x4": _get_cond_4x4(svd_anat_active),
            "column_norms": {
                PARAMETER_NAMES[j]: float(col_norms_active[j]) for j in range(4)
            },
            "least_informative_right_singular_vector_v4": [
                float(x) for x in svd_anat_active.right_singular_vectors_V[:, -1]
            ],
        },
        "sensor_points_adim": sensor_pts_anat.tolist(),
    }


# =============================================================================
# 5. Generazione Grafico Pubblicabile (Inglese, zero sigle interne)
# =============================================================================
def plot_flow_transition_and_anatomical_svd(
    sweep_records: Sequence[Dict[str, Any]],
    anat_res: Dict[str, Any],
    out_base: Path,
) -> Tuple[Path, Path]:
    positive_recs = [r for r in sweep_records if r["kappa_q"] > 0.0]
    pe_vals = np.array([r["peclet_number"] for r in positive_recs], dtype=np.float64)
    s1 = np.array([r["sigma_1"] for r in positive_recs], dtype=np.float64)
    s2 = np.array([r["sigma_2"] for r in positive_recs], dtype=np.float64)
    s3 = np.array([r["sigma_3"] for r in positive_recs], dtype=np.float64)
    s4 = np.array([r["sigma_4"] for r in positive_recs], dtype=np.float64)

    n_a = np.array([r["col_norm_alpha"] for r in positive_recs], dtype=np.float64)
    n_b = np.array([r["col_norm_beta"] for r in positive_recs], dtype=np.float64)
    n_d = np.array([r["col_norm_D"] for r in positive_recs], dtype=np.float64)
    n_e = np.array([r["col_norm_E"] for r in positive_recs], dtype=np.float64)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3), dpi=300)
    ax_col, ax_sv, ax_anat = axes

    ax_col.loglog(pe_vals, n_d, "o-", label=r"$\|\mathbf{J}_{:,\tilde{D}}\|_2$", linewidth=1.7)
    ax_col.loglog(pe_vals, n_a, "s-", label=r"$\|\mathbf{J}_{:,\tilde{\alpha}}\|_2$", linewidth=1.7)
    ax_col.loglog(pe_vals, n_e, "^-", label=r"$\|\mathbf{J}_{:,\tilde{E}}\|_2$ (mechanical)", linewidth=1.7)
    ax_col.loglog(pe_vals, n_b, "d-", label=r"$\|\mathbf{J}_{:,\tilde{\beta}}\|_2$ (viscoelastic)", linewidth=1.7)
    ax_col.set_title("(a) Parameter Sensitivity vs. Péclet Number")
    ax_col.set_xlabel(r"Global Péclet Number $\mathrm{Pe} = \|q_h\|_{\infty} L_0 / (\phi_0 D)$")
    ax_col.set_ylabel(r"Jacobian Column Norm $\|\mathbf{J}_{:,j}\|_2$")
    ax_col.grid(True, which="both", linestyle=":", alpha=0.55)
    ax_col.legend(loc="lower right", frameon=True)

    ax_sv.loglog(pe_vals, s1, "o-", label=r"$\sigma_1$", linewidth=1.7)
    ax_sv.loglog(pe_vals, s2, "s-", label=r"$\sigma_2$", linewidth=1.7)
    ax_sv.loglog(pe_vals, s3, "^-", label=r"$\sigma_3$ (emerges for $q \neq 0$)", linewidth=1.7)
    ax_sv.loglog(pe_vals, s4, "d-", label=r"$\sigma_4$ ($\beta$-dominated mode)", linewidth=1.7)
    ax_sv.set_title("(b) Singular Value Spectrum vs. Péclet Number")
    ax_sv.set_xlabel(r"Global Péclet Number $\mathrm{Pe}$")
    ax_sv.set_ylabel(r"Singular Value $\sigma_k(\mathbf{J})$")
    ax_sv.grid(True, which="both", linestyle=":", alpha=0.55)
    ax_sv.legend(loc="lower right", frameon=True)

    params = [r"$\tilde{\alpha}$", r"$\tilde{\beta}$", r"$\tilde{D}$", r"$\tilde{E}$"]
    x_pos = np.arange(4)
    width = 0.36
    c_zero = [max(anat_res["zero_flow"]["column_norms"][p], 1.0e-16) for p in PARAMETER_NAMES]
    c_act = [max(anat_res["active_flow"]["column_norms"][p], 1.0e-16) for p in PARAMETER_NAMES]
    cond_anat = anat_res["active_flow"]["condition_number_4x4"]
    ax_anat.bar(
        x_pos - 0.5 * width,
        c_zero,
        width=width,
        color="#d95f02",
        alpha=0.85,
        edgecolor="black",
        label=r"BraTS Zero Flow ($q \equiv 0$, rank 2)",
    )
    ax_anat.bar(
        x_pos + 0.5 * width,
        c_act,
        width=width,
        color="#1b9e77",
        alpha=0.85,
        edgecolor="black",
        label=f"BraTS Active Flow (rank 4, $\\kappa={cond_anat:.1f}$)",
    )
    ax_anat.set_yscale("log")
    ax_anat.set_ylim(1.0e-17, 1.0e1)
    ax_anat.set_xticks(x_pos)
    ax_anat.set_xticklabels(params)
    ax_anat.set_title("(c) Anatomical Proxy (BraTS 2D) Sensitivity")
    ax_anat.set_xlabel("Normalized Parameter")
    ax_anat.set_ylabel(r"Column Norm $\|\mathbf{J}_{:,j}^{\mathrm{BraTS}}\|_2$ (log scale)")
    ax_anat.grid(True, which="both", linestyle=":", alpha=0.55)
    ax_anat.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    png_path = out_base.with_suffix(".png")
    pdf_path = out_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main() -> int:
    out_dir = Path("artifacts/13_reviewer_identifiability_extensions")
    out_dir.mkdir(parents=True, exist_ok=True)

    master = load_master_config("configs/scales_and_physics.yaml")
    with open("configs/twin_experiment_synthetic.yaml", "r", encoding="utf-8") as fh:
        twin_cfg = yaml.safe_load(fh)

    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_dagger = master.parameter_space_theta.theta_dagger_array()

    print("[1/3] Esecuzione Audit Q1 Esatto vs Griddata (Target N=32 e Jacobiano N=16)...")
    q1_audit = run_q1_vs_griddata_audit(master, twin_cfg, transform, theta_dagger)

    print("[2/3] Esecuzione Sweep di Transizione del Flusso kappa_q / Péclet...")
    sweep_records = run_flow_transition_peclet_sweep(master, twin_cfg, transform, theta_dagger)

    sweep_csv_path = out_dir / "flow_transition_peclet_sweep.csv"
    with open(sweep_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_records[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_records)

    print("[3/3] Esecuzione Identificabilità Locale SVD sul Dominio Anatomico BraTS (3.0 mm)...")
    anat_ident = run_anatomical_brats_identifiability(master, transform, theta_dagger)

    fig_png, fig_pdf = plot_flow_transition_and_anatomical_svd(
        sweep_records,
        anat_ident,
        out_dir / "fig_flow_transition_and_anatomical_identifiability",
    )

    full_report = sanitize_for_json(
        {
            "q1_vs_griddata_observation_operator_audit": q1_audit,
            "flow_transition_peclet_sweep": sweep_records,
            "anatomical_brats_identifiability": anat_ident,
            "generated_files": {
                "sweep_csv": str(sweep_csv_path),
                "figure_png": str(fig_png),
                "figure_pdf": str(fig_pdf),
            },
        }
    )
    report_path = out_dir / "reviewer_identifiability_extensions_summary.json"
    report_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")

    print("\n=== RIEPILOGO VERIFICHE BLOCCO 1 ===")
    print(json.dumps(full_report["q1_vs_griddata_observation_operator_audit"], indent=2))
    print(
        json.dumps(
            {
                "brats_zero_flow": full_report["anatomical_brats_identifiability"]["zero_flow"],
                "brats_active_flow": full_report["anatomical_brats_identifiability"]["active_flow"],
            },
            indent=2,
        )
    )
    print(f"\nArtefatti salvati in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())