#!/usr/bin/env python3
"""
experiments/02_run_identifiability_study.py
Studio scientifico rigoroso di identificabilità locale e pratica per RETECO v1:
- Verifica quantitativa esplicita del protocollo a flusso nullo (q == 0 lungo la traiettoria
  e attraverso le perturbazioni parametriche);
- Audit del Jacobiano normalizzato, della pesatura osservativa e della stabilità rispetto
  al passo delle differenze finite;
- Analisi SVD completa con gestione rigorosa del rango deficiente;
- Sezioni unidimensionali del funzionale di costo (One-at-a-Time Cost Slices);
- Profili ottimizzati dell'obiettivo (attivabili tramite `--profiles`) con budget controllato;
- Esportazione dei Jacobiani (.npz, .csv), del report JSON completo e delle Figure A, B, C, D
  in doppio formato vettoriale PDF e PNG a 300 dpi.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT_DIR / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from reteco import _core, load_master_config
from reteco.config_schema import ParameterCoordinateTransform
from reteco.forward.target_generator import build_standard_driving_Callbacks
from reteco.forward.wrapper import RetecoForwardProblem2D
from reteco.identifiability.profile_likelihood import (
    build_misfit_objective_from_observations,
    compute_one_at_a_time_cost_slices,
    compute_optimized_objective_profiles,
)
from reteco.identifiability.sensitivity_jacobian import (
    PARAMETER_NAMES,
    ObservationWeightingSpec,
    compute_normalized_jacobian_with_audit,
    verify_fd_step_stability,
)
from reteco.identifiability.svd_conditioning import (
    analyze_jacobian_svd,
    sanitize_for_json,
)
from reteco.observation.operator import DiscreteObservationOperator
from reteco.visualization.identifiability_plots import (
    plot_figure_a_sensitivity_and_svd,
    plot_figure_b_local_correlation,
    plot_figure_c_cost_slices,
    plot_figure_d_optimized_profiles,
)


def _save_jacobian_csv(path: Path, J: np.ndarray) -> None:
    header = ",".join(PARAMETER_NAMES)
    np.savetxt(path, J, delimiter=",", header=header, comments="", fmt="%.12e")


def run_identifiability_study(
    scales_config_path: str | Path = "configs/scales_and_physics.yaml",
    twin_config_path: str | Path = "configs/twin_experiment_synthetic.yaml",
    output_dir: str | Path = "artifacts/02_identifiability",
    eps_step: float = 1.0e-4,
    rank_threshold: float = 1.0e-8,
    run_fd_stability_audit: bool = True,
    fd_audit_steps: Sequence[float] = (1.0e-3, 1.0e-4, 1.0e-5),
    slice_points: int = 21,
    run_optimized_profiles: bool = False,
    profile_points: int = 9,
    max_evals_per_profile_point: int = 25,
    max_total_profile_evals: int = 900,
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    master = load_master_config(scales_config_path)
    with Path(twin_config_path).open("r", encoding="utf-8") as fh:
        twin_cfg = yaml.safe_load(fh)

    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_dagger = master.parameter_space_theta.theta_dagger_array()
    theta_tilde_dagger = transform.to_normalized(theta_dagger)

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
        fe_deg_disp=int(inv_cfg.get("fe_deg_disp", 2)),
        fe_deg_press=int(inv_cfg.get("fe_deg_press", 1)),
        fe_deg_tracer=int(inv_cfg.get("fe_deg_tracer", 1)),
    )

    c0, F_mech_func, S_fluid_func = build_standard_driving_Callbacks(
        problem, twin_cfg["driving_protocol"]
    )
    obs_cfg = twin_cfg["observation_protocol"]
    observed_fields = tuple(str(f) for f in obs_cfg.get("observed_fields", ["c"]))
    obs_op = DiscreteObservationOperator.from_regular_grid(
        nx=int(obs_cfg["sensor_grid_nx"]),
        ny=int(obs_cfg["sensor_grid_ny"]),
        margin=float(obs_cfg["sensor_margin"]),
        observation_times=obs_cfg["observation_times"],
        observed_fields=list(observed_fields),
    )

    # Specifica di pesatura: le osservazioni di c_h sono già adimensionali (c_phys / c_ref).
    # Non inventiamo un modello di rumore per l'analisi deterministica di identificabilità.
    weighting_spec = ObservationWeightingSpec(
        weighting_mode="unweighted_dimensionless_state_identity",
        weights_sqrt_diag=None,
        is_statistically_calibrated=False,
        observed_fields=observed_fields,
        provenance_description=(
            "Observations produced by DiscreteObservationOperator are already in dimensionless "
            "RETECO units (c / c_ref, p / E_0, d / L_0). Since observed_fields="
            f"{list(observed_fields)} share a homogeneous dimensionless scale and no empirical "
            "noise covariance is assumed at the baseline identifiability stage, W^{1/2} = I is used."
        ),
    )

    # =========================================================================
    # 1. Protocollo a Flusso Nullo (q == 0) con verifica quantitativa esplicita
    # =========================================================================
    zero_flow_vel_records: List[float] = []
    zero_flow_p_l2_records: List[float] = []
    zero_flow_d_l2_records: List[float] = []
    zero_flow_compat_records: List[float] = []

    def G_zero_flow(theta: np.ndarray) -> np.ndarray:
        traj = problem.solve_trajectory(
            theta_vec=theta,
            c0=c0,
            F_mech_func=None,
            S_fluid_func=None,
        )
        zero_flow_vel_records.append(float(traj.max_darcy_velocity))
        zero_flow_compat_records.append(float(traj.initial_compatibility_residual))
        max_p_l2 = max(problem.compute_press_L2_norm(traj.p_hist[n]) for n in range(len(traj.times)))
        max_d_l2 = max(problem.compute_disp_L2_norm(traj.d_hist[n]) for n in range(len(traj.times)))
        zero_flow_p_l2_records.append(float(max_p_l2))
        zero_flow_d_l2_records.append(float(max_d_l2))
        return obs_op.apply(problem, traj)

    audit_zero = compute_normalized_jacobian_with_audit(
        forward_obs_callable=G_zero_flow,
        theta_ref=theta_dagger,
        transform=transform,
        eps_step=eps_step,
        weighting_spec=weighting_spec,
    )
    rep_zero = analyze_jacobian_svd(
        audit_zero.jacobian_weighted,
        rank_threshold=rank_threshold,
        weighting_spec=weighting_spec,
    )

    zero_flow_tol = 1.0e-14
    max_q_zero_ref = float(zero_flow_vel_records[0])
    max_q_zero_all = float(max(zero_flow_vel_records))
    zero_flow_verified = bool(max_q_zero_all <= zero_flow_tol)

    zero_flow_verification = {
        "verification_method": (
            "Quantitative evaluation of Darcy velocity q_h = -K_h grad(p_h), pore pressure "
            "||p_h||_{L^2(Omega)}, and displacement ||d_h||_{L^2(Omega)} across all time steps "
            "and across all finite-difference parameter perturbations (not a nominal assumption)."
        ),
        "numeric_tolerance": zero_flow_tol,
        "tolerance_justification": (
            "IEEE 754 double-precision machine epsilon (~2.22e-16) multiplied by sparse direct "
            "LU solver conditioning factor."
        ),
        "max_darcy_velocity_at_theta_ref": max_q_zero_ref,
        "max_darcy_velocity_across_all_jacobian_perturbations": max_q_zero_all,
        "max_pressure_l2_norm_across_perturbations": float(max(zero_flow_p_l2_records)),
        "max_displacement_l2_norm_across_perturbations": float(max(zero_flow_d_l2_records)),
        "max_initial_compatibility_residual": float(max(zero_flow_compat_records)),
        "num_trajectories_checked": len(zero_flow_vel_records),
        "zero_flow_verified": zero_flow_verified,
    }

    # =========================================================================
    # 2. Protocollo a Flusso Attivo (q != 0)
    # =========================================================================
    active_flow_vel_records: List[float] = []

    def G_active_flow(theta: np.ndarray) -> np.ndarray:
        traj = problem.solve_trajectory(
            theta_vec=theta,
            c0=c0,
            F_mech_func=F_mech_func,
            S_fluid_func=S_fluid_func,
        )
        active_flow_vel_records.append(float(traj.max_darcy_velocity))
        return obs_op.apply(problem, traj)

    audit_active = compute_normalized_jacobian_with_audit(
        forward_obs_callable=G_active_flow,
        theta_ref=theta_dagger,
        transform=transform,
        eps_step=eps_step,
        weighting_spec=weighting_spec,
    )
    rep_active = analyze_jacobian_svd(
        audit_active.jacobian_weighted,
        rank_threshold=rank_threshold,
        weighting_spec=weighting_spec,
    )

    # =========================================================================
    # 3. Verifica di stabilità rispetto al passo delle differenze finite
    # =========================================================================
    fd_stability_report = None
    if run_fd_stability_audit:
        fd_stability_report = verify_fd_step_stability(
            forward_obs_callable=G_active_flow,
            theta_ref=theta_dagger,
            transform=transform,
            steps=fd_audit_steps,
            reference_step=eps_step,
            weighting_spec=weighting_spec,
        )

    # =========================================================================
    # 4. Sezioni 1D del funzionale di costo (One-at-a-Time Cost Slices)
    # =========================================================================
    obj_active = build_misfit_objective_from_observations(
        G_active_flow, audit_active.baseline_observation, weighting_spec
    )
    obj_zero = build_misfit_objective_from_observations(
        G_zero_flow, audit_zero.baseline_observation, weighting_spec
    )

    slices_active = compute_one_at_a_time_cost_slices(
        objective_callable=obj_active,
        theta_ref=theta_dagger,
        transform=transform,
        num_points=slice_points,
    )
    slices_zero = compute_one_at_a_time_cost_slices(
        objective_callable=obj_zero,
        theta_ref=theta_dagger,
        transform=transform,
        num_points=slice_points,
    )

    # =========================================================================
    # 5. Profili dell'obiettivo ottimizzati (opzionali tramite --profiles)
    # =========================================================================
    optimized_profiles_report = None
    if run_optimized_profiles:
        optimized_profiles_report = compute_optimized_objective_profiles(
            objective_callable=obj_active,
            theta_ref=theta_dagger,
            transform=transform,
            num_grid_points=profile_points,
            max_evals_per_point=max_evals_per_profile_point,
            max_total_evaluations=max_total_profile_evals,
            optimizer_method="L-BFGS-B",
            is_statistical_profile_likelihood=weighting_spec.is_statistically_calibrated,
        )

    # =========================================================================
    # 6. Salvataggio Jacobiani (.csv, .npz) e Generazione Figure A, B, C, D
    # =========================================================================
    csv_zero_raw = out_dir / "jacobian_zero_flow_raw.csv"
    csv_active_raw = out_dir / "jacobian_active_flow_raw.csv"
    csv_active_weighted = out_dir / "jacobian_active_flow_weighted.csv"
    npz_bundle = out_dir / "jacobians_and_slices.npz"

    _save_jacobian_csv(csv_zero_raw, audit_zero.jacobian_raw)
    _save_jacobian_csv(csv_active_raw, audit_active.jacobian_raw)
    _save_jacobian_csv(csv_active_weighted, audit_active.jacobian_weighted)

    npz_payload: Dict[str, Any] = {
        "parameter_names": np.array(PARAMETER_NAMES),
        "theta_dagger_physical": theta_dagger,
        "theta_dagger_normalized": theta_tilde_dagger,
        "y0_zero_flow": audit_zero.baseline_observation,
        "y0_active_flow": audit_active.baseline_observation,
        "J_zero_raw": audit_zero.jacobian_raw,
        "J_zero_weighted": audit_zero.jacobian_weighted,
        "J_active_raw": audit_active.jacobian_raw,
        "J_active_weighted": audit_active.jacobian_weighted,
        "singular_values_zero": rep_zero.singular_values,
        "singular_values_active": rep_active.singular_values,
        "V_active": rep_active.right_singular_vectors_V,
    }
    for pname in PARAMETER_NAMES:
        npz_payload[f"oat_slice_{pname}_grid_phys"] = slices_active.slices[pname].grid_physical
        npz_payload[f"oat_slice_{pname}_cost_active"] = slices_active.slices[pname].cost_values
        npz_payload[f"oat_slice_{pname}_cost_zero"] = slices_zero.slices[pname].cost_values
        if optimized_profiles_report is not None:
            pr = optimized_profiles_report.profiles[pname]
            npz_payload[f"opt_profile_{pname}_grid_phys"] = pr.grid_physical
            npz_payload[f"opt_profile_{pname}_cost"] = pr.profile_cost_values
            npz_payload[f"opt_profile_{pname}_thetas_phys"] = pr.optimized_thetas_physical

    np.savez_compressed(npz_bundle, **npz_payload)

    fig_a_files = plot_figure_a_sensitivity_and_svd(
        rep_active, rep_zero, out_dir / "figA_sensitivity_and_svd.pdf"
    )
    # Manteniamo anche il nome legacy per eventuali script/test esterni
    plot_figure_a_sensitivity_and_svd(
        rep_active, rep_zero, out_dir / "fig_svd_identifiability.pdf"
    )
    fig_b_files = plot_figure_b_local_correlation(
        rep_active, rep_zero, out_dir / "figB_local_correlation.pdf"
    )
    fig_c_files = plot_figure_c_cost_slices(
        slices_active, slices_zero, transform, out_dir / "figC_one_at_a_time_cost_slices.pdf"
    )

    fig_d_files: Optional[Dict[str, str]] = None
    if optimized_profiles_report is not None:
        fig_d_files = plot_figure_d_optimized_profiles(
            optimized_profiles_report, out_dir / "figD_optimized_objective_profiles.pdf"
        )

    # =========================================================================
    # 7. Costruzione e serializzazione di `identifiability_report.json`
    # =========================================================================
    n_sensors = int(obs_cfg["sensor_grid_nx"]) * int(obs_cfg["sensor_grid_ny"])
    n_times = len(obs_cfg["observation_times"])
    n_obs = int(audit_active.baseline_observation.size)

    report_json = sanitize_for_json(
        {
            "study_id": "RETECO-v1-Identifiability-Study",
            "epistemic_and_methodological_notes": {
                "local_vs_global_identifiability": (
                    "Full column rank (rank = 4) of the sensitivity Jacobian at theta_dagger "
                    "demonstrates local first-order regular identifiability in a neighborhood "
                    "of the reference parameter, NOT global structural identifiability across Theta."
                ),
                "geometric_vs_statistical_correlation": (
                    "Without an empirically calibrated noise covariance matrix Sigma_eps, "
                    "the matrix R derived from (J^T J)^{-1} is a local geometric sensitivity "
                    "correlation diagnostic and must not be interpreted as calibrated statistical "
                    "estimator correlation or confidence intervals."
                ),
                "slices_vs_profiles": (
                    "1D sections with fixed nuisance parameters are reported as One-at-a-Time (OAT) "
                    "Cost Slices. Constrained nuisance-optimized curves (when --profiles is enabled) "
                    "are reported as Optimized Objective Profiles."
                ),
            },
            "reference_configuration": {
                "scales_config_path": str(scales_config_path),
                "twin_config_path": str(twin_config_path),
                "parameter_names": list(PARAMETER_NAMES),
                "theta_dagger_physical": {
                    PARAMETER_NAMES[j]: float(theta_dagger[j]) for j in range(4)
                },
                "theta_dagger_normalized": {
                    PARAMETER_NAMES[j]: float(theta_tilde_dagger[j]) for j in range(4)
                },
                "parameter_bounds": master.parameter_space_theta.bounds.model_dump(),
                "discretization": inv_cfg,
            },
            "observation_structure": {
                "observed_fields": list(observed_fields),
                "sensor_grid_nx": int(obs_cfg["sensor_grid_nx"]),
                "sensor_grid_ny": int(obs_cfg["sensor_grid_ny"]),
                "num_spatial_sensors": n_sensors,
                "observation_times_adim": [float(t) for t in obs_cfg["observation_times"]],
                "num_observation_times": n_times,
                "total_observations_N_obs": n_obs,
                "state_units": "dimensionless_RETECO_units (c/c_ref, p/E_0, d/L_0)",
            },
            "jacobian_and_normalization_audit": {
                "zero_flow": audit_zero.to_metadata_dict(),
                "active_flow": audit_active.to_metadata_dict(),
            },
            "zero_flow_protocol_verification": zero_flow_verification,
            "zero_flow_proposition_check": rep_zero.to_json_dict(),
            "active_flow_protocol": {
                "max_darcy_velocity_at_theta_ref": float(active_flow_vel_records[0]),
                "max_darcy_velocity_across_perturbations": float(max(active_flow_vel_records)),
                **rep_active.to_json_dict(),
            },
            "fd_step_stability_audit": (
                fd_stability_report.to_metadata_dict() if fd_stability_report is not None else None
            ),
            "one_at_a_time_cost_slices": {
                "active_flow": slices_active.to_json_dict(),
                "zero_flow": slices_zero.to_json_dict(),
            },
            "optimized_objective_profiles": (
                optimized_profiles_report.to_json_dict()
                if optimized_profiles_report is not None
                else {
                    "executed": False,
                    "note": "Run with `--profiles` CLI flag to compute constrained optimized objective profiles.",
                }
            ),
            "generated_artifacts": {
                "report_json": str(out_dir / "identifiability_report.json"),
                "jacobians_and_slices_npz": str(npz_bundle),
                "jacobian_zero_flow_raw_csv": str(csv_zero_raw),
                "jacobian_active_flow_raw_csv": str(csv_active_raw),
                "jacobian_active_flow_weighted_csv": str(csv_active_weighted),
                "figure_A_sensitivity_and_svd": fig_a_files,
                "figure_B_local_correlation": fig_b_files,
                "figure_C_one_at_a_time_cost_slices": fig_c_files,
                "figure_D_optimized_objective_profiles": fig_d_files,
            },
        }
    )

    report_path = out_dir / "identifiability_report.json"
    report_path.write_text(
        json.dumps(report_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report_json


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="RETECO v1 — Scientific Identifiability Study (Local SVD, Cost Slices & Optimized Profiles)"
    )
    parser.add_argument(
        "--scales-config",
        type=str,
        default="configs/scales_and_physics.yaml",
        help="Percorso alla configurazione fisica e delle scale.",
    )
    parser.add_argument(
        "--twin-config",
        type=str,
        default="configs/twin_experiment_synthetic.yaml",
        help="Percorso alla configurazione del twin experiment.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/02_identifiability",
        help="Directory di output per report, CSV/NPZ e figure.",
    )
    parser.add_argument(
        "--eps-step",
        type=float,
        default=1.0e-4,
        help="Passo h delle differenze finite nelle coordinate normalizzate [0, 1]^4.",
    )
    parser.add_argument(
        "--rank-threshold",
        type=float,
        default=1.0e-8,
        help="Soglia relativa tau_rel per il rango numerico SVD (sigma_k > tau_rel * sigma_1).",
    )
    parser.add_argument(
        "--no-fd-audit",
        action="store_true",
        help="Disabilita la verifica di stabilità rispetto al passo delle differenze finite.",
    )
    parser.add_argument(
        "--slice-points",
        type=int,
        default=21,
        help="Numero di punti per parametro nelle sezioni 1D (One-at-a-Time Cost Slices).",
    )
    parser.add_argument(
        "--profiles",
        action="store_true",
        help="Esegue anche il calcolo dei profili dell'obiettivo ottimizzati (Figura D).",
    )
    parser.add_argument(
        "--profile-points",
        type=int,
        default=9,
        help="Numero di punti sulla griglia di ciascun parametro per i profili ottimizzati.",
    )
    parser.add_argument(
        "--max-evals-per-point",
        type=int,
        default=25,
        help="Numero massimo di valutazioni FEM per singolo punto del profilo ottimizzato.",
    )
    parser.add_argument(
        "--max-total-profile-evals",
        type=int,
        default=900,
        help="Budget massimo complessivo di valutazioni FEM per l'analisi dei profili ottimizzati.",
    )
    args = parser.parse_args(argv)

    report = run_identifiability_study(
        scales_config_path=args.scales_config,
        twin_config_path=args.twin_config,
        output_dir=args.output_dir,
        eps_step=args.eps_step,
        rank_threshold=args.rank_threshold,
        run_fd_stability_audit=(not args.no_fd_audit),
        slice_points=args.slice_points,
        run_optimized_profiles=bool(args.profiles),
        profile_points=args.profile_points,
        max_evals_per_profile_point=args.max_evals_per_point,
        max_total_profile_evals=args.max_total_profile_evals,
    )

    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "zero_flow_verified": report["zero_flow_protocol_verification"]["zero_flow_verified"],
                "zero_flow_rank": report["zero_flow_proposition_check"]["numerical_rank"],
                "active_flow_rank": report["active_flow_protocol"]["numerical_rank"],
                "active_flow_condition_number": report["active_flow_protocol"]["condition_number_full_4x4"],
                "active_flow_collinearity_index": report["active_flow_protocol"]["collinearity_index_full_4x4"],
                "profiles_executed": bool(args.profiles),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())