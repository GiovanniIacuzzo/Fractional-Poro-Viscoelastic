"""
experiments/03b_validate_twin_forcing.py
Fase 1 — Verifica numerica del problema diretto con forzanti corrette e controllo
diagnostico interno same-mesh (senza esecuzione di ottimizzazioni CPSO).

Salva tutti gli artefatti esclusivamente in:
    artifacts/03b_twin_forcing_validation/
"""

from __future__ import annotations
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import yaml

from reteco import _core, load_master_config
from reteco.config_schema import ParameterCoordinateTransform
from reteco.forward.wrapper import RetecoForwardProblem2D
from reteco.forward.target_generator import (
    FORCING_PROTOCOL_VERSION,
    build_displacement_mass_matrix_csr,
    build_scalar_lagrange_stiffness_matrix_csr,
    build_standard_driving_Callbacks,
    compute_spatial_forcing_vectors_and_diagnostics,
)
from reteco.observation.operator import DiscreteObservationOperator
from reteco.observation.noise_model import generate_gaussian_noisy_observations
from reteco.inverse.regularization import TikhonovRegularizer
from reteco.inverse.objective_functional import InverseObjectiveFunctional


DISCRETIZATION_LEVELS: Tuple[Tuple[int, int], ...] = (
    (16, 80),
    (24, 120),
    (32, 160),
)


def _extract_history_list(traj: Any, candidates: Tuple[str, ...]) -> Optional[List[np.ndarray]]:
    for attr in candidates:
        if hasattr(traj, attr):
            val = getattr(traj, attr)
            if val is not None and len(val) > 0:
                return [np.asarray(v, dtype=np.float64).ravel() for v in val]
    return None


def _compute_field_norms_from_trajectory(
    problem: RetecoForwardProblem2D,
    traj: Any,
    K_h_coeff: float,
) -> Dict[str, Any]:
    """
    Calcola le norme L2(Omega) ed energetiche dei campi esposti da `traj` e `problem`.
    """
    M_d = build_displacement_mass_matrix_csr(problem)
    K_0 = problem.K0_csr
    M_p = problem.M_p_csr
    M_c = problem.M_c_csr

    n_sub = int(problem.subdivisions_N)
    K_lap_p = build_scalar_lagrange_stiffness_matrix_csr(
        problem.press_pts, subdivisions_N=n_sub, fe_degree=1
    )
    K_lap_c = build_scalar_lagrange_stiffness_matrix_csr(
        problem.tracer_pts, subdivisions_N=n_sub, fe_degree=1
    )

    d_hist = _extract_history_list(traj, ("d_history", "d_hist", "displacement_history", "d"))
    p_hist = _extract_history_list(traj, ("p_history", "p_hist", "pressure_history", "p"))
    c_hist = _extract_history_list(traj, ("c_history", "c_hist", "tracer_history", "c"))

    out: Dict[str, Any] = {
        "max_darcy_velocity_pointwise": float(getattr(traj, "max_darcy_velocity", np.nan)),
        "initial_compatibility_residual": float(
            getattr(traj, "initial_compatibility_residual", 0.0)
        ),
    }

    if d_hist is not None:
        d_l2_all = [float(np.sqrt(max(0.0, float(d @ (M_d @ d))))) for d in d_hist]
        d_k0_all = [float(np.sqrt(max(0.0, float(d @ (K_0 @ d))))) for d in d_hist]
        out["d_L2_final"] = d_l2_all[-1]
        out["d_L2_max_over_time"] = float(np.max(d_l2_all))
        out["d_strain_energy_seminorm_final"] = d_k0_all[-1]
        out["d_strain_energy_seminorm_max"] = float(np.max(d_k0_all))
    else:
        out["d_L2_final"] = None
        out["d_L2_max_over_time"] = None
        out["d_strain_energy_seminorm_final"] = None
        out["d_strain_energy_seminorm_max"] = None

    if p_hist is not None:
        p_l2_all = [float(np.sqrt(max(0.0, float(p @ (M_p @ p))))) for p in p_hist]
        # q_h = -K_h grad(p_h) => ||q_h||_{L2(Omega)} = K_h * sqrt(p^T K_lap_p p)
        q_l2_all = [
            float(K_h_coeff * np.sqrt(max(0.0, float(p @ (K_lap_p @ p)))))
            for p in p_hist
        ]
        out["p_L2_final"] = p_l2_all[-1]
        out["p_L2_max_over_time"] = float(np.max(p_l2_all))
        out["q_darcy_L2_final"] = q_l2_all[-1]
        out["q_darcy_L2_max_over_time"] = float(np.max(q_l2_all))
    else:
        out["p_L2_final"] = None
        out["p_L2_max_over_time"] = None
        out["q_darcy_L2_final"] = None
        out["q_darcy_L2_max_over_time"] = None

    if c_hist is not None:
        c_l2_all = [float(np.sqrt(max(0.0, float(c @ (M_c @ c))))) for c in c_hist]
        c_h1_semi_all = [float(np.sqrt(max(0.0, float(c @ (K_lap_c @ c))))) for c in c_hist]
        out["c_L2_final"] = c_l2_all[-1]
        out["c_L2_max_over_time"] = float(np.max(c_l2_all))
        out["c_H1_seminorm_final"] = c_h1_semi_all[-1]
    else:
        out["c_L2_final"] = None
        out["c_L2_max_over_time"] = None
        out["c_H1_seminorm_final"] = None

    diags = getattr(traj, "diagnostics", None)
    if diags:
        masses = np.array([float(d.tracer_total_mass) for d in diags], dtype=np.float64)
        el_energies = np.array([float(d.elastic_energy) for d in diags], dtype=np.float64)
        fl_energies = np.array([float(d.fluid_storage_energy) for d in diags], dtype=np.float64)
        out["tracer_mass_initial"] = float(masses[0])
        out["tracer_mass_final"] = float(masses[-1])
        out["tracer_mass_max_rel_variation"] = float(
            np.max(np.abs(masses - masses[0])) / max(abs(masses[0]), 1.0e-30)
        )
        out["elastic_energy_final"] = float(el_energies[-1])
        out["elastic_energy_max"] = float(np.max(el_energies))
        out["fluid_storage_energy_final"] = float(fl_energies[-1])
        out["fluid_storage_energy_max"] = float(np.max(fl_energies))
        out["cumulative_darcy_dissip_final"] = float(diags[-1].cumulative_darcy_dissip)
        out["cumulative_visco_dissip_final"] = float(diags[-1].cumulative_visco_dissip)

    return out


def main() -> None:
    out_dir = Path("artifacts/03b_twin_forcing_validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    master = load_master_config("configs/scales_and_physics.yaml")
    with open("configs/twin_experiment_synthetic.yaml", "r", encoding="utf-8") as f:
        twin_cfg = yaml.safe_load(f)

    theta_dagger = master.parameter_space_theta.theta_dagger_array()
    K_h_coeff = float(master.fixed_physics.K_h)

    tar_cfg = twin_cfg["target_discretization"]
    final_time_T = float(tar_cfg["final_time_T"])
    grading_r = float(tar_cfg["grading_r"])
    fe_deg_disp = int(tar_cfg.get("fe_deg_disp", 2))
    fe_deg_press = int(tar_cfg.get("fe_deg_press", 1))
    fe_deg_tracer = int(tar_cfg.get("fe_deg_tracer", 1))

    obs_cfg = twin_cfg["observation_protocol"]
    obs_op_c = DiscreteObservationOperator.from_regular_grid(
        nx=int(obs_cfg["sensor_grid_nx"]),
        ny=int(obs_cfg["sensor_grid_ny"]),
        margin=float(obs_cfg["sensor_margin"]),
        observation_times=obs_cfg["observation_times"],
        observed_fields=obs_cfg["observed_fields"],
    )

    log_lines: List[str] = [
        f"=== RETECO Fase 1: Log Contabilizzazione Simulazioni ({datetime.now(timezone.utc).isoformat()}) ===",
        f"Forcing protocol version: {FORCING_PROTOCOL_VERSION}",
        f"Theta_dagger: alpha={theta_dagger[0]}, beta={theta_dagger[1]}, D={theta_dagger[2]}, E={theta_dagger[3]}",
        "",
    ]

    level_results: Dict[str, Dict[str, Any]] = {}
    observations_by_level: Dict[str, np.ndarray] = {}
    problems_by_level: Dict[str, RetecoForwardProblem2D] = {}

    sim_counter = 0
    for N_sub, M_steps in DISCRETIZATION_LEVELS:
        level_key = f"N{N_sub}_M{M_steps}"
        t0_setup = time.perf_counter()
        grid = _core.TimeGrid.make_graded(final_time_T, M_steps, grading_r)
        prob = RetecoForwardProblem2D(
            subdivisions_N=N_sub,
            physics_model=master.fixed_physics,
            scales=master.characteristic_scales,
            time_grid=grid,
            fe_deg_disp=fe_deg_disp,
            fe_deg_press=fe_deg_press,
            fe_deg_tracer=fe_deg_tracer,
        )
        forcing_info = compute_spatial_forcing_vectors_and_diagnostics(
            prob, twin_cfg["driving_protocol"]
        )
        c0, F_mech_func, S_fluid_func = build_standard_driving_Callbacks(
            prob, twin_cfg["driving_protocol"]
        )
        setup_time = time.perf_counter() - t0_setup

        t0_solve = time.perf_counter()
        traj = prob.solve_trajectory(
            theta_vec=theta_dagger,
            c0=c0,
            F_mech_func=F_mech_func,
            S_fluid_func=S_fluid_func,
            compute_diagnostics=True,
        )
        y_obs = np.asarray(obs_op_c.apply(prob, traj), dtype=np.float64).ravel()
        solve_time = time.perf_counter() - t0_solve
        sim_counter += 1

        field_norms = _compute_field_norms_from_trajectory(prob, traj, K_h_coeff)

        times_arr = np.array([float(d.time) for d in traj.diagnostics], dtype=np.float64)
        sin_factors = np.sin(float(forcing_info["omega_t"]) * np.maximum(times_arr, 0.0))
        max_abs_sin = float(np.max(np.abs(sin_factors)))

        rec = {
            "level_key": level_key,
            "subdivisions_N": N_sub,
            "time_steps_M": M_steps,
            "final_time_T": final_time_T,
            "grading_r": grading_r,
            "num_disp_dofs": int(prob.num_disp_dofs),
            "num_press_dofs": int(prob.num_press_dofs),
            "num_tracer_dofs": int(prob.num_tracer_dofs),
            "num_observations": int(y_obs.size),
            "forcing_protocol_version": FORCING_PROTOCOL_VERSION,
            "fluid_source_spatial_integral": float(forcing_info["fluid_source_spatial_integral"]),
            "fluid_source_continuous_integral_ref": float(
                forcing_info["fluid_source_continuous_integral_ref"]
            ),
            "fluid_source_peak_abs_integral": float(
                abs(forcing_info["fluid_source_spatial_integral"]) * max_abs_sin
            ),
            "mech_resultant_x_spatial": float(forcing_info["mech_resultant_x_spatial"]),
            "mech_resultant_y_spatial": float(forcing_info["mech_resultant_y_spatial"]),
            "mech_resultant_x_continuous_ref": float(
                forcing_info["mech_resultant_x_continuous_ref"]
            ),
            "mech_resultant_y_continuous_ref": float(
                forcing_info["mech_resultant_y_continuous_ref"]
            ),
            "mech_interior_resultant_x": float(forcing_info["mech_interior_resultant_x"]),
            "mech_interior_resultant_y": float(forcing_info["mech_interior_resultant_y"]),
            "obs_L2_norm": float(np.linalg.norm(y_obs)),
            "obs_Linf_norm": float(np.max(np.abs(y_obs))),
            "setup_wall_time_s": float(setup_time),
            "solve_wall_time_s": float(solve_time),
            **field_norms,
        }

        level_results[level_key] = rec
        observations_by_level[level_key] = y_obs
        problems_by_level[level_key] = prob

        log_lines.append(
            f"[Sim #{sim_counter}] Forward evaluation {level_key}: "
            f"DOFs=(d:{prob.num_disp_dofs}, p:{prob.num_press_dofs}, c:{prob.num_tracer_dofs}), "
            f"max|q|={rec['max_darcy_velocity_pointwise']:.4e}, "
            f"||y_obs||_2={rec['obs_L2_norm']:.6e}, solve_time={solve_time:.3f}s"
        )
        print(log_lines[-1])

    ref_key = "N32_M160"
    y_ref = observations_by_level[ref_key]
    ref_rec = level_results[ref_key]

    for key, rec in level_results.items():
        y_cur = observations_by_level[key]
        rec["rel_discrepancy_obs_vs_N32_M160"] = float(
            np.linalg.norm(y_cur - y_ref) / np.linalg.norm(y_ref)
        )
        rec["max_abs_discrepancy_obs_vs_N32_M160"] = float(np.max(np.abs(y_cur - y_ref)))

        for metric_name in (
            "d_L2_final",
            "p_L2_final",
            "c_L2_final",
            "q_darcy_L2_max_over_time",
            "max_darcy_velocity_pointwise",
            "elastic_energy_max",
            "fluid_storage_energy_max",
        ):
            val_cur = rec.get(metric_name)
            val_ref = ref_rec.get(metric_name)
            if val_cur is not None and val_ref is not None and abs(val_ref) > 0.0:
                rec[f"rel_diff_{metric_name}_vs_N32_M160"] = float(
                    abs(val_cur - val_ref) / abs(val_ref)
                )
            else:
                rec[f"rel_diff_{metric_name}_vs_N32_M160"] = None

    # Controllo diagnostico interno same-mesh (N=16, M=80 -> N=16, M=80)
    prob_16 = problems_by_level["N16_M80"]
    c0_16, F_16, Sf_16 = build_standard_driving_Callbacks(
        prob_16, twin_cfg["driving_protocol"]
    )

    def G_inv_16(theta: np.ndarray) -> np.ndarray:
        tr = prob_16.solve_trajectory(
            theta, c0=c0_16, F_mech_func=F_16, S_fluid_func=Sf_16
        )
        return obs_op_c.apply(prob_16, tr)

    y_target_same_mesh = observations_by_level["N16_M80"].copy()
    noise_cfg = twin_cfg["noise_and_regularization"]
    base_seed = int(noise_cfg["base_seed"])
    lambda_reg = float(noise_cfg["regularization_lambda"])
    prior_theta_tilde_0 = np.array(noise_cfg["prior_theta_tilde_0"], dtype=np.float64)

    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    reg = TikhonovRegularizer(
        transform,
        lambda_reg=lambda_reg,
        theta_tilde_0=prior_theta_tilde_0,
    )

    sample_same_mesh = generate_gaussian_noisy_observations(
        y_clean=y_target_same_mesh,
        eta_level=0.0,
        seed=base_seed,
    )
    obj_same_mesh = InverseObjectiveFunctional(
        G_inv_16,
        sample_same_mesh.y_noisy,
        sample_same_mesh.weight_matrix_diag,
        transform,
        reg,
        max_evaluations=10,
    )

    t0_sm = time.perf_counter()
    j_total_same = float(obj_same_mesh(theta_dagger))
    sm_wall_time = time.perf_counter() - t0_sm
    sim_counter += 1

    t0_rec = time.perf_counter()
    y_reconstructed_same = np.asarray(G_inv_16(theta_dagger), dtype=np.float64).ravel()
    rec_wall_time = time.perf_counter() - t0_rec
    sim_counter += 1

    sm_hist_rec = obj_same_mesh.history[0]
    j_data_same = float(
        sm_hist_rec["j_data"] if isinstance(sm_hist_rec, dict) else sm_hist_rec.j_data
    )
    j_reg_same = float(
        sm_hist_rec["j_reg"] if isinstance(sm_hist_rec, dict) else sm_hist_rec.j_reg
    )

    rel_res_same = float(
        np.linalg.norm(y_reconstructed_same - y_target_same_mesh)
        / np.linalg.norm(y_target_same_mesh)
    )
    max_abs_diff_same = float(np.max(np.abs(y_reconstructed_same - y_target_same_mesh)))

    sample_twin_mesh = generate_gaussian_noisy_observations(
        y_clean=y_ref,
        eta_level=0.0,
        seed=base_seed,
    )
    w_diff_twin = sample_twin_mesh.weight_matrix_diag * (y_reconstructed_same - y_ref)
    j_data_twin_at_true = 0.5 * float(np.dot(w_diff_twin, w_diff_twin))
    j_reg_twin_at_true = float(reg.evaluate(theta_dagger))
    j_total_twin_at_true = j_data_twin_at_true + j_reg_twin_at_true

    same_mesh_control = {
        "control_type": "same_mesh_direct_evaluation_at_theta_dagger_no_cpso",
        "methodological_note": (
            "Controllo diagnostico interno di consistenza tra percorso forward e operatore "
            "osservativo su (N=16, M=80) a rumore nullo. Non costituisce prova di "
            "identificabilità globale o di unicità del minimo."
        ),
        "discretization": {"subdivisions_N": 16, "time_steps_M": 80},
        "theta_dagger": theta_dagger.tolist(),
        "eta_level": 0.0,
        "regularization_lambda": lambda_reg,
        "prior_theta_tilde_0": prior_theta_tilde_0.tolist(),
        "relative_observation_residual_at_true": rel_res_same,
        "max_abs_observation_difference_at_true": max_abs_diff_same,
        "J_data_at_theta_dagger_same_mesh": j_data_same,
        "J_reg_at_theta_dagger": j_reg_same,
        "J_total_at_theta_dagger_same_mesh": j_total_same,
        "twin_mesh_comparison_at_theta_dagger": {
            "target_discretization": "N32_M160",
            "inverse_discretization": "N16_M80",
            "relative_discretization_gap_L2": level_results["N16_M80"][
                "rel_discrepancy_obs_vs_N32_M160"
            ],
            "J_data_at_theta_dagger_twin_mesh": j_data_twin_at_true,
            "J_reg_at_theta_dagger_twin_mesh": j_reg_twin_at_true,
            "J_total_at_theta_dagger_twin_mesh": j_total_twin_at_true,
        },
        "wall_time_seconds": float(sm_wall_time + rec_wall_time),
    }

    log_lines.append(
        f"[Sim #{sim_counter - 1} & #{sim_counter}] Same-mesh internal control (N16_M80 at theta_dagger): "
        f"rel_res={rel_res_same:.3e}, J_data={j_data_same:.3e}, J_reg={j_reg_same:.6e}, J_total={j_total_same:.6e}"
    )
    log_lines.append(f"Total forward FEM evaluations executed: {sim_counter}")

    summary_payload = {
        "experiment_id": "RETECO-v1-PHASE1-TWIN-FORCING-VALIDATION-03B",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "forcing_protocol_version": FORCING_PROTOCOL_VERSION,
        "scales_and_physics_config": "configs/scales_and_physics.yaml",
        "twin_experiment_config": "configs/twin_experiment_synthetic.yaml",
        "driving_protocol_parameters": twin_cfg["driving_protocol"],
        "observation_protocol_parameters": twin_cfg["observation_protocol"],
        "theta_dagger": {
            "alpha": float(theta_dagger[0]),
            "beta": float(theta_dagger[1]),
            "D": float(theta_dagger[2]),
            "E": float(theta_dagger[3]),
        },
        "field_accessibility_notes": {
            "displacement_d": "Vettori nodali e diagnostici di energia elastica esposti; calcolata norma L2(Omega) esatta tramite M_d.",
            "pressure_p": "Vettori nodali e diagnostici di immagazzinamento esposti; calcolata norma L2(Omega) esatta tramite M_p.",
            "darcy_velocity_q": "Velocità massima puntuale `max_darcy_velocity` esposta dal solver C++; norma L2(Omega) ||q_h||_{L2} = K_h * ||grad(p_h)||_{L2} calcolata tramite la matrice di rigidezza Q1.",
            "tracer_c": "Vettori nodali e massa integrata M(t_n) esposti; calcolate norme L2(Omega) e seminorma H1(Omega).",
        },
        "discretization_levels": level_results,
        "same_mesh_internal_control": same_mesh_control,
        "total_fem_simulations_executed": sim_counter,
    }

    json_path = out_dir / "forcing_validation_summary.json"
    csv_path = out_dir / "forcing_validation_discretizations.csv"
    md_path = out_dir / "forcing_validation_report.md"
    log_path = out_dir / "simulation_accounting.log"

    json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    csv_fields = [
        "level_key",
        "subdivisions_N",
        "time_steps_M",
        "num_disp_dofs",
        "num_press_dofs",
        "num_tracer_dofs",
        "fluid_source_spatial_integral",
        "fluid_source_continuous_integral_ref",
        "mech_resultant_x_spatial",
        "mech_resultant_y_spatial",
        "mech_resultant_x_continuous_ref",
        "mech_resultant_y_continuous_ref",
        "max_darcy_velocity_pointwise",
        "q_darcy_L2_max_over_time",
        "d_L2_final",
        "p_L2_final",
        "c_L2_final",
        "tracer_mass_max_rel_variation",
        "obs_L2_norm",
        "rel_discrepancy_obs_vs_N32_M160",
        "max_abs_discrepancy_obs_vs_N32_M160",
        "solve_wall_time_s",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for key in ("N16_M80", "N24_M120", "N32_M160"):
            row_data = {k: level_results[key].get(k) for k in csv_fields}
            writer.writerow(row_data)

    md_lines = [
        "# Report Fase 1: Validazione delle Forzanti del Twin Experiment e Controllo Same-Mesh (`03b`)",
        "",
        f"- **Versione protocollo forzanti**: `{FORCING_PROTOCOL_VERSION}`",
        f"- **Parametri veri (`theta_dagger`)**: `alpha = {theta_dagger[0]}`, `beta = {theta_dagger[1]}`, `D = {theta_dagger[2]}`, `E = {theta_dagger[3]}`",
        f"- **Simulazioni FEM totali eseguite**: `{sim_counter}` (nessuna ottimizzazione CPSO eseguita)",
        "",
        "## 1. Verifica delle Forzanti e delle Discretizzazioni Dirette",
        "| Discretizzazione | DOFs $(d, p, c)$ | $\\int_\\Omega s_{f,\\mathrm{sp}}\\,\\mathrm{d}x$ | $R_x$ (Mecc.) | $R_y$ (Mecc.) | $\\max \\Vert{}\\mathbf{q}_h\\Vert{}$ | $\\Vert{}\\mathbf{q}_h\\Vert{}_{L^2,\\max}$ | $\\Vert{}\\mathbf{d}_h(T)\\Vert{}_{L^2}$ | $\\Vert{}p_h(T)\\Vert{}_{L^2}$ | $\\Vert{}c_h(T)\\Vert{}_{L^2}$ | $\\Vert{}\\mathbf{y}_h\\Vert{}_2$ | Discrepanza Rel. Oss. vs `N32_M160` | Tempo [s] |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("N16_M80", "N24_M120", "N32_M160"):
        r = level_results[key]
        d_l2_s = f"{r['d_L2_final']:.4e}" if r["d_L2_final"] is not None else "n/d"
        p_l2_s = f"{r['p_L2_final']:.4e}" if r["p_L2_final"] is not None else "n/d"
        c_l2_s = f"{r['c_L2_final']:.4e}" if r["c_L2_final"] is not None else "n/d"
        q_l2_s = f"{r['q_darcy_L2_max_over_time']:.4e}" if r["q_darcy_L2_max_over_time"] is not None else "n/d"
        md_lines.append(
            f"| `{key}` | `({r['num_disp_dofs']}, {r['num_press_dofs']}, {r['num_tracer_dofs']})` | "
            f"`{r['fluid_source_spatial_integral']:.6f}` | `{r['mech_resultant_x_spatial']:.6f}` | "
            f"`{r['mech_resultant_y_spatial']:.6f}` | `{r['max_darcy_velocity_pointwise']:.4e}` | "
            f"`{q_l2_s}` | `{d_l2_s}` | `{p_l2_s}` | `{c_l2_s}` | "
            f"`{r['obs_L2_norm']:.6e}` | `{r['rel_discrepancy_obs_vs_N32_M160']:.4e}` | `{r['solve_wall_time_s']:.2f}` |"
        )

    md_lines.extend(
        [
            "",
            f"- **Riferimenti analitici continui**: $\\int_\\Omega s_{{f,\\mathrm{{sp}}}}\\,\\mathrm{{d}}x = {ref_rec['fluid_source_continuous_integral_ref']:.6f}$, "
            f"$R_x^{{\\mathrm{{ref}}}} = {ref_rec['mech_resultant_x_continuous_ref']:.6f}$, "
            f"$R_y^{{\\mathrm{{ref}}}} = {ref_rec['mech_resultant_y_continuous_ref']:.6f}$.",
            "",
            "## 2. Controllo Diagnostico Interno Same-Mesh (`N=16, M=80` in `theta_dagger`)",
            "- **Avvertenza metodologica**: Questo controllo verifica esclusivamente la consistenza tra il solutore diretto e l'operatore osservativo a discretizzazione fissata (`N=16, M=80`) e rumore nullo (`eta=0`). Non costituisce prova di identificabilità né di unicità del minimo.",
            f"- **Residuo relativo osservativo in `theta_dagger` (same-mesh)**: `{same_mesh_control['relative_observation_residual_at_true']:.3e}`",
            f"- **Differenza assoluta massima in `theta_dagger` (same-mesh)**: `{same_mesh_control['max_abs_observation_difference_at_true']:.3e}`",
            f"- **Termine dati `J_data(theta_dagger)` (same-mesh `N16->N16`)**: `{same_mesh_control['J_data_at_theta_dagger_same_mesh']:.6e}`",
            f"- **Termine dati `J_data(theta_dagger)` (twin-mesh `N32->N16`)**: `{j_data_twin_at_true:.6e}`",
            f"- **Termine di regolarizzazione `J_reg(theta_dagger)` (`lambda = {lambda_reg:.1e}`)**: `{j_reg_same:.6e}`",
            f"- **Costo totale `J_total(theta_dagger)` (same-mesh `N16->N16`)**: `{j_total_same:.6e}`",
            f"- **Costo totale `J_total(theta_dagger)` (twin-mesh `N32->N16`)**: `{j_total_twin_at_true:.6e}`",
            "",
        ]
    )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"[03b] Verifica completata. Artefatti salvati in: {out_dir}")


if __name__ == "__main__":
    main()