"""
experiments/06_cpso_budget_diagnostic.py
RETECO — Fase 3.6: Budget diagnostico CPSO (600 valutazioni FEM = 10 particelle x 60 iterazioni)
esclusivamente sul Caso 1 senza rumore (eta = 0.0, Modalità A deterministica, lambda_A = 1.0e-4).
"""

from __future__ import annotations
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List
import numpy as np
import yaml

from reteco import _core, load_master_config
from reteco.config_schema import ParameterCoordinateTransform
from reteco.forward.wrapper import RetecoForwardProblem2D
from reteco.forward.target_generator import (
    FORCING_PROTOCOL_VERSION,
    build_standard_driving_Callbacks,
)
from reteco.observation.operator import DiscreteObservationOperator
from reteco.observation.noise_model import generate_gaussian_noisy_observations
from reteco.inverse.regularization import TikhonovRegularizer
from reteco.inverse.objective_functional import InverseObjectiveFunctional
from reteco.inverse.cpso import run_cpso_optimization
from reteco.inverse.diagnostics import (
    extract_run_diagnostics_from_history,
    serialize_run_record,
    write_convergence_history_csv,
)

DIAG_NUM_PARTICLES: int = 10
DIAG_MAX_ITERATIONS_600: int = 60
DIAG_PLANNED_FEM_BUDGET_600: int = DIAG_NUM_PARTICLES * DIAG_MAX_ITERATIONS_600  # 600

REFERENCE_PILOT_120: Dict[str, Any] = {
    "case_id": "case_1_noise_free_eta_0pct_mode_a_120evals",
    "planned_fem_budget": 120,
    "optimizer_fem_evaluations": 120,
    "num_particles": 10,
    "max_iterations": 12,
    "best_cost_J_total": 2.1903258237383324e-05,
    "J_total_at_theta_dagger": 4.984481939398449e-06,
    "beta_hat": 0.36509264798853436,
    "E_hat": 2.156829277257016,
}


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()


def _load_pilot_120_reference() -> Dict[str, Any]:
    ref = dict(REFERENCE_PILOT_120)
    pilot_json = Path("artifacts/05_pilot_inverse/pilot_inverse_summary.json")
    if pilot_json.exists():
        data = json.loads(pilot_json.read_text(encoding="utf-8"))
        for c in data.get("cases", []):
            if float(c.get("eta_level", -1.0)) == 0.0:
                ref.update(
                    {
                        "source": str(pilot_json),
                        "alpha_hat": float(c["alpha_hat"]),
                        "beta_hat": float(c["beta_hat"]),
                        "D_hat": float(c["D_hat"]),
                        "E_hat": float(c["E_hat"]),
                        "err_rel_alpha_pct": float(c["err_rel_alpha_pct"]),
                        "err_rel_beta_pct": float(c["err_rel_beta_pct"]),
                        "err_rel_D_pct": float(c["err_rel_D_pct"]),
                        "err_rel_E_pct": float(c["err_rel_E_pct"]),
                        "best_cost_J_total": float(c["best_cost_J_total"]),
                        "best_cost_J_data": float(c["best_cost_J_data"]),
                        "best_cost_J_reg": float(c["best_cost_J_reg"]),
                        "best_eval_index": int(c["best_eval_index"]),
                        "weighted_residual_l2_norm": float(c["weighted_residual_l2_norm"]),
                        "relative_residual_vs_observed_data": float(
                            c["relative_residual_vs_observed_data"]
                        ),
                        "relative_residual_vs_clean_target": float(
                            c["relative_residual_vs_clean_target"]
                        ),
                        "wall_time_seconds": float(c["wall_time_seconds"]),
                        "J_total_at_theta_dagger": float(
                            c["reference_evaluation_at_theta_dagger"]["J_total_at_true"]
                        ),
                        "J_data_at_theta_dagger": float(
                            c["reference_evaluation_at_theta_dagger"]["J_data_at_true"]
                        ),
                        "J_reg_at_theta_dagger": float(
                            c["reference_evaluation_at_theta_dagger"]["J_reg_at_true"]
                        ),
                    }
                )
                break
    else:
        ref["source"] = "prompt_reference_constants"
    return ref


def _extract_budget_milestones(
    obj: InverseObjectiveFunctional, theta_true: np.ndarray
) -> List[Dict[str, Any]]:
    milestones = [10, 30, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600]
    out: List[Dict[str, Any]] = []
    for m in milestones:
        if m <= len(obj.history):
            prefix = obj.history[:m]
            best_idx = int(np.argmin([r.j_total for r in prefix]))
            best_r = prefix[best_idx]
            errs = 100.0 * np.abs(best_r.theta - theta_true) / np.abs(theta_true)
            out.append(
                {
                    "fem_evaluation_milestone": m,
                    "best_eval_index_up_to_milestone": int(best_r.eval_index),
                    "best_j_total": float(best_r.j_total),
                    "best_j_data": float(best_r.j_data),
                    "best_j_reg": float(best_r.j_reg),
                    "relative_residual_l2": float(best_r.relative_residual_l2),
                    "weighted_residual_l2_norm": float(best_r.weighted_residual_l2_norm),
                    "alpha_hat": float(best_r.theta[0]),
                    "beta_hat": float(best_r.theta[1]),
                    "D_hat": float(best_r.theta[2]),
                    "E_hat": float(best_r.theta[3]),
                    "err_rel_alpha_pct": float(errs[0]),
                    "err_rel_beta_pct": float(errs[1]),
                    "err_rel_D_pct": float(errs[2]),
                    "err_rel_E_pct": float(errs[3]),
                }
            )
    return out


def _write_comparison_csv(
    comparison_rows: List[Dict[str, Any]], csv_path: Path
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_label",
        "planned_fem_budget",
        "optimizer_fem_evaluations",
        "num_particles",
        "max_iterations",
        "data_seed",
        "optimizer_seed",
        "lambda_reg",
        "best_eval_index",
        "best_cost_J_total",
        "best_cost_J_data",
        "best_cost_J_reg",
        "J_total_at_theta_dagger",
        "ratio_J_best_over_J_true",
        "weighted_residual_l2_norm",
        "relative_residual_vs_observed_data",
        "relative_residual_vs_clean_target",
        "alpha_true",
        "beta_true",
        "D_true",
        "E_true",
        "alpha_hat",
        "beta_hat",
        "D_hat",
        "E_hat",
        "err_rel_alpha_pct",
        "err_rel_beta_pct",
        "err_rel_D_pct",
        "err_rel_E_pct",
        "consistency_all_checks_passed",
        "wall_time_seconds",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    out_dir = Path("artifacts/06_cpso_budget_diagnostic")
    out_dir.mkdir(parents=True, exist_ok=True)

    master = load_master_config("configs/scales_and_physics.yaml")
    with open("configs/twin_experiment_synthetic.yaml", "r", encoding="utf-8") as f:
        twin_cfg = yaml.safe_load(f)

    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_dagger = master.parameter_space_theta.theta_dagger_array()

    obs_cfg = twin_cfg["observation_protocol"]
    obs_op = DiscreteObservationOperator.from_regular_grid(
        nx=int(obs_cfg["sensor_grid_nx"]),
        ny=int(obs_cfg["sensor_grid_ny"]),
        margin=float(obs_cfg["sensor_margin"]),
        observation_times=obs_cfg["observation_times"],
        observed_fields=obs_cfg["observed_fields"],
    )

    # 1. Generazione (e verifica di riproducibilità) del target sintetico (N=32, M=160)
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
        fe_deg_disp=int(tar_cfg.get("fe_deg_disp", 2)),
        fe_deg_press=int(tar_cfg.get("fe_deg_press", 1)),
        fe_deg_tracer=int(tar_cfg.get("fe_deg_tracer", 1)),
    )
    c0_tar, F_tar, Sf_tar = build_standard_driving_Callbacks(
        prob_tar, twin_cfg["driving_protocol"]
    )
    t0_tar = time.perf_counter()
    traj_tar = prob_tar.solve_trajectory(
        theta_dagger, c0=c0_tar, F_mech_func=F_tar, S_fluid_func=Sf_tar
    )
    y_target_clean = np.asarray(obs_op.apply(prob_tar, traj_tar), dtype=np.float64).ravel()
    target_wall_time = time.perf_counter() - t0_tar
    y_target_sha256 = _sha256_array(y_target_clean)
    n_obs = int(y_target_clean.size)

    # 2. Solutore inverso su discretizzazione (N=16, M=80)
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
        fe_deg_disp=int(inv_cfg.get("fe_deg_disp", 2)),
        fe_deg_press=int(inv_cfg.get("fe_deg_press", 1)),
        fe_deg_tracer=int(inv_cfg.get("fe_deg_tracer", 1)),
    )
    c0_inv, F_inv, Sf_inv = build_standard_driving_Callbacks(
        prob_inv, twin_cfg["driving_protocol"]
    )

    def G_inv(theta: np.ndarray) -> np.ndarray:
        tr = prob_inv.solve_trajectory(
            theta, c0=c0_inv, F_mech_func=F_inv, S_fluid_func=Sf_inv
        )
        return np.asarray(obs_op.apply(prob_inv, tr), dtype=np.float64).ravel()

    noise_cfg = twin_cfg["noise_and_regularization"]
    data_seed = int(noise_cfg["base_seed"])               # 20261016
    optimizer_seed = data_seed + 1                        # 20261017
    lambda_a = float(noise_cfg["regularization_lambda"])  # 1.0e-4
    prior_theta_tilde_0 = np.asarray(noise_cfg["prior_theta_tilde_0"], dtype=np.float64).ravel()

    reg = TikhonovRegularizer(
        transform=transform,
        lambda_reg=lambda_a,
        theta_tilde_0=prior_theta_tilde_0,
    )

    sample_0 = generate_gaussian_noisy_observations(
        y_target_clean,
        eta_level=0.0,
        seed=data_seed,
        weighting_mode="mode_a_deterministic_normalized",
        reference_scale_type="linf",
    )

    # 3. Controllo di riproducibilità a parità di seed (2 iterazioni = 20 valutazioni FEM)
    obj_repro_1 = InverseObjectiveFunctional.from_mode_a_deterministic(
        G_inv,
        sample_0.y_noisy,
        transform,
        reg,
        y_reference=y_target_clean,
        scale_type="linf",
        max_evaluations=20,
    )
    obj_repro_2 = InverseObjectiveFunctional.from_mode_a_deterministic(
        G_inv,
        sample_0.y_noisy,
        transform,
        reg,
        y_reference=y_target_clean,
        scale_type="linf",
        max_evaluations=20,
    )
    res_repro_1 = run_cpso_optimization(
        obj_repro_1, seed=optimizer_seed, num_particles=DIAG_NUM_PARTICLES, max_iterations=2
    )
    res_repro_2 = run_cpso_optimization(
        obj_repro_2, seed=optimizer_seed, num_particles=DIAG_NUM_PARTICLES, max_iterations=2
    )
    seed_repro_verified = bool(
        res_repro_1.best_cost_J == res_repro_2.best_cost_J
        and np.array_equal(res_repro_1.theta_hat, res_repro_2.theta_hat)
        and np.array_equal(res_repro_1.convergence_eval_trace, res_repro_2.convergence_eval_trace)
    )
    if not seed_repro_verified:
        raise RuntimeError("Fallito il controllo di riproducibilità a parità di seed del CPSO.")

    # 4. Esecuzione principale a budget esteso: 600 valutazioni FEM (10 particelle x 60 iterazioni)
    obj_600 = InverseObjectiveFunctional.from_mode_a_deterministic(
        G_inv,
        sample_0.y_noisy,
        transform,
        reg,
        y_reference=y_target_clean,
        scale_type="linf",
        max_evaluations=DIAG_PLANNED_FEM_BUDGET_600,
    )
    if obj_600.max_evaluations != 600 or (DIAG_NUM_PARTICLES * DIAG_MAX_ITERATIONS_600) != 600:
        raise RuntimeError(
            f"Budget disallineato: obj.max_evaluations={obj_600.max_evaluations}, "
            f"cpso_budget={DIAG_NUM_PARTICLES * DIAG_MAX_ITERATIONS_600}."
        )

    t0_600 = time.perf_counter()
    opt_res_600 = run_cpso_optimization(
        obj_600,
        seed=optimizer_seed,
        num_particles=DIAG_NUM_PARTICLES,
        max_iterations=DIAG_MAX_ITERATIONS_600,
    )
    wall_time_600 = time.perf_counter() - t0_600

    rec_600 = extract_run_diagnostics_from_history(
        run_id="case_1_noise_free_eta_0pct_mode_a_600evals",
        obj=obj_600,
        opt_result=opt_res_600,
        theta_true=theta_dagger,
        y_target_clean=y_target_clean,
        y_noisy=sample_0.y_noisy,
        nominal_noise_std=0.0,
        eta_level=0.0,
        data_seed=data_seed,
        optimizer_seed=optimizer_seed,
        planned_fem_budget=DIAG_PLANNED_FEM_BUDGET_600,
        num_particles=DIAG_NUM_PARTICLES,
        max_iterations=DIAG_MAX_ITERATIONS_600,
        wall_time_seconds=wall_time_600,
        forward_operator_inv=G_inv,
        transform=transform,
        regularizer=reg,
        seed_reproducibility_verified=seed_repro_verified,
    )

    # 5. Valutazione diagnostica nel punto vero theta_dagger (fuori dallo sciame e dalla cronologia)
    history_len_before_true_eval = len(obj_600.history)
    fem_count_before_true_eval = obj_600.num_fem_evaluations
    y_inv_at_true = G_inv(theta_dagger)
    bd_at_true = obj_600.evaluate_from_prediction(
        theta_dagger, y_inv_at_true, record_in_history=False
    )
    assert len(obj_600.history) == history_len_before_true_eval
    assert obj_600.num_fem_evaluations == fem_count_before_true_eval

    discretization_rel_gap = float(
        np.linalg.norm(y_target_clean - y_inv_at_true) / np.linalg.norm(y_target_clean)
    )

    ref_120 = _load_pilot_120_reference()
    milestones_600 = _extract_budget_milestones(obj_600, theta_dagger)

    row_120: Dict[str, Any] = {
        "run_label": "pilot_120_evals",
        "planned_fem_budget": 120,
        "optimizer_fem_evaluations": 120,
        "num_particles": 10,
        "max_iterations": 12,
        "data_seed": data_seed,
        "optimizer_seed": optimizer_seed,
        "lambda_reg": lambda_a,
        "best_eval_index": ref_120.get("best_eval_index", ""),
        "best_cost_J_total": float(ref_120["best_cost_J_total"]),
        "best_cost_J_data": ref_120.get("best_cost_J_data", ""),
        "best_cost_J_reg": ref_120.get("best_cost_J_reg", ""),
        "J_total_at_theta_dagger": float(ref_120["J_total_at_theta_dagger"]),
        "ratio_J_best_over_J_true": float(
            ref_120["best_cost_J_total"] / ref_120["J_total_at_theta_dagger"]
        ),
        "weighted_residual_l2_norm": ref_120.get("weighted_residual_l2_norm", ""),
        "relative_residual_vs_observed_data": ref_120.get(
            "relative_residual_vs_observed_data", ""
        ),
        "relative_residual_vs_clean_target": ref_120.get(
            "relative_residual_vs_clean_target", ""
        ),
        "alpha_true": float(theta_dagger[0]),
        "beta_true": float(theta_dagger[1]),
        "D_true": float(theta_dagger[2]),
        "E_true": float(theta_dagger[3]),
        "alpha_hat": ref_120.get("alpha_hat", ""),
        "beta_hat": float(ref_120["beta_hat"]),
        "D_hat": ref_120.get("D_hat", ""),
        "E_hat": float(ref_120["E_hat"]),
        "err_rel_alpha_pct": ref_120.get("err_rel_alpha_pct", ""),
        "err_rel_beta_pct": float(
            100.0 * abs(float(ref_120["beta_hat"]) - theta_dagger[1]) / theta_dagger[1]
        ),
        "err_rel_D_pct": ref_120.get("err_rel_D_pct", ""),
        "err_rel_E_pct": float(
            100.0 * abs(float(ref_120["E_hat"]) - theta_dagger[3]) / theta_dagger[3]
        ),
        "consistency_all_checks_passed": True,
        "wall_time_seconds": ref_120.get("wall_time_seconds", ""),
    }

    errs_600 = rec_600.individual_relative_errors_pct
    row_600: Dict[str, Any] = {
        "run_label": "diagnostic_600_evals",
        "planned_fem_budget": DIAG_PLANNED_FEM_BUDGET_600,
        "optimizer_fem_evaluations": rec_600.optimizer_fem_evaluations,
        "num_particles": DIAG_NUM_PARTICLES,
        "max_iterations": DIAG_MAX_ITERATIONS_600,
        "data_seed": data_seed,
        "optimizer_seed": optimizer_seed,
        "lambda_reg": lambda_a,
        "best_eval_index": rec_600.best_eval_index,
        "best_cost_J_total": rec_600.best_cost_J_total,
        "best_cost_J_data": rec_600.best_cost_J_data,
        "best_cost_J_reg": rec_600.best_cost_J_reg,
        "J_total_at_theta_dagger": float(bd_at_true.j_total),
        "ratio_J_best_over_J_true": float(rec_600.best_cost_J_total / bd_at_true.j_total),
        "weighted_residual_l2_norm": rec_600.weighted_residual_l2_norm,
        "relative_residual_vs_observed_data": rec_600.relative_residual_vs_noisy_data,
        "relative_residual_vs_clean_target": rec_600.relative_residual_vs_clean_target,
        "alpha_true": float(theta_dagger[0]),
        "beta_true": float(theta_dagger[1]),
        "D_true": float(theta_dagger[2]),
        "E_true": float(theta_dagger[3]),
        "alpha_hat": float(rec_600.theta_hat[0]),
        "beta_hat": float(rec_600.theta_hat[1]),
        "D_hat": float(rec_600.theta_hat[2]),
        "E_hat": float(rec_600.theta_hat[3]),
        "err_rel_alpha_pct": float(errs_600[0]),
        "err_rel_beta_pct": float(errs_600[1]),
        "err_rel_D_pct": float(errs_600[2]),
        "err_rel_E_pct": float(errs_600[3]),
        "consistency_all_checks_passed": bool(
            rec_600.consistency_checks["all_checks_passed"]
        ),
        "wall_time_seconds": rec_600.wall_time_seconds,
    }

    summary_json_path = out_dir / "cpso_budget_diagnostic_summary.json"
    comp_csv_path = out_dir / "cpso_budget_comparison_120_vs_600.csv"
    hist_csv_path = out_dir / "cpso_600_convergence_history.csv"
    milestones_csv_path = out_dir / "cpso_600_budget_milestones.csv"

    summary_payload = {
        "experiment_id": "RETECO-v1-PHASE3.6-CPSO-BUDGET-DIAGNOSTIC-600",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "forcing_protocol_version": FORCING_PROTOCOL_VERSION,
        "target_provenance": {
            "N_target": int(tar_cfg["subdivisions_N"]),
            "M_target": int(tar_cfg["time_steps_M"]),
            "N_obs": n_obs,
            "y_target_clean_sha256": y_target_sha256,
            "y_target_clean_linf": float(np.max(np.abs(y_target_clean))),
            "y_target_clean_l2": float(np.linalg.norm(y_target_clean)),
            "discretization_relative_gap_L2": discretization_rel_gap,
            "target_generation_wall_time_seconds": target_wall_time,
        },
        "reference_evaluation_at_theta_dagger": {
            "theta_dagger": theta_dagger.tolist(),
            "J_data_at_true": float(bd_at_true.j_data),
            "J_reg_at_true": float(bd_at_true.j_reg),
            "J_total_at_true": float(bd_at_true.j_total),
            "weighted_residual_l2_at_true": float(bd_at_true.weighted_residual_l2_norm),
            "relative_residual_vs_observed_at_true": float(bd_at_true.relative_residual_l2),
            "recorded_in_optimizer_history": False,
        },
        "consistency_checks_600": rec_600.consistency_checks,
        "comparison_120_vs_600": {
            "pilot_120": row_120,
            "diagnostic_600": row_600,
            "cost_reduction_factor_120_to_600": float(
                row_120["best_cost_J_total"] / rec_600.best_cost_J_total
            ),
            "cost_gap_ratio_600_vs_theta_dagger": float(
                rec_600.best_cost_J_total / bd_at_true.j_total
            ),
        },
        "trajectory_milestones_600": milestones_600,
        "diagnostic_600_full_record": serialize_run_record(rec_600),
    }

    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    _write_comparison_csv([row_120, row_600], comp_csv_path)
    write_convergence_history_csv([rec_600], hist_csv_path)

    with open(milestones_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(milestones_600[0].keys()))
        writer.writeheader()
        writer.writerows(milestones_600)

    print(
        f"[06_budget_diagnostic] Completato | J_600={rec_600.best_cost_J_total:.6e} "
        f"(vs J_120={row_120['best_cost_J_total']:.6e}, J_true={bd_at_true.j_total:.6e}) | "
        f"BestEval={rec_600.best_eval_index}/600 | "
        f"beta_hat={rec_600.theta_hat[1]:.6f} (err={errs_600[1]:.2f}%), "
        f"E_hat={rec_600.theta_hat[3]:.6f} (err={errs_600[3]:.2f}%)"
    )


if __name__ == "__main__":
    main()