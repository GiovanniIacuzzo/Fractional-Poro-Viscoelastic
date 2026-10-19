"""
experiments/03_run_twin_robustness.py
Twin Experiment anti-inverse-crime:
- Genera il target y_clean sulla discretizzazione fine h_tar (N=32, M=160)
- Quantifica la discrepanza di discretizzazione ||G_{h_tar}(theta_dagger) - G_{h_inv}(theta_dagger)||
- Esegue l'inversione CPSO sulla discretizzazione standard h_inv (N=16, M=80)
  per i livelli di rumore eta in {0%, 1%, 5%, 10%}.
- Salva la diagnostica arricchita (decomposizione J_data/J_reg, residui relativi rispetto
  ai dati rumorosi e al target pulito, norma effettiva del rumore, contabilizzazione
  distinta delle valutazioni FEM e grafici esplorativi) in `artifacts/03_twin_robustness/diagnostics/`
  senza sovrascrivere silenziosamente il file JSON originario se già esistente.
"""

from __future__ import annotations
import json
from pathlib import Path
import time
from typing import List
import numpy as np
import yaml

from reteco import _core, load_master_config
from reteco.config_schema import ParameterCoordinateTransform
from reteco.forward.wrapper import RetecoForwardProblem2D
from reteco.forward.target_generator import build_standard_driving_Callbacks
from reteco.observation.operator import DiscreteObservationOperator
from reteco.observation.noise_model import generate_gaussian_noisy_observations
from reteco.inverse.regularization import TikhonovRegularizer
from reteco.inverse.objective_functional import InverseObjectiveFunctional
from reteco.inverse.cpso import run_cpso_optimization
from reteco.inverse.diagnostics import (
    PARAM_NAMES,
    SingleRunDiagnosticRecord,
    extract_run_diagnostics_from_history,
    plot_noise_study_parameter_errors,
    plot_noise_study_relative_residuals,
    serialize_run_record,
    write_convergence_history_csv,
    write_noise_study_markdown_report,
    write_runs_summary_csv,
)
from reteco.visualization.benchmark_plots import compute_individual_relative_errors_pct


def main() -> None:
    out_dir = Path("artifacts/03_twin_robustness")
    diag_dir = out_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

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

    # 1. Generazione Target sulla mesh fine h_tar
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
    c0_tar, F_tar, Sf_tar = build_standard_driving_Callbacks(
        prob_tar, twin_cfg["driving_protocol"]
    )
    traj_tar = prob_tar.solve_trajectory(
        theta_dagger, c0=c0_tar, F_mech_func=F_tar, S_fluid_func=Sf_tar
    )
    y_target_clean = obs_op.apply(prob_tar, traj_tar)

    # 2. Configurazione Solutore di Inversione sulla mesh più grezza h_inv
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
    c0_inv, F_inv, Sf_inv = build_standard_driving_Callbacks(
        prob_inv, twin_cfg["driving_protocol"]
    )

    def G_inv(theta: np.ndarray) -> np.ndarray:
        tr = prob_inv.solve_trajectory(
            theta, c0=c0_inv, F_mech_func=F_inv, S_fluid_func=Sf_inv
        )
        return obs_op.apply(prob_inv, tr)

    y_inv_at_true = G_inv(theta_dagger)
    discretization_rel_gap = float(
        np.linalg.norm(y_target_clean - y_inv_at_true) / np.linalg.norm(y_target_clean)
    )
    print(f"Discrepanza relativa di discretizzazione (h_tar vs h_inv): {discretization_rel_gap:.4e}")

    noise_cfg = twin_cfg["noise_and_regularization"]
    lambda_reg = float(noise_cfg["regularization_lambda"])
    prior_theta_tilde_0 = np.array(noise_cfg["prior_theta_tilde_0"], dtype=np.float64)
    reg = TikhonovRegularizer(
        transform,
        lambda_reg=lambda_reg,
        theta_tilde_0=prior_theta_tilde_0,
    )

    results_log = {
        "discretization_relative_gap": discretization_rel_gap,
        "noise_levels": {},
    }
    diagnostic_records: List[SingleRunDiagnosticRecord] = []

    planned_budget = 120
    num_particles = 10
    max_iterations = 12

    for eta in noise_cfg["noise_levels_eta"]:
        eta_val = float(eta)
        seed_eta = int(noise_cfg["base_seed"]) + int(round(eta_val * 1000))
        opt_seed = seed_eta + 1
        sample = generate_gaussian_noisy_observations(y_target_clean, eta_val, seed_eta)

        # Il funzionale obiettivo riceve esclusivamente sample.y_noisy (mai y_target_clean per eta > 0)
        obj = InverseObjectiveFunctional(
            G_inv,
            sample.y_noisy,
            sample.weight_matrix_diag,
            transform,
            reg,
            max_evaluations=planned_budget,
        )
        t0_run = time.perf_counter()
        res = run_cpso_optimization(
            obj,
            seed=opt_seed,
            num_particles=num_particles,
            max_iterations=max_iterations,
        )
        run_wall_time = time.perf_counter() - t0_run

        rel_errs = compute_individual_relative_errors_pct(res.theta_hat, theta_dagger)
        diag_rec = extract_run_diagnostics_from_history(
            run_id=f"noise_eta_{int(round(eta_val * 100)):02d}pct",
            obj=obj,
            opt_result=res,
            theta_true=theta_dagger,
            y_target_clean=y_target_clean,
            y_noisy=sample.y_noisy,
            nominal_noise_std=sample.sigma_std,
            eta_level=eta_val,
            data_seed=seed_eta,
            optimizer_seed=opt_seed,
            planned_fem_budget=planned_budget,
            num_particles=num_particles,
            max_iterations=max_iterations,
            wall_time_seconds=run_wall_time,
            forward_operator_inv=G_inv,
            transform=transform,
            regularizer=reg,
        )
        diagnostic_records.append(diag_rec)

        print(
            f"  Rumore eta={eta_val*100:4.1f}% | J_tot={diag_rec.best_cost_J_total:.3e} "
            f"(J_data={diag_rec.best_cost_J_data:.3e}, J_reg={diag_rec.best_cost_J_reg:.3e}) | "
            f"Res.Noisy={diag_rec.relative_residual_vs_noisy_data:.3e} | "
            f"Res.Clean={diag_rec.relative_residual_vs_clean_target:.3e} | "
            f"Errori rel [%]: alpha={rel_errs[0]:.2f}, beta={rel_errs[1]:.2f}, "
            f"D={rel_errs[2]:.2f}, E={rel_errs[3]:.2f}"
        )

        results_log["noise_levels"][str(eta)] = {
            "seed": seed_eta,
            "data_seed": seed_eta,
            "optimizer_seed": opt_seed,
            "theta_hat": res.theta_hat.tolist(),
            "individual_relative_errors_pct": rel_errs.tolist(),
            "best_cost_J": res.best_cost_J,
            "best_cost_J_data": diag_rec.best_cost_J_data,
            "best_cost_J_reg": diag_rec.best_cost_J_reg,
            "nominal_noise_std": diag_rec.nominal_noise_std,
            "effective_relative_noise_norm": diag_rec.effective_relative_noise_norm,
            "relative_residual_vs_noisy_data": diag_rec.relative_residual_vs_noisy_data,
            "relative_residual_vs_clean_target": diag_rec.relative_residual_vs_clean_target,
            "num_fem_evaluations": res.num_fem_evaluations,
            "optimizer_fem_evaluations": diag_rec.optimizer_fem_evaluations,
            "diagnostic_fem_evaluations": diag_rec.diagnostic_fem_evaluations,
            "termination_condition": diag_rec.termination_condition,
            "wall_time_seconds": diag_rec.wall_time_seconds,
        }

    # Preserva intatto il file JSON originario se già presente sul disco
    legacy_json_path = out_dir / "twin_robustness_results.json"
    if not legacy_json_path.exists():
        legacy_json_path.write_text(
            json.dumps(results_log, indent=2), encoding="utf-8"
        )
    else:
        print(
            f"File originario {legacy_json_path} già presente: preservato senza sovrascrittura."
        )

    # Salva i nuovi artefatti diagnostici nella sottocartella dedicata `diagnostics/`
    diag_json_payload = {
        "experiment_id": "RETECO-v1-TWIN-ROBUSTNESS-DIAGNOSTICS-03",
        "methodological_note": (
            "Per eta > 0 i pesi W_ii = 1 / (sqrt(N_obs) * sigma) dipendono da sigma = eta * ||y_clean||_inf. "
            "Pertanto il valore grezzo di J non è direttamente confrontabile tra diversi livelli di rumore; "
            "occorre esaminare separatamente J_data, J_reg, i residui relativi non pesati e gli errori sui parametri."
        ),
        "discretization_relative_gap": discretization_rel_gap,
        "regularization_lambda": lambda_reg,
        "theta_true": {
            name: float(theta_dagger[i]) for i, name in enumerate(PARAM_NAMES)
        },
        "noise_levels": results_log["noise_levels"],
        "detailed_runs": [serialize_run_record(r) for r in diagnostic_records],
    }

    (diag_dir / "twin_robustness_diagnostics.json").write_text(
        json.dumps(diag_json_payload, indent=2), encoding="utf-8"
    )
    write_runs_summary_csv(diagnostic_records, diag_dir / "twin_robustness_summary.csv")
    write_convergence_history_csv(
        diagnostic_records, diag_dir / "twin_robustness_convergence_history.csv"
    )
    plot_noise_study_parameter_errors(
        diagnostic_records, diag_dir / "fig_noise_parameter_errors.png"
    )
    plot_noise_study_relative_residuals(
        diagnostic_records,
        discretization_rel_gap=discretization_rel_gap,
        out_path=diag_dir / "fig_noise_relative_residuals.png",
    )
    write_noise_study_markdown_report(
        diagnostic_records,
        discretization_rel_gap=discretization_rel_gap,
        lambda_reg=lambda_reg,
        report_path=diag_dir / "twin_robustness_diagnostic_report.md",
    )
    print(f"Diagnostica arricchita dello studio con rumore salvata in: {diag_dir}")


if __name__ == "__main__":
    main()