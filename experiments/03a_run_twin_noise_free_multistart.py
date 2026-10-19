"""
experiments/03a_run_twin_noise_free_multistart.py
Studio diagnostico multistart CPSO senza rumore (eta = 0.0):
- Genera una sola volta il target pulito y_target_clean sulla mesh fine h_tar (N=32, M=160);
- Riutilizza esattamente lo stesso target e gli stessi pesi osservativi in 5 esecuzioni
  indipendenti di CPSO sulla discretizzazione h_inv (N=16, M=80);
- Mantiene lambda = 1e-4, 10 particelle, 12 iterazioni (120 valutazioni FEM per run)
  per isolare l'effetto dell'inizializzazione casuale di CPSO;
- Contabilizza separatamente le valutazioni FEM dell'ottimizzatore e quelle diagnostiche;
- Salva JSON, CSV riassuntivo, CSV delle storie di convergenza, report Markdown e 3 figure PNG.
"""

from __future__ import annotations
import json
from pathlib import Path
import time
from typing import Dict, List
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
    plot_multistart_convergence,
    plot_multistart_objective_decomposition,
    plot_multistart_parameter_recovery,
    serialize_run_record,
    write_convergence_history_csv,
    write_multistart_markdown_report,
    write_runs_summary_csv,
)

NUM_MULTISTART_RUNS: int = 5
NUM_PARTICLES: int = 10
MAX_ITERATIONS: int = 12
PLANNED_FEM_BUDGET: int = NUM_PARTICLES * MAX_ITERATIONS  # 120 valutazioni FEM esatte


def _extract_bounds_dict(master_cfg: object) -> Dict[str, tuple[float, float]]:
    bounds_obj = master_cfg.parameter_space_theta.bounds
    out: Dict[str, tuple[float, float]] = {}
    for name in PARAM_NAMES:
        b = getattr(bounds_obj, name, None)
        if isinstance(b, (list, tuple)) and len(b) == 2:
            out[name] = (float(b[0]), float(b[1]))
        elif hasattr(b, "min") and hasattr(b, "max"):
            out[name] = (float(b.min), float(b.max))
        elif hasattr(b, "lower") and hasattr(b, "upper"):
            out[name] = (float(b.lower), float(b.upper))
    return out


def main() -> None:
    out_dir = Path("artifacts/03a_twin_noise_free_multistart")
    out_dir.mkdir(parents=True, exist_ok=True)

    master = load_master_config("configs/scales_and_physics.yaml")
    with open("configs/twin_experiment_synthetic.yaml", "r", encoding="utf-8") as f:
        twin_cfg = yaml.safe_load(f)

    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_dagger = master.parameter_space_theta.theta_dagger_array()
    bounds_dict = _extract_bounds_dict(master)

    obs_cfg = twin_cfg["observation_protocol"]
    obs_op = DiscreteObservationOperator.from_regular_grid(
        nx=int(obs_cfg["sensor_grid_nx"]),
        ny=int(obs_cfg["sensor_grid_ny"]),
        margin=float(obs_cfg["sensor_margin"]),
        observation_times=obs_cfg["observation_times"],
        observed_fields=obs_cfg["observed_fields"],
    )

    # 1. Generazione unica del target pulito sulla discretizzazione fine h_tar (N=32, M=160)
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
    t0_target = time.perf_counter()
    traj_tar = prob_tar.solve_trajectory(
        theta_dagger, c0=c0_tar, F_mech_func=F_tar, S_fluid_func=Sf_tar
    )
    y_target_clean = obs_op.apply(prob_tar, traj_tar)
    target_generation_wall_time = time.perf_counter() - t0_target

    # 2. Configurazione del solutore di inversione sulla discretizzazione standard h_inv (N=16, M=80)
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

    # Valutazione diagnostica preliminare in theta_dagger per quantificare il gap di discretizzazione
    y_inv_at_true = G_inv(theta_dagger)
    discretization_rel_gap = float(
        np.linalg.norm(y_target_clean - y_inv_at_true) / np.linalg.norm(y_target_clean)
    )
    print(f"[03a] Discrepanza relativa di discretizzazione (h_tar vs h_inv): {discretization_rel_gap:.4e}")

    # 3. Costruzione unica del campione osservativo senza rumore (eta = 0.0) e del regolarizzatore
    noise_cfg = twin_cfg["noise_and_regularization"]
    base_seed = int(noise_cfg["base_seed"])
    lambda_reg = float(noise_cfg["regularization_lambda"])
    prior_theta_tilde_0 = np.array(noise_cfg["prior_theta_tilde_0"], dtype=np.float64)

    sample_clean = generate_gaussian_noisy_observations(
        y_clean=y_target_clean,
        eta_level=0.0,
        seed=base_seed,
    )
    reg = TikhonovRegularizer(
        transform,
        lambda_reg=lambda_reg,
        theta_tilde_0=prior_theta_tilde_0,
    )

    # Cinque semi riproducibili e documentati: il primo (base_seed + 1 = 20261017) coincide
    # esattamente con l'esecuzione eta=0.0 di experiments/03_run_twin_robustness.py.
    optimizer_seeds: List[int] = [base_seed + 1 + k for k in range(NUM_MULTISTART_RUNS)]
    diagnostic_records: List[SingleRunDiagnosticRecord] = []

    for run_idx, opt_seed in enumerate(optimizer_seeds):
        run_id = f"multistart_run_{run_idx + 1:02d}_seed_{opt_seed}"
        obj = InverseObjectiveFunctional(
            G_inv,
            sample_clean.y_noisy,
            sample_clean.weight_matrix_diag,
            transform,
            reg,
            max_evaluations=PLANNED_FEM_BUDGET,
        )

        t0_run = time.perf_counter()
        res = run_cpso_optimization(
            obj,
            seed=opt_seed,
            num_particles=NUM_PARTICLES,
            max_iterations=MAX_ITERATIONS,
        )
        run_wall_time = time.perf_counter() - t0_run

        rec = extract_run_diagnostics_from_history(
            run_id=run_id,
            obj=obj,
            opt_result=res,
            theta_true=theta_dagger,
            y_target_clean=y_target_clean,
            y_noisy=sample_clean.y_noisy,
            nominal_noise_std=sample_clean.sigma_std,
            eta_level=0.0,
            data_seed=base_seed,
            optimizer_seed=opt_seed,
            planned_fem_budget=PLANNED_FEM_BUDGET,
            num_particles=NUM_PARTICLES,
            max_iterations=MAX_ITERATIONS,
            wall_time_seconds=run_wall_time,
            forward_operator_inv=G_inv,
            transform=transform,
            regularizer=reg,
        )
        diagnostic_records.append(rec)

        errs = rec.individual_relative_errors_pct
        print(
            f"  [{run_idx + 1}/{NUM_MULTISTART_RUNS}] Seed={opt_seed} | "
            f"J_tot={rec.best_cost_J_total:.3e} (J_data={rec.best_cost_J_data:.3e}, J_reg={rec.best_cost_J_reg:.3e}) | "
            f"Res.Rel={rec.relative_residual_vs_clean_target:.3e} | "
            f"Err[%]: alpha={errs[0]:.2f}, beta={errs[1]:.2f}, D={errs[2]:.2f}, E={errs[3]:.2f} | "
            f"Val.FEM={rec.optimizer_fem_evaluations}+{rec.diagnostic_fem_evaluations} diag ({rec.wall_time_seconds:.1f}s)"
        )

    # 4. Calcolo statistiche descrittive tra i 5 seed (senza assumere convergenza globale)
    theta_hat_matrix = np.vstack([r.theta_hat for r in diagnostic_records])
    err_pct_matrix = np.vstack([r.individual_relative_errors_pct for r in diagnostic_records])
    j_tot_vec = np.array([r.best_cost_J_total for r in diagnostic_records], dtype=np.float64)
    j_dat_vec = np.array([r.best_cost_J_data for r in diagnostic_records], dtype=np.float64)
    j_reg_vec = np.array([r.best_cost_J_reg for r in diagnostic_records], dtype=np.float64)
    res_clean_vec = np.array([r.relative_residual_vs_clean_target for r in diagnostic_records], dtype=np.float64)

    summary_payload = {
        "experiment_id": "RETECO-v1-TWIN-NOISE-FREE-MULTISTART-03A",
        "configuration": {
            "target_discretization": tar_cfg,
            "inverse_discretization": inv_cfg,
            "eta_level": 0.0,
            "regularization_lambda": lambda_reg,
            "prior_theta_tilde_0": prior_theta_tilde_0.tolist(),
            "num_multistart_runs": NUM_MULTISTART_RUNS,
            "optimizer_seeds": optimizer_seeds,
            "cpso_num_particles": NUM_PARTICLES,
            "cpso_max_iterations": MAX_ITERATIONS,
            "planned_fem_budget_per_run": PLANNED_FEM_BUDGET,
        },
        "evaluation_accounting": {
            "target_fine_mesh_evaluations": 1,
            "discretization_gap_diagnostic_evaluations_inv_mesh": 1,
            "planned_optimizer_evaluations_per_run": PLANNED_FEM_BUDGET,
            "total_optimizer_evaluations_consumed": int(
                sum(r.optimizer_fem_evaluations for r in diagnostic_records)
            ),
            "total_post_run_diagnostic_evaluations_inv_mesh": int(
                sum(r.diagnostic_fem_evaluations for r in diagnostic_records)
            ),
            "target_generation_wall_time_seconds": target_generation_wall_time,
        },
        "discretization_relative_gap": discretization_rel_gap,
        "theta_true": {
            name: float(theta_dagger[i]) for i, name in enumerate(PARAM_NAMES)
        },
        "across_seeds_summary": {
            "theta_hat_mean": {
                name: float(np.mean(theta_hat_matrix[:, i])) for i, name in enumerate(PARAM_NAMES)
            },
            "theta_hat_std": {
                name: float(np.std(theta_hat_matrix[:, i], ddof=1)) for i, name in enumerate(PARAM_NAMES)
            },
            "theta_hat_min": {
                name: float(np.min(theta_hat_matrix[:, i])) for i, name in enumerate(PARAM_NAMES)
            },
            "theta_hat_max": {
                name: float(np.max(theta_hat_matrix[:, i])) for i, name in enumerate(PARAM_NAMES)
            },
            "relative_error_pct_mean": {
                name: float(np.mean(err_pct_matrix[:, i])) for i, name in enumerate(PARAM_NAMES)
            },
            "relative_error_pct_std": {
                name: float(np.std(err_pct_matrix[:, i], ddof=1)) for i, name in enumerate(PARAM_NAMES)
            },
            "best_cost_J_total_min": float(np.min(j_tot_vec)),
            "best_cost_J_total_max": float(np.max(j_tot_vec)),
            "best_cost_J_data_min": float(np.min(j_dat_vec)),
            "best_cost_J_data_max": float(np.max(j_dat_vec)),
            "best_cost_J_reg_min": float(np.min(j_reg_vec)),
            "best_cost_J_reg_max": float(np.max(j_reg_vec)),
            "relative_residual_clean_min": float(np.min(res_clean_vec)),
            "relative_residual_clean_max": float(np.max(res_clean_vec)),
        },
        "runs": [serialize_run_record(r) for r in diagnostic_records],
    }

    # 5. Salvataggio artefatti JSON, CSV, figure PNG e report Markdown
    json_path = out_dir / "multistart_summary.json"
    runs_csv_path = out_dir / "multistart_runs_summary.csv"
    hist_csv_path = out_dir / "multistart_convergence_history.csv"
    fig_rec_path = out_dir / "fig_multistart_parameter_recovery.png"
    fig_conv_path = out_dir / "fig_multistart_convergence.png"
    fig_dec_path = out_dir / "fig_multistart_objective_decomposition.png"
    report_md_path = out_dir / "multistart_diagnostic_report.md"

    json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    write_runs_summary_csv(diagnostic_records, runs_csv_path)
    write_convergence_history_csv(diagnostic_records, hist_csv_path)

    plot_multistart_parameter_recovery(diagnostic_records, bounds_dict, fig_rec_path)
    plot_multistart_convergence(diagnostic_records, fig_conv_path)
    plot_multistart_objective_decomposition(diagnostic_records, fig_dec_path)
    write_multistart_markdown_report(
        diagnostic_records,
        discretization_rel_gap=discretization_rel_gap,
        lambda_reg=lambda_reg,
        prior_theta_tilde_0=prior_theta_tilde_0.tolist(),
        target_cfg=tar_cfg,
        inverse_cfg=inv_cfg,
        report_path=report_md_path,
    )

    print(f"[03a] Artefatti multistart completati e salvati in: {out_dir}")


if __name__ == "__main__":
    main()