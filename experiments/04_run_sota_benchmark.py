"""
experiments/04_run_sota_benchmark.py
Confronto equo multi-run a budget fissato di valutazioni FEM tra:
CPSO, L-BFGS-B, Powell, Nelder-Mead e Differential-Evolution.
"""

from __future__ import annotations
import json
from pathlib import Path
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
from reteco.inverse.cpso import OptimizationRunResult, run_cpso_optimization
from reteco.inverse.sota_runners import run_scipy_baseline
from reteco.visualization.benchmark_plots import (
    export_sota_latex_table,
    plot_sota_convergence_by_fem_evaluations,
)


def main() -> None:
    out_dir = Path("artifacts/04_sota_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    master = load_master_config("configs/scales_and_physics.yaml")
    with open("configs/twin_experiment_synthetic.yaml", "r", encoding="utf-8") as f:
        twin_cfg = yaml.safe_load(f)
    with open("configs/sota_comparison.yaml", "r", encoding="utf-8") as f:
        sota_cfg = yaml.safe_load(f)

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

    # Target su discretizzazione fine h_tar
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
    y_clean = obs_op.apply(
        prob_tar,
        prob_tar.solve_trajectory(theta_dagger, c0=c0_tar, F_mech_func=F_tar, S_fluid_func=Sf_tar),
    )

    # Solutore inverso su discretizzazione h_inv
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

    budget_cfg = sota_cfg["evaluation_budget"]
    max_evals = int(budget_cfg["max_fem_evaluations"])
    n_reps = int(budget_cfg["num_independent_replicates"])
    master_seed = int(budget_cfg["master_seed"])

    sample_5pct = generate_gaussian_noisy_observations(y_clean, eta_level=0.05, seed=master_seed)
    reg = TikhonovRegularizer(transform, lambda_reg=1.0e-4)

    algorithms = ["CPSO", "L-BFGS-B", "Powell", "Nelder-Mead", "Differential-Evolution"]
    results_by_alg: Dict[str, List[OptimizationRunResult]] = {alg: [] for alg in algorithms}

    cpso_cfg = sota_cfg["optimizers"]["CPSO"]

    for rep in range(n_reps):
        run_seed = master_seed + 100 * (rep + 1)
        print(f"=== Replica {rep + 1}/{n_reps} (seed={run_seed}) ===")
        for alg in algorithms:
            obj = InverseObjectiveFunctional(
                G_inv,
                sample_5pct.y_noisy,
                sample_5pct.weight_matrix_diag,
                transform,
                reg,
                max_evaluations=max_evals,
            )
            if alg == "CPSO":
                res = run_cpso_optimization(
                    obj,
                    seed=run_seed,
                    num_particles=int(cpso_cfg["num_particles"]),
                    max_iterations=int(cpso_cfg["max_iterations"]),
                    omega_inertia=float(cpso_cfg["omega_inertia"]),
                    c1_cognitive=float(cpso_cfg["c1_cognitive"]),
                    c2_social=float(cpso_cfg["c2_social"]),
                    dt_step=float(cpso_cfg["dt_step"]),
                )
            else:
                res = run_scipy_baseline(alg, obj, seed=run_seed)

            results_by_alg[alg].append(res)
            print(
                f"  {alg:22s} | Evals={res.num_fem_evaluations:3d} | "
                f"J={res.best_cost_J:.3e} | theta_hat={np.round(res.theta_hat, 4)}"
            )

    export_sota_latex_table(results_by_alg, theta_dagger, out_dir / "table_sota_comparison.tex")
    plot_sota_convergence_by_fem_evaluations(
        results_by_alg, out_dir / "fig_sota_convergence_fair_budget.pdf"
    )
    print(f"Benchmark SOTA completato. Output salvati in: {out_dir}")


if __name__ == "__main__":
    main()