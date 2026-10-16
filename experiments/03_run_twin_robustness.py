"""
experiments/03_run_twin_robustness.py
Twin Experiment anti-inverse-crime:
- Genera il target y_clean sulla discretizzazione fine h_tar (N=32, M=160)
- Quantifica la discrepanza di discretizzazione ||G_{h_tar}(theta_dagger) - G_{h_inv}(theta_dagger)||
- Esegue l'inversione CPSO sulla discretizzazione standard h_inv (N=16, M=80)
  per i livelli di rumore eta in {0%, 1%, 5%, 10%}.
"""

from __future__ import annotations
import json
from pathlib import Path
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
from reteco.visualization.benchmark_plots import compute_individual_relative_errors_pct


def main() -> None:
    out_dir = Path("artifacts/03_twin_robustness")
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
    reg = TikhonovRegularizer(
        transform,
        lambda_reg=float(noise_cfg["regularization_lambda"]),
        theta_tilde_0=np.array(noise_cfg["prior_theta_tilde_0"], dtype=np.float64),
    )

    results_log = {
        "discretization_relative_gap": discretization_rel_gap,
        "noise_levels": {},
    }

    for eta in noise_cfg["noise_levels_eta"]:
        seed_eta = int(noise_cfg["base_seed"]) + int(round(eta * 1000))
        sample = generate_gaussian_noisy_observations(y_target_clean, float(eta), seed_eta)

        obj = InverseObjectiveFunctional(
            G_inv,
            sample.y_noisy,
            sample.weight_matrix_diag,
            transform,
            reg,
            max_evaluations=120,
        )
        res = run_cpso_optimization(
            obj, seed=seed_eta + 1, num_particles=10, max_iterations=12
        )
        rel_errs = compute_individual_relative_errors_pct(res.theta_hat, theta_dagger)
        print(
            f"  Rumore eta={eta*100:4.1f}% | J={res.best_cost_J:.3e} | "
            f"Errori rel [%]: alpha={rel_errs[0]:.2f}, beta={rel_errs[1]:.2f}, "
            f"D={rel_errs[2]:.2f}, E={rel_errs[3]:.2f}"
        )
        results_log["noise_levels"][str(eta)] = {
            "seed": seed_eta,
            "theta_hat": res.theta_hat.tolist(),
            "individual_relative_errors_pct": rel_errs.tolist(),
            "best_cost_J": res.best_cost_J,
            "num_fem_evaluations": res.num_fem_evaluations,
        }

    (out_dir / "twin_robustness_results.json").write_text(
        json.dumps(results_log, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()