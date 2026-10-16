"""
experiments/02_run_identifiability_study.py
Esegue lo studio di identificabilità locale (Jacobiano normalizzato + SVD) e pratica
(profili 1D di J_h), confrontando il caso con flusso di Darcy nullo (Proposizione 2)
e il caso con protocollo di eccitazione meccanico-idraulica attivo.
"""

from __future__ import annotations
import json
from pathlib import Path
import yaml

from reteco import _core, load_master_config
from reteco.config_schema import ParameterCoordinateTransform
from reteco.forward.wrapper import RetecoForwardProblem2D
from reteco.forward.target_generator import build_standard_driving_Callbacks
from reteco.observation.operator import DiscreteObservationOperator
from reteco.identifiability.sensitivity_jacobian import compute_normalized_jacobian
from reteco.identifiability.svd_conditioning import analyze_jacobian_svd
from reteco.visualization.identifiability_plots import plot_svd_identifiability_summary


def main() -> None:
    out_dir = Path("artifacts/02_identifiability")
    out_dir.mkdir(parents=True, exist_ok=True)

    master = load_master_config("configs/scales_and_physics.yaml")
    with open("configs/twin_experiment_synthetic.yaml", "r", encoding="utf-8") as f:
        twin_cfg = yaml.safe_load(f)

    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_dagger = master.parameter_space_theta.theta_dagger_array()

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

    c0, F_mech_func, S_fluid_func = build_standard_driving_Callbacks(
        problem, twin_cfg["driving_protocol"]
    )
    obs_cfg = twin_cfg["observation_protocol"]
    obs_op = DiscreteObservationOperator.from_regular_grid(
        nx=int(obs_cfg["sensor_grid_nx"]),
        ny=int(obs_cfg["sensor_grid_ny"]),
        margin=float(obs_cfg["sensor_margin"]),
        observation_times=obs_cfg["observation_times"],
        observed_fields=obs_cfg["observed_fields"],
    )

    # 1. Caso q == 0 (Proposizione 2)
    J_zero = compute_normalized_jacobian(
        lambda th: obs_op.apply(problem, problem.solve_trajectory(th, c0=c0)),
        theta_dagger,
        transform,
    )
    rep_zero = analyze_jacobian_svd(J_zero)

    # 2. Caso q != 0 (Protocollo attivo)
    J_active = compute_normalized_jacobian(
        lambda th: obs_op.apply(
            problem,
            problem.solve_trajectory(
                th, c0=c0, F_mech_func=F_mech_func, S_fluid_func=S_fluid_func
            ),
        ),
        theta_dagger,
        transform,
    )
    rep_active = analyze_jacobian_svd(J_active)

    plot_svd_identifiability_summary(
        rep_active, rep_zero, out_dir / "fig_svd_identifiability.pdf"
    )

    report_json = {
        "zero_flow_proposition_check": {
            "column_norms_alpha_beta_D_E": rep_zero.column_norms.tolist(),
            "singular_values": rep_zero.singular_values.tolist(),
            "numerical_rank": rep_zero.numerical_rank,
        },
        "active_flow_protocol": {
            "column_norms_alpha_beta_D_E": rep_active.column_norms.tolist(),
            "singular_values": rep_active.singular_values.tolist(),
            "numerical_rank": rep_active.numerical_rank,
            "condition_number": rep_active.condition_number,
            "collinearity_index": rep_active.collinearity_index,
            "fisher_correlation_matrix": rep_active.fisher_correlation_matrix.tolist(),
        },
    }
    (out_dir / "identifiability_report.json").write_text(
        json.dumps(report_json, indent=2), encoding="utf-8"
    )
    print(f"Studio di identificabilità completato e salvato in: {out_dir}")


if __name__ == "__main__":
    main()