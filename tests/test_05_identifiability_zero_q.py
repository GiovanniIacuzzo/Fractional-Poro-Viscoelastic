"""
tests/test_05_identifiability_zero_q.py
Gate 7 del Contratto Matematico (Proposizione Flusso Nullo, Sez. 8 e Sez. 9.7):
1) Test negativo obbligatorio: con d0=0, p0=0, F=0, s_f=0 (quindi q == 0 per ogni E, beta)
   e osservando solo c, le colonne del Jacobiano rispetto a beta ed E devono essere nulle.
2) Test positivo: attivando le forzanti meccanico-idrauliche, q != 0 e tutte e 4 le colonne
   del Jacobiano sono non nulle con rango numerico pieno pari a 4.
"""

from __future__ import annotations
import numpy as np
import yaml

from reteco import _core, load_master_config
from reteco.config_schema import ParameterCoordinateTransform
from reteco.forward.wrapper import RetecoForwardProblem2D
from reteco.forward.target_generator import build_standard_driving_Callbacks
from reteco.observation.operator import DiscreteObservationOperator
from reteco.identifiability.sensitivity_jacobian import compute_normalized_jacobian
from reteco.identifiability.svd_conditioning import analyze_jacobian_svd


def test_proposition_zero_darcy_flow_yields_zero_beta_E_sensitivity() -> None:
    """Verifica numerica esatta della Proposizione `prop:zeroq`."""
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_ref = master.parameter_space_theta.theta_dagger_array()

    grid = _core.TimeGrid.make_graded(1.0, 30, 1.5)
    problem = RetecoForwardProblem2D(
        subdivisions_N=10,
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid,
    )
    c0 = problem.build_gaussian_initial_tracer()
    obs_op = DiscreteObservationOperator.from_regular_grid(
        nx=5, ny=5, observation_times=[0.25, 0.5, 0.75, 1.0], observed_fields=["c"]
    )

    def G_zero_flow(theta: np.ndarray) -> np.ndarray:
        traj = problem.solve_trajectory(
            theta_vec=theta,
            c0=c0,
            F_mech_func=None,
            S_fluid_func=None,
        )
        assert traj.max_darcy_velocity < 1.0e-15
        return obs_op.apply(problem, traj)

    J_zero = compute_normalized_jacobian(G_zero_flow, theta_ref, transform)
    rep_zero = analyze_jacobian_svd(J_zero, rank_threshold=1.0e-7)

    # Colonne: 0 -> alpha~, 1 -> beta~, 2 -> D~, 3 -> E~
    assert rep_zero.column_norms[0] > 1.0e-2, "La sensibilità rispetto ad alpha deve essere > 0."
    assert rep_zero.column_norms[2] > 1.0e-2, "La sensibilità rispetto a D deve essere > 0."
    assert rep_zero.column_norms[1] < 1.0e-13, (
        f"Con q=0 la colonna di beta deve essere nulla, trovato {rep_zero.column_norms[1]}"
    )
    assert rep_zero.column_norms[3] < 1.0e-13, (
        f"Con q=0 la colonna di E deve essere nulla, trovato {rep_zero.column_norms[3]}"
    )
    assert rep_zero.numerical_rank == 2


def test_active_driving_protocol_restores_full_rank_four() -> None:
    """Verifica che il protocollo con sorgente idraulica/meccanica attivi il canale indiretto su (beta, E)."""
    master = load_master_config("configs/scales_and_physics.yaml")
    with open("configs/twin_experiment_synthetic.yaml", "r", encoding="utf-8") as f:
        twin_cfg = yaml.safe_load(f)

    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_ref = master.parameter_space_theta.theta_dagger_array()

    grid = _core.TimeGrid.make_graded(1.0, 30, 1.5)
    problem = RetecoForwardProblem2D(
        subdivisions_N=10,
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid,
    )
    c0, F_mech_func, S_fluid_func = build_standard_driving_Callbacks(
        problem, twin_cfg["driving_protocol"]
    )
    obs_op = DiscreteObservationOperator.from_regular_grid(
        nx=5, ny=5, observation_times=[0.25, 0.5, 0.75, 1.0], observed_fields=["c"]
    )

    def G_active_flow(theta: np.ndarray) -> np.ndarray:
        traj = problem.solve_trajectory(
            theta_vec=theta,
            c0=c0,
            F_mech_func=F_mech_func,
            S_fluid_func=S_fluid_func,
        )
        return obs_op.apply(problem, traj)

    J_active = compute_normalized_jacobian(G_active_flow, theta_ref, transform)
    rep_active = analyze_jacobian_svd(J_active, rank_threshold=1.0e-6)

    assert np.all(rep_active.column_norms > 0.0), f"Trovate colonne nulle: {rep_active.column_norms}"
    assert rep_active.numerical_rank == 4, (
        f"Rango numerico inferiore a 4: {rep_active.numerical_rank}, sigma={rep_active.singular_values}"
    )