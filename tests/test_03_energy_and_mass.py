"""
tests/test_03_energy_and_mass.py
Gate 4 e Gate 5 del Contratto Matematico (Sez. 9.4 e 9.5) + Identità Energetica Discreta Esatta:
- Conservazione della massa spaziale integrata del tracciante M(t_n) = M(0) quando s_c = 0 e F_c . n = 0
- Verifica dell'identità energetica discreta esatta ad ogni passo temporale:
    R_n - R_{n-1} = - 0.5 * E * ||Delta d^n||_{K0}^2 - 0.5 * S_p * ||Delta p^n||_{Mp}^2
- Convergenza a zero del residuo cumulativo |R_N| al raffinarsi del passo temporale dt.
"""

from __future__ import annotations
import numpy as np
import yaml
from reteco import _core, load_master_config
from reteco.forward.wrapper import RetecoForwardProblem2D
from reteco.forward.target_generator import build_standard_driving_Callbacks


def test_discrete_tracer_mass_conservation_under_active_darcy_flow() -> None:
    master = load_master_config("configs/scales_and_physics.yaml")
    with open("configs/twin_experiment_synthetic.yaml", "r", encoding="utf-8") as f:
        twin_cfg = yaml.safe_load(f)

    grid = _core.TimeGrid.make_graded(1.0, 40, 1.5)
    problem = RetecoForwardProblem2D(
        subdivisions_N=12,
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid,
    )

    c0, F_mech_func, S_fluid_func = build_standard_driving_Callbacks(
        problem, twin_cfg["driving_protocol"]
    )
    theta = master.parameter_space_theta.theta_dagger_array()

    traj = problem.solve_trajectory(
        theta_vec=theta,
        c0=c0,
        F_mech_func=F_mech_func,
        S_fluid_func=S_fluid_func,
        S_tracer_func=None,
        compute_diagnostics=True,
    )

    assert traj.max_darcy_velocity > 1.0e-4

    masses = np.array([d.tracer_total_mass for d in traj.diagnostics], dtype=np.float64)
    m0 = masses[0]
    rel_mass_variation = np.max(np.abs(masses - m0)) / m0
    assert rel_mass_variation < 1.0e-12, (
        f"Violazione della conservazione discreta della massa: {rel_mass_variation:.3e}"
    )


def test_exact_stepwise_discrete_energy_identity_and_refinement() -> None:
    """
    Verifica a ogni singolo passo temporale che la discrepanza energetica R_n soddisfi esattamente:
        R_n - R_{n-1} = - (0.5 * E * ||Delta d^n||_{K0}^2 + 0.5 * S_p * ||Delta p^n||_{Mp}^2)
    a precisione di macchina, e che |R_N| decresca con ordine O(dt) al raffinarsi del passo.
    """
    master = load_master_config("configs/scales_and_physics.yaml")
    theta = master.parameter_space_theta.theta_dagger_array()

    residuals_by_M = []
    for M_steps in [25, 50, 100]:
        grid = _core.TimeGrid.make_graded(0.5, M_steps, 1.0)
        problem = RetecoForwardProblem2D(
            subdivisions_N=10,
            physics_model=master.fixed_physics,
            scales=master.characteristic_scales,
            time_grid=grid,
        )

        px = problem.press_pts[:, 0]
        py = problem.press_pts[:, 1]
        p0 = 0.5 * np.cos(np.pi * px) * np.cos(np.pi * py)
        c0 = np.ones(problem.num_tracer_dofs, dtype=np.float64)

        traj = problem.solve_trajectory(
            theta_vec=theta,
            c0=c0,
            p0=p0,
            d0=None,
            F_mech_func=None,
            S_fluid_func=None,
            enforce_initial_compatibility=True,
            compute_diagnostics=True,
        )

        assert traj.initial_compatibility_residual < 1.0e-12
        e0 = traj.diagnostics[0].total_energy
        assert e0 > 0.0

        cum_num_dissip = 0.0
        for diag in traj.diagnostics[1:]:
            assert diag.total_energy <= e0 + 1.0e-12
            assert diag.cumulative_visco_dissip >= -1.0e-12
            assert diag.cumulative_darcy_dissip >= 0.0
            assert diag.step_numerical_dissipation >= 0.0
            # Verifica esatta dell'identità energetica discreta passo-passo (Sez. 2.1 ChatGPT)
            assert diag.step_discrete_identity_error < 1.0e-12 * e0, (
                f"Identità energetica discreta violata: err={diag.step_discrete_identity_error:.3e}"
            )
            cum_num_dissip += diag.step_numerical_dissipation
            assert abs(diag.energy_balance_residual + cum_num_dissip) < 1.0e-11 * e0

        residuals_by_M.append(abs(traj.diagnostics[-1].energy_balance_residual))

    # Verifica che la dissipazione numerica cumulativa |R_N| si dimezzi raddoppiando M_steps
    assert residuals_by_M[1] < 0.60 * residuals_by_M[0]
    assert residuals_by_M[2] < 0.60 * residuals_by_M[1]