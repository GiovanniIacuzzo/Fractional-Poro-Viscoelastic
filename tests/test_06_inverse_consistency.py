"""
tests/test_06_inverse_consistency.py
Gate 8 e Gate 9 del Contratto Matematico (Sez. 8, Sez. 9.8, 9.9):
- Biiezione esatta tra spazio fisico/adimensionale Theta e cubo normalizzato [0, 1]^4
- Continuità dei 3 regimi analitici di CPSO (radici reali distinte, coincidenti, complesse)
- Riproducibilità bit-a-bit col seed e rispetto del budget di valutazioni FEM.
"""

from __future__ import annotations
import numpy as np

from reteco import load_master_config
from reteco.config_schema import ParameterCoordinateTransform
from reteco.observation.noise_model import generate_gaussian_noisy_observations
from reteco.inverse.regularization import TikhonovRegularizer
from reteco.inverse.objective_functional import InverseObjectiveFunctional
from reteco.inverse.cpso import _solve_cpso_analytical_step, run_cpso_optimization
from reteco.inverse.sota_runners import run_scipy_baseline


def test_parameter_normalization_bijection() -> None:
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_true = master.parameter_space_theta.theta_dagger_array()

    tilde = transform.to_normalized(theta_true)
    assert np.all((0.0 < tilde) & (tilde < 1.0))

    recovered = transform.from_normalized(tilde)
    assert np.allclose(theta_true, recovered, rtol=1.0e-14, atol=1.0e-14)


def test_cpso_analytical_step_three_spectral_regimes_continuity() -> None:
    """
    Verifica che il solutore analitico dell'ODE del 2° ordine di CPSO sia continuo
    attraversando la soglia critica Delta = a^2 - 4b = 0.
    """
    x0 = np.array([0.2, 0.8, 0.5, 0.1], dtype=np.float64)
    v0 = np.array([0.05, -0.04, 0.02, 0.0], dtype=np.float64)
    attr = np.array([0.6, 0.3, 0.5, 0.9], dtype=np.float64)
    a_damp = 0.4  # a^2 = 0.16 => soglia critica b_crit = 0.04
    b_crit = 0.04

    x_coinc, v_coinc = _solve_cpso_analytical_step(
        x0, v0, attr, a_damp, np.full(4, b_crit), dt=1.0
    )
    x_real, v_real = _solve_cpso_analytical_step(
        x0, v0, attr, a_damp, np.full(4, b_crit - 1.0e-7), dt=1.0
    )
    x_comp, v_comp = _solve_cpso_analytical_step(
        x0, v0, attr, a_damp, np.full(4, b_crit + 1.0e-7), dt=1.0
    )

    assert np.allclose(x_coinc, x_real, atol=1.0e-6)
    assert np.allclose(x_coinc, x_comp, atol=1.0e-6)
    assert np.allclose(v_coinc, v_real, atol=1.0e-6)
    assert np.allclose(v_coinc, v_comp, atol=1.0e-6)


def test_cpso_and_sota_seed_reproducibility_and_budget_enforcement() -> None:
    """
    Verifica su un operatore surrogato veloce che:
    1) Due esecuzioni CPSO con lo stesso seed producano esattamente lo stesso vettore bit-a-bit;
    2) CPSO e i metodi SOTA rispettino rigorosamente `max_evaluations`.
    """
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_dagger = master.parameter_space_theta.theta_dagger_array()

    # Mappa sintetica iniettiva ben condizionata per testare l'infrastruttura di ottimizzazione
    A_synth = np.array(
        [
            [1.2, 0.1, -0.2, 0.0],
            [0.0, 0.9, 0.1, -0.1],
            [-0.1, 0.0, 1.5, 0.2],
            [0.1, -0.2, 0.0, 1.1],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )

    def mock_G(theta: np.ndarray) -> np.ndarray:
        t_tilde = transform.to_normalized(theta)
        return A_synth @ t_tilde

    y_clean = mock_G(theta_dagger)
    sample = generate_gaussian_noisy_observations(y_clean, eta_level=0.01, seed=42)
    reg = TikhonovRegularizer(transform, lambda_reg=1.0e-4)

    budget = 36
    obj1 = InverseObjectiveFunctional(
        mock_G, sample.y_noisy, sample.weight_matrix_diag, transform, reg, max_evaluations=budget
    )
    res1 = run_cpso_optimization(obj1, seed=12345, num_particles=6, max_iterations=6)

    obj2 = InverseObjectiveFunctional(
        mock_G, sample.y_noisy, sample.weight_matrix_diag, transform, reg, max_evaluations=budget
    )
    res2 = run_cpso_optimization(obj2, seed=12345, num_particles=6, max_iterations=6)

    assert res1.num_fem_evaluations == budget
    assert np.array_equal(res1.theta_hat, res2.theta_hat)
    assert res1.best_cost_J == res2.best_cost_J

    for alg in ["L-BFGS-B", "Powell", "Nelder-Mead", "Differential-Evolution"]:
        obj_b = InverseObjectiveFunctional(
            mock_G, sample.y_noisy, sample.weight_matrix_diag, transform, reg, max_evaluations=budget
        )
        res_b = run_scipy_baseline(alg, obj_b, seed=12345)
        assert res_b.num_fem_evaluations <= budget
        assert transform.is_within_bounds(res_b.theta_hat)