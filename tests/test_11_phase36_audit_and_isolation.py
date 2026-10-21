"""
tests/test_11_phase36_audit_and_isolation.py
RETECO — Fase 3.6: Test unitari dedicati per:
1. `verify_history_and_best_candidate_consistency()` integrato nel flusso diagnostico;
2. Rilevamento automatico di violazioni su J_total = J_data + J_reg, monotonia o contatori;
3. Isolamento del target pulito in Modalità B (eta > 0): due campioni con uguali osservazioni
   rumorose e uguale deviazione standard ma vettori puliti differenti producono lo stesso funzionale;
4. Riproducibilità esatta del CPSO a parità di seed e rispetto del budget a 600 valutazioni.
"""

from __future__ import annotations
from dataclasses import replace
import numpy as np
import pytest

from reteco import load_master_config
from reteco.config_schema import ParameterCoordinateTransform
from reteco.observation.noise_model import (
    NoisyObservationSample,
    generate_gaussian_noisy_observations,
)
from reteco.inverse.regularization import TikhonovRegularizer
from reteco.inverse.objective_functional import InverseObjectiveFunctional
from reteco.inverse.cpso import run_cpso_optimization
from reteco.inverse.diagnostics import extract_run_diagnostics_from_history


def _build_analytic_surrogate(theta_true: np.ndarray, n_obs: int = 180):
    grid_s = np.linspace(0.1, 1.0, n_obs, dtype=np.float64)

    def forward_op(theta: np.ndarray) -> np.ndarray:
        th = np.asarray(theta, dtype=np.float64).ravel()
        return (
            th[0] * np.exp(-th[2] * grid_s)
            + 0.25 * th[1] * np.sin(2.0 * np.pi * grid_s)
            + 0.10 * th[3] * (grid_s ** 2)
        )

    return forward_op


def test_history_and_best_candidate_consistency_in_diagnostics() -> None:
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_true = master.parameter_space_theta.theta_dagger_array()
    fwd = _build_analytic_surrogate(theta_true)
    y_clean = fwd(theta_true)

    reg = TikhonovRegularizer(
        transform=transform,
        lambda_reg=1.0e-4,
        theta_tilde_0=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64),
    )
    obj = InverseObjectiveFunctional.from_mode_a_deterministic(
        fwd,
        y_clean,
        transform,
        reg,
        y_reference=y_clean,
        max_evaluations=40,
    )
    res = run_cpso_optimization(obj, seed=20261017, num_particles=10, max_iterations=4)

    rec = extract_run_diagnostics_from_history(
        run_id="test_consistency_40",
        obj=obj,
        opt_result=res,
        theta_true=theta_true,
        y_target_clean=y_clean,
        y_noisy=y_clean,
        nominal_noise_std=0.0,
        eta_level=0.0,
        data_seed=20261016,
        optimizer_seed=20261017,
        planned_fem_budget=40,
        num_particles=10,
        max_iterations=4,
        wall_time_seconds=0.01,
        forward_operator_inv=fwd,
        transform=transform,
        regularizer=reg,
    )

    checks = rec.consistency_checks
    assert checks["all_checks_passed"] is True
    assert checks["eval_count_matches_history_length"] is True
    assert checks["num_fem_evaluations"] == 40
    assert checks["history_length"] == 40
    assert checks["j_total_equals_j_data_plus_j_reg_all_records"] is True
    assert checks["best_cost_history_monotone_non_increasing"] is True
    assert checks["historical_min_matches_best_cost_eval_index_and_theta_hat"] is True
    assert checks["normalized_and_physical_coordinates_consistent"] is True

    # Verifica che una corruzione dell'identità J_total = J_data + J_reg sollevi RuntimeError
    corrupted_rec = replace(obj.history[5], j_total=obj.history[5].j_total + 1.0e-3)
    obj.history[5] = corrupted_rec
    with pytest.raises(RuntimeError, match="Controllo di coerenza della cronologia fallito"):
        obj.verify_history_and_best_candidate_consistency(res, raise_on_failure=True)


def test_mode_b_clean_target_isolation_with_different_clean_vectors() -> None:
    """
    Fase 3.6 - Punto 2: Due campioni con uguali osservazioni rumorose e uguale deviazione
    standard nominale ma vettori puliti differenti devono produrre lo stesso identico funzionale.
    """
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_true = master.parameter_space_theta.theta_dagger_array()
    fwd = _build_analytic_surrogate(theta_true)
    y_clean_1 = fwd(theta_true)

    sample_1 = generate_gaussian_noisy_observations(
        y_clean_1,
        eta_level=0.05,
        seed=20261066,
        weighting_mode="mode_b_noise_weighted",
        covariance_scaling="exact_inverse_covariance",
    )

    # Costruiamo un secondo campione con le STESSE osservazioni rumorose e la STESSA sigma_std,
    # ma con un vettore pulito `y_clean_2` completamente diverso.
    y_clean_2 = 5.0 * y_clean_1 - 12.345
    # Manteniamo tutti i metadati del campione generato (i nomi effettivi dei campi
    # includono `signal_reference_scale`) e sostituiamo soltanto il target pulito.
    # Sigma nominale e osservazioni rumorose devono restare identiche per isolare il test.
    sample_2 = replace(
        sample_1,
        y_clean=y_clean_2,
        y_noisy=sample_1.y_noisy.copy(),
        noise_vector=sample_1.noise_vector.copy(),
        weight_matrix_diag=sample_1.weight_matrix_diag.copy(),
        inverse_covariance_diag=(
            None if sample_1.inverse_covariance_diag is None
            else sample_1.inverse_covariance_diag.copy()
        ),
    )

    reg = TikhonovRegularizer(
        transform=transform,
        lambda_reg=7.2,
        theta_tilde_0=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64),
    )

    obj_1 = InverseObjectiveFunctional.from_mode_b_noise_model(
        fwd, sample_1, transform, reg, covariance_scaling="exact_inverse_covariance"
    )
    obj_2 = InverseObjectiveFunctional.from_mode_b_noise_model(
        fwd, sample_2, transform, reg, covariance_scaling="exact_inverse_covariance"
    )

    assert not np.allclose(sample_1.y_clean, sample_2.y_clean)
    assert np.array_equal(obj_1.y_observed, obj_2.y_observed)
    assert np.array_equal(obj_1.weight_matrix_diag, obj_2.weight_matrix_diag)
    assert obj_1.residual_reference_scale == obj_2.residual_reference_scale
    assert obj_1.uses_clean_target_vector_in_evaluation is False
    assert obj_2.uses_clean_target_vector_in_evaluation is False

    for u in [0.15, 0.50, 0.85]:
        th_test = transform.from_normalized(np.full(4, u, dtype=np.float64))
        bd_1 = obj_1.evaluate_breakdown(th_test)
        bd_2 = obj_2.evaluate_breakdown(th_test)
        assert bd_1.j_data == bd_2.j_data
        assert bd_1.j_reg == bd_2.j_reg
        assert bd_1.j_total == bd_2.j_total
        assert bd_1.weighted_residual_l2_norm == bd_2.weighted_residual_l2_norm


def test_cpso_600_budget_enforcement_and_seed_reproducibility() -> None:
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_true = master.parameter_space_theta.theta_dagger_array()
    fwd = _build_analytic_surrogate(theta_true)
    y_clean = fwd(theta_true)

    reg = TikhonovRegularizer(
        transform=transform,
        lambda_reg=1.0e-4,
        theta_tilde_0=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64),
    )

    obj_a = InverseObjectiveFunctional.from_mode_a_deterministic(
        fwd,
        y_clean,
        transform,
        reg,
        y_reference=y_clean,
        max_evaluations=600,
    )
    obj_b = InverseObjectiveFunctional.from_mode_a_deterministic(
        fwd,
        y_clean,
        transform,
        reg,
        y_reference=y_clean,
        max_evaluations=600,
    )

    res_a = run_cpso_optimization(obj_a, seed=20261017, num_particles=10, max_iterations=60)
    res_b = run_cpso_optimization(obj_b, seed=20261017, num_particles=10, max_iterations=60)

    assert obj_a.max_evaluations == 600
    assert obj_a.num_fem_evaluations == 600
    assert len(obj_a.history) == 600
    assert res_a.num_fem_evaluations == 600

    rep_a = obj_a.verify_history_and_best_candidate_consistency(res_a, raise_on_failure=True)
    rep_b = obj_b.verify_history_and_best_candidate_consistency(res_b, raise_on_failure=True)
    assert rep_a["all_checks_passed"] is True
    assert rep_b["all_checks_passed"] is True

    assert res_a.best_cost_J == res_b.best_cost_J
    assert np.array_equal(res_a.theta_hat, res_b.theta_hat)
    assert np.array_equal(res_a.convergence_eval_trace, res_b.convergence_eval_trace)