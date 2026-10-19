"""
tests/test_10_objective_functional_modes.py
Fase 2 — Suite di test obbligatori sulla coerenza matematica della funzione obiettivo (Sezione 6):
1. Decomposizione esatta J_total = J_data + J_reg;
2. Riproducibilità della funzione obiettivo e della generazione dei campioni osservativi;
3. Assenza di divisioni per zero quando eta = 0 (sia in Modalità A sia in Modalità B);
4. Coerenza della normalizzazione tra candidati valutati con gli stessi dati;
5. Coerenza della trasformazione dei parametri psi(theta) e del prior theta_tilde_0;
6. Comportamento al variare di lambda (scaling lineare di J_reg e invarianza di J_data);
7. Equivalenza esatta fra la modalità storica/legacy e la nuova modalità compatibile;
8. Distinzione rigorosa fra rumore nominale (eta, sigma_nom) e norma effettiva della realizzazione.
"""

from __future__ import annotations
import numpy as np
import pytest

from reteco import load_master_config
from reteco.config_schema import ParameterCoordinateTransform
from reteco.observation.noise_model import (
    build_mode_a_deterministic_weights,
    build_mode_b_noise_weights,
    generate_gaussian_noisy_observations,
)
from reteco.inverse.regularization import TikhonovRegularizer
from reteco.inverse.objective_functional import InverseObjectiveFunctional


def _make_surrogate_forward(n_obs: int = 36):
    """Operatore diretto sintetico deterministico per testare il funzionale obiettivo."""
    grid = np.linspace(0.05, 0.95, n_obs, dtype=np.float64)

    def G_surrogate(theta: np.ndarray) -> np.ndarray:
        th = np.asarray(theta, dtype=np.float64).ravel()
        alpha, beta, D, E = th[0], th[1], th[2], th[3]
        return (
            1.5 * np.exp(-alpha * grid)
            + 0.4 * np.cos(np.pi * beta * grid)
            + 3.0 * D * (1.0 - grid ** 2)
            + 0.2 * np.log1p(E) * np.sin(np.pi * grid)
        )

    return G_surrogate


def test_01_exact_additive_decomposition_j_total_j_data_j_reg() -> None:
    """1. Verifica la decomposizione esatta J_total = J_data + J_reg."""
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_true = master.parameter_space_theta.theta_dagger_array()
    G_op = _make_surrogate_forward()
    y_clean = G_op(theta_true)

    sample = generate_gaussian_noisy_observations(y_clean, eta_level=0.05, seed=20261016)
    reg = TikhonovRegularizer(transform, lambda_reg=1.0e-3)

    obj_a = InverseObjectiveFunctional.from_mode_a_deterministic(
        G_op, sample.y_noisy, transform, reg, y_reference=y_clean
    )
    obj_b = InverseObjectiveFunctional.from_mode_b_noise_model(
        G_op, sample, transform, reg
    )

    candidate = np.array([0.55, 0.62, 0.14, 3.40], dtype=np.float64)
    for obj in (obj_a, obj_b):
        val = obj(candidate)
        bd = obj.history[-1]
        assert np.isclose(val, bd.j_total, rtol=1e-15, atol=1e-16)
        assert np.isclose(bd.j_total, bd.j_data + bd.j_reg, rtol=1e-15, atol=1e-16)
        assert np.isclose(bd.data_share_of_total + bd.reg_share_of_total, 1.0, rtol=1e-14)
        # Accesso anche tramite dizionario per retrocompatibilità
        assert bd["j_total"] == bd.j_total
        assert bd["j_data"] == bd.j_data
        assert bd["j_reg"] == bd.j_reg


def test_02_objective_functional_and_noise_reproducibility() -> None:
    """2. Verifica la riproducibilità bit-a-bit a parità di seed e candidato."""
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_true = master.parameter_space_theta.theta_dagger_array()
    G_op = _make_surrogate_forward()
    y_clean = G_op(theta_true)

    s1 = generate_gaussian_noisy_observations(y_clean, eta_level=0.05, seed=777)
    s2 = generate_gaussian_noisy_observations(y_clean, eta_level=0.05, seed=777)
    np.testing.assert_array_equal(s1.y_noisy, s2.y_noisy)
    np.testing.assert_array_equal(s1.weight_matrix_diag, s2.weight_matrix_diag)

    reg = TikhonovRegularizer(transform, lambda_reg=2.5e-4)
    obj1 = InverseObjectiveFunctional.from_mode_b_noise_model(G_op, s1, transform, reg)
    obj2 = InverseObjectiveFunctional.from_mode_b_noise_model(G_op, s2, transform, reg)

    cand = np.array([0.42, 0.31, 0.05, 1.85], dtype=np.float64)
    bd1 = obj1.evaluate_breakdown(cand)
    bd2 = obj2.evaluate_breakdown(cand)

    assert bd1.j_total == bd2.j_total
    assert bd1.j_data == bd2.j_data
    assert bd1.j_reg == bd2.j_reg
    assert bd1.unweighted_residual_l2_norm == bd2.unweighted_residual_l2_norm


def test_03_no_division_by_zero_when_eta_is_zero() -> None:
    """3. Verifica l'assenza di divisioni per zero per eta = 0 (anche su vettore nullo)."""
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_true = master.parameter_space_theta.theta_dagger_array()
    G_op = _make_surrogate_forward()
    y_clean = G_op(theta_true)

    s_a = generate_gaussian_noisy_observations(
        y_clean, eta_level=0.0, seed=10, weighting_mode="mode_a_deterministic_normalized"
    )
    s_b = generate_gaussian_noisy_observations(
        y_clean, eta_level=0.0, seed=10, weighting_mode="mode_b_noise_weighted"
    )
    assert np.all(np.isfinite(s_a.weight_matrix_diag)) and np.all(s_a.weight_matrix_diag > 0.0)
    assert np.all(np.isfinite(s_b.weight_matrix_diag)) and np.all(s_b.weight_matrix_diag > 0.0)
    assert s_b.is_covariance_invertible is False

    # Verifica che eta_zero_policy='raise' segnali esplicitamente la singolarità di Sigma=0
    with pytest.raises(ValueError, match="Sigma = 0"):
        build_mode_b_noise_weights(y_clean, eta_level=0.0, eta_zero_policy="raise")

    # Anche un vettore identicamente nullo non deve causare ZeroDivisionError
    y_zeros = np.zeros(10, dtype=np.float64)
    w_zero, s_ref_zero = build_mode_a_deterministic_weights(y_zeros)
    assert s_ref_zero == 1.0
    assert np.all(np.isfinite(w_zero))


def test_04_normalization_consistency_across_candidates_on_same_data() -> None:
    """
    4. Verifica che per candidati valutati sugli stessi dati:
       - il rapporto J_data(theta_1) / J_data(theta_2) coincida esattamente con il rapporto
         dei quadrati dei residui L2 non pesati ||r(theta_1)||_2^2 / ||r(theta_2)||_2^2;
       - per eta > 0, J_data^(B,mean)(theta) == J_data^(A)(theta) / (eta^2)
         e J_data^(B,cov)(theta) == N_obs * J_data^(B,mean)(theta).
    """
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_true = master.parameter_space_theta.theta_dagger_array()
    G_op = _make_surrogate_forward(n_obs=40)
    y_clean = G_op(theta_true)

    eta = 0.05
    sample = generate_gaussian_noisy_observations(y_clean, eta_level=eta, seed=2026)
    reg = TikhonovRegularizer(transform, lambda_reg=1.0e-4)

    obj_a = InverseObjectiveFunctional.from_mode_a_deterministic(
        G_op, sample.y_noisy, transform, reg, y_reference=y_clean
    )
    obj_b_mean = InverseObjectiveFunctional.from_mode_b_noise_model(
        G_op, sample, transform, reg, covariance_scaling="mean_normalized"
    )
    obj_b_cov = InverseObjectiveFunctional.from_mode_b_noise_model(
        G_op, sample, transform, reg, covariance_scaling="exact_inverse_covariance"
    )

    c1 = np.array([0.35, 0.25, 0.03, 1.20], dtype=np.float64)
    c2 = np.array([0.80, 0.70, 0.22, 4.50], dtype=np.float64)

    bd_a_1 = obj_a.evaluate_breakdown(c1)
    bd_a_2 = obj_a.evaluate_breakdown(c2)
    bd_bm_1 = obj_b_mean.evaluate_breakdown(c1)
    bd_bm_2 = obj_b_mean.evaluate_breakdown(c2)
    bd_bc_1 = obj_b_cov.evaluate_breakdown(c1)

    ratio_unweighted_sq = (bd_a_1.unweighted_residual_l2_norm ** 2) / (
        bd_a_2.unweighted_residual_l2_norm ** 2
    )
    assert np.isclose(bd_a_1.j_data / bd_a_2.j_data, ratio_unweighted_sq, rtol=1e-14)
    assert np.isclose(bd_bm_1.j_data / bd_bm_2.j_data, ratio_unweighted_sq, rtol=1e-14)

    # Relazione esatta di scala tra Modalità A e Modalità B
    assert np.isclose(bd_bm_1.j_data, bd_a_1.j_data / (eta ** 2), rtol=1e-14)
    assert np.isclose(bd_bc_1.j_data, y_clean.size * bd_bm_1.j_data, rtol=1e-14)


def test_05_parameter_transformation_and_prior_consistency() -> None:
    """
    5. Verifica la coerenza tra la trasformazione psi(theta) in [0, 1]^4 e il prior:
       - J_reg(theta_0) == 0 esattamente nel punto fisico corrispondente al prior;
       - invarianza di D_tilde ed E_tilde rispetto alla base del logaritmo (ln vs log10).
    """
    master = load_master_config("configs/scales_and_physics.yaml")
    bounds = master.parameter_space_theta.bounds
    transform = ParameterCoordinateTransform(bounds)

    prior_tilde = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    reg = TikhonovRegularizer(transform, lambda_reg=1.0e-2, theta_tilde_0=prior_tilde)
    theta_0_phys = reg.prior_physical_theta

    bd_prior = reg.evaluate_breakdown(theta_0_phys)
    assert np.isclose(bd_prior.j_reg, 0.0, atol=1e-16)
    assert np.isclose(bd_prior.prior_distance_l2_norm, 0.0, atol=1e-15)
    np.testing.assert_allclose(bd_prior.theta_tilde, prior_tilde, rtol=1e-14, atol=1e-15)

    # Verifica esplicita dell'invarianza di base del logaritmo per D ed E
    theta_test = np.array([0.65, 0.35, 0.12, 3.20], dtype=np.float64)
    tilde_code = transform.to_normalized(theta_test)
    d_min, d_max = bounds.D[0], bounds.D[1]
    e_min, e_max = bounds.E[0], bounds.E[1]
    d_tilde_log10 = (np.log10(theta_test[2]) - np.log10(d_min)) / (
        np.log10(d_max) - np.log10(d_min)
    )
    e_tilde_log10 = (np.log10(theta_test[3]) - np.log10(e_min)) / (
        np.log10(e_max) - np.log10(e_min)
    )
    assert np.isclose(tilde_code[2], d_tilde_log10, rtol=1e-15)
    assert np.isclose(tilde_code[3], e_tilde_log10, rtol=1e-15)


def test_06_lambda_sensitivity_scaling_and_j_data_invariance() -> None:
    """
    6. Verifica che al variare di lambda:
       - J_data resti rigorosamente invariato;
       - J_reg scali linearmente con lambda;
       - per lambda = 0 si abbia J_reg = 0 e J_total = J_data.
    """
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_true = master.parameter_space_theta.theta_dagger_array()
    G_op = _make_surrogate_forward()
    y_clean = G_op(theta_true)

    sample = generate_gaussian_noisy_observations(y_clean, eta_level=0.01, seed=99)
    reg_unit = TikhonovRegularizer(transform, lambda_reg=1.0)
    obj = InverseObjectiveFunctional.from_mode_a_deterministic(
        G_op, sample.y_noisy, transform, reg_unit, y_reference=y_clean
    )

    cand = np.array([0.72, 0.38, 0.09, 2.80], dtype=np.float64)
    y_pred = G_op(cand)

    bd_unit = obj.evaluate_from_prediction(cand, y_pred, lambda_override=1.0)
    bd_zero = obj.evaluate_from_prediction(cand, y_pred, lambda_override=0.0)
    assert bd_zero.j_reg == 0.0
    assert bd_zero.j_total == bd_zero.j_data
    assert bd_zero.j_data == bd_unit.j_data

    lambdas = [1.0e-7, 1.0e-5, 1.0e-4, 1.0e-2, 1.0e-1]
    prev_total = bd_zero.j_total
    for lam in lambdas:
        bd_lam = obj.evaluate_from_prediction(cand, y_pred, lambda_override=lam)
        assert bd_lam.j_data == bd_unit.j_data
        assert np.isclose(bd_lam.j_reg, lam * bd_unit.j_reg, rtol=1e-15, atol=1e-16)
        assert np.isclose(bd_lam.j_total, bd_lam.j_data + bd_lam.j_reg, rtol=1e-15)
        assert bd_lam.j_total > prev_total
        prev_total = bd_lam.j_total


def test_07_legacy_mode_and_configurable_mode_exact_equivalence() -> None:
    """
    7. Verifica l'equivalenza numerica esatta tra l'istanziazione storica di
       InverseObjectiveFunctional e la nuova Modalità B (per eta >= 0) e Modalità A (per eta = 0).
    """
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_true = master.parameter_space_theta.theta_dagger_array()
    G_op = _make_surrogate_forward()
    y_clean = G_op(theta_true)
    reg = TikhonovRegularizer(transform, lambda_reg=1.0e-4)
    cand = np.array([0.61, 0.52, 0.11, 2.15], dtype=np.float64)

    for eta in (0.0, 0.01, 0.05, 0.10):
        sample_legacy = generate_gaussian_noisy_observations(y_clean, eta, seed=20261016)
        obj_legacy = InverseObjectiveFunctional(
            G_op, sample_legacy.y_noisy, sample_legacy.weight_matrix_diag, transform, reg
        )
        obj_mode_b = InverseObjectiveFunctional.from_mode_b_noise_model(
            G_op, sample_legacy, transform, reg, covariance_scaling="mean_normalized"
        )

        j_leg = obj_legacy(cand)
        j_mb = obj_mode_b(cand)
        assert j_leg == j_mb
        assert obj_legacy.history[-1].j_data == obj_mode_b.history[-1].j_data
        assert obj_legacy.history[-1].j_reg == obj_mode_b.history[-1].j_reg

        if eta == 0.0:
            obj_mode_a = InverseObjectiveFunctional.from_mode_a_deterministic(
                G_op, sample_legacy.y_noisy, transform, reg, y_reference=y_clean
            )
            assert obj_mode_a(cand) == j_leg


def test_08_nominal_noise_vs_realized_noise_norm_distinction() -> None:
    """
    8. Verifica la distinzione esplicita tra livello nominale eta, deviazione standard
       nominale sigma_nom = eta * ||y_clean||_inf, deviazione standard empirica della
       realizzazione e norma relativa effettiva ||epsilon||_2 / ||y_clean||_2.
    """
    y_clean = np.linspace(0.5, 4.0, 180, dtype=np.float64)
    eta = 0.05
    sample = generate_gaussian_noisy_observations(y_clean, eta_level=eta, seed=20261016)

    expected_s_ref = float(np.max(np.abs(y_clean)))
    expected_sigma_nom = eta * expected_s_ref
    expected_noise_l2 = float(np.linalg.norm(sample.noise_vector))
    expected_rel_noise = expected_noise_l2 / float(np.linalg.norm(y_clean))
    expected_emp_std = float(np.sqrt(np.mean(sample.noise_vector ** 2)))

    assert sample.eta_level == eta
    assert np.isclose(sample.signal_reference_scale, expected_s_ref, rtol=1e-15)
    assert np.isclose(sample.sigma_std, expected_sigma_nom, rtol=1e-15)
    assert np.isclose(sample.effective_noise_l2_norm, expected_noise_l2, rtol=1e-15)
    assert np.isclose(sample.effective_relative_noise_norm, expected_rel_noise, rtol=1e-15)
    assert np.isclose(sample.empirical_noise_std, expected_emp_std, rtol=1e-15)
    # La deviazione standard empirica di una realizzazione finita differisce da quella nominale
    assert not np.isclose(sample.empirical_noise_std, sample.sigma_std, rtol=1e-6)
    # La norma relativa L2 del rumore differisce da eta perché ||y_clean||_2 != sqrt(N_obs)*||y_clean||_inf
    assert not np.isclose(sample.effective_relative_noise_norm, sample.eta_level, rtol=1e-3)