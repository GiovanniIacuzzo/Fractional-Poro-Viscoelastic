"""
tests/test_10_objective_functional_modes.py
Fase 2.1 — Suite completa dei 9 test richiesti per l'audit matematico finale
della funzione obiettivo inversa (Sezione 5):
1. Decomposizione esatta di J_total in J_data + J_reg;
2. Identità W^T W = Sigma^{-1} e J_data = 0.5 * r^T Sigma^{-1} r = 0.5 * ||W r||_2^2
   per la modalità con covarianza esatta (`exact_inverse_covariance`);
3. Relazioni di normalizzazione tra Modalità A, Modalità B normalizzata (`mean_normalized`)
   e Modalità B con covarianza esatta (`exact_inverse_covariance`), più equivalenza
   del rapporto J_reg / J_data quando lambda è riscalato dello stesso fattore;
4. Distinzione tra deviazione standard nominale e rumore effettivamente generato;
5. Identità della realizzazione del rumore quando si confrontano modalità con uguale seed
   e uguali dati;
6. Comportamento deterministico a eta = 0 e gestione rigorosa del riferimento nullo;
7. Indipendenza da lambda di J_data per una predizione e un dataset fissati, e rifiuto
   esplicito di valori di lambda non ammissibili (negativi, NaN, inf);
8. Proporzionalità di J_reg a lambda e correttezza della trasformazione dei parametri
   nello spazio di regolarizzazione [0, 1]^4;
9. Uso coerente e invariante della scala di riferimento tra i diversi livelli di rumore
   in Modalità A (e rifiuto dell'omissione ambigua di `y_reference`).
"""

from __future__ import annotations
import numpy as np
import pytest

from reteco import load_master_config
from reteco.config_schema import ParameterCoordinateTransform
from reteco.observation.noise_model import (
    build_mode_a_deterministic_weights,
    build_mode_b_noise_weights,
    compute_positive_reference_scale,
    generate_gaussian_noisy_observations,
)
from reteco.inverse.regularization import TikhonovRegularizer
from reteco.inverse.objective_functional import InverseObjectiveFunctional


def _make_surrogate_forward(n_obs: int = 36):
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
    """1. Decomposizione esatta di J_total in J_data + J_reg."""
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
    obj_b_mean = InverseObjectiveFunctional.from_mode_b_noise_model(
        G_op, sample, transform, reg, covariance_scaling="mean_normalized"
    )
    obj_b_cov = InverseObjectiveFunctional.from_mode_b_noise_model(
        G_op, sample, transform, reg, covariance_scaling="exact_inverse_covariance"
    )

    candidate = np.array([0.55, 0.62, 0.14, 3.40], dtype=np.float64)
    for obj in (obj_a, obj_b_mean, obj_b_cov):
        val = obj(candidate)
        bd = obj.history[-1]
        assert np.isclose(val, bd.j_total, rtol=1e-15, atol=1e-16)
        assert np.isclose(bd.j_total, bd.j_data + bd.j_reg, rtol=1e-15, atol=1e-16)
        assert np.isclose(bd.data_share_of_total + bd.reg_share_of_total, 1.0, rtol=1e-14)


def test_02_exact_inverse_covariance_identity_wt_w_equals_sigma_inv() -> None:
    """
    2. Verifica che per eta > 0 la modalità con covarianza esatta soddisfi esattamente:
       - W^T W = Sigma^{-1} con Sigma = sigma_nom^2 * I_{N_obs};
       - J_data(theta) = 0.5 * r(theta)^T Sigma^{-1} r(theta) = 0.5 * ||W r(theta)||_2^2.
    """
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_true = master.parameter_space_theta.theta_dagger_array()
    G_op = _make_surrogate_forward(n_obs=48)
    y_clean = G_op(theta_true)

    eta = 0.05
    sample_cov = generate_gaussian_noisy_observations(
        y_clean,
        eta_level=eta,
        seed=20261016,
        weighting_mode="mode_b_noise_weighted",
        covariance_scaling="exact_inverse_covariance",
    )
    assert sample_cov.is_covariance_invertible is True
    assert sample_cov.inverse_covariance_diag is not None

    sigma_nom = eta * float(np.max(np.abs(y_clean)))
    expected_sigma_inv_diag = np.full(y_clean.size, 1.0 / (sigma_nom ** 2), dtype=np.float64)

    # Verifica matriciale W^T W = Sigma^{-1}
    W_mat = np.diag(sample_cov.weight_matrix_diag)
    Sigma_inv_mat = np.diag(expected_sigma_inv_diag)
    np.testing.assert_allclose(W_mat.T @ W_mat, Sigma_inv_mat, rtol=1e-14, atol=0.0)

    reg = TikhonovRegularizer(transform, lambda_reg=1.0e-4)
    obj_cov = InverseObjectiveFunctional.from_mode_b_noise_model(
        G_op, sample_cov, transform, reg, covariance_scaling="exact_inverse_covariance"
    )
    assert obj_cov.verify_exact_inverse_covariance_identity() is True

    cand = np.array([0.48, 0.58, 0.09, 3.10], dtype=np.float64)
    bd = obj_cov.evaluate_breakdown(cand)
    r_vec = G_op(cand) - sample_cov.y_noisy
    expected_quad = 0.5 * float(r_vec.T @ Sigma_inv_mat @ r_vec)
    expected_norm_sq = 0.5 * float(np.linalg.norm(W_mat @ r_vec) ** 2)

    assert np.isclose(bd.j_data, expected_quad, rtol=1e-14, atol=0.0)
    assert np.isclose(bd.j_data, expected_norm_sq, rtol=1e-14, atol=0.0)
    # `ObjectiveEvaluationBreakdown` espone `j_data` e la norma pesata; non
    # dichiariamo una seconda proprietà ridondante non prevista dalla sua API.
    assert np.isclose(0.5 * bd.weighted_residual_l2_norm**2, bd.j_data, rtol=1e-14, atol=0.0)


def test_03_normalization_relations_between_mode_a_and_mode_b_variants() -> None:
    """
    3. Verifica per ogni eta > 0 le relazioni esatte:
       - J_data^(B,mean) = J_data^(A) / eta^2;
       - J_data^(B,cov)  = N_obs * J_data^(A) / eta^2;
       e verifica che riscalando lambda dello stesso fattore (lambda_B_mean = lambda_A / eta^2,
       lambda_B_cov = N_obs * lambda_A / eta^2) le quote J_data / J_total e J_reg / J_total
       rimangono identiche a precisione di macchina.
    """
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_true = master.parameter_space_theta.theta_dagger_array()
    n_obs = 50
    G_op = _make_surrogate_forward(n_obs=n_obs)
    y_clean = G_op(theta_true)

    cand = np.array([0.38, 0.66, 0.15, 4.10], dtype=np.float64)
    y_pred = G_op(cand)
    lam_a = 1.0e-4
    reg = TikhonovRegularizer(transform, lambda_reg=lam_a)

    for eta in (0.01, 0.05, 0.10):
        sample = generate_gaussian_noisy_observations(y_clean, eta_level=eta, seed=1234)
        obj_a = InverseObjectiveFunctional.from_mode_a_deterministic(
            G_op, sample.y_noisy, transform, reg, y_reference=y_clean
        )
        obj_bm = InverseObjectiveFunctional.from_mode_b_noise_model(
            G_op, sample, transform, reg, covariance_scaling="mean_normalized"
        )
        obj_bc = InverseObjectiveFunctional.from_mode_b_noise_model(
            G_op, sample, transform, reg, covariance_scaling="exact_inverse_covariance"
        )

        bd_a = obj_a.evaluate_from_prediction(cand, y_pred, lambda_override=lam_a)
        bd_bm = obj_bm.evaluate_from_prediction(cand, y_pred, lambda_override=lam_a)
        bd_bc = obj_bc.evaluate_from_prediction(cand, y_pred, lambda_override=lam_a)

        assert np.isclose(bd_bm.j_data, bd_a.j_data / (eta ** 2), rtol=1e-14)
        assert np.isclose(bd_bc.j_data, (n_obs * bd_a.j_data) / (eta ** 2), rtol=1e-14)
        assert np.isclose(bd_bc.j_data, n_obs * bd_bm.j_data, rtol=1e-14)

        # Verifica dell'equivalenza del rapporto J_reg / J_total con lambda riscalato
        lam_bm_eq = TikhonovRegularizer.compute_equivalent_mode_b_lambda(
            lam_a, eta, n_obs, "mean_normalized"
        )
        lam_bc_eq = TikhonovRegularizer.compute_equivalent_mode_b_lambda(
            lam_a, eta, n_obs, "exact_inverse_covariance"
        )
        bd_bm_eq = obj_bm.evaluate_from_prediction(cand, y_pred, lambda_override=lam_bm_eq)
        bd_bc_eq = obj_bc.evaluate_from_prediction(cand, y_pred, lambda_override=lam_bc_eq)

        assert np.isclose(bd_bm_eq.reg_share_of_total, bd_a.reg_share_of_total, rtol=1e-14)
        assert np.isclose(bd_bc_eq.reg_share_of_total, bd_a.reg_share_of_total, rtol=1e-14)
        assert np.isclose(bd_bm_eq.j_total, bd_a.j_total / (eta ** 2), rtol=1e-14)
        assert np.isclose(bd_bc_eq.j_total, (n_obs * bd_a.j_total) / (eta ** 2), rtol=1e-14)


def test_04_nominal_noise_std_vs_realized_noise_distinction() -> None:
    """4. Distinzione tra deviazione standard nominale e rumore effettivamente generato."""
    y_clean = np.linspace(0.5, 4.0, 180, dtype=np.float64)
    eta = 0.05
    sample = generate_gaussian_noisy_observations(y_clean, eta_level=eta, seed=20261016)

    expected_s_ref = float(np.max(np.abs(y_clean)))
    expected_sigma_nom = eta * expected_s_ref
    expected_noise_l2 = float(np.linalg.norm(sample.noise_vector))
    expected_rel_noise = expected_noise_l2 / float(np.linalg.norm(y_clean))
    expected_emp_std = float(np.sqrt(np.mean(sample.noise_vector ** 2)))

    assert np.isclose(sample.sigma_std, expected_sigma_nom, rtol=1e-15)
    assert np.isclose(sample.effective_noise_l2_norm, expected_noise_l2, rtol=1e-15)
    assert np.isclose(sample.effective_relative_noise_norm, expected_rel_noise, rtol=1e-15)
    assert np.isclose(sample.empirical_noise_std, expected_emp_std, rtol=1e-15)
    assert not np.isclose(sample.empirical_noise_std, sample.sigma_std, rtol=1e-6)
    assert not np.isclose(sample.effective_relative_noise_norm, sample.eta_level, rtol=1e-3)


def test_05_identical_noise_realization_across_modes_with_same_seed_and_data() -> None:
    """5. Identità bit-a-bit della realizzazione del rumore tra modalità con uguale seed e dati."""
    y_clean = np.linspace(0.2, 3.5, 120, dtype=np.float64)
    eta = 0.05
    seed = 20261019

    s_mode_a = generate_gaussian_noisy_observations(
        y_clean,
        eta_level=eta,
        seed=seed,
        weighting_mode="mode_a_deterministic_normalized",
    )
    s_mode_b_mean = generate_gaussian_noisy_observations(
        y_clean,
        eta_level=eta,
        seed=seed,
        weighting_mode="mode_b_noise_weighted",
        covariance_scaling="mean_normalized",
    )
    s_mode_b_cov = generate_gaussian_noisy_observations(
        y_clean,
        eta_level=eta,
        seed=seed,
        weighting_mode="mode_b_noise_weighted",
        covariance_scaling="exact_inverse_covariance",
    )

    np.testing.assert_array_equal(s_mode_a.noise_vector, s_mode_b_mean.noise_vector)
    np.testing.assert_array_equal(s_mode_a.noise_vector, s_mode_b_cov.noise_vector)
    np.testing.assert_array_equal(s_mode_a.y_noisy, s_mode_b_mean.y_noisy)
    np.testing.assert_array_equal(s_mode_a.y_noisy, s_mode_b_cov.y_noisy)
    assert s_mode_a.sigma_std == s_mode_b_mean.sigma_std == s_mode_b_cov.sigma_std


def test_06_deterministic_behavior_at_eta_zero_and_zero_reference_guard() -> None:
    """
    6. Comportamento deterministico a eta = 0 (senza falsa covarianza invertibile)
       e gestione rigorosa di un riferimento nullo senza normalizzazioni silenziosamente arbitrarie.
    """
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
    assert s_a.sigma_std == 0.0
    assert s_b.sigma_std == 0.0
    assert s_a.is_covariance_invertible is False
    assert s_b.is_covariance_invertible is False
    assert s_b.inverse_covariance_diag is None
    np.testing.assert_array_equal(s_a.y_noisy, y_clean)
    np.testing.assert_array_equal(s_b.y_noisy, y_clean)

    with pytest.raises(ValueError, match="Sigma = 0"):
        build_mode_b_noise_weights(y_clean, eta_level=0.0, eta_zero_policy="raise")

    reg = TikhonovRegularizer(transform, lambda_reg=1.0e-4)
    obj_b0 = InverseObjectiveFunctional.from_mode_b_noise_model(G_op, s_b, transform, reg)
    assert obj_b0.verify_exact_inverse_covariance_identity() is False
    assert "non_statistical" in obj_b0.objective_mode

    # Riferimento identicamente nullo: deve sollevare ValueError per default (mai 1.0 silenzioso)
    y_zeros = np.zeros(16, dtype=np.float64)
    with pytest.raises(ValueError, match="norma nulla"):
        compute_positive_reference_scale(y_zeros)
    with pytest.raises(ValueError, match="norma nulla"):
        build_mode_a_deterministic_weights(y_zeros)

    # Funziona senza divisioni per zero solo con fallback esplicito e dichiarato > 0
    w_fb, s_fb = build_mode_a_deterministic_weights(
        y_zeros,
        zero_reference_policy="explicit_fallback",
        zero_reference_fallback_scale=2.5,
    )
    assert s_fb == 2.5
    assert np.all(np.isfinite(w_fb)) and np.all(w_fb > 0.0)


def test_07_j_data_independence_from_lambda_and_invalid_lambda_rejection() -> None:
    """
    7. Indipendenza di J_data da lambda per predizione e dataset fissati, e rifiuto esplicito
       di valori di lambda non ammissibili (negativi, NaN, inf, bool).
    """
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_true = master.parameter_space_theta.theta_dagger_array()
    G_op = _make_surrogate_forward()
    y_clean = G_op(theta_true)

    sample = generate_gaussian_noisy_observations(y_clean, eta_level=0.05, seed=99)
    reg = TikhonovRegularizer(transform, lambda_reg=1.0e-4)
    obj = InverseObjectiveFunctional.from_mode_a_deterministic(
        G_op, sample.y_noisy, transform, reg, y_reference=y_clean
    )

    cand = np.array([0.72, 0.38, 0.09, 2.80], dtype=np.float64)
    y_pred = G_op(cand)

    j_data_values = [
        obj.evaluate_from_prediction(cand, y_pred, lambda_override=lam).j_data
        for lam in (0.0, 1.0e-8, 1.0e-4, 1.0, 1.0e3)
    ]
    assert all(val == j_data_values[0] for val in j_data_values)

    for bad_lam in (-1.0e-5, -1.0, float("nan"), float("inf"), -float("inf"), True):
        with pytest.raises(ValueError):
            TikhonovRegularizer(transform, lambda_reg=bad_lam)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            reg.evaluate_with_lambda(cand, bad_lam)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            obj.evaluate_from_prediction(cand, y_pred, lambda_override=bad_lam)  # type: ignore[arg-type]


def test_08_j_reg_proportionality_to_lambda_and_parameter_transform_correctness() -> None:
    """
    8. Proporzionalità di J_reg a lambda e correttezza della trasformazione dei parametri
       nello spazio di regolarizzazione [0, 1]^4.
    """
    master = load_master_config("configs/scales_and_physics.yaml")
    bounds = master.parameter_space_theta.bounds
    transform = ParameterCoordinateTransform(bounds)

    prior_tilde = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    reg_unit = TikhonovRegularizer(transform, lambda_reg=1.0, theta_tilde_0=prior_tilde)
    theta_0_phys = reg_unit.prior_physical_theta

    bd_at_prior = reg_unit.evaluate_breakdown(theta_0_phys)
    assert np.isclose(bd_at_prior.j_reg, 0.0, atol=1e-16)
    np.testing.assert_allclose(bd_at_prior.theta_tilde, prior_tilde, rtol=1e-14, atol=1e-15)

    cand = np.array([0.65, 0.35, 0.12, 3.20], dtype=np.float64)
    base_unscaled = reg_unit.evaluate_unscaled_half_norm_sq(cand)
    assert base_unscaled > 0.0

    for lam in (0.0, 1.0e-7, 3.5e-5, 1.0e-3, 0.25):
        bd = reg_unit.evaluate_breakdown(cand, lambda_override=lam)
        assert np.isclose(bd.j_reg, lam * base_unscaled, rtol=1e-15, atol=1e-16)


def test_09_reference_scale_invariance_across_noise_levels_in_mode_a() -> None:
    """
    9. Verifica che in Modalità A:
       - omettere `y_reference` senza `allow_observed_as_reference=True` sollevi ValueError
         (impedendo che la scala dipenda silenziosamente da `y_observed`);
       - passando `y_reference=y_clean`, la scala di riferimento s_ref e la matrice di
         peso W^(A) rimangano identiche bit-a-bit al variare di eta in {0%, 1%, 5%, 10%}.
    """
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_true = master.parameter_space_theta.theta_dagger_array()
    G_op = _make_surrogate_forward()
    y_clean = G_op(theta_true)
    reg = TikhonovRegularizer(transform, lambda_reg=1.0e-4)

    sample_noisy = generate_gaussian_noisy_observations(y_clean, eta_level=0.10, seed=42)
    with pytest.raises(ValueError, match="y_reference"):
        InverseObjectiveFunctional.from_mode_a_deterministic(
            G_op, sample_noisy.y_noisy, transform, reg
        )

    scales_seen = []
    weights_seen = []
    for eta in (0.0, 0.01, 0.05, 0.10):
        smp = generate_gaussian_noisy_observations(
            y_clean,
            eta_level=eta,
            seed=20261016 + int(eta * 1000),
            weighting_mode="mode_a_deterministic_normalized",
        )
        obj_a = InverseObjectiveFunctional.from_mode_a_deterministic(
            G_op, smp.y_noisy, transform, reg, y_reference=y_clean
        )
        scales_seen.append(obj_a.residual_reference_scale)
        weights_seen.append(obj_a.weight_matrix_diag.copy())

    for s_val in scales_seen[1:]:
        assert s_val == scales_seen[0]
    for w_vec in weights_seen[1:]:
        np.testing.assert_array_equal(w_vec, weights_seen[0])