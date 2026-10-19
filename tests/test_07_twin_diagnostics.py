"""
tests/test_07_twin_diagnostics.py
Test unitari per la fase diagnostica dell'esperimento inverso sintetico:
1. Riproducibilità della generazione del rumore a parità di seed;
2. Deviazione standard nominale e pesi effettivamente applicati nei casi eta=0 e eta>0
   (verifica esatta di W^T W = Sigma^{-1} / N_obs per eta>0);
3. Coerenza additiva fra J totale, J_data e J_reg;
4. Coerenza fra il miglior candidato memorizzato da CPSO e il record corrispondente in history;
5. Contabilizzazione distinta delle valutazioni dell'ottimizzatore e di quelle diagnostiche;
6. Generazione corretta di tutti gli artefatti (JSON, CSV per esecuzione, CSV della storia,
   PNG e report Markdown) su un modello diretto sintetico veloce.
"""

from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import pytest

from reteco import load_master_config
from reteco.config_schema import ParameterCoordinateTransform
from reteco.observation.noise_model import generate_gaussian_noisy_observations
from reteco.inverse.regularization import TikhonovRegularizer
from reteco.inverse.objective_functional import InverseObjectiveFunctional
from reteco.inverse.cpso import run_cpso_optimization
from reteco.inverse.diagnostics import (
    extract_run_diagnostics_from_history,
    plot_multistart_convergence,
    plot_multistart_objective_decomposition,
    plot_multistart_parameter_recovery,
    plot_noise_study_parameter_errors,
    plot_noise_study_relative_residuals,
    serialize_run_record,
    write_convergence_history_csv,
    write_multistart_markdown_report,
    write_noise_study_markdown_report,
    write_runs_summary_csv,
)


def _make_fast_synthetic_forward(sensor_grid_size: int = 20):
    """
    Costruisce un operatore osservativo sintetico rapido G_syn(theta) ben condizionato
    per testare l'intera pipeline inversa e diagnostica in pochi millisecondi.
    """
    s = np.linspace(0.1, 1.0, sensor_grid_size, dtype=np.float64)

    def G_syn(theta: np.ndarray) -> np.ndarray:
        th = np.asarray(theta, dtype=np.float64).ravel()
        alpha, beta, D, E = th[0], th[1], th[2], th[3]
        return (
            1.2 * np.exp(-alpha * s)
            + 0.6 * np.sin(np.pi * beta * s)
            + 150.0 * D * (s ** 2)
            + 0.15 * np.log1p(E) * np.cos(0.5 * np.pi * s)
        )

    return G_syn


def test_noise_reproducibility_and_weights_eta_zero_and_positive() -> None:
    y_clean = np.array([0.5, -2.0, 1.25, 4.0, -0.75, 3.1], dtype=np.float64)
    n_obs = y_clean.size
    peak = float(np.max(np.abs(y_clean)))  # 4.0

    # 1. Caso eta = 0.0 (noise-free)
    s0_a = generate_gaussian_noisy_observations(y_clean, eta_level=0.0, seed=20261016)
    s0_b = generate_gaussian_noisy_observations(y_clean, eta_level=0.0, seed=99999999)
    assert s0_a.sigma_std == 0.0
    np.testing.assert_array_equal(s0_a.noise_vector, np.zeros_like(y_clean))
    np.testing.assert_array_equal(s0_a.y_noisy, y_clean)
    expected_w0 = 1.0 / (np.sqrt(n_obs) * peak)
    np.testing.assert_allclose(s0_a.weight_matrix_diag, expected_w0, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(s0_b.weight_matrix_diag, expected_w0, rtol=1e-14, atol=1e-15)

    # 2. Caso eta > 0: riproducibilità a parità di seed e differenza per seed diversi
    eta = 0.05
    s1 = generate_gaussian_noisy_observations(y_clean, eta_level=eta, seed=20261016)
    s2 = generate_gaussian_noisy_observations(y_clean, eta_level=eta, seed=20261016)
    s3 = generate_gaussian_noisy_observations(y_clean, eta_level=eta, seed=20261017)

    np.testing.assert_array_equal(s1.noise_vector, s2.noise_vector)
    np.testing.assert_array_equal(s1.y_noisy, s2.y_noisy)
    assert not np.allclose(s1.noise_vector, s3.noise_vector)

    # Verifica esatta di sigma = eta * max(abs(y_clean)) e W^T W = Sigma^{-1} / N_obs
    expected_sigma = eta * peak
    assert np.isclose(s1.sigma_std, expected_sigma, rtol=1e-14, atol=1e-15)

    expected_w_pos = 1.0 / (np.sqrt(n_obs) * expected_sigma)
    np.testing.assert_allclose(s1.weight_matrix_diag, expected_w_pos, rtol=1e-14, atol=1e-15)

    w_sq_diag = s1.weight_matrix_diag ** 2
    sigma_inv_over_n = (1.0 / (expected_sigma ** 2)) / float(n_obs)
    np.testing.assert_allclose(w_sq_diag, sigma_inv_over_n, rtol=1e-14, atol=1e-15)


def test_morozov_reference_discrepancy_value() -> None:
    ref_default = TikhonovRegularizer.morozov_target_discrepancy(tau_safety=1.05)
    assert np.isclose(ref_default, 0.5 * (1.05 ** 2), rtol=1e-14)
    assert np.isclose(TikhonovRegularizer.morozov_target_discrepancy(1.0), 0.5, rtol=1e-14)


def test_objective_decomposition_best_record_and_evaluation_accounting(tmp_path: Path) -> None:
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_true = master.parameter_space_theta.theta_dagger_array()

    forward_calls = {"total": 0}
    G_base = _make_fast_synthetic_forward(sensor_grid_size=24)

    def G_counted(theta: np.ndarray) -> np.ndarray:
        forward_calls["total"] += 1
        return G_base(theta)

    y_clean = G_base(theta_true)
    sample = generate_gaussian_noisy_observations(y_clean, eta_level=0.0, seed=20261016)
    reg = TikhonovRegularizer(
        transform,
        lambda_reg=1.0e-4,
        theta_tilde_0=np.full(4, 0.5, dtype=np.float64),
    )

    num_particles = 4
    max_iterations = 3
    planned_budget = num_particles * max_iterations  # 12 valutazioni

    obj = InverseObjectiveFunctional(
        G_counted,
        sample.y_noisy,
        sample.weight_matrix_diag,
        transform,
        reg,
        max_evaluations=planned_budget,
    )
    res = run_cpso_optimization(
        obj, seed=20261017, num_particles=num_particles, max_iterations=max_iterations
    )

    # Prima della valutazione diagnostica, le chiamate al forward solver devono coincidere col budget
    assert res.num_fem_evaluations == planned_budget
    assert len(obj.history) == planned_budget
    assert forward_calls["total"] == planned_budget

    rec = extract_run_diagnostics_from_history(
        run_id="test_run_01",
        obj=obj,
        opt_result=res,
        theta_true=theta_true,
        y_target_clean=y_clean,
        y_noisy=sample.y_noisy,
        nominal_noise_std=sample.sigma_std,
        eta_level=0.0,
        data_seed=20261016,
        optimizer_seed=20261017,
        planned_fem_budget=planned_budget,
        num_particles=num_particles,
        max_iterations=max_iterations,
        wall_time_seconds=0.05,
        forward_operator_inv=G_counted,
        transform=transform,
        regularizer=reg,
    )

    # Verifica contabilizzazione distinta: 12 valutazioni dell'ottimizzatore + 1 valutazione diagnostica
    assert rec.optimizer_fem_evaluations == planned_budget
    assert rec.diagnostic_fem_evaluations == 1
    assert len(obj.history) == planned_budget
    assert forward_calls["total"] == planned_budget + 1
    assert rec.termination_condition == "budget_exhausted_max_iterations_reached"

    # Verifica coerenza tra J_total, J_data e J_reg sul miglior candidato e su tutta la history
    assert np.isclose(
        rec.best_cost_J_total,
        rec.best_cost_J_data + rec.best_cost_J_reg,
        rtol=1e-12,
        atol=1e-14,
    )
    assert np.isclose(rec.best_cost_J_total, res.best_cost_J, rtol=1e-12, atol=1e-14)
    assert np.isclose(rec.best_cost_J_reg, reg.evaluate(res.theta_hat), rtol=1e-12, atol=1e-14)

    y_hat_check = G_base(res.theta_hat)
    w_res = sample.weight_matrix_diag * (y_hat_check - sample.y_noisy)
    expected_j_data = 0.5 * float(np.dot(w_res, w_res))
    assert np.isclose(rec.best_cost_J_data, expected_j_data, rtol=1e-11, atol=1e-13)

    # Verifica monotonia di best_j_so_far e coerenza finale di best_theta_so_far
    best_j_seq = [row.best_j_so_far for row in rec.history]
    assert all(best_j_seq[i + 1] <= best_j_seq[i] + 1e-15 for i in range(len(best_j_seq) - 1))
    assert np.isclose(best_j_seq[-1], rec.best_cost_J_total, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(rec.history[-1].best_theta_so_far, res.theta_hat, rtol=1e-10, atol=1e-12)


def test_artifacts_and_plots_generation_synthetic(tmp_path: Path) -> None:
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_true = master.parameter_space_theta.theta_dagger_array()
    G_syn = _make_fast_synthetic_forward(sensor_grid_size=18)
    y_clean = G_syn(theta_true)

    reg = TikhonovRegularizer(transform, lambda_reg=1.0e-4)
    records = []

    for idx, seed in enumerate((20261017, 20261018)):
        sample = generate_gaussian_noisy_observations(y_clean, eta_level=0.0, seed=20261016)
        obj = InverseObjectiveFunctional(
            G_syn,
            sample.y_noisy,
            sample.weight_matrix_diag,
            transform,
            reg,
            max_evaluations=8,
        )
        res = run_cpso_optimization(obj, seed=seed, num_particles=4, max_iterations=2)
        rec = extract_run_diagnostics_from_history(
            run_id=f"syn_run_{idx + 1:02d}",
            obj=obj,
            opt_result=res,
            theta_true=theta_true,
            y_target_clean=y_clean,
            y_noisy=sample.y_noisy,
            nominal_noise_std=sample.sigma_std,
            eta_level=0.0,
            data_seed=20261016,
            optimizer_seed=seed,
            planned_fem_budget=8,
            num_particles=4,
            max_iterations=2,
            wall_time_seconds=0.02,
            forward_operator_inv=G_syn,
            transform=transform,
            regularizer=reg,
        )
        records.append(rec)

    json_path = tmp_path / "multistart_summary.json"
    runs_csv = tmp_path / "multistart_runs_summary.csv"
    hist_csv = tmp_path / "multistart_convergence_history.csv"
    fig1 = tmp_path / "fig_multistart_parameter_recovery.png"
    fig2 = tmp_path / "fig_multistart_convergence.png"
    fig3 = tmp_path / "fig_multistart_objective_decomposition.png"
    fig4 = tmp_path / "fig_noise_parameter_errors.png"
    fig5 = tmp_path / "fig_noise_relative_residuals.png"
    md1 = tmp_path / "multistart_diagnostic_report.md"
    md2 = tmp_path / "twin_robustness_diagnostic_report.md"

    payload = {"runs": [serialize_run_record(r) for r in records]}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_runs_summary_csv(records, runs_csv)
    write_convergence_history_csv(records, hist_csv)
    plot_multistart_parameter_recovery(records, None, fig1)
    plot_multistart_convergence(records, fig2)
    plot_multistart_objective_decomposition(records, fig3)
    plot_noise_study_parameter_errors(records, fig4)
    plot_noise_study_relative_residuals(records, discretization_rel_gap=1.0e-3, out_path=fig5)
    write_multistart_markdown_report(
        records,
        discretization_rel_gap=1.0e-3,
        lambda_reg=1.0e-4,
        prior_theta_tilde_0=[0.5, 0.5, 0.5, 0.5],
        target_cfg={"subdivisions_N": 32, "time_steps_M": 160, "final_time_T": 1.0, "grading_r": 1.5},
        inverse_cfg={"subdivisions_N": 16, "time_steps_M": 80, "final_time_T": 1.0, "grading_r": 1.5},
        report_path=md1,
    )
    write_noise_study_markdown_report(
        records,
        discretization_rel_gap=1.0e-3,
        lambda_reg=1.0e-4,
        report_path=md2,
    )

    for p in (json_path, runs_csv, hist_csv, fig1, fig2, fig3, fig4, fig5, md1, md2):
        assert p.exists() and p.stat().st_size > 0

    with open(runs_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2

    with open(hist_csv, "r", encoding="utf-8") as f:
        h_rows = list(csv.DictReader(f))
    assert len(h_rows) == 16  # 2 run * 8 valutazioni ciascuna