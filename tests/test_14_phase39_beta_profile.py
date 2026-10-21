"""
tests/test_14_phase39_beta_profile.py
RETECO — Fase 3.9: Test unitari per i profili ottimizzati rispetto a beta:
1. Costruzione della griglia in beta su [0.15, 0.90] con inclusione degli 8 punti informativi;
2. Separazione matematica tra Phi_data(beta) (solo J_data) e Phi_lambda(beta) (J_total);
3. Scansione bidirezionale (forward e backward), rispetto del budget uniforme per punto,
   distinzione rigorosa tra convergenza e budget esaurito (mai entrambi True), e identità
   J_total = J_data + J_reg;
4. Generazione dei CSV e della figura a 3 pannelli (PNG 300 dpi + PDF vettoriale).
"""

from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np

from reteco import load_master_config
from reteco.config_schema import ParameterCoordinateTransform
from reteco.inverse.regularization import TikhonovRegularizer
from reteco.inverse.objective_functional import InverseObjectiveFunctional
from reteco.identifiability.profile_likelihood import (
    compute_bidirectional_optimized_1d_profile,
)


def _load_exp09_module():
    spec = importlib.util.spec_from_file_location(
        "exp09_beta_profile", "experiments/09_beta_profile_analysis.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_phase39_beta_grid_covers_bounds_and_mandatory_points() -> None:
    exp09 = _load_exp09_module()
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)

    grid, info = exp09.build_phase39_beta_grid(transform)
    assert info["all_mandatory_betas_included"] is True
    assert np.isclose(grid[0], transform.b_min)
    assert np.isclose(grid[-1], transform.b_max)
    assert np.all(np.diff(grid) > 0.0)

    for b_m in exp09.MANDATORY_INFORMATIVE_BETAS:
        if transform.b_min <= b_m <= transform.b_max:
            assert np.any(np.isclose(grid, b_m, rtol=0.0, atol=1.0e-9))


def test_bidirectional_profiles_separation_budget_and_convergence_flags(tmp_path: Path) -> None:
    exp09 = _load_exp09_module()
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_dagger = master.parameter_space_theta.theta_dagger_array()

    grid_s = np.linspace(0.1, 1.0, 180, dtype=np.float64)

    # Operatore surrogato con lieve accoppiamento tra alpha, D, E e beta
    def surrogate_G(theta: np.ndarray) -> np.ndarray:
        th = np.asarray(theta, dtype=np.float64).ravel()
        return (
            (th[0] + 0.02 * (th[1] - 0.45)) * np.exp(-th[2] * grid_s)
            + 0.015 * th[1] * np.sin(2.0 * np.pi * grid_s)
            + 0.10 * th[3] * (grid_s ** 2)
        )

    y_clean = surrogate_G(theta_dagger)
    canonical_reg = TikhonovRegularizer(
        transform=transform,
        lambda_reg=1.0e-4,
        theta_tilde_0=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64),
    )
    obj_diag = InverseObjectiveFunctional.from_mode_a_deterministic(
        surrogate_G,
        y_clean,
        transform,
        canonical_reg,
        y_reference=y_clean,
        allow_observed_as_reference=True,
        scale_type="linf",
        max_evaluations=500,
    )

    short_grid = np.array([0.3880473413, 0.45, 0.525, 0.6032850194], dtype=np.float64)

    # 1. Test con budget stretto (6 valutazioni per punto) per verificare budget_exhausted=True -> converged=False
    prof_tight = compute_bidirectional_optimized_1d_profile(
        profile_type="data_only",
        beta_grid=short_grid,
        forward_operator=surrogate_G,
        y_observed=y_clean,
        transform=transform,
        canonical_regularizer=canonical_reg,
        planned_point_budget=6,
    )
    for rec in prof_tight.forward_records + prof_tight.backward_records:
        assert rec.fem_evaluations_used <= 6
        if rec.budget_exhausted:
            assert rec.converged is False

    # 2. Test con budget standard (28 valutazioni per punto) sui due profili distinti Phi_data e Phi_lambda
    prof_data = compute_bidirectional_optimized_1d_profile(
        profile_type="data_only",
        beta_grid=short_grid,
        forward_operator=surrogate_G,
        y_observed=y_clean,
        transform=transform,
        canonical_regularizer=canonical_reg,
        planned_point_budget=28,
    )
    prof_reg = compute_bidirectional_optimized_1d_profile(
        profile_type="regularized_total",
        beta_grid=short_grid,
        forward_operator=surrogate_G,
        y_observed=y_clean,
        transform=transform,
        canonical_regularizer=canonical_reg,
        planned_point_budget=28,
    )

    checks_d = exp09.verify_profile_result_integrity(
        prof_data, forward_operator=surrogate_G, obj_diag=obj_diag
    )
    checks_r = exp09.verify_profile_result_integrity(
        prof_reg, forward_operator=surrogate_G, obj_diag=obj_diag
    )
    assert checks_d["all_checks_passed"] is True
    assert checks_r["all_checks_passed"] is True

    comp_rows = exp09.write_combined_comparison_csv(
        prof_data, prof_reg, tmp_path / "beta_profiles_comparison.csv"
    )
    assert len(comp_rows) == short_grid.size

    bd_true = obj_diag.evaluate_from_prediction(
        theta_dagger, y_clean, record_in_history=False
    )
    mock_cpso = [
        {
            "optimizer_seed": 20261017,
            "beta_hat": 0.3880473413,
            "best_cost_J_data": float(prof_data.combined_best_records[0].j_data),
            "best_cost_J_total": float(prof_reg.combined_best_records[0].j_total),
        }
    ]
    png_path, pdf_path = exp09.plot_optimized_profiles_figure(
        prof_data=prof_data,
        prof_reg=prof_reg,
        cpso_solutions=mock_cpso,
        cond_curves={},
        bd_at_theta_dagger=bd_true,
        out_path=tmp_path / "fig_beta_optimized_profiles.png",
    )
    assert png_path.exists() and png_path.stat().st_size > 0
    assert pdf_path.exists() and pdf_path.stat().st_size > 0