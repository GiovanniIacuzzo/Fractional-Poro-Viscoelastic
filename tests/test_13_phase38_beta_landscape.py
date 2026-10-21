"""
tests/test_13_phase38_beta_landscape.py
RETECO — Fase 3.8: Test unitari per l'analisi delle sezioni condizionali rispetto a beta:
1. Costruzione della griglia di ~31 punti sull'intero intervallo fisico ammissibile con
   inclusione obbligatoria di beta in {0.3880473413, 0.45, 0.4791427903, 0.6032850194};
2. Trasformazione coordinate tramite ParameterCoordinateTransform e valutazione tramite
   InverseObjectiveFunctional senza incrementare il contatore/history dell'ottimizzatore;
3. Verifica di J_total = J_data + J_reg, ammissibilità fisica, conteggio valutazioni e
   generazione della figura a 3 pannelli (PNG 300 dpi + PDF vettoriale).
"""

from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np

from reteco import load_master_config
from reteco.config_schema import ParameterCoordinateTransform
from reteco.inverse.regularization import TikhonovRegularizer
from reteco.inverse.objective_functional import InverseObjectiveFunctional


def _load_exp08_module():
    spec = importlib.util.spec_from_file_location(
        "exp08_beta_landscape", "experiments/08_beta_landscape_analysis.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_beta_grid_includes_required_values_and_stays_within_bounds() -> None:
    exp08 = _load_exp08_module()
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)

    grid, info = exp08.build_beta_scan_grid(
        transform=transform,
        special_betas=exp08.REQUIRED_SPECIAL_BETAS_PROMPT,
        num_uniform_points=31,
    )

    assert info["all_admissible_specials_included"] is True
    assert 31 <= grid.size <= 35
    assert np.isclose(grid[0], transform.b_min)
    assert np.isclose(grid[-1], transform.b_max)
    assert np.all(np.diff(grid) > 0.0)

    for b_req in exp08.REQUIRED_SPECIAL_BETAS_PROMPT:
        assert np.any(np.isclose(grid, b_req, rtol=0.0, atol=1.0e-9))


def test_conditional_sections_consistency_and_three_panel_plot(tmp_path: Path) -> None:
    exp08 = _load_exp08_module()
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_dagger = master.parameter_space_theta.theta_dagger_array()

    grid_s = np.linspace(0.1, 1.0, 180, dtype=np.float64)

    def surrogate_G(theta: np.ndarray) -> np.ndarray:
        th = np.asarray(theta, dtype=np.float64).ravel()
        return (
            th[0] * np.exp(-th[2] * grid_s)
            + 0.02 * th[1] * np.sin(2.0 * np.pi * grid_s)
            + 0.10 * th[3] * (grid_s ** 2)
        )

    y_clean = surrogate_G(theta_dagger)
    reg = TikhonovRegularizer(
        transform=transform,
        lambda_reg=1.0e-4,
        theta_tilde_0=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64),
    )
    obj_diag = InverseObjectiveFunctional.from_mode_a_deterministic(
        surrogate_G,
        y_clean,
        transform,
        reg,
        y_reference=y_clean,
        allow_observed_as_reference=True,
        scale_type="linf",
        max_evaluations=1000,
    )

    beta_grid, _ = exp08.build_beta_scan_grid(
        transform=transform,
        special_betas=exp08.REQUIRED_SPECIAL_BETAS_PROMPT,
        num_uniform_points=31,
    )

    mock_solutions = []
    for seed, vals in exp08.EXPECTED_PHASE37_SOLUTIONS.items():
        th_s = np.array([vals["alpha"], vals["beta"], vals["D"], vals["E"]], dtype=np.float64)
        bd_s = obj_diag.evaluate_from_prediction(th_s, surrogate_G(th_s), record_in_history=False)
        mock_solutions.append(
            {
                "optimizer_seed": seed,
                "run_id": f"cpso_600_seed_{seed}",
                "theta_hat": th_s,
                "alpha_hat": vals["alpha"],
                "beta_hat": vals["beta"],
                "D_hat": vals["D"],
                "E_hat": vals["E"],
                "best_cost_J_total": float(bd_s.j_total),
                "best_cost_J_data": float(bd_s.j_data),
                "best_cost_J_reg": float(bd_s.j_reg),
                "best_eval_index": 500,
                "relative_residual_vs_clean_target": float(bd_s.relative_residual_l2),
            }
        )

    all_rows, by_seed, checks = exp08.evaluate_conditional_beta_sections(
        solutions=mock_solutions,
        beta_grid=beta_grid,
        forward_operator_inv=surrogate_G,
        obj_functional=obj_diag,
        transform=transform,
        beta_true_ref=0.45,
    )

    assert checks["j_total_equals_j_data_plus_j_reg_all_points"] is True
    assert checks["all_betas_physically_admissible"] is True
    assert checks["physical_and_normalized_coordinates_consistent"] is True
    assert checks["fem_evaluation_count_exact"] is True
    assert checks["diagnostic_scan_fem_evaluations"] == 3 * beta_grid.size
    assert checks["optimizer_fem_evaluations_incremented"] == 0
    assert checks["optimizer_history_length_on_diagnostic_obj"] == 0
    assert len(all_rows) == 3 * beta_grid.size

    bd_true = obj_diag.evaluate_from_prediction(
        theta_dagger, surrogate_G(theta_dagger), record_in_history=False
    )
    png_path, pdf_path = exp08.plot_three_panel_beta_landscape(
        by_seed=by_seed,
        solutions=mock_solutions,
        bd_at_theta_dagger=bd_true,
        out_path=tmp_path / "fig_beta_conditional_landscape.png",
        beta_true_ref=0.45,
    )
    assert png_path.exists() and png_path.stat().st_size > 0
    assert pdf_path.exists() and pdf_path.stat().st_size > 0