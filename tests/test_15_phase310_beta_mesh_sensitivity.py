"""
tests/test_15_phase310_beta_mesh_sensitivity.py
RETECO — Fase 3.10: Test unitari per la sensibilità del profilo dati Phi_data(beta)
alla discretizzazione inversa.
"""

from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

from reteco import load_master_config
from reteco.config_schema import ParameterCoordinateTransform
from reteco.inverse.regularization import TikhonovRegularizer
from reteco.inverse.objective_functional import InverseObjectiveFunctional
from reteco.identifiability.profile_likelihood import (
    compute_bidirectional_optimized_1d_profile,
)


def _load_exp10_module():
    spec = importlib.util.spec_from_file_location(
        "exp10_mesh_sens", "experiments/10_beta_mesh_sensitivity.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_local_beta_grid_exact_values() -> None:
    exp10 = _load_exp10_module()
    expected = [
        0.425,
        0.4375,
        0.4500,
        0.4625,
        0.4750,
        0.4791427903,
        0.4875,
        0.5000,
        0.5250,
    ]
    assert len(exp10.LOCAL_BETA_GRID) == 9
    assert np.allclose(exp10.LOCAL_BETA_GRID, expected, rtol=0.0, atol=1.0e-12)


def test_reuse_from_phase39_supplementary_check_and_overlay_plot(tmp_path: Path) -> None:
    exp10 = _load_exp10_module()
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_dagger = master.parameter_space_theta.theta_dagger_array()

    grid_s = np.linspace(0.1, 1.0, 180, dtype=np.float64)

    def make_surrogate(disc_err_scale: float):
        def _fwd(theta: np.ndarray) -> np.ndarray:
            th = np.asarray(theta, dtype=np.float64).ravel()
            base = (
                th[0] * np.exp(-th[2] * grid_s)
                + 0.04 * (th[1] - 0.45) ** 2 * np.sin(2.0 * np.pi * grid_s)
                + 0.10 * th[3] * (grid_s ** 2)
            )
            disc_shift = disc_err_scale * 1.0e-3 * np.cos(np.pi * grid_s)
            return base + disc_shift

        return _fwd

    fwd_target = make_surrogate(0.0)
    fwd_16 = make_surrogate(1.0)
    fwd_24 = make_surrogate(0.4)

    y_clean = fwd_target(theta_dagger)
    canonical_reg = TikhonovRegularizer(
        transform=transform,
        lambda_reg=1.0e-4,
        theta_tilde_0=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64),
    )

    p39_betas = [0.425, 0.45, 0.475, 0.4791427903, 0.50, 0.525]
    prof_p39_sub = compute_bidirectional_optimized_1d_profile(
        profile_type="data_only",
        beta_grid=p39_betas,
        forward_operator=fwd_16,
        y_observed=y_clean,
        transform=transform,
        canonical_regularizer=canonical_reg,
        planned_point_budget=12,
    )
    mock_p39_json = tmp_path / "beta_profile_summary.json"
    mock_p39_json.write_text(
        json.dumps(
            {
                "frozen_protocol": {
                    "target_sha256": exp10.EXPECTED_TARGET_SHA256,
                    "N_inv": 16,
                    "M_inv": 80,
                    "objective_mode": "mode_a_deterministic_normalized_linf",
                    "true_parameters_used_in_optimization": False,
                },
                "budget_specification": {
                    "planned_per_point_per_direction_fem_budget": 12,
                },
                "profile_data_summary": {
                    "forward_records": [r.to_dict() for r in prof_p39_sub.forward_records],
                    "backward_records": [r.to_dict() for r in prof_p39_sub.backward_records],
                },
            }
        ),
        encoding="utf-8",
    )

    prof_16, reused_flags, reuse_audit = exp10.build_or_reuse_n16_m80_profile_on_local_grid(
        beta_grid=exp10.LOCAL_BETA_GRID,
        forward_operator_16=fwd_16,
        y_observed=y_clean,
        transform=transform,
        canonical_regularizer=canonical_reg,
        planned_point_budget=12,
        cpso_anchors=[],
        phase39_summary_path=mock_p39_json,
    )
    assert reuse_audit["phase39_reused"] is True
    assert len(reuse_audit["reused_betas"]) == 6
    assert len(reuse_audit["newly_computed_betas"]) == 3
    assert sum(1 for v in reused_flags.values() if v) == 6

    prof_24_tight = compute_bidirectional_optimized_1d_profile(
        profile_type="data_only",
        beta_grid=exp10.LOCAL_BETA_GRID,
        forward_operator=fwd_24,
        y_observed=y_clean,
        transform=transform,
        canonical_regularizer=canonical_reg,
        planned_point_budget=4,
    )
    supp_list = exp10.run_supplementary_verifications_for_nonconverged_central_points(
        resolution_label="N24_M120",
        N_inv=24,
        M_inv=120,
        prof=prof_24_tight,
        forward_operator=fwd_24,
        y_observed=y_clean,
        transform=transform,
        canonical_regularizer=canonical_reg,
        supplementary_budget=16,
    )
    assert len(supp_list) > 0

    obj_16 = InverseObjectiveFunctional.from_mode_a_deterministic(
        fwd_16,
        y_clean,
        transform,
        canonical_reg,
        y_reference=y_clean,
        allow_observed_as_reference=True,
    )
    obj_24 = InverseObjectiveFunctional.from_mode_a_deterministic(
        fwd_24,
        y_clean,
        transform,
        canonical_reg,
        y_reference=y_clean,
        allow_observed_as_reference=True,
    )
    bd_true_16 = obj_16.evaluate_from_prediction(
        theta_dagger, fwd_16(theta_dagger), record_in_history=False
    )
    bd_true_24 = obj_24.evaluate_from_prediction(
        theta_dagger, fwd_24(theta_dagger), record_in_history=False
    )

    png_path, pdf_path = exp10.plot_beta_mesh_sensitivity_figure(
        prof_16=prof_16,
        prof_24=prof_24_tight,
        bd_true_16=bd_true_16,
        bd_true_24=bd_true_24,
        supplementary_verifications=supp_list,
        out_path=tmp_path / "fig_beta_mesh_sensitivity.png",
    )
    assert png_path.exists() and png_path.stat().st_size > 0
    assert pdf_path.exists() and pdf_path.stat().st_size > 0