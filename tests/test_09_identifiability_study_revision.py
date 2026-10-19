"""
tests/test_09_identifiability_study_revision.py
Test unitari rapidi per la revisione finale dello studio di identificabilità:
1) Rango, condizionamento, indice di collinearità e modo meno informativo su matrici
   sintetiche a rango pieno e a rango deficiente;
2) Gestione trasparente delle correlazioni non definite quando il rango è < 4;
3) Differenze finite centrali nei punti interni, schemi unidirezionali del 2° ordine
   vicino ai bordi e audit di stabilità rispetto al passo h;
4) Correttezza delle sezioni unidimensionali (One-at-a-Time Cost Slices) e
   convergenza effettiva (36/36) dei profili ottimizzati (Optimized Objective Profiles);
5) Verifica automatica delle annotazioni della Figura B rispetto al JSON;
6) Verifica rigorosa del limite cumulativo per punto `nfev <= max_evals_per_point`
   e della quadratura contabile esatta tra valutazioni per punto, valutazione
   preliminare di riferimento e totale globale in una scansione bidirezionale.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest

from reteco import load_master_config
from reteco.config_schema import ParameterCoordinateTransform
from reteco.identifiability.profile_likelihood import (
    compute_one_at_a_time_cost_slices,
    compute_optimized_objective_profiles,
)
from reteco.identifiability.sensitivity_jacobian import (
    ObservationWeightingSpec,
    compute_normalized_jacobian_with_audit,
    verify_fd_step_stability,
)
from reteco.identifiability.svd_conditioning import (
    analyze_jacobian_svd,
    sanitize_for_json,
)
from reteco.visualization.identifiability_plots import (
    build_figure_b_cell_annotations,
    plot_figure_a_sensitivity_and_svd,
    plot_figure_b_local_correlation,
    plot_figure_c_cost_slices,
    plot_figure_d_optimized_profiles,
)


def _get_test_transform() -> ParameterCoordinateTransform:
    master = load_master_config("configs/scales_and_physics.yaml")
    return ParameterCoordinateTransform(master.parameter_space_theta.bounds)


def test_01_svd_full_rank_vs_rank_deficient_and_undefined_correlations() -> None:
    J_full = np.vstack([np.diag([4.0, 2.0, 1.0, 0.5]), np.zeros((6, 4))])
    rep_full = analyze_jacobian_svd(J_full, rank_threshold=1.0e-8)

    assert rep_full.numerical_rank == 4
    assert np.isclose(rep_full.condition_number, 8.0, rtol=1.0e-12)
    assert np.isclose(rep_full.collinearity_index, 1.0, rtol=1.0e-12)
    assert rep_full.correlation_matrix_defined is True
    assert np.allclose(rep_full.fisher_correlation_matrix, np.eye(4), atol=1.0e-12)
    assert rep_full.least_informative_mode["dominant_parameter"] == "E"
    assert np.isclose(rep_full.least_informative_mode["singular_value"], 0.5)

    J_def = np.zeros((10, 4), dtype=np.float64)
    J_def[0, 0] = 3.0
    J_def[1, 2] = 1.5
    rep_def = analyze_jacobian_svd(J_def, rank_threshold=1.0e-8)

    assert rep_def.numerical_rank == 2
    assert not np.isfinite(rep_def.condition_number)
    assert not np.isfinite(rep_def.collinearity_index)
    assert np.isclose(rep_def.active_subset_collinearity_index, 1.0)
    assert rep_def.correlation_matrix_defined is False
    assert np.all(np.isnan(rep_def.fisher_correlation_matrix))
    assert set(rep_def.unidentifiable_parameters) == {"beta", "E"}


def test_02_jacobian_boundary_stencils_weighting_and_fd_stability() -> None:
    transform = _get_test_transform()
    A_true = np.array(
        [
            [2.0, -0.5, 1.2, 0.3],
            [0.4, 1.5, -0.8, 0.9],
            [-1.1, 0.2, 2.5, -0.6],
            [0.0, 0.8, -0.3, 1.7],
            [1.3, -0.4, 0.5, -1.2],
        ],
        dtype=np.float64,
    )

    def G_quad(theta_phys: np.ndarray) -> np.ndarray:
        t_norm = transform.to_normalized(theta_phys)
        return A_true @ t_norm + 0.1 * (A_true @ (t_norm ** 2))

    t_bdy_norm = np.array([0.0, 0.45, 0.60, 1.0], dtype=np.float64)
    theta_bdy_phys = transform.from_normalized(t_bdy_norm)
    J_exact = A_true * (1.0 + 0.2 * t_bdy_norm[None, :])

    w_sqrt = np.array([1.0, 2.0, 0.5, 3.0, 1.5], dtype=np.float64)
    w_spec = ObservationWeightingSpec(
        weighting_mode="custom_test_weights",
        weights_sqrt_diag=w_sqrt,
        is_statistically_calibrated=True,
    )

    audit = compute_normalized_jacobian_with_audit(
        G_quad, theta_bdy_phys, transform, eps_step=1.0e-4, weighting_spec=w_spec
    )
    assert audit.scheme_per_parameter["alpha"] == "forward_2nd_order_boundary"
    assert audit.scheme_per_parameter["beta"] == "central_2nd_order"
    assert audit.scheme_per_parameter["D"] == "central_2nd_order"
    assert audit.scheme_per_parameter["E"] == "backward_2nd_order_boundary"
    assert np.allclose(audit.jacobian_raw, J_exact, rtol=1.0e-9, atol=1.0e-9)
    assert np.allclose(audit.jacobian_weighted, w_sqrt[:, None] * J_exact, rtol=1.0e-9)

    stab = verify_fd_step_stability(
        G_quad, theta_bdy_phys, transform, steps=(1.0e-3, 1.0e-4, 1.0e-5), reference_step=1.0e-4
    )
    assert stab.frobenius_rel_diff_vs_reference["1.0e-03"] < 1.0e-9
    assert stab.frobenius_rel_diff_vs_reference["1.0e-05"] < 1.0e-8


def test_03_cost_slices_vs_optimized_profiles_convergence_and_metadata() -> None:
    transform = _get_test_transform()
    u_star = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    theta_star_phys = transform.from_normalized(u_star)

    def quad_coupled_obj(theta_phys: np.ndarray) -> float:
        u = transform.to_normalized(theta_phys)
        return float(
            0.5 * ((u[0] - 0.5) ** 2)
            + 2.0 * ((u[1] - (0.5 + 1.4 * (u[0] - 0.5))) ** 2)
            + 1.5 * ((u[2] - 0.5) ** 2)
            + 1.0 * ((u[3] - 0.5) ** 2)
        )

    oat_rep = compute_one_at_a_time_cost_slices(
        quad_coupled_obj, theta_star_phys, transform, num_points=9
    )
    prof_rep = compute_optimized_objective_profiles(
        quad_coupled_obj,
        theta_star_phys,
        transform,
        num_grid_points=9,
        max_evals_per_point=160,
        max_iter_per_point=60,
        max_total_evaluations=3000,
        bidirectional_scan=True,
    )

    assert prof_rep.all_profiles_complete_and_converged is True
    assert prof_rep.total_converged_points == 36
    assert prof_rep.total_non_converged_points == 0
    assert prof_rep.total_incomplete_points == 0
    assert prof_rep.budget_exhausted is False
    assert prof_rep.per_point_budget_respected is True
    assert prof_rep.accounting_consistency_verified is True

    pr_alpha = prof_rep.profiles["alpha"]
    assert pr_alpha.all_points_converged is True
    assert pr_alpha.nuisance_hit_bounds_mask[0] is True
    assert pr_alpha.nuisance_hit_bounds_mask[-1] is True
    assert "beta" in pr_alpha.nuisance_at_bounds_names[0]

    for rec in pr_alpha.pointwise_optimization_records:
        assert rec["success"] is True
        assert isinstance(rec["status"], int)
        assert isinstance(rec["message"], str) and len(rec["message"]) > 0
        assert isinstance(rec["nit"], int) and rec["nit"] >= 0
        assert isinstance(rec["nfev"], int) and 0 <= rec["nfev"] <= 160

    prof_partial = compute_optimized_objective_profiles(
        quad_coupled_obj,
        theta_star_phys,
        transform,
        num_grid_points=9,
        max_evals_per_point=120,
        max_iter_per_point=40,
        max_total_evaluations=18,
    )
    assert prof_partial.budget_exhausted is True
    assert prof_partial.all_profiles_complete_and_converged is False
    assert prof_partial.total_incomplete_points > 0


def test_04_figure_b_annotations_diagonal_zeros_and_undefined_cells() -> None:
    theta_ang = np.arccos(-0.9963)
    J_zero = np.zeros((6, 4), dtype=np.float64)
    J_zero[0, 0] = 1.0
    J_zero[0, 2] = np.cos(theta_ang)
    J_zero[1, 2] = np.sin(theta_ang)

    rep_zero = analyze_jacobian_svd(J_zero, rank_threshold=1.0e-8)
    json_zero = rep_zero.to_json_dict()
    cos_json = json_zero["local_correlation_analysis"]["column_cosine_similarity_matrix"]

    assert np.isclose(cos_json[0][0], 1.0, atol=1.0e-15)
    assert np.isclose(cos_json[2][2], 1.0, atol=1.0e-15)
    assert np.isclose(cos_json[0][2], -0.9963, atol=1.0e-4)
    assert cos_json[1][1] is None
    assert cos_json[3][3] is None

    table_zero = build_figure_b_cell_annotations(rep_zero)
    assert table_zero[0][0]["formatted_number"] == "+1.00"
    assert table_zero[0][0]["display_text"] == "+1.00\n(diag)"
    assert table_zero[2][2]["formatted_number"] == "+1.00"
    assert table_zero[2][2]["display_text"] == "+1.00\n(diag)"
    assert table_zero[0][2]["formatted_number"] == "-0.996"
    assert table_zero[0][2]["display_text"] == "-0.996\n(cos sim)"
    assert table_zero[2][0]["display_text"] == "-0.996\n(cos sim)"
    for idx in (1, 3):
        assert table_zero[idx][idx]["is_defined"] is False
        assert table_zero[idx][idx]["display_text"] == "UNDEFINED\n(Rank < 4)"
        assert table_zero[0][idx]["is_defined"] is False

    J_orth = np.vstack([np.diag([2.0, 1.5, 1.0, 0.8]), np.zeros((4, 4))])
    rep_orth = analyze_jacobian_svd(J_orth, rank_threshold=1.0e-8)
    table_orth = build_figure_b_cell_annotations(rep_orth)
    for i in range(4):
        for j in range(4):
            if i == j:
                assert table_orth[i][j]["display_text"] == "+1.00"
            else:
                assert table_orth[i][j]["display_text"] == "+0.00"


def test_05_json_serialization_and_figures_generation(tmp_path: Path) -> None:
    transform = _get_test_transform()
    u_star = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    theta_star_phys = transform.from_normalized(u_star)

    J_act = np.vstack([np.diag([1.2, 0.35, 0.9, 0.15]), np.ones((4, 4)) * 0.05])
    J_zero = J_act.copy()
    J_zero[:, 1] = 0.0
    J_zero[:, 3] = 0.0

    rep_act = analyze_jacobian_svd(J_act)
    rep_zero = analyze_jacobian_svd(J_zero)

    payload = sanitize_for_json(
        {
            "active": rep_act.to_json_dict(),
            "zero": rep_zero.to_json_dict(),
            "raw_nan_inf": [float("nan"), float("inf"), -float("inf"), 1.25],
        }
    )
    json_str = json.dumps(payload, allow_nan=False)
    parsed = json.loads(json_str)
    assert parsed["zero"]["local_correlation_analysis"]["fisher_correlation_matrix"] is None
    assert parsed["raw_nan_inf"] == [None, None, None, 1.25]

    obj_fn = lambda th: 0.5 * float(np.sum((transform.to_normalized(th) - u_star) ** 2))
    oat_rep = compute_one_at_a_time_cost_slices(obj_fn, theta_star_phys, transform, num_points=7)
    prof_rep = compute_optimized_objective_profiles(
        obj_fn, theta_star_phys, transform, num_grid_points=5, max_evals_per_point=80
    )
    assert prof_rep.all_profiles_complete_and_converged is True

    f_a = plot_figure_a_sensitivity_and_svd(rep_act, rep_zero, tmp_path / "figA.pdf")
    f_b = plot_figure_b_local_correlation(rep_act, rep_zero, tmp_path / "figB.pdf")
    f_c = plot_figure_c_cost_slices(oat_rep, oat_rep, transform, tmp_path / "figC.pdf")
    f_d = plot_figure_d_optimized_profiles(prof_rep, tmp_path / "figD.pdf")

    for group in (f_a, f_b, f_c, f_d):
        assert Path(group["pdf"]).exists() and Path(group["pdf"]).stat().st_size > 0
        assert Path(group["png"]).exists() and Path(group["png"]).stat().st_size > 0


def test_06_strict_per_point_budget_and_exact_accounting_in_bidirectional_scan() -> None:
    """
    Requisiti 1, 2, 3, 4:
    Verifica in una scansione bidirezionale sintetica (sia con budget per punto stretto
    `max_evals_per_point=22` che con budget completo `max_evals_per_point=140` su una
    valle curva tipo Rosenbrock) che:
    1) Per ogni singolo punto di ogni parametro valga rigorosamente `nfev <= max_evals_per_point`;
    2) La somma dei contatori disaggregati (`oat + warm_probes + primary_opt + recovery_opt`)
       coincida con `nfev` per ogni punto;
    3) La somma delle valutazioni di tutti i punti più `setup_reference_evaluations` (1)
       coincida esattamente con `total_forward_evaluations` e con il numero reale di
       chiamate alla funzione obiettivo.
    """
    transform = _get_test_transform()
    u_star = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    theta_star_phys = transform.from_normalized(u_star)

    call_count = 0

    def curved_valley_objective(theta_phys: np.ndarray) -> float:
        nonlocal call_count
        call_count += 1
        u = transform.to_normalized(theta_phys)
        # Valle non lineare fortemente accoppiata (tipo Rosenbrock 4D) che richiede molte iterazioni
        d0 = u[0] - 0.5
        d1 = u[1] - (0.5 + 0.8 * d0 + 0.6 * (d0 ** 2))
        d2 = u[2] - (0.5 - 0.5 * d0 + 0.4 * d1)
        d3 = u[3] - (0.5 + 0.7 * d2)
        return float(0.4 * (d0 ** 2) + 8.0 * (d1 ** 2) + 6.0 * (d2 ** 2) + 5.0 * (d3 ** 2))

    # Caso A: Limite per punto molto stretto (max_evals_per_point = 22) con scansione bidirezionale attiva
    call_count = 0
    cap_tight = 22
    rep_tight = compute_optimized_objective_profiles(
        curved_valley_objective,
        theta_star_phys,
        transform,
        num_grid_points=7,
        max_evals_per_point=cap_tight,
        max_iter_per_point=50,
        max_total_evaluations=2000,
        bidirectional_scan=True,
    )

    assert rep_tight.per_point_budget_respected is True
    assert rep_tight.accounting_consistency_verified is True
    assert rep_tight.setup_reference_evaluations == 1
    assert rep_tight.total_forward_evaluations == call_count

    sum_all_pts_tight = 0
    for pname, prof in rep_tight.profiles.items():
        assert prof.max_point_evaluations_observed <= cap_tight
        assert prof.total_evaluations_for_parameter == sum(prof.evaluations_per_point)
        sum_all_pts_tight += prof.total_evaluations_for_parameter
        for rec in prof.pointwise_optimization_records:
            assert 0 <= rec["nfev"] <= cap_tight, (
                f"Punto {pname}[{rec['grid_index']}] ha nfev={rec['nfev']} > {cap_tight}"
            )
            bd = rec["nfev_breakdown"]
            assert bd["oat"] + bd["warm_probes"] + bd["primary_opt"] + bd["recovery_opt"] == rec["nfev"]
            if not rec["success"]:
                assert rec["termination_category"] in (
                    "failed_local_eval_limit",
                    "failed_other",
                    "failed_global_budget_exhausted",
                )

    assert rep_tight.setup_reference_evaluations + sum_all_pts_tight == rep_tight.total_forward_evaluations

    # Caso B: Budget locale adeguato (max_evals_per_point = 140) con recupero di secondo stadio attivo
    call_count = 0
    cap_std = 140
    rep_std = compute_optimized_objective_profiles(
        curved_valley_objective,
        theta_star_phys,
        transform,
        num_grid_points=7,
        max_evals_per_point=cap_std,
        max_iter_per_point=60,
        max_total_evaluations=4000,
        bidirectional_scan=True,
    )
    assert rep_std.per_point_budget_respected is True
    assert rep_std.accounting_consistency_verified is True
    assert rep_std.total_forward_evaluations == call_count
    assert rep_std.all_profiles_complete_and_converged is True
    for prof in rep_std.profiles.values():
        assert prof.max_point_evaluations_observed <= cap_std