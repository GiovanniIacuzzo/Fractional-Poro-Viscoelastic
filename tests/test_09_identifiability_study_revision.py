"""
tests/test_09_identifiability_study_revision.py
Test unitari rapidi per la revisione scientifica dello studio di identificabilità:
1) Rango, condizionamento, indice di collinearità e modo meno informativo su matrici
   sintetiche a rango pieno e a rango deficiente;
2) Gestione trasparente delle correlazioni non definite quando il rango è < 4;
3) Differenze finite centrali nei punti interni, schemi unidirezionali del 2° ordine
   vicino ai bordi e audit di stabilità rispetto al passo h;
4) Correttezza delle sezioni unidimensionali (One-at-a-Time Cost Slices);
5) Correttezza dei profili dell'obiettivo ottimizzati (Optimized Objective Profiles)
   su un funzionale analitico quadratico accoppiato con soluzione esatta nota
   (sia con minimo interno sia con parametri di disturbo vincolati al bordo);
6) Serializzazione JSON rigorosa di valori finiti, NaN, Inf e quantità non definite,
   e generazione delle Figure A, B, C, D in doppio formato PDF e PNG (300 dpi).
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
    plot_figure_a_sensitivity_and_svd,
    plot_figure_b_local_correlation,
    plot_figure_c_cost_slices,
    plot_figure_d_optimized_profiles,
)


def _get_test_transform() -> ParameterCoordinateTransform:
    master = load_master_config("configs/scales_and_physics.yaml")
    return ParameterCoordinateTransform(master.parameter_space_theta.bounds)


def test_01_svd_full_rank_vs_rank_deficient_and_undefined_correlations() -> None:
    """
    Verifica rango, condizionamento e comportamento della matrice di correlazione
    su matrici sintetiche a rango pieno (rank=4) e a rango deficiente (rank=2).
    """
    # 1. Matrice ortogonale scalata a rango pieno (4): diag(4.0, 2.0, 1.0, 0.5)
    J_full = np.vstack([np.diag([4.0, 2.0, 1.0, 0.5]), np.zeros((6, 4))])
    rep_full = analyze_jacobian_svd(J_full, rank_threshold=1.0e-8)

    assert rep_full.numerical_rank == 4
    assert np.isclose(rep_full.condition_number, 8.0, rtol=1.0e-12)
    assert np.isclose(rep_full.collinearity_index, 1.0, rtol=1.0e-12)
    assert rep_full.correlation_matrix_defined is True
    assert np.allclose(rep_full.fisher_correlation_matrix, np.eye(4), atol=1.0e-12)
    assert rep_full.least_informative_mode["dominant_parameter"] == "E"
    assert np.isclose(rep_full.least_informative_mode["singular_value"], 0.5)

    # 2. Matrice a rango deficiente (rank=2): colonne beta (1) ed E (3) nulle
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
    """
    Verifica l'esattezza del 2° ordine sia nei punti interni sia sul bordo di [0,1]^4,
    la pesatura J_w = W^{1/2} J e lo studio di stabilità rispetto al passo h.
    """
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

    # Funzione quadratica in coordinate normalizzate: G(theta) = A_true * theta_tilde + 0.1 * theta_tilde^2
    # Poiché gli schemi centrali e unidirezionali a 3 punti sono del 2° ordine,
    # devono essere ESATTI sui polinomi quadratici!
    def G_quad(theta_phys: np.ndarray) -> np.ndarray:
        t_norm = transform.to_normalized(theta_phys)
        return A_true @ t_norm + 0.1 * (A_true @ (t_norm ** 2))

    # Punto con alpha~ sul bordo inferiore (0.0), E~ sul bordo superiore (1.0) e gli altri interni
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


def test_03_cost_slices_vs_optimized_objective_profiles_analytical() -> None:
    """
    Confronta su una forma quadratica accoppiata analitica:
        J(u) = 0.5 * (u_0 - 0.5)^2 + 2.0 * (u_1 - u_0)^2 + 1.5 * (u_2 - 0.5)^2 + 1.0 * (u_3 - 0.5)^2
    - La sezione OAT lungo u_0 (con u_1=0.5 fisso) vale:
          J_OAT(u_0) = 0.5*(u_0 - 0.5)^2 + 2.0*(0.5 - u_0)^2 = 2.5 * (u_0 - 0.5)^2;
    - Il profilo ottimizzato lungo u_0 (ottimizzando u_1 = u_0) vale:
          P_0(u_0) = 0.5 * (u_0 - 0.5)^2  (strettamente inferiore a J_OAT per ogni u_0 != 0.5!).
    """
    transform = _get_test_transform()
    u_star = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    theta_star_phys = transform.from_normalized(u_star)

    def quad_coupled_obj(theta_phys: np.ndarray) -> float:
        u = transform.to_normalized(theta_phys)
        return float(
            0.5 * ((u[0] - 0.5) ** 2)
            + 2.0 * ((u[1] - u[0]) ** 2)
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
        max_evals_per_point=40,
        max_total_evaluations=800,
    )

    sl_alpha = oat_rep.slices["alpha"]
    pr_alpha = prof_rep.profiles["alpha"]

    expected_oat = 2.5 * ((sl_alpha.grid_normalized - 0.5) ** 2)
    expected_prof = 0.5 * ((pr_alpha.grid_normalized - 0.5) ** 2)

    assert np.allclose(sl_alpha.cost_values, expected_oat, atol=1.0e-10)
    assert np.allclose(pr_alpha.profile_cost_values, expected_prof, atol=1.0e-4)
    # Verifica che per u_0 != 0.5 il profilo ottimizzato sia strettamente minore della sezione OAT
    non_ref = np.abs(pr_alpha.grid_normalized - 0.5) > 0.05
    assert np.all(pr_alpha.profile_cost_values[non_ref] < pr_alpha.oat_slice_cost_values[non_ref])


def test_04_json_serialization_and_figures_generation(tmp_path: Path) -> None:
    """
    Verifica la serializzazione JSON di valori finiti, NaN, Inf e risultati non disponibili
    e controlla la generazione delle Figure A, B, C, D in PDF e PNG.
    """
    transform = _get_test_transform()
    u_star = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    theta_star_phys = transform.from_normalized(u_star)

    J_act = np.vstack([np.diag([1.2, 0.35, 0.9, 0.15]), np.ones((4, 4)) * 0.05])
    J_zero = J_act.copy()
    J_zero[:, 1] = 0.0
    J_zero[:, 3] = 0.0

    rep_act = analyze_jacobian_svd(J_act)
    rep_zero = analyze_jacobian_svd(J_zero)

    # Verifica che json.dumps(..., allow_nan=False) abbia successo senza eccezioni
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
        obj_fn, theta_star_phys, transform, num_grid_points=5, max_evals_per_point=15
    )

    f_a = plot_figure_a_sensitivity_and_svd(rep_act, rep_zero, tmp_path / "figA.pdf")
    f_b = plot_figure_b_local_correlation(rep_act, rep_zero, tmp_path / "figB.pdf")
    f_c = plot_figure_c_cost_slices(oat_rep, oat_rep, transform, tmp_path / "figC.pdf")
    f_d = plot_figure_d_optimized_profiles(prof_rep, tmp_path / "figD.pdf")

    for group in (f_a, f_b, f_c, f_d):
        assert Path(group["pdf"]).exists() and Path(group["pdf"]).stat().st_size > 0
        assert Path(group["png"]).exists() and Path(group["png"]).stat().st_size > 0