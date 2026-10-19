"""
tests/test_09_twin_forcing_consistency.py
Verifica rigorosa della costruzione discreta delle forzanti del twin experiment (Fase 1):
1. Assemblaggio meccanico: coincidenza tra vettore assemblato con matrice di massa M_d
   e integrale FEM di riferimento (quadratura diretta di Gauss-Legendre e risultante analitico),
   più verifica di coincidenza esatta a precisione di macchina con le matrici M_p e M_c di deal.II;
2. Componenti direzionali: rispetto di amplitude_x e amplitude_y senza mescolamento tra DOFs x e y;
3. Assemblaggio della sorgente fluida: corrispondenza con l'integrale contro le funzioni test w_h;
4. Bilancio integrale della sorgente fluida: misura di int_Omega s_f dx nel caso standard
   (integrale non nullo prescritto) e nel caso opzionale a media integrale nulla;
5. Compatibilità iniziale a t = 0: F_mech(0) = 0, S_fluid(0) = 0 e residuo di equilibrio nullo;
6. Riproducibilità: invarianza bit-a-bit delle forzanti e della traiettoria forward;
7. Assenza di contaminazioni: conservazione della massa del tracciante e proprietà dei test storici.
"""

from __future__ import annotations
import copy
import math
import numpy as np
import yaml

from reteco import _core, load_master_config
from reteco.forward.wrapper import RetecoForwardProblem2D
from reteco.forward.target_generator import (
    build_displacement_mass_matrix_csr,
    build_scalar_lagrange_mass_matrix_csr,
    build_standard_driving_Callbacks,
    compute_spatial_forcing_vectors_and_diagnostics,
    identify_displacement_component_masks,
    integrate_vector_body_force_quadrature,
)


def _make_test_problem(subdivisions_N: int = 8, time_steps_M: int = 20) -> RetecoForwardProblem2D:
    master = load_master_config("configs/scales_and_physics.yaml")
    grid = _core.TimeGrid.make_graded(1.0, time_steps_M, 1.5)
    return RetecoForwardProblem2D(
        subdivisions_N=subdivisions_N,
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid,
    )


def test_01_mechanical_mass_matrix_and_weak_integral_consistency() -> None:
    """
    1. Verifica che il costruttore della matrice di massa scalare coincida a precisione di
       macchina (< 1e-14) con M_p_csr e M_c_csr assemblate in C++ da deal.II;
    2. Verifica che per un campo polinomiale quadratico in [Q_2]^2, M_d @ f_nodal coincida
       a precisione di macchina con l'integrale di quadratura di Gauss-Legendre;
    3. Verifica che per il profilo sin(pi*x)*sin(pi*y) del protocollo, il vettore M_d @ f_nodal
       e i risultanti integrali convergano all'integrale FEM di riferimento.
    """
    problem = _make_test_problem(subdivisions_N=8, time_steps_M=10)

    # 1a. Coincidenza esatta con M_p_csr e M_c_csr di deal.II (grado 1)
    m_p_py = build_scalar_lagrange_mass_matrix_csr(
        problem.press_pts, subdivisions_N=8, fe_degree=1
    )
    diff_mp = np.max(np.abs((m_p_py - problem.M_p_csr).toarray()))
    assert diff_mp < 1.0e-14, f"Discrepanza rispetto a M_p di deal.II: {diff_mp:.3e}"

    m_c_py = build_scalar_lagrange_mass_matrix_csr(
        problem.tracer_pts, subdivisions_N=8, fe_degree=1
    )
    diff_mc = np.max(np.abs((m_c_py - problem.M_c_csr).toarray()))
    assert diff_mc < 1.0e-14, f"Discrepanza rispetto a M_c di deal.II: {diff_mc:.3e}"

    # 1b. Coincidenza esatta su campo polinomiale quadratico f_s(x,y) in [Q_2]^2
    mask_x, mask_y = identify_displacement_component_masks(problem)
    M_d = build_displacement_mass_matrix_csr(problem)
    pts = problem.disp_pts

    fx_poly = lambda x, y: 1.5 * (x ** 2) * y + 0.5 * x - 0.25 * (y ** 2)
    fy_poly = lambda x, y: -0.8 * x * (y ** 2) + 1.2 * y + 0.3

    f_nodal_poly = np.zeros(problem.num_disp_dofs, dtype=np.float64)
    f_nodal_poly[mask_x] = fx_poly(pts[mask_x, 0], pts[mask_x, 1])
    f_nodal_poly[mask_y] = fy_poly(pts[mask_y, 0], pts[mask_y, 1])

    F_mass_poly = M_d @ f_nodal_poly
    F_quad_poly = integrate_vector_body_force_quadrature(
        problem, fx_poly, fy_poly, apply_dirichlet_homogenization=False
    )
    np.testing.assert_allclose(F_mass_poly, F_quad_poly, rtol=1e-13, atol=1e-14)

    # 1c. Verifica sul profilo trigonometrico del protocollo twin_experiment_synthetic.yaml
    with open("configs/twin_experiment_synthetic.yaml", "r", encoding="utf-8") as f:
        twin_cfg = yaml.safe_load(f)

    info = compute_spatial_forcing_vectors_and_diagnostics(
        problem, twin_cfg["driving_protocol"]
    )
    amp_x = info["amplitude_x"]
    amp_y = info["amplitude_y"]

    fx_trig = lambda x, y: amp_x * np.sin(np.pi * x) * np.sin(np.pi * y)
    fy_trig = lambda x, y: amp_y * np.sin(np.pi * x) * np.sin(np.pi * y)
    F_quad_trig = integrate_vector_body_force_quadrature(
        problem, fx_trig, fy_trig, apply_dirichlet_homogenization=True
    )

    rel_diff_vec = np.linalg.norm(info["F_spatial_base"] - F_quad_trig) / np.linalg.norm(F_quad_trig)
    assert rel_diff_vec < 1.0e-3, f"Errore relativo vettore carico FEM: {rel_diff_vec:.3e}"
    assert np.isclose(
        info["mech_resultant_x_spatial"],
        info["mech_resultant_x_continuous_ref"],
        rtol=5.0e-5,
    )
    assert np.isclose(
        info["mech_resultant_y_spatial"],
        info["mech_resultant_y_continuous_ref"],
        rtol=5.0e-5,
    )


def test_02_directional_components_separation_and_scaling() -> None:
    """
    Verifica che le componenti x e y della forza rispettino rigorosamente amplitude_x e
    amplitude_y senza mescolarsi o replicarsi.
    """
    problem = _make_test_problem(subdivisions_N=6, time_steps_M=10)
    mask_x, mask_y = identify_displacement_component_masks(problem)

    with open("configs/twin_experiment_synthetic.yaml", "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)["driving_protocol"]

    # Caso 1: Forza puramente lungo x (amplitude_x = 2.0, amplitude_y = 0.0)
    cfg_only_x = copy.deepcopy(base_cfg)
    cfg_only_x["mechanical_body_force"]["amplitude_x"] = 2.0
    cfg_only_x["mechanical_body_force"]["amplitude_y"] = 0.0
    _, F_only_x, _ = build_standard_driving_Callbacks(problem, cfg_only_x)
    vec_x = F_only_x(0.5)
    assert np.linalg.norm(vec_x[mask_x]) > 1.0e-2
    assert np.linalg.norm(vec_x[mask_y]) == 0.0

    # Caso 2: Forza puramente lungo y (amplitude_x = 0.0, amplitude_y = 3.5)
    cfg_only_y = copy.deepcopy(base_cfg)
    cfg_only_y["mechanical_body_force"]["amplitude_x"] = 0.0
    cfg_only_y["mechanical_body_force"]["amplitude_y"] = 3.5
    _, F_only_y, _ = build_standard_driving_Callbacks(problem, cfg_only_y)
    vec_y = F_only_y(0.5)
    assert np.linalg.norm(vec_y[mask_x]) == 0.0
    assert np.linalg.norm(vec_y[mask_y]) > 1.0e-2

    # Caso 3: Rapporto esatto amplitude_x / amplitude_y nella configurazione standard (1.5 vs 1.0)
    info_std = compute_spatial_forcing_vectors_and_diagnostics(problem, base_cfg)
    ratio_res = info_std["mech_resultant_x_spatial"] / info_std["mech_resultant_y_spatial"]
    expected_ratio = (
        base_cfg["mechanical_body_force"]["amplitude_x"]
        / base_cfg["mechanical_body_force"]["amplitude_y"]
    )
    assert np.isclose(ratio_res, expected_ratio, rtol=1.0e-14)


def test_03_and_04_fluid_source_assembly_and_integral_balance() -> None:
    """
    Verifica l'assemblaggio di S_fluid_func contro le funzioni test di pressione w_h
    e misura il bilancio integrale int_Omega s_f dx sia per la sorgente prescritta
    dal protocollo (integrale non nullo), sia quando viene attivato enforce_zero_mean.
    """
    problem = _make_test_problem(subdivisions_N=12, time_steps_M=10)
    with open("configs/twin_experiment_synthetic.yaml", "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)["driving_protocol"]

    # 1. Sorgente standard prescritta dal protocollo (enforce_zero_mean = False)
    info = compute_spatial_forcing_vectors_and_diagnostics(problem, base_cfg)
    s_int_disc = info["fluid_source_spatial_integral"]
    s_int_cont = info["fluid_source_continuous_integral_ref"]

    # L'integrale continuo vale circa 0.8 * (2*pi*0.12^2 - 0.5) ~ -0.327611 (non nullo)
    assert s_int_cont < -0.30
    assert np.isclose(s_int_disc, s_int_cont, rtol=2.0e-3), (
        f"Integrale discreto s_f ({s_int_disc:.6f}) distante dal valore continuo ({s_int_cont:.6f})"
    )

    # Verifica che per una funzione test costante w_h = 1, (s_f(t), 1)_Omega = sin(omega*t) * s_int_disc
    _, _, S_fluid_func = build_standard_driving_Callbacks(problem, base_cfg)
    t_test = 0.25
    omega = info["omega_t"]
    ones_p = np.ones(problem.num_press_dofs, dtype=np.float64)
    integral_at_t = float(ones_p @ S_fluid_func(t_test))
    assert np.isclose(
        integral_at_t, math.sin(omega * t_test) * s_int_disc, rtol=1.0e-14
    )

    # 2. Verifica dell'opzione esplicita a media integrale nulla (enforce_zero_mean = True)
    cfg_zero_mean = copy.deepcopy(base_cfg)
    cfg_zero_mean["fluid_injection_source"]["enforce_zero_mean"] = True
    info_zm = compute_spatial_forcing_vectors_and_diagnostics(problem, cfg_zero_mean)
    assert abs(info_zm["fluid_source_spatial_integral"]) < 1.0e-14


def test_05_initial_compatibility_at_t_zero() -> None:
    """
    Verifica che a t = 0 le forzanti siano esattamente nulle e che lo stato iniziale
    d_0 = 0, p_0 = 0 soddisfi la condizione di compatibilità meccanica a precisione di macchina.
    """
    problem = _make_test_problem(subdivisions_N=8, time_steps_M=10)
    with open("configs/twin_experiment_synthetic.yaml", "r", encoding="=utf-8" if False else "utf-8") as f:
        driving_cfg = yaml.safe_load(f)["driving_protocol"]

    _, F_mech_func, S_fluid_func = build_standard_driving_Callbacks(problem, driving_cfg)
    F0 = F_mech_func(0.0)
    S0 = S_fluid_func(0.0)

    assert np.all(F0 == 0.0)
    assert np.all(S0 == 0.0)

    d0 = np.zeros(problem.num_disp_dofs, dtype=np.float64)
    p0 = np.zeros(problem.num_press_dofs, dtype=np.float64)
    res_norm = _core.InitialCompatibility.compatibility_residual_norm(
        problem.fem, 2.50, d0, p0, F0
    )
    assert res_norm == 0.0


def test_06_and_07_reproducibility_and_mass_conservation_integrity() -> None:
    """
    Verifica la riproducibilità bit-a-bit della traiettoria con le nuove forzanti e
    l'assenza di regressioni sulla conservazione della massa del tracciante con flusso attivo.
    """
    master = load_master_config("configs/scales_and_physics.yaml")
    with open("configs/twin_experiment_synthetic.yaml", "r", encoding="utf-8") as f:
        twin_cfg = yaml.safe_load(f)

    theta = master.parameter_space_theta.theta_dagger_array()

    prob1 = _make_test_problem(subdivisions_N=8, time_steps_M=20)
    c0_1, F_1, Sf_1 = build_standard_driving_Callbacks(prob1, twin_cfg["driving_protocol"])
    traj1 = prob1.solve_trajectory(
        theta_vec=theta,
        c0=c0_1,
        F_mech_func=F_1,
        S_fluid_func=Sf_1,
        compute_diagnostics=True,
    )

    prob2 = _make_test_problem(subdivisions_N=8, time_steps_M=20)
    c0_2, F_2, Sf_2 = build_standard_driving_Callbacks(prob2, twin_cfg["driving_protocol"])
    traj2 = prob2.solve_trajectory(
        theta_vec=theta,
        c0=c0_2,
        F_mech_func=F_2,
        S_fluid_func=Sf_2,
        compute_diagnostics=True,
    )

    # Riproducibilità esatta
    np.testing.assert_array_equal(F_1(0.35), F_2(0.35))
    np.testing.assert_array_equal(Sf_1(0.35), Sf_2(0.35))
    assert traj1.max_darcy_velocity == traj2.max_darcy_velocity
    assert traj1.max_darcy_velocity > 1.0e-4

    # Conservazione della massa del tracciante M(t_n) = M(0)
    masses_1 = np.array([d.tracer_total_mass for d in traj1.diagnostics], dtype=np.float64)
    masses_2 = np.array([d.tracer_total_mass for d in traj2.diagnostics], dtype=np.float64)
    np.testing.assert_array_equal(masses_1, masses_2)

    rel_mass_var = np.max(np.abs(masses_1 - masses_1[0])) / masses_1[0]
    assert rel_mass_var < 1.0e-12