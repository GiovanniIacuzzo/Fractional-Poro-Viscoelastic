"""
tests/test_04_mms_coupled.py
Gate 3 e Gate 6 del Contratto Matematico (Sez. 7 e Sez. 9.3, 9.6):
Verifica mediante Soluzione Manufatta (MMS) del sistema completo a 3 campi (d*, p*, c*)
con controllo completo di tutti gli ordini spaziali (L2 e H1 su d, p, c), verifica di isolamento
dall'errore temporale e controllo dei tassi temporali su tutti e 3 i campi.
"""

from __future__ import annotations
import math
from reteco import _core, load_master_config


def _rate(e_coarse: float, e_fine: float, h_coarse: float, h_fine: float) -> float:
    return math.log(e_coarse / e_fine) / math.log(h_coarse / h_fine)


def test_coupled_mms_spatial_convergence_all_fields_and_temporal_isolation() -> None:
    """
    Studio di convergenza spaziale per Taylor-Hood Q_2/Q_1 + Q_1 tracciante:
    - Verifica soglie sugli ordini L^2 e H^1 per TUTTI e tre i campi (d, p, c).
    - Verifica sulla mesh più fine che raddoppiando il numero di passi temporali
      gli errori L^2 cambino in misura trascurabile (< 2%), dimostrando che l'errore
      temporale non contamina la stima dell'ordine spaziale.
    """
    master = load_master_config("configs/scales_and_physics.yaml")
    phys = master.fixed_physics.to_cpp(master.characteristic_scales)
    theta = _core.ParameterTheta(0.70, 0.45, 0.08, 2.50)

    mms = _core.MMSManufacturedSystem(phys, theta, 0.4)
    T_end = 0.25
    grid_base = _core.TimeGrid.make_graded(T_end, 300, 1.0)

    N_list = [6, 12, 24]
    reports = [mms.run_full_mms_test(N, grid_base, 2, 1, 1) for N in N_list]

    for i in range(1, len(reports)):
        assert reports[i].d_L2 < reports[i - 1].d_L2
        assert reports[i].d_H1 < reports[i - 1].d_H1
        assert reports[i].p_L2 < reports[i - 1].p_L2
        assert reports[i].p_H1 < reports[i - 1].p_H1
        assert reports[i].c_L2 < reports[i - 1].c_L2
        assert reports[i].c_H1 < reports[i - 1].c_H1

    r_coarse, r_fine = reports[-2], reports[-1]
    rate_d_l2 = _rate(r_coarse.d_L2, r_fine.d_L2, r_coarse.h_mesh, r_fine.h_mesh)
    rate_d_h1 = _rate(r_coarse.d_H1, r_fine.d_H1, r_coarse.h_mesh, r_fine.h_mesh)
    rate_p_l2 = _rate(r_coarse.p_L2, r_fine.p_L2, r_coarse.h_mesh, r_fine.h_mesh)
    rate_p_h1 = _rate(r_coarse.p_H1, r_fine.p_H1, r_coarse.h_mesh, r_fine.h_mesh)
    rate_c_l2 = _rate(r_coarse.c_L2, r_fine.c_L2, r_coarse.h_mesh, r_fine.h_mesh)
    rate_c_h1 = _rate(r_coarse.c_H1, r_fine.c_H1, r_coarse.h_mesh, r_fine.h_mesh)

    # Soglie esplicite per tutti e 6 gli indicatori spaziali
    assert rate_d_l2 > 2.50, f"Ordine L2 spostamento Q2 insufficiente: {rate_d_l2:.2f}"
    assert rate_d_h1 > 1.85, f"Ordine H1 spostamento Q2 insufficiente: {rate_d_h1:.2f}"
    assert rate_p_l2 > 1.85, f"Ordine L2 pressione Q1 insufficiente: {rate_p_l2:.2f}"
    assert rate_p_h1 > 0.90, f"Ordine H1 pressione Q1 insufficiente: {rate_p_h1:.2f}"
    assert rate_c_l2 > 1.85, f"Ordine L2 tracciante Q1 insufficiente: {rate_c_l2:.2f}"
    assert rate_c_h1 > 0.90, f"Ordine H1 tracciante Q1 insufficiente: {rate_c_h1:.2f}"

    # Controllo di isolamento dall'errore temporale sulla mesh più fine (Sez. 2.2 ChatGPT)
    grid_double = _core.TimeGrid.make_graded(T_end, 600, 1.0)
    rep_fine_dt = mms.run_full_mms_test(N_list[-1], grid_double, 2, 1, 1)
    rel_change_p = abs(rep_fine_dt.p_L2 - r_fine.p_L2) / r_fine.p_L2
    rel_change_c = abs(rep_fine_dt.c_L2 - r_fine.c_L2) / r_fine.c_L2
    assert rel_change_p < 0.03, f"Errore spaziale pressione contaminato da dt: {rel_change_p:.2%}"
    assert rel_change_c < 0.03, f"Errore spaziale tracciante contaminato da dt: {rel_change_c:.2%}"


def test_coupled_mms_temporal_convergence_all_three_fields() -> None:
    """
    Studio di convergenza temporale M = [10, 20, 40, 80] con verifica quantitativa
    esplicita dell'ordine di convergenza per TUTTI e tre i campi (d, p, c).
    """
    master = load_master_config("configs/scales_and_physics.yaml")
    phys = master.fixed_physics.to_cpp(master.characteristic_scales)
    theta = _core.ParameterTheta(0.65, 0.45, 0.08, 2.50)

    mms = _core.MMSManufacturedSystem(phys, theta, 0.4)
    M_list = [10, 20, 40, 80]
    reports = []
    for M in M_list:
        grid = _core.TimeGrid.make_graded(0.5, M, 1.0)
        rep = mms.run_full_mms_test(24, grid, 2, 2, 2)
        reports.append(rep)

    for i in range(1, len(reports)):
        assert reports[i].d_L2 < reports[i - 1].d_L2
        assert reports[i].p_L2 < reports[i - 1].p_L2
        assert reports[i].c_L2 < reports[i - 1].c_L2

    rate_d_t = math.log(reports[-2].d_L2 / reports[-1].d_L2) / math.log(2.0)
    rate_p_t = math.log(reports[-2].p_L2 / reports[-1].p_L2) / math.log(2.0)
    rate_c_t = math.log(reports[-2].c_L2 / reports[-1].c_L2) / math.log(2.0)

    assert rate_d_t > 0.85, f"Tasso temporale spostamento d insufficiente: {rate_d_t:.2f}"
    assert rate_p_t > 0.85, f"Tasso temporale pressione p insufficiente: {rate_p_t:.2f}"
    assert rate_c_t > 0.85, f"Tasso temporale tracciante c insufficiente: {rate_c_t:.2f}"