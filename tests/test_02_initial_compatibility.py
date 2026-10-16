"""
tests/test_02_initial_compatibility.py
Gate 2 del Contratto Matematico (Sez. 4 e Sez. 9.2):
Verifica della condizione di compatibilità meccanica iniziale a t = 0:
    a_E(d_0, v) - b * (p_0, div v) = <F(0), v>   per ogni v in V.
"""

from __future__ import annotations
import numpy as np
from reteco import _core, load_master_config


def test_homogeneous_zero_initial_state_is_compatible() -> None:
    """Se p0 = 0 e F(0) = 0, allora d0 = 0 deve soddisfare esattamente la compatibilità iniziale."""
    master = load_master_config("configs/scales_and_physics.yaml")
    phys = master.fixed_physics.to_cpp(master.characteristic_scales)
    fem = _core.InvariantFEMMatrices2D(12, phys, 2, 1, 1, True)

    d0 = np.zeros(fem.num_disp_dofs(), dtype=np.float64)
    p0 = np.zeros(fem.num_press_dofs(), dtype=np.float64)
    F0 = np.zeros(fem.num_disp_dofs(), dtype=np.float64)

    res = _core.InitialCompatibility.compatibility_residual_norm(fem, 2.5, d0, p0, F0)
    assert res == 0.0


def test_nonhomogeneous_initial_compatibility_solver() -> None:
    """
    Se p0 != 0 o F(0) != 0, un dato arbitrario d0 = 0 viola la compatibilità, mentre
    InitialCompatibility.compute_compatible_d0 deve annullare il residuo in V a precisione di macchina.
    """
    master = load_master_config("configs/scales_and_physics.yaml")
    phys = master.fixed_physics.to_cpp(master.characteristic_scales)
    fem = _core.InvariantFEMMatrices2D(10, phys, 2, 1, 1, True)

    pts_p = np.asarray(fem.get_press_support_points())
    p0 = np.cos(np.pi * pts_p[:, 0]) * np.cos(np.pi * pts_p[:, 1])
    F0 = np.zeros(fem.num_disp_dofs(), dtype=np.float64)
    E_val = 2.5

    # 1. Verifica che d0 = 0 produca un residuo non nullo
    d0_wrong = np.zeros(fem.num_disp_dofs(), dtype=np.float64)
    res_wrong = _core.InitialCompatibility.compatibility_residual_norm(fem, E_val, d0_wrong, p0, F0)
    assert res_wrong > 1.0e-3

    # 2. Calcolo di d0 compatibile e verifica che il residuo scenda sotto 1e-12
    d0_compat = _core.InitialCompatibility.compute_compatible_d0(fem, E_val, p0, F0)
    res_compat = _core.InitialCompatibility.compatibility_residual_norm(fem, E_val, d0_compat, p0, F0)
    assert res_compat < 1.0e-12, f"Residuo di compatibilità troppo alto: {res_compat}"