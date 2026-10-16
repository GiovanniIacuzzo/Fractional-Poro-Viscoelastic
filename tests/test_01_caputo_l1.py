"""
tests/test_01_caputo_l1.py
Gate 1 del Contratto Matematico (Sez. 6 e Sez. 9.1):
- Verifica dei pesi L1 su griglia uniforme e graduata t_n = T*(n/N)^r
- Convergenza empirica su u(t) = t^2 rispetto a Gamma(3)/Gamma(3-gamma)*t^{2-gamma}
- Verifica del limite classico gamma -> 1^- (Backward Euler)
"""

from __future__ import annotations
import math
import numpy as np
import pytest
from scipy.special import gamma

from reteco import _core


@pytest.mark.parametrize("gamma_order", [0.25, 0.50, 0.80])
@pytest.mark.parametrize("grading_r", [1.0, 1.8])
def test_caputo_l1_convergence_on_t_squared(gamma_order: float, grading_r: float) -> None:
    """
    Controlla che l'approssimazione L1 converga al valore analitico esatto di CD_t^gamma(t^2)
    sia su griglie uniformi (r=1.0) sia su griglie graduate (r=1.8).
    """
    T = 0.8
    exact_val = (2.0 / gamma(3.0 - gamma_order)) * (T ** (2.0 - gamma_order))

    steps_list = [20, 40, 80, 160]
    errors = []

    for N in steps_list:
        grid = _core.TimeGrid.make_graded(T, N, grading_r)
        mem = _core.CaputoL1Memory(gamma_order, grid, 1)
        mem.reset(np.array([0.0], dtype=np.float64))

        approx_T = 0.0
        for n in range(1, N + 1):
            t_n = grid.time_at(n)
            u_n = np.array([t_n * t_n], dtype=np.float64)
            if n == N:
                approx_T = float(mem.evaluate_discrete_derivative(n, u_n)[0])
            mem.commit_step(n, u_n)

        errors.append(abs(approx_T - exact_val))

    # L'errore deve decrescere strettamente al raffinarsi della griglia
    for i in range(1, len(errors)):
        assert errors[i] < errors[i - 1], f"Errore non decrescente: {errors}"

    # Calcolo dell'ordine empirico sull'ultimo raffinamento (atteso ~ 2 - gamma)
    rate_last = math.log(errors[-2] / errors[-1]) / math.log(steps_list[-1] / steps_list[-2])
    expected_rate = 2.0 - gamma_order
    assert rate_last > expected_rate - 0.15, (
        f"Tasso di convergenza L1 troppo basso: ottenuto {rate_last:.3f}, atteso ~{expected_rate:.3f}"
    )


def test_caputo_l1_backward_euler_limit() -> None:
    """
    Verifica che per gamma -> 1^- (es. gamma = 1 - 1e-7) l'operatore L1 su griglia non uniforme
    tenda alla derivata prima discreta all'indietro (u^n - u^{n-1}) / dt_n.
    """
    T = 1.0
    N = 25
    grid = _core.TimeGrid.make_graded(T, N, 1.6)
    gamma_near_one = 1.0 - 1.0e-7

    mem = _core.CaputoL1Memory(gamma_near_one, grid, 2)
    u0 = np.array([1.0, -0.5], dtype=np.float64)
    mem.reset(u0)

    u_prev = u0.copy()
    for n in range(1, N + 1):
        t_n = grid.time_at(n)
        dt_n = grid.dt(n)
        u_n = np.array([math.cos(t_n), math.exp(-0.5 * t_n) - 1.5], dtype=np.float64)

        caputo_val = mem.evaluate_discrete_derivative(n, u_n)
        backward_euler = (u_n - u_prev) / dt_n

        rel_diff = np.linalg.norm(caputo_val - backward_euler) / np.linalg.norm(backward_euler)
        assert rel_diff < 1.0e-5, f"Limite gamma->1 violato al passo {n}: rel_diff={rel_diff}"

        mem.commit_step(n, u_n)
        u_prev = u_n.copy()


def test_caputo_constant_function_is_exact_zero() -> None:
    """Per una funzione costante u(t) = u_0 != 0, la derivata di Caputo L1 deve essere 0 a precisione di macchina."""
    grid = _core.TimeGrid.make_graded(1.2, 30, 1.4)
    mem = _core.CaputoL1Memory(0.63, grid, 5)
    u_const = np.array([2.5, -1.2, 0.8, 4.1, 0.0], dtype=np.float64)
    mem.reset(u_const)

    for n in range(1, 31):
        d_val = mem.evaluate_discrete_derivative(n, u_const)
        assert np.max(np.abs(d_val)) < 1.0e-14
        mem.commit_step(n, u_const)