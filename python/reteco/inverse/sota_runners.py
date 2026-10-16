"""
python/reteco/inverse/sota_runners.py
Wrapper equi per gli ottimizzatori di riferimento (L-BFGS-B, Powell, Nelder-Mead,
Differential Evolution) vincolati allo stesso funzionale J_h(theta), agli stessi bounds
e allo stesso budget massimo di valutazioni FEM (Contratto Sez. 8 e 9.9).
"""

from __future__ import annotations
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

from .objective_functional import InverseObjectiveFunctional, BudgetExhaustedException
from .cpso import OptimizationRunResult


def _extract_run_result(
    objective: InverseObjectiveFunctional,
    alg_name: str,
    seed: int,
    wall_sec: float,
) -> OptimizationRunResult:
    if objective.best_theta is None or len(objective.history) == 0:
        raise RuntimeError(f"Nessuna valutazione registrata per l'algoritmo {alg_name}.")

    best_theta = objective.best_theta.copy()
    best_tilde = objective.transform.to_normalized(best_theta)
    eval_trace = np.array([rec.best_j_so_far for rec in objective.history], dtype=np.float64)

    theta_trace = np.zeros((len(objective.history), 4), dtype=np.float64)
    running_best_theta = objective.history[0].theta.copy()
    running_best_j = objective.history[0].j_total
    for k, rec in enumerate(objective.history):
        if rec.j_total <= running_best_j:
            running_best_j = rec.j_total
            running_best_theta = rec.theta.copy()
        theta_trace[k, :] = running_best_theta

    return OptimizationRunResult(
        algorithm_name=alg_name,
        seed=int(seed),
        theta_hat=best_theta,
        theta_tilde_hat=best_tilde,
        best_cost_J=float(objective.best_cost),
        num_fem_evaluations=int(objective.num_fem_evaluations),
        wall_time_sec=wall_sec,
        convergence_eval_trace=eval_trace,
        theta_history_trace=theta_trace,
    )


def run_scipy_baseline(
    algorithm_name: str,
    objective: InverseObjectiveFunctional,
    seed: int,
) -> OptimizationRunResult:
    """
    Esegue uno tra {'L-BFGS-B', 'Powell', 'Nelder-Mead', 'Differential-Evolution'}
    nello spazio normalizzato [0, 1]^4 con arresto rigoroso al raggiungimento di
    `objective.max_evaluations`.
    """
    objective.reset_counters()
    rng = np.random.default_rng(int(seed))
    x0_tilde = rng.uniform(0.15, 0.85, size=4)
    box_bounds = [(0.0, 1.0)] * 4

    t0 = time.perf_counter()
    try:
        if algorithm_name == "L-BFGS-B":
            minimize(
                objective.call_from_normalized,
                x0_tilde,
                method="L-BFGS-B",
                bounds=box_bounds,
                options={"maxfun": objective.max_evaluations or 240, "ftol": 1e-12, "eps": 1e-3},
            )
        elif algorithm_name == "Powell":
            minimize(
                objective.call_from_normalized,
                x0_tilde,
                method="Powell",
                bounds=box_bounds,
                options={"maxfev": objective.max_evaluations or 240, "ftol": 1e-10},
            )
        elif algorithm_name == "Nelder-Mead":
            minimize(
                objective.call_from_normalized,
                x0_tilde,
                method="Nelder-Mead",
                bounds=box_bounds,
                options={"maxfev": objective.max_evaluations or 240, "fatol": 1e-10, "xatol": 1e-8},
            )
        elif algorithm_name == "Differential-Evolution":
            differential_evolution(
                objective.call_from_normalized,
                bounds=box_bounds,
                seed=int(seed),
                popsize=3,
                maxiter=50,
                polish=False,
            )
        else:
            raise ValueError(f"Algoritmo baseline non riconosciuto: {algorithm_name}")
    except BudgetExhaustedException:
        pass

    wall_sec = time.perf_counter() - t0
    return _extract_run_result(objective, algorithm_name, seed, wall_sec)