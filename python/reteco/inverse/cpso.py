"""
python/reteco/inverse/cpso.py
Implementazione matematicamente verificata del Continuous Particle Swarm Optimization (CPSO)
basata sulla soluzione analitica dell'ODE del secondo ordine (Orlando & Ricciardello, 2020),
con gestione esatta dei 3 regimi spettrali (Delta > 0, Delta == 0, Delta < 0) e seed deterministico.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import math
import time
import numpy as np

from .objective_functional import InverseObjectiveFunctional, BudgetExhaustedException


@dataclass
class OptimizationRunResult:
    algorithm_name: str
    seed: int
    theta_hat: np.ndarray
    theta_tilde_hat: np.ndarray
    best_cost_J: float
    num_fem_evaluations: int
    wall_time_sec: float
    convergence_eval_trace: np.ndarray   # (N_evals,) miglior costo J ad ogni valutazione FEM
    theta_history_trace: np.ndarray      # (N_evals, 4) miglior parametro theta ad ogni valutazione FEM


def _solve_cpso_analytical_step(
    x0: np.ndarray,
    v0: np.ndarray,
    attractor: np.ndarray,
    a_damp: float,
    b_stiff: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Risolve analiticamente al tempo t = dt il sistema di Cauchy componente per componente:
        y''(t) + a * y'(t) + b * y(t) = 0,   con y(t) = x(t) - attractor,
        y(0) = x0 - attractor,  y'(0) = v0,
    dove a = 1 - omega > 0 e b = phi_1 + phi_2 >= 0.
    Gestisce con continuità numerica i 3 casi: Delta > 0, Delta == 0, Delta < 0.
    """
    y0 = x0 - attractor
    n_dim = x0.size
    y_next = np.empty(n_dim, dtype=np.float64)
    v_next = np.empty(n_dim, dtype=np.float64)

    for d in range(n_dim):
        bi = float(b_stiff[d])
        y_init = float(y0[d])
        v_init = float(v0[d])

        if bi <= 1.0e-15:
            # Caso puro smorzamento y'' + a y' = 0
            if abs(a_damp) <= 1.0e-12:
                y_next[d] = y_init + v_init * dt
                v_next[d] = v_init
            else:
                exp_at = math.exp(-a_damp * dt)
                y_next[d] = y_init + (v_init / a_damp) * (1.0 - exp_at)
                v_next[d] = v_init * exp_at
            continue

        disc = a_damp * a_damp - 4.0 * bi
        half_a = 0.5 * a_damp

        if abs(disc) <= 1.0e-10:
            # Caso 2: Radici reali coincidenti r = -a / 2
            r = -half_a
            exp_rt = math.exp(r * dt)
            c1 = y_init
            c2 = v_init - r * y_init
            y_next[d] = (c1 + c2 * dt) * exp_rt
            v_next[d] = (r * (c1 + c2 * dt) + c2) * exp_rt
        elif disc > 0.0:
            # Caso 1: Radici reali distinte r1, r2
            sqrt_d = math.sqrt(disc)
            r1 = -half_a + 0.5 * sqrt_d
            r2 = -half_a - 0.5 * sqrt_d
            c1 = (v_init - r2 * y_init) / (r1 - r2)
            c2 = (r1 * y_init - v_init) / (r1 - r2)
            e1 = math.exp(r1 * dt)
            e2 = math.exp(r2 * dt)
            y_next[d] = c1 * e1 + c2 * e2
            v_next[d] = c1 * r1 * e1 + c2 * r2 * e2
        else:
            # Caso 3: Radici complesse coniugate -a/2 +/- i * omega_d
            omega_d = 0.5 * math.sqrt(-disc)
            exp_at = math.exp(-half_a * dt)
            cos_wt = math.cos(omega_d * dt)
            sin_wt = math.sin(omega_d * dt)
            c1 = y_init
            c2 = (v_init + half_a * y_init) / omega_d
            y_next[d] = exp_at * (c1 * cos_wt + c2 * sin_wt)
            v_next[d] = exp_at * (
                (-half_a * c1 + omega_d * c2) * cos_wt
                + (-half_a * c2 - omega_d * c1) * sin_wt
            )

    return attractor + y_next, v_next


def run_cpso_optimization(
    objective: InverseObjectiveFunctional,
    seed: int,
    num_particles: int = 12,
    max_iterations: int = 20,
    omega_inertia: float = 0.65,
    c1_cognitive: float = 1.40,
    c2_social: float = 1.40,
    dt_step: float = 1.0,
) -> OptimizationRunResult:
    """
    Esegue l'ottimizzazione CPSO nello spazio normalizzato [0, 1]^4 rispettando
    il budget esatto di valutazioni FEM registrato da `objective`.
    """
    objective.reset_counters()
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(seed))

    n_dim = 4
    # Inizializzazione Latin-like stratificata in [0.05, 0.95]^4
    X = rng.uniform(0.05, 0.95, size=(num_particles, n_dim))
    V = rng.uniform(-0.15, 0.15, size=(num_particles, n_dim))

    P_best = X.copy()
    P_cost = np.full(num_particles, float("inf"), dtype=np.float64)

    G_best = X[0].copy()
    G_cost = float("inf")

    a_damp = 1.0 - float(omega_inertia)
    if a_damp <= 0.0:
        raise ValueError("CPSO richiede omega_inertia < 1.0 per lo smorzamento positivo.")

    try:
        # Iterazione 0: valutazione iniziale dello sciame
        for i in range(num_particles):
            cost_i = objective.call_from_normalized(X[i])
            P_cost[i] = cost_i
            if cost_i < G_cost:
                G_cost = cost_i
                G_best = X[i].copy()

        # Ciclo evolutivo continuo
        for _ in range(1, max_iterations):
            for i in range(num_particles):
                phi1 = c1_cognitive * rng.uniform(0.0, 1.0, size=n_dim)
                phi2 = c2_social * rng.uniform(0.0, 1.0, size=n_dim)
                b_stiff = phi1 + phi2

                # Calcolo dell'attrattore baricentrico (phi1*p_i + phi2*g) / (phi1 + phi2)
                attractor = np.where(
                    b_stiff > 1.0e-14,
                    (phi1 * P_best[i] + phi2 * G_best) / np.maximum(b_stiff, 1.0e-14),
                    X[i],
                )

                x_new, v_new = _solve_cpso_analytical_step(
                    X[i], V[i], attractor, a_damp, b_stiff, dt_step
                )

                # Riflessione smorzata sui bordi del cubo unitario [0, 1]^4
                for d in range(n_dim):
                    if x_new[d] < 0.0:
                        x_new[d] = 0.0
                        v_new[d] *= -0.5
                    elif x_new[d] > 1.0:
                        x_new[d] = 1.0
                        v_new[d] *= -0.5

                X[i] = x_new
                V[i] = v_new

                cost_i = objective.call_from_normalized(X[i])
                if cost_i < P_cost[i]:
                    P_cost[i] = cost_i
                    P_best[i] = X[i].copy()
                if cost_i < G_cost:
                    G_cost = cost_i
                    G_best = X[i].copy()

    except BudgetExhaustedException:
        pass

    wall_sec = time.perf_counter() - t0
    best_theta = (
        objective.best_theta
        if objective.best_theta is not None
        else objective.transform.from_normalized(G_best)
    )
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
        algorithm_name="CPSO",
        seed=int(seed),
        theta_hat=best_theta,
        theta_tilde_hat=best_tilde,
        best_cost_J=float(objective.best_cost),
        num_fem_evaluations=int(objective.num_fem_evaluations),
        wall_time_sec=wall_sec,
        convergence_eval_trace=eval_trace,
        theta_history_trace=theta_trace,
    )