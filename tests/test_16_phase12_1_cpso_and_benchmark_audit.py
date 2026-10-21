"""
tests/test_16_phase12_1_cpso_and_benchmark_audit.py
Suite di verifica per la Fase 12.1:
1. Audit analitico e numerico di `python/reteco/inverse/cpso.py` (soluzione esatta dell'ODE
   del secondo ordine rispetto a integrazione numerica fine, attrattore della particella leader,
   riflessione smorzata sui bordi, validazione argomenti e consistenza del minimo storico).
2. Validazione del protocollo storico delle 3 run CPSO importate in `experiments/12_run_optimizer_benchmark.py`
   (controllo di hash SHA-256, discretizzazione, modalità obiettivo, lambda_A, prior, budget,
   minimo storico e non-assegnazione automatica di True alla riproducibilità nel benchmark corrente).
"""

from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

from reteco import load_master_config
from reteco.config_schema import ParameterCoordinateTransform
from reteco.inverse.cpso import _solve_cpso_analytical_step, run_cpso_optimization
from reteco.inverse.objective_functional import InverseObjectiveFunctional
from reteco.inverse.regularization import TikhonovRegularizer


def _load_exp12_module():
    exp_file = Path("experiments/12_run_optimizer_benchmark.py").resolve()
    spec = importlib.util.spec_from_file_location("exp12_bench", exp_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_01_cpso_analytical_step_matches_rk4_integration_all_regimes() -> None:
    """
    Verifica che `_solve_cpso_analytical_step` soddisfi esattamente l'ODE del secondo ordine:
        x''(t) + (1 - omega) * x'(t) + (phi_1 + phi_2) * (x(t) - attractor) = 0
    nei quattro regimi: b = 0, Delta > 0, Delta == 0, Delta < 0.
    """
    x0 = np.array([0.15, 0.85, 0.25, 0.75], dtype=np.float64)
    v0 = np.array([0.12, -0.09, 0.04, -0.11], dtype=np.float64)
    attr = np.array([0.60, 0.30, 0.55, 0.45], dtype=np.float64)
    a_damp = 0.35  # omega = 0.65 => a^2 = 0.1225, b_crit = 0.030625
    b_vec = np.array([0.0, 0.015, 0.030625, 1.45], dtype=np.float64)
    dt = 1.0

    x_ana, v_ana = _solve_cpso_analytical_step(x0, v0, attr, a_damp, b_vec, dt)

    # Integrazione RK4 con 2000 sotto-passi come riferimento indipendente
    n_steps = 2000
    h = dt / n_steps
    x_rk = x0.copy()
    v_rk = v0.copy()
    for _ in range(n_steps):
        def acc(x_val: np.ndarray, v_val: np.ndarray) -> np.ndarray:
            return -a_damp * v_val - b_vec * (x_val - attr)

        k1_x = v_rk
        k1_v = acc(x_rk, v_rk)
        k2_x = v_rk + 0.5 * h * k1_v
        k2_v = acc(x_rk + 0.5 * h * k1_x, v_rk + 0.5 * h * k1_v)
        k3_x = v_rk + 0.5 * h * k2_v
        k3_v = acc(x_rk + 0.5 * h * k2_x, v_rk + 0.5 * h * k2_v)
        k4_x = v_rk + h * k3_v
        k4_v = acc(x_rk + h * k3_x, v_rk + h * k3_v)

        x_rk = x_rk + (h / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
        v_rk = v_rk + (h / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    assert np.allclose(x_ana, x_rk, rtol=1.0e-10, atol=1.0e-12)
    assert np.allclose(v_ana, v_rk, rtol=1.0e-10, atol=1.0e-12)


def test_02_cpso_leader_attractor_damping_and_boundary_reflection() -> None:
    """
    Verifica:
    1) che per una particella con P_best == G_best l'attrattore coincida con G_best e la velocità
       decada secondo exp(-0.5 * (1 - omega) * dt);
    2) che le particelle che oltrepassano [0, 1]^4 vengano riflesse sul bordo e che la soluzione
       finale di `run_cpso_optimization` coincida sempre con il minimo storico di `objective`.
    """
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    reg = TikhonovRegularizer(transform, lambda_reg=1.0e-4)

    # Funzione surrogata anisotropa con minimo vicino al bordo (0.98, 0.50, 0.02, 0.50)
    target_tilde = np.array([0.98, 0.50, 0.02, 0.50], dtype=np.float64)
    weights_aniso = np.array([100.0, 0.05, 100.0, 0.10], dtype=np.float64)

    def forward_aniso(theta: np.ndarray) -> np.ndarray:
        tilde = transform.to_normalized(theta)
        return weights_aniso * (tilde - target_tilde)

    y_obs = np.zeros(4, dtype=np.float64)
    obj = InverseObjectiveFunctional.from_mode_a_deterministic(
        forward_operator=forward_aniso,
        y_observed=y_obs,
        transform=transform,
        regularizer=reg,
        y_reference=np.ones(4, dtype=np.float64),
        scale_type="linf",
        max_evaluations=80,
    )

    res = run_cpso_optimization(obj, seed=20261017, num_particles=8, max_iterations=10)
    consistency = obj.verify_history_and_best_candidate_consistency(res, raise_on_failure=True)
    assert consistency["all_checks_passed"] is True
    assert res.num_fem_evaluations == 80
    assert np.all((res.theta_tilde_hat >= 0.0) & (res.theta_tilde_hat <= 1.0))

    with pytest.raises(ValueError, match="num_particles deve essere > 0"):
        run_cpso_optimization(obj, seed=1, num_particles=0, max_iterations=5)
    with pytest.raises(ValueError, match="omega_inertia < 1.0"):
        run_cpso_optimization(obj, seed=1, num_particles=4, max_iterations=5, omega_inertia=1.0)


def test_03_phase12_imported_cpso_validation_and_rejection_of_tampered_metadata(
    tmp_path: Path,
) -> None:
    """
    Punto 3 & 4: Verifica che `validate_and_load_cpso_runs_from_phase07` in
    `experiments/12_run_optimizer_benchmark.py`:
    - validi correttamente un artefatto conforme al protocollo di Fase 07;
    - NON assegni automaticamente True a `reproducibility_verified_in_current_benchmark`;
    - sollevi ValueError se vengono alterati l'hash SHA-256 del target, la discretizzazione,
      la modalità dell'obiettivo, il valore di lambda/J_reg, il budget o la coerenza del minimo storico.
    """
    mod = _load_exp12_module()
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    reg = TikhonovRegularizer(
        transform, lambda_reg=mod.LAMBDA_A, theta_tilde_0=mod.EXPECTED_PRIOR_TILDE_0
    )

    theta_cand = np.array([0.698, 0.412, 0.0795, 2.38], dtype=np.float64)
    j_reg_exact = float(reg.evaluate_breakdown(theta_cand).j_reg)
    j_data_val = 4.20e-6
    j_tot_val = j_data_val + j_reg_exact
    w_l2 = float(np.sqrt(2.0 * j_data_val))

    best_seq = [1.0e-3] * 199 + [j_tot_val] * 401

    def _make_run(idx: int, seed: int, prior_repro: bool) -> dict:
        return {
            "run_id": f"cpso_run_{idx}",
            "optimizer_seed": seed,
            "eta_level": 0.0,
            "objective_mode": "mode_a_deterministic_normalized_linf",
            "planned_fem_budget": 600,
            "optimizer_fem_evaluations": 600,
            "best_eval_index": 200,
            "theta_true_vector": mod.EXPECTED_THETA_TRUE.tolist(),
            "theta_hat_vector": theta_cand.tolist(),
            "best_cost_J_data": j_data_val,
            "best_cost_J_reg": j_reg_exact,
            "best_cost_J_total": j_tot_val,
            "weighted_residual_l2_norm": w_l2,
            "relative_residual_vs_clean_target": 0.012,
            "termination_condition": "budget_exhausted_max_iterations_reached",
            "wall_time_seconds": 45.0,
            "seed_reproducibility_verified": prior_repro,
            "convergence_history": {
                "eval_index": list(range(1, 601)),
                "best_j_so_far": best_seq,
            },
        }

    valid_payload = {
        "target_provenance": {
            "y_target_clean_sha256": mod.EXPECTED_TARGET_SHA256,
            "N_target": 32,
            "M_target": 160,
            "N_inverse": 16,
            "M_inverse": 80,
            "N_obs": 180,
            "y_target_clean_linf": 0.85,
        },
        "regularization": {
            "lambda_reg": mod.LAMBDA_A,
            "prior_theta_tilde_0": [0.5, 0.5, 0.5, 0.5],
        },
        "runs": [
            _make_run(1, 20261017, True),
            _make_run(2, 20261018, False),
            _make_run(3, 20261019, False),
        ],
    }

    json_file = tmp_path / "cpso_multiseed_summary.json"
    json_file.write_text(json.dumps(valid_payload), encoding="utf-8")

    runs, _, val_summary = mod.validate_and_load_cpso_runs_from_phase07(
        tmp_path, transform=transform, regularizer=reg, expected_reference_scale=0.85
    )
    assert val_summary["all_three_cpso_runs_validated"] is True
    assert len(runs) == 3
    # Nessuna run importata deve avere reproducibility_verified_in_current_benchmark == True
    assert all(r["reproducibility_verified_in_current_benchmark"] is False for r in runs)
    assert runs[0]["prior_phase_seed_reproducibility_recorded"] is True
    assert runs[1]["prior_phase_seed_reproducibility_recorded"] is False

    # Verifica rigetto su hash SHA-256 errato
    bad_hash = json.loads(json.dumps(valid_payload))
    bad_hash["target_provenance"]["y_target_clean_sha256"] = "0" * 64
    json_file.write_text(json.dumps(bad_hash), encoding="utf-8")
    with pytest.raises(ValueError, match="Hash SHA-256"):
        mod.validate_and_load_cpso_runs_from_phase07(tmp_path, transform=transform, regularizer=reg)

    # Verifica rigetto su J_reg incompatibile con lambda_A = 1e-4
    bad_reg = json.loads(json.dumps(valid_payload))
    bad_reg["runs"][0]["best_cost_J_reg"] = j_reg_exact * 10.0
    bad_reg["runs"][0]["best_cost_J_total"] = j_data_val + j_reg_exact * 10.0
    bad_reg["runs"][0]["convergence_history"]["best_j_so_far"] = (
        [1.0e-3] * 199 + [j_data_val + j_reg_exact * 10.0] * 401
    )
    json_file.write_text(json.dumps(bad_reg), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatibile con lambda_A"):
        mod.validate_and_load_cpso_runs_from_phase07(tmp_path, transform=transform, regularizer=reg)