"""
experiments/05_pilot_inverse.py
RETECO — Fase 3.3, 3.4 & 3.6: Pilot inverso controllato end-to-end con
verifica effettiva dell'adimensionalizzazione e dell'isolamento del target pulito.
"""

from __future__ import annotations
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List
import numpy as np
import yaml

from reteco import _core, load_master_config
from reteco.config_schema import ParameterCoordinateTransform
from reteco.forward.wrapper import RetecoForwardProblem2D
from reteco.forward.target_generator import (
    FORCING_PROTOCOL_VERSION,
    build_standard_driving_Callbacks,
)
from reteco.observation.operator import DiscreteObservationOperator
from reteco.observation.noise_model import generate_gaussian_noisy_observations
from reteco.inverse.regularization import TikhonovRegularizer
from reteco.inverse.objective_functional import InverseObjectiveFunctional
from reteco.inverse.cpso import run_cpso_optimization
from reteco.inverse.diagnostics import (
    SingleRunDiagnosticRecord,
    extract_run_diagnostics_from_history,
    serialize_run_record,
    write_convergence_history_csv,
)

PILOT_NUM_PARTICLES: int = 10
PILOT_MAX_ITERATIONS: int = 12
PILOT_PLANNED_FEM_BUDGET: int = PILOT_NUM_PARTICLES * PILOT_MAX_ITERATIONS  # 120


def verify_canonical_pilot_configuration(
    twin_cfg: Dict[str, Any], master_cfg: Any
) -> Dict[str, Any]:
    """
    Controlla la configurazione canonica del protocollo senza assegnare staticamente True
    a `adimensionalized_observations` e `clean_target_isolation_verified`.
    """
    tar_cfg = twin_cfg["target_discretization"]
    inv_cfg = twin_cfg["inverse_discretization"]
    obs_cfg = twin_cfg["observation_protocol"]

    n_tar = int(tar_cfg["subdivisions_N"])
    m_tar = int(tar_cfg["time_steps_M"])
    n_inv = int(inv_cfg["subdivisions_N"])
    m_inv = int(inv_cfg["time_steps_M"])

    if (n_tar, m_tar) != (32, 160):
        raise ValueError(
            f"Discretizzazione target inattesa: (N={n_tar}, M={m_tar}), attesa (N=32, M=160)."
        )
    if (n_inv, m_inv) != (16, 80):
        raise ValueError(
            f"Discretizzazione inversa inattesa: (N={n_inv}, M={m_inv}), attesa (N=16, M=80)."
        )

    observed_fields = list(obs_cfg["observed_fields"])
    obs_times = [float(t) for t in obs_cfg["observation_times"]]
    nx = int(obs_cfg["sensor_grid_nx"])
    ny = int(obs_cfg["sensor_grid_ny"])
    margin = float(obs_cfg["sensor_margin"])
    n_obs_expected = nx * ny * len(obs_times) * len(observed_fields)

    # Ispezione concreta delle scale caratteristiche e del percorso dei dati:
    # `DiscreteObservationOperator.apply` campiona direttamente i gradi di libertà
    # di `c` senza dividere a runtime per `C_char`. Pertanto l'adimensionalizzazione
    # del campo grezzo `c` è un'ipotesi modellistica (scale unitarie nel file YAML)
    # e NON una trasformazione eseguita e verificata a runtime sul vettore osservato.
    scales_obj = getattr(master_cfg, "characteristic_scales", None)
    c_char_val = getattr(scales_obj, "C_char", None) if scales_obj is not None else None

    return {
        "target_discretization": {
            "N_target": n_tar,
            "M_target": m_tar,
            "T": float(tar_cfg["final_time_T"]),
            "grading_r": float(tar_cfg["grading_r"]),
        },
        "inverse_discretization": {
            "N_inv": n_inv,
            "M_inv": m_inv,
            "T": float(inv_cfg["final_time_T"]),
            "grading_r": float(inv_cfg["grading_r"]),
        },
        "observed_fields": observed_fields,
        "observation_times": obs_times,
        "sensor_grid": {
            "nx": nx,
            "ny": ny,
            "margin": margin,
            "num_spatial_sensors": nx * ny,
        },
        "expected_N_obs": n_obs_expected,
        "single_commensurable_field_observed": (observed_fields == ["c"]),
        "adimensionalized_observations": False,
        "adimensionalization_audit_status": (
            "modeling_assumption_not_runtime_enforced: l'operatore osservativo estrae "
            "direttamente il singolo campo scalare c senza divisione runtime per C_char "
            f"(C_char={c_char_val}); la commensurabilità è garantita dall'osservazione "
            "del solo campo c e dalla normalizzazione tramite la matrice peso W."
        ),
        "noise_covariance_model": "Sigma = sigma_nom^2 * I_{N_obs} with sigma_nom = eta * ||y_clean||_inf",
        "clean_target_isolation_verified": None,
    }


def _write_pilot_runs_csv(
    case_entries: List[Dict[str, Any]],
    csv_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "objective_mode",
        "eta_level",
        "lambda_reg",
        "data_seed",
        "optimizer_seed",
        "N_target",
        "M_target",
        "N_inv",
        "M_inv",
        "N_obs",
        "alpha_true",
        "beta_true",
        "D_true",
        "E_true",
        "alpha_hat",
        "beta_hat",
        "D_hat",
        "E_hat",
        "err_rel_alpha_pct",
        "err_rel_beta_pct",
        "err_rel_D_pct",
        "err_rel_E_pct",
        "best_cost_J_total",
        "best_cost_J_data",
        "best_cost_J_reg",
        "data_share_pct",
        "reg_share_pct",
        "best_eval_index",
        "weighted_residual_l2_norm",
        "relative_residual_vs_observed_data",
        "relative_residual_vs_clean_target",
        "nominal_noise_std",
        "effective_relative_noise_norm",
        "empirical_noise_std",
        "planned_fem_budget",
        "optimizer_fem_evaluations",
        "diagnostic_fem_evaluations",
        "termination_condition",
        "wall_time_seconds",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in case_entries:
            writer.writerow({k: entry[k] for k in fieldnames})


def main() -> None:
    out_dir = Path("artifacts/05_pilot_inverse")
    out_dir.mkdir(parents=True, exist_ok=True)

    master = load_master_config("configs/scales_and_physics.yaml")
    with open("configs/twin_experiment_synthetic.yaml", "r", encoding="utf-8") as f:
        twin_cfg = yaml.safe_load(f)

    protocol_verification = verify_canonical_pilot_configuration(twin_cfg, master)
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_dagger = master.parameter_space_theta.theta_dagger_array()

    obs_cfg = twin_cfg["observation_protocol"]
    obs_op = DiscreteObservationOperator.from_regular_grid(
        nx=int(obs_cfg["sensor_grid_nx"]),
        ny=int(obs_cfg["sensor_grid_ny"]),
        margin=float(obs_cfg["sensor_margin"]),
        observation_times=obs_cfg["observation_times"],
        observed_fields=obs_cfg["observed_fields"],
    )

    tar_cfg = twin_cfg["target_discretization"]
    grid_tar = _core.TimeGrid.make_graded(
        float(tar_cfg["final_time_T"]),
        int(tar_cfg["time_steps_M"]),
        float(tar_cfg["grading_r"]),
    )
    prob_tar = RetecoForwardProblem2D(
        subdivisions_N=int(tar_cfg["subdivisions_N"]),
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid_tar,
        fe_deg_disp=int(tar_cfg.get("fe_deg_disp", 2)),
        fe_deg_press=int(tar_cfg.get("fe_deg_press", 1)),
        fe_deg_tracer=int(tar_cfg.get("fe_deg_tracer", 1)),
    )
    c0_tar, F_tar, Sf_tar = build_standard_driving_Callbacks(
        prob_tar, twin_cfg["driving_protocol"]
    )
    t0_tar = time.perf_counter()
    traj_tar = prob_tar.solve_trajectory(
        theta_dagger, c0=c0_tar, F_mech_func=F_tar, S_fluid_func=Sf_tar
    )
    y_target_clean = np.asarray(obs_op.apply(prob_tar, traj_tar), dtype=np.float64).ravel()
    target_wall_time = time.perf_counter() - t0_tar

    n_obs = int(y_target_clean.size)
    if n_obs != protocol_verification["expected_N_obs"]:
        raise RuntimeError(
            f"Dimensione osservazioni {n_obs} diversa da expected_N_obs={protocol_verification['expected_N_obs']}."
        )

    inv_cfg = twin_cfg["inverse_discretization"]
    grid_inv = _core.TimeGrid.make_graded(
        float(inv_cfg["final_time_T"]),
        int(inv_cfg["time_steps_M"]),
        float(inv_cfg["grading_r"]),
    )
    prob_inv = RetecoForwardProblem2D(
        subdivisions_N=int(inv_cfg["subdivisions_N"]),
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid_inv,
        fe_deg_disp=int(inv_cfg.get("fe_deg_disp", 2)),
        fe_deg_press=int(inv_cfg.get("fe_deg_press", 1)),
        fe_deg_tracer=int(inv_cfg.get("fe_deg_tracer", 1)),
    )
    c0_inv, F_inv, Sf_inv = build_standard_driving_Callbacks(
        prob_inv, twin_cfg["driving_protocol"]
    )

    def G_inv(theta: np.ndarray) -> np.ndarray:
        tr = prob_inv.solve_trajectory(
            theta, c0=c0_inv, F_mech_func=F_inv, S_fluid_func=Sf_inv
        )
        return np.asarray(obs_op.apply(prob_inv, tr), dtype=np.float64).ravel()

    y_inv_at_true = G_inv(theta_dagger)
    discretization_rel_gap = float(
        np.linalg.norm(y_target_clean - y_inv_at_true) / np.linalg.norm(y_target_clean)
    )

    noise_cfg = twin_cfg["noise_and_regularization"]
    base_seed = int(noise_cfg["base_seed"])
    lambda_a_base = float(noise_cfg["regularization_lambda"])
    prior_theta_tilde_0 = np.asarray(noise_cfg["prior_theta_tilde_0"], dtype=np.float64).ravel()
    pilot_optimizer_seed = base_seed + 1  # 20261017

    eta_noisy = 0.05
    lambda_b_equiv = TikhonovRegularizer.compute_equivalent_mode_b_lambda(
        lambda_mode_a=lambda_a_base,
        eta_level=eta_noisy,
        n_obs=n_obs,
        covariance_scaling="exact_inverse_covariance",
    )

    pilot_cases_spec = [
        {
            "case_id": "case_1_noise_free_eta_0pct_mode_a",
            "eta_level": 0.0,
            "data_seed": base_seed,
            "optimizer_seed": pilot_optimizer_seed,
            "mode_type": "mode_a",
            "lambda_reg": lambda_a_base,
            "lambda_derivation": "Configurazione base lambda_A = 1.0e-4 da configs/twin_experiment_synthetic.yaml",
        },
        {
            "case_id": "case_2_noisy_eta_5pct_mode_b_exact_cov",
            "eta_level": eta_noisy,
            "data_seed": base_seed + int(round(eta_noisy * 1000)),
            "optimizer_seed": pilot_optimizer_seed,
            "mode_type": "mode_b_exact_cov",
            "lambda_reg": lambda_b_equiv,
            "lambda_derivation": (
                f"Valore iniziale di pilot algebricamente equivalente: "
                f"lambda_B = (N_obs / eta^2) * lambda_A = ({n_obs} / {eta_noisy}^2) * {lambda_a_base:.1e} = {lambda_b_equiv:.6g}"
            ),
        },
    ]

    diagnostic_records: List[SingleRunDiagnosticRecord] = []
    case_summary_entries: List[Dict[str, Any]] = []
    isolation_audit_report: Optional[Dict[str, Any]] = None

    for spec in pilot_cases_spec:
        cid = spec["case_id"]
        eta_val = float(spec["eta_level"])
        d_seed = int(spec["data_seed"])
        opt_seed = int(spec["optimizer_seed"])
        lam_val = float(spec["lambda_reg"])

        reg = TikhonovRegularizer(
            transform=transform,
            lambda_reg=lam_val,
            theta_tilde_0=prior_theta_tilde_0,
        )

        if spec["mode_type"] == "mode_a":
            sample = generate_gaussian_noisy_observations(
                y_target_clean,
                eta_level=0.0,
                seed=d_seed,
                weighting_mode="mode_a_deterministic_normalized",
                reference_scale_type="linf",
            )
            obj = InverseObjectiveFunctional.from_mode_a_deterministic(
                G_inv,
                sample.y_noisy,
                transform,
                reg,
                y_reference=sample.y_noisy,
                allow_observed_as_reference=True,
                scale_type="linf",
                max_evaluations=PILOT_PLANNED_FEM_BUDGET,
            )
        else:
            sample = generate_gaussian_noisy_observations(
                y_target_clean,
                eta_level=eta_val,
                seed=d_seed,
                weighting_mode="mode_b_noise_weighted",
                covariance_scaling="exact_inverse_covariance",
            )
            # Fase 3.6 - Punto 2: Verifica concreta dell'isolamento del target pulito
            isolation_audit_report = (
                InverseObjectiveFunctional.verify_clean_target_isolation_mode_b(
                    sample,
                    transform,
                    reg,
                    covariance_scaling="exact_inverse_covariance",
                )
            )
            protocol_verification["clean_target_isolation_verified"] = bool(
                isolation_audit_report["clean_target_isolation_verified"]
            )
            protocol_verification["clean_target_isolation_audit"] = isolation_audit_report
            if not isolation_audit_report["clean_target_isolation_verified"]:
                raise RuntimeError(
                    f"Verifica di isolamento del target pulito fallita: {isolation_audit_report}"
                )

            obj = InverseObjectiveFunctional.from_mode_b_noise_model(
                G_inv,
                sample,
                transform,
                reg,
                covariance_scaling="exact_inverse_covariance",
                max_evaluations=PILOT_PLANNED_FEM_BUDGET,
            )
            if not obj.verify_exact_inverse_covariance_identity():
                raise RuntimeError("Verifica dell'identità W^T W = Sigma^{-1} fallita nel Caso 2.")
            if obj.uses_clean_target_vector_in_evaluation:
                raise RuntimeError("Rilevato uso improprio del vettore pulito durante l'inversione.")

        t0_run = time.perf_counter()
        opt_res = run_cpso_optimization(
            obj,
            seed=opt_seed,
            num_particles=PILOT_NUM_PARTICLES,
            max_iterations=PILOT_MAX_ITERATIONS,
        )
        run_wall_time = time.perf_counter() - t0_run

        rec = extract_run_diagnostics_from_history(
            run_id=cid,
            obj=obj,
            opt_result=opt_res,
            theta_true=theta_dagger,
            y_target_clean=y_target_clean,
            y_noisy=sample.y_noisy,
            nominal_noise_std=sample.sigma_std,
            eta_level=eta_val,
            data_seed=d_seed,
            optimizer_seed=opt_seed,
            planned_fem_budget=PILOT_PLANNED_FEM_BUDGET,
            num_particles=PILOT_NUM_PARTICLES,
            max_iterations=PILOT_MAX_ITERATIONS,
            wall_time_seconds=run_wall_time,
            forward_operator_inv=G_inv,
            transform=transform,
            regularizer=reg,
        )
        diagnostic_records.append(rec)

        weighted_res_l2 = float(np.sqrt(max(0.0, 2.0 * rec.best_cost_J_data)))
        data_share_pct = (
            100.0 * rec.best_cost_J_data / rec.best_cost_J_total
            if rec.best_cost_J_total > 0.0
            else 0.0
        )
        reg_share_pct = (
            100.0 * rec.best_cost_J_reg / rec.best_cost_J_total
            if rec.best_cost_J_total > 0.0
            else 0.0
        )

        bd_at_true = obj.evaluate_from_prediction(
            theta_dagger, y_inv_at_true, record_in_history=False
        )

        errs = rec.individual_relative_errors_pct
        serialized = serialize_run_record(rec)
        serialized.update(
            {
                "case_id": cid,
                "objective_mode": obj.objective_mode,
                "lambda_reg": lam_val,
                "lambda_derivation": spec["lambda_derivation"],
                "N_target": int(tar_cfg["subdivisions_N"]),
                "M_target": int(tar_cfg["time_steps_M"]),
                "N_inv": int(inv_cfg["subdivisions_N"]),
                "M_inv": int(inv_cfg["time_steps_M"]),
                "N_obs": n_obs,
                "alpha_true": float(theta_dagger[0]),
                "beta_true": float(theta_dagger[1]),
                "D_true": float(theta_dagger[2]),
                "E_true": float(theta_dagger[3]),
                "alpha_hat": float(rec.theta_hat[0]),
                "beta_hat": float(rec.theta_hat[1]),
                "D_hat": float(rec.theta_hat[2]),
                "E_hat": float(rec.theta_hat[3]),
                "err_rel_alpha_pct": float(errs[0]),
                "err_rel_beta_pct": float(errs[1]),
                "err_rel_D_pct": float(errs[2]),
                "err_rel_E_pct": float(errs[3]),
                "data_share_pct": float(data_share_pct),
                "reg_share_pct": float(reg_share_pct),
                "weighted_residual_l2_norm": weighted_res_l2,
                "relative_residual_vs_observed_data": float(rec.relative_residual_vs_noisy_data),
                "empirical_noise_std": float(sample.empirical_noise_std),
                "is_covariance_invertible": bool(obj.is_covariance_invertible),
                "wt_w_equals_sigma_inv": bool(obj.verify_exact_inverse_covariance_identity()),
                "reference_evaluation_at_theta_dagger": {
                    "J_data_at_true": float(bd_at_true.j_data),
                    "J_reg_at_true": float(bd_at_true.j_reg),
                    "J_total_at_true": float(bd_at_true.j_total),
                    "weighted_residual_l2_at_true": float(bd_at_true.weighted_residual_l2_norm),
                    "relative_residual_vs_observed_at_true": float(bd_at_true.relative_residual_l2),
                },
            }
        )
        case_summary_entries.append(serialized)

    json_path = out_dir / "pilot_inverse_summary.json"
    runs_csv_path = out_dir / "pilot_inverse_runs_summary.csv"
    hist_csv_path = out_dir / "pilot_inverse_convergence_history.csv"

    summary_payload = {
        "experiment_id": "RETECO-v1-PHASE3-PILOT-INVERSE-05",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "forcing_protocol_version": FORCING_PROTOCOL_VERSION,
        "phase_3_2_protocol_verification": protocol_verification,
        "discretization_relative_gap_L2": discretization_rel_gap,
        "target_generation_wall_time_seconds": target_wall_time,
        "cases": case_summary_entries,
    }
    json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    _write_pilot_runs_csv(case_summary_entries, runs_csv_path)
    write_convergence_history_csv(diagnostic_records, hist_csv_path)


if __name__ == "__main__":
    main()