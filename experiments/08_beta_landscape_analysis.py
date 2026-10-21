"""
experiments/08_beta_landscape_analysis.py
RETECO — Fase 3.8: Analisi della funzione obiettivo rispetto a beta tramite sezioni
condizionali 1D passanti per le tre soluzioni CPSO della Fase 3.7 (seed 20261017,
20261018, 20261019), senza eseguire nuove ottimizzazioni CPSO.

Salva tutti gli artefatti in `artifacts/08_beta_landscape/`.
"""

from __future__ import annotations
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Sequence, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from reteco.visualization.twin_diagnostics_plots import (
    _Apply_paper_rc_style,
    save_figure_png_and_pdf,
)

EXPECTED_TARGET_SHA256: str = (
    "1bca9e29c60d77c1450fd2fcea845d2320c9e44945dd7d3ccda52aa33ef261af"
)
FIXED_DATA_SEED: int = 20261016
FIXED_LAMBDA_REG: float = 1.0e-4
FIXED_PRIOR_TILDE: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5)
BASE_UNIFORM_BETA_POINTS: int = 31

EXPECTED_PHASE37_SOLUTIONS: Dict[int, Dict[str, float]] = {
    20261017: {
        "alpha": 0.7033132338,
        "beta": 0.3880473413,
        "D": 0.0795239185,
        "E": 2.4787911063,
    },
    20261018: {
        "alpha": 0.6955680388,
        "beta": 0.4791427903,
        "D": 0.0789215374,
        "E": 2.4804893704,
    },
    20261019: {
        "alpha": 0.6837503054,
        "beta": 0.6032850194,
        "D": 0.0775466528,
        "E": 2.4870340345,
    },
}

REQUIRED_SPECIAL_BETAS_PROMPT: Tuple[float, float, float, float] = (
    0.3880473413,
    0.45,
    0.4791427903,
    0.6032850194,
)


def compute_sha256_vector(vec: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(vec, dtype=np.float64).ravel())
    return hashlib.sha256(arr.tobytes()).hexdigest()


def hash_directory_files(
    directory: Path, patterns: Sequence[str] = ("*.json", "*.csv", "*.npy")
) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    if not directory.exists():
        return hashes
    for pat in patterns:
        for fpath in sorted(directory.glob(pat)):
            hashes[str(fpath)] = hashlib.sha256(fpath.read_bytes()).hexdigest()
    return hashes


def verify_target_hash_or_abort(
    y_target_clean: np.ndarray,
    expected_sha256: str = EXPECTED_TARGET_SHA256,
    context_label: str = "phase38_target_check",
) -> str:
    actual_sha256 = compute_sha256_vector(y_target_clean)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"[ABORT - {context_label}] Hash SHA-256 del target non corrispondente!\n"
            f"  Atteso : {expected_sha256}\n"
            f"  Ottenuto: {actual_sha256}\n"
            "Esecuzione interrotta."
        )
    return actual_sha256


def load_verified_clean_target(
    master: Any,
    twin_cfg: Dict[str, Any],
    obs_op: DiscreteObservationOperator,
    theta_dagger: np.ndarray,
    expected_sha256: str = EXPECTED_TARGET_SHA256,
) -> Tuple[np.ndarray, str, str]:
    candidate_paths = [
        Path("artifacts/07_cpso_multiseed/y_target_clean.npy"),
        Path("artifacts/06_cpso_budget_diagnostic/y_target_clean.npy"),
        Path("artifacts/05_pilot_inverse/y_target_clean.npy"),
    ]
    for cpath in candidate_paths:
        if cpath.exists():
            y_loaded = np.asarray(np.load(cpath), dtype=np.float64).ravel()
            sha = verify_target_hash_or_abort(
                y_loaded,
                expected_sha256=expected_sha256,
                context_label=f"loaded_from_{cpath}",
            )
            return y_loaded, sha, f"reused_saved_vector:{cpath}"

    tar_cfg = twin_cfg["target_discretization"]
    n_tar = int(tar_cfg["subdivisions_N"])
    m_tar = int(tar_cfg["time_steps_M"])
    if (n_tar, m_tar) != (32, 160):
        raise RuntimeError(f"Discretizzazione target non conforme: {(n_tar, m_tar)} != (32, 160).")

    grid_tar = _core.TimeGrid.make_graded(
        float(tar_cfg["final_time_T"]),
        m_tar,
        float(tar_cfg["grading_r"]),
    )
    prob_tar = RetecoForwardProblem2D(
        subdivisions_N=n_tar,
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
    traj_tar = prob_tar.solve_trajectory(
        theta_dagger, c0=c0_tar, F_mech_func=F_tar, S_fluid_func=Sf_tar
    )
    y_target_clean = np.asarray(obs_op.apply(prob_tar, traj_tar), dtype=np.float64).ravel()
    sha = verify_target_hash_or_abort(
        y_target_clean,
        expected_sha256=expected_sha256,
        context_label="regenerated_N32_M160",
    )
    return y_target_clean, sha, "regenerated_and_hash_verified_N32_M160"


def _extract_theta_vector_from_run_dict(r: Dict[str, Any]) -> np.ndarray:
    if "theta_hat_vector" in r and r["theta_hat_vector"] is not None:
        return np.asarray(r["theta_hat_vector"], dtype=np.float64).ravel()
    if "theta_hat" in r:
        th = r["theta_hat"]
        if isinstance(th, dict):
            return np.array(
                [float(th["alpha"]), float(th["beta"]), float(th["D"]), float(th["E"])],
                dtype=np.float64,
            )
        return np.asarray(th, dtype=np.float64).ravel()
    return np.array(
        [float(r["alpha_hat"]), float(r["beta_hat"]), float(r["D_hat"]), float(r["E_hat"])],
        dtype=np.float64,
    )


def load_and_verify_phase37_solutions(
    phase37_summary_path: Path = Path("artifacts/07_cpso_multiseed/cpso_multiseed_summary.json"),
) -> List[Dict[str, Any]]:
    """
    Carica le tre soluzioni CPSO da `artifacts/07_cpso_multiseed/cpso_multiseed_summary.json`
    e verifica che corrispondano ai valori certificati di Fase 3.7.
    """
    if not phase37_summary_path.exists():
        raise FileNotFoundError(
            f"Artefatto Fase 3.7 non trovato: {phase37_summary_path}. "
            "Non è consentito rilanciare ottimizzazioni CPSO in Fase 3.8."
        )

    payload = json.loads(phase37_summary_path.read_text(encoding="utf-8"))
    prov = payload.get("target_provenance", {})
    target_sha = str(
        prov.get("target_sha256", prov.get("y_target_clean_sha256", EXPECTED_TARGET_SHA256))
    )
    if target_sha != EXPECTED_TARGET_SHA256:
        raise RuntimeError(
            f"Hash del target in {phase37_summary_path} ({target_sha}) non coincide con {EXPECTED_TARGET_SHA256}."
        )

    runs_list = payload.get("runs", payload.get("comparison_table_rows", []))
    if len(runs_list) != 3:
        raise RuntimeError(f"Attese 3 run in {phase37_summary_path}, trovate {len(runs_list)}.")

    verified_solutions: List[Dict[str, Any]] = []
    for r in runs_list:
        seed = int(r["optimizer_seed"])
        if seed not in EXPECTED_PHASE37_SOLUTIONS:
            raise RuntimeError(f"Seed CPSO inatteso in Fase 3.7: {seed}.")

        th_vec = _extract_theta_vector_from_run_dict(r)
        ref = EXPECTED_PHASE37_SOLUTIONS[seed]
        ref_vec = np.array([ref["alpha"], ref["beta"], ref["D"], ref["E"]], dtype=np.float64)
        if not np.allclose(th_vec, ref_vec, rtol=0.0, atol=1.0e-8):
            raise RuntimeError(
                f"Parametri del seed {seed} in {phase37_summary_path} ({th_vec}) "
                f"differiscono dai valori di riferimento ({ref_vec}) oltre 1e-8."
            )

        verified_solutions.append(
            {
                "optimizer_seed": seed,
                "run_id": str(r.get("run_id", f"cpso_600_seed_{seed}")),
                "theta_hat": th_vec.copy(),
                "alpha_hat": float(th_vec[0]),
                "beta_hat": float(th_vec[1]),
                "D_hat": float(th_vec[2]),
                "E_hat": float(th_vec[3]),
                "best_cost_J_total": float(r["best_cost_J_total"]),
                "best_cost_J_data": float(r["best_cost_J_data"]),
                "best_cost_J_reg": float(r["best_cost_J_reg"]),
                "best_eval_index": int(r["best_eval_index"]),
                "relative_residual_vs_clean_target": float(
                    r["relative_residual_vs_clean_target"]
                ),
            }
        )

    verified_solutions.sort(key=lambda x: x["optimizer_seed"])
    return verified_solutions


def build_beta_scan_grid(
    transform: ParameterCoordinateTransform,
    special_betas: Sequence[float],
    num_uniform_points: int = BASE_UNIFORM_BETA_POINTS,
    dedup_tol: float = 1.0e-11,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    b_min, b_max = float(transform.b_min), float(transform.b_max)
    uniform_grid = np.linspace(b_min, b_max, int(num_uniform_points), dtype=np.float64)

    probe_alpha = 0.5 * (transform.a_min + transform.a_max)
    probe_D = float(np.sqrt(transform.d_min * transform.d_max))
    probe_E = float(np.sqrt(transform.e_min * transform.e_max))

    admissible_specials: List[float] = []
    skipped_specials: List[float] = []
    for b_val in special_betas:
        b_f = float(b_val)
        probe_theta = np.array([probe_alpha, b_f, probe_D, probe_E], dtype=np.float64)
        if transform.is_within_bounds(probe_theta):
            admissible_specials.append(b_f)
        else:
            skipped_specials.append(b_f)

    combined = list(uniform_grid)
    for b_sp in admissible_specials:
        diffs = [abs(x - b_sp) for x in combined]
        min_idx = int(np.argmin(diffs))
        if diffs[min_idx] <= dedup_tol:
            combined[min_idx] = b_sp
        else:
            combined.append(b_sp)

    grid_sorted = np.array(sorted(combined), dtype=np.float64)

    inclusion_map: Dict[str, bool] = {}
    for b_sp in admissible_specials:
        inclusion_map[f"{b_sp:.10f}"] = bool(
            np.any(np.isclose(grid_sorted, b_sp, rtol=0.0, atol=1.0e-9))
        )

    grid_info = {
        "beta_min": b_min,
        "beta_max": b_max,
        "num_uniform_base_points": int(num_uniform_points),
        "total_grid_points": int(grid_sorted.size),
        "admissible_special_betas": [float(x) for x in admissible_specials],
        "non_admissible_special_betas": [float(x) for x in skipped_specials],
        "special_betas_inclusion_verified": inclusion_map,
        "all_admissible_specials_included": all(inclusion_map.values()),
    }
    return grid_sorted, grid_info


def evaluate_conditional_beta_sections(
    *,
    solutions: Sequence[Dict[str, Any]],
    beta_grid: np.ndarray,
    forward_operator_inv: Callable[[np.ndarray], np.ndarray],
    obj_functional: InverseObjectiveFunctional,
    transform: ParameterCoordinateTransform,
    beta_true_ref: float = 0.45,
) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]], Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    by_seed: Dict[int, List[Dict[str, Any]]] = {}

    fem_eval_counter = 0
    max_decomp_err = 0.0
    max_coord_phys_err = 0.0
    max_coord_tilde_err = 0.0
    all_admissible = True
    seed_optimum_reproduction_checks: Dict[int, Dict[str, Any]] = {}

    for sol in solutions:
        seed = int(sol["optimizer_seed"])
        alpha_fixed = float(sol["alpha_hat"])
        beta_seed_opt = float(sol["beta_hat"])
        D_fixed = float(sol["D_hat"])
        E_fixed = float(sol["E_hat"])

        seed_rows: List[Dict[str, Any]] = []
        for idx, beta_val in enumerate(beta_grid):
            beta_f = float(beta_val)
            theta_vec = np.array([alpha_fixed, beta_f, D_fixed, E_fixed], dtype=np.float64)

            is_adm = bool(transform.is_within_bounds(theta_vec, tol=1.0e-12))
            if not is_adm:
                all_admissible = False
                raise RuntimeError(
                    f"Punto non ammissibile rilevato per seed={seed}, beta={beta_f}: {theta_vec}"
                )

            theta_tilde = transform.to_normalized(theta_vec)
            theta_back = transform.from_normalized(theta_tilde)
            tilde_in_unit_box = bool(
                np.all(theta_tilde >= -1.0e-12) and np.all(theta_tilde <= 1.0 + 1.0e-12)
            )
            if not tilde_in_unit_box:
                all_admissible = False
                raise RuntimeError(
                    f"Coordinate normalizzate fuori da [0,1]^4 per seed={seed}, beta={beta_f}: {theta_tilde}"
                )

            phys_err = float(np.max(np.abs(theta_vec - theta_back)))
            max_coord_phys_err = max(max_coord_phys_err, phys_err)

            y_pred = forward_operator_inv(theta_vec)
            fem_eval_counter += 1

            bd = obj_functional.evaluate_from_prediction(
                theta_vec,
                y_pred,
                record_in_history=False,
                eval_index_override=fem_eval_counter,
            )

            tilde_err = float(np.max(np.abs(theta_tilde - bd.theta_tilde)))
            max_coord_tilde_err = max(max_coord_tilde_err, tilde_err)

            decomp_err = abs(float(bd.j_total) - (float(bd.j_data) + float(bd.j_reg)))
            max_decomp_err = max(max_decomp_err, decomp_err)
            if not np.isclose(
                float(bd.j_total), float(bd.j_data) + float(bd.j_reg), rtol=1.0e-12, atol=1.0e-14
            ):
                raise RuntimeError(
                    f"Violazione identità J_total = J_data + J_reg per seed={seed}, beta={beta_f}: "
                    f"J_total={bd.j_total}, J_data+J_reg={bd.j_data + bd.j_reg}"
                )

            is_seed_beta = bool(np.isclose(beta_f, beta_seed_opt, rtol=0.0, atol=1.0e-8))
            is_true_beta = bool(np.isclose(beta_f, beta_true_ref, rtol=0.0, atol=1.0e-12))

            if is_seed_beta:
                j_tot_match = bool(
                    np.isclose(float(bd.j_total), float(sol["best_cost_J_total"]), rtol=1.0e-5, atol=1.0e-9)
                )
                j_dat_match = bool(
                    np.isclose(float(bd.j_data), float(sol["best_cost_J_data"]), rtol=1.0e-5, atol=1.0e-9)
                )
                j_reg_match = bool(
                    np.isclose(float(bd.j_reg), float(sol["best_cost_J_reg"]), rtol=1.0e-5, atol=1.0e-9)
                )
                seed_optimum_reproduction_checks[seed] = {
                    "beta_evaluated": beta_f,
                    "beta_phase37": beta_seed_opt,
                    "j_total_scan": float(bd.j_total),
                    "j_total_phase37": float(sol["best_cost_J_total"]),
                    "j_data_scan": float(bd.j_data),
                    "j_data_phase37": float(sol["best_cost_J_data"]),
                    "j_reg_scan": float(bd.j_reg),
                    "j_reg_phase37": float(sol["best_cost_J_reg"]),
                    "matches_phase37_within_tolerance": bool(
                        j_tot_match and j_dat_match and j_reg_match
                    ),
                }

            row = {
                "optimizer_seed": seed,
                "source_run_id": sol["run_id"],
                "point_index": idx + 1,
                "is_seed_optimum_beta": is_seed_beta,
                "is_true_beta_ref": is_true_beta,
                "alpha": alpha_fixed,
                "beta": beta_f,
                "D": D_fixed,
                "E": E_fixed,
                "alpha_tilde": float(theta_tilde[0]),
                "beta_tilde": float(theta_tilde[1]),
                "D_tilde": float(theta_tilde[2]),
                "E_tilde": float(theta_tilde[3]),
                "is_admissible": is_adm,
                "j_data": float(bd.j_data),
                "j_reg": float(bd.j_reg),
                "j_total": float(bd.j_total),
                "data_share_pct": float(100.0 * bd.data_share_of_total),
                "reg_share_pct": float(100.0 * bd.reg_share_of_total),
                "relative_residual_l2": float(bd.relative_residual_l2),
                "weighted_residual_l2_norm": float(bd.weighted_residual_l2_norm),
                "unweighted_residual_l2_norm": float(bd.unweighted_residual_l2_norm),
                "unweighted_residual_linf_norm": float(bd.unweighted_residual_linf_norm),
            }
            seed_rows.append(row)
            all_rows.append(row)

        by_seed[seed] = seed_rows

    expected_evals = len(solutions) * int(beta_grid.size)
    checks_summary = {
        "j_total_equals_j_data_plus_j_reg_all_points": True,
        "max_abs_j_decomposition_error": max_decomp_err,
        "all_betas_physically_admissible": all_admissible,
        "physical_and_normalized_coordinates_consistent": bool(
            max_coord_phys_err <= 1.0e-12 and max_coord_tilde_err <= 1.0e-12
        ),
        "max_abs_coordinate_roundtrip_error": max_coord_phys_err,
        "max_abs_coordinate_tilde_error": max_coord_tilde_err,
        "diagnostic_scan_fem_evaluations": fem_eval_counter,
        "expected_scan_fem_evaluations": expected_evals,
        "fem_evaluation_count_exact": (fem_eval_counter == expected_evals),
        "optimizer_fem_evaluations_incremented": int(obj_functional.num_fem_evaluations),
        "optimizer_history_length_on_diagnostic_obj": len(obj_functional.history),
        "counted_as_optimizer_iterations": False,
        "seed_optimum_reproduction_checks": seed_optimum_reproduction_checks,
    }
    return all_rows, by_seed, checks_summary


def summarize_section_landscape(
    by_seed: Dict[int, List[Dict[str, Any]]],
    solutions: Sequence[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    sol_map = {int(s["optimizer_seed"]): s for s in solutions}
    stats_by_seed: Dict[int, Dict[str, Any]] = {}

    for seed, rows in by_seed.items():
        betas = np.array([r["beta"] for r in rows], dtype=np.float64)
        j_data = np.array([r["j_data"] for r in rows], dtype=np.float64)
        j_reg = np.array([r["j_reg"] for r in rows], dtype=np.float64)
        j_tot = np.array([r["j_total"] for r in rows], dtype=np.float64)
        rel_res = np.array([r["relative_residual_l2"] for r in rows], dtype=np.float64)

        idx_min_data = int(np.argmin(j_data))
        idx_max_data = int(np.argmax(j_data))
        idx_min_reg = int(np.argmin(j_reg))
        idx_max_reg = int(np.argmax(j_reg))
        idx_min_tot = int(np.argmin(j_tot))
        idx_max_tot = int(np.argmax(j_tot))

        mask_data_10pct_min = j_data <= 1.10 * j_data[idx_min_data]
        mask_tot_10pct_min = j_tot <= 1.10 * j_tot[idx_min_tot]

        flat_betas_data = betas[mask_data_10pct_min]
        flat_betas_tot = betas[mask_tot_10pct_min]

        idx_seed_beta = int(np.argmin(np.abs(betas - float(sol_map[seed]["beta_hat"]))))
        idx_true_beta = int(np.argmin(np.abs(betas - 0.45)))

        stats_by_seed[seed] = {
            "optimizer_seed": seed,
            "fixed_parameters": {
                "alpha": float(sol_map[seed]["alpha_hat"]),
                "D": float(sol_map[seed]["D_hat"]),
                "E": float(sol_map[seed]["E_hat"]),
                "cpso_beta_hat": float(sol_map[seed]["beta_hat"]),
            },
            "J_data": {
                "min": float(j_data[idx_min_data]),
                "beta_at_min": float(betas[idx_min_data]),
                "max": float(j_data[idx_max_data]),
                "beta_at_max": float(betas[idx_max_data]),
                "excursion_max_minus_min": float(j_data[idx_max_data] - j_data[idx_min_data]),
                "ratio_max_over_min": float(j_data[idx_max_data] / max(j_data[idx_min_data], 1.0e-30)),
                "at_cpso_beta_hat": float(j_data[idx_seed_beta]),
                "at_beta_045": float(j_data[idx_true_beta]),
                "diff_cpso_minus_min": float(j_data[idx_seed_beta] - j_data[idx_min_data]),
                "flat_zone_within_10pct_of_min_beta_interval": [
                    float(np.min(flat_betas_data)),
                    float(np.max(flat_betas_data)),
                ],
                "flat_zone_width_beta": float(np.max(flat_betas_data) - np.min(flat_betas_data)),
            },
            "J_reg": {
                "min": float(j_reg[idx_min_reg]),
                "beta_at_min": float(betas[idx_min_reg]),
                "max": float(j_reg[idx_max_reg]),
                "beta_at_max": float(betas[idx_max_reg]),
                "excursion_max_minus_min": float(j_reg[idx_max_reg] - j_reg[idx_min_reg]),
                "ratio_max_over_min": float(j_reg[idx_max_reg] / max(j_reg[idx_min_reg], 1.0e-30)),
                "at_cpso_beta_hat": float(j_reg[idx_seed_beta]),
                "at_beta_045": float(j_reg[idx_true_beta]),
            },
            "J_total": {
                "min": float(j_tot[idx_min_tot]),
                "beta_at_min": float(betas[idx_min_tot]),
                "max": float(j_tot[idx_max_tot]),
                "beta_at_max": float(betas[idx_max_tot]),
                "excursion_max_minus_min": float(j_tot[idx_max_tot] - j_tot[idx_min_tot]),
                "ratio_max_over_min": float(j_tot[idx_max_tot] / max(j_tot[idx_min_tot], 1.0e-30)),
                "at_cpso_beta_hat": float(j_tot[idx_seed_beta]),
                "at_beta_045": float(j_tot[idx_true_beta]),
                "diff_cpso_minus_min": float(j_tot[idx_seed_beta] - j_tot[idx_min_tot]),
                "flat_zone_within_10pct_of_min_beta_interval": [
                    float(np.min(flat_betas_tot)),
                    float(np.max(flat_betas_tot)),
                ],
                "flat_zone_width_beta": float(np.max(flat_betas_tot) - np.min(flat_betas_tot)),
            },
            "relative_residual_l2": {
                "min": float(np.min(rel_res)),
                "max": float(np.max(rel_res)),
                "at_cpso_beta_hat": float(rel_res[idx_seed_beta]),
                "at_beta_045": float(rel_res[idx_true_beta]),
            },
        }

    return stats_by_seed


def plot_three_panel_beta_landscape(
    by_seed: Dict[int, List[Dict[str, Any]]],
    solutions: Sequence[Dict[str, Any]],
    bd_at_theta_dagger: Any,
    out_path: str | Path,
    beta_true_ref: float = 0.45,
) -> Tuple[Path, Path]:
    colors = {
        20261017: "#1f77b4",
        20261018: "#d62728",
        20261019: "#2ca02c",
    }
    linestyles = {
        20261017: "-",
        20261018: "--",
        20261019: "-.",
    }
    sol_map = {int(s["optimizer_seed"]): s for s in solutions}

    with plt.rc_context(_Apply_paper_rc_style()):
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
        ax_data, ax_reg, ax_tot = axes

        panel_specs = [
            (
                ax_data,
                "j_data",
                r"(a) Data Fidelity $J_{\mathrm{data}}(\beta \mid \widehat{\alpha}_s, \widehat{D}_s, \widehat{E}_s)$",
                r"$J_{\mathrm{data}}(\boldsymbol{\theta})$",
                float(bd_at_theta_dagger.j_data),
            ),
            (
                ax_reg,
                "j_reg",
                r"(b) Tikhonov Penalty $J_{\mathrm{reg}}(\beta \mid \widehat{\alpha}_s, \widehat{D}_s, \widehat{E}_s)$",
                r"$J_{\mathrm{reg}}(\boldsymbol{\theta})$ ($\lambda = 10^{-4}$)",
                float(bd_at_theta_dagger.j_reg),
            ),
            (
                ax_tot,
                "j_total",
                r"(c) Total Functional $J_{\mathrm{total}} = J_{\mathrm{data}} + J_{\mathrm{reg}}$",
                r"$J_{\mathrm{total}}(\boldsymbol{\theta})$",
                float(bd_at_theta_dagger.j_total),
            ),
        ]

        legend_handles = []
        legend_labels = []

        for p_idx, (ax, key, title, ylabel, ref_val_true) in enumerate(panel_specs):
            for seed in sorted(by_seed.keys()):
                rows = by_seed[seed]
                betas = np.array([r["beta"] for r in rows], dtype=np.float64)
                vals = np.array([r[key] for r in rows], dtype=np.float64)
                c = colors[seed]
                ls = linestyles[seed]
                beta_hat_s = float(sol_map[seed]["beta_hat"])

                line = ax.plot(
                    betas,
                    vals,
                    color=c,
                    linestyle=ls,
                    linewidth=1.7,
                    label=f"Seed {seed} ($\\widehat{{\\beta}}={beta_hat_s:.3f}$)",
                    zorder=3,
                )[0]

                idx_opt = int(np.argmin(np.abs(betas - beta_hat_s)))
                pt_opt = ax.plot(
                    betas[idx_opt],
                    vals[idx_opt],
                    marker="o",
                    markersize=6.2,
                    color=c,
                    markeredgecolor="black",
                    markeredgewidth=0.7,
                    linestyle="none",
                    zorder=5,
                )[0]

                idx_min = int(np.argmin(vals))
                pt_min = ax.plot(
                    betas[idx_min],
                    vals[idx_min],
                    marker="D",
                    markersize=5.2,
                    markerfacecolor="white",
                    markeredgecolor=c,
                    markeredgewidth=1.2,
                    linestyle="none",
                    zorder=4,
                )[0]

                if p_idx == 0:
                    legend_handles.append(line)
                    legend_labels.append(
                        f"Seed {seed} fixed $(\\widehat{{\\alpha}}, \\widehat{{D}}, \\widehat{{E}})$, "
                        f"CPSO $\\widehat{{\\beta}}={beta_hat_s:.3f}$"
                    )

            vline = ax.axvline(
                beta_true_ref,
                color="#444444",
                linestyle=":",
                linewidth=1.35,
                zorder=2,
            )
            pt_true = ax.plot(
                beta_true_ref,
                ref_val_true,
                marker="*",
                markersize=8.5,
                color="#111111",
                markeredgecolor="white",
                markeredgewidth=0.4,
                linestyle="none",
                zorder=6,
            )[0]

            if p_idx == 0:
                legend_handles.extend([pt_opt, pt_min, vline, pt_true])
                legend_labels.extend(
                    [
                        r"CPSO returned $\widehat{\beta}_s$ (filled circle)",
                        r"1D section minimum $\arg\min_\beta J$ (open diamond)",
                        r"Diagnostic reference $\beta^\dagger = 0.45$",
                        r"Value at $\boldsymbol{\theta}^\dagger = (0.70, 0.45, 0.08, 2.50)$",
                    ]
                )

            ax.set_title(title)
            ax.set_xlabel(r"Viscoelastic Fractional Order $\beta$")
            ax.set_ylabel(ylabel)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            ax.grid(True)

        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=3,
            frameon=True,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
        paths = save_figure_png_and_pdf(fig, out_path)
        plt.close(fig)
        return paths


def write_scans_csv(rows: Sequence[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "optimizer_seed",
        "source_run_id",
        "point_index",
        "is_seed_optimum_beta",
        "is_true_beta_ref",
        "alpha",
        "beta",
        "D",
        "E",
        "alpha_tilde",
        "beta_tilde",
        "D_tilde",
        "E_tilde",
        "is_admissible",
        "j_data",
        "j_reg",
        "j_total",
        "data_share_pct",
        "reg_share_pct",
        "relative_residual_l2",
        "weighted_residual_l2_norm",
        "unweighted_residual_l2_norm",
        "unweighted_residual_linf_norm",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_landscape_markdown_report(
    *,
    grid_info: Dict[str, Any],
    stats_by_seed: Dict[int, Dict[str, Any]],
    checks_summary: Dict[str, Any],
    bd_at_true: Any,
    target_sha256: str,
    target_source: str,
    profile_module_available: bool,
) -> str:
    lines: List[str] = [
        "# RETECO — Fase 3.8: Analisi delle Sezioni Condizionali della Funzione Obiettivo rispetto a $\\beta$",
        "",
        "## 1. Protocollo Congelato e Controlli di Integrità",
        f"- **Target sintetico**: `N = 32, M = 160`, SHA-256 verificato: `{target_sha256}` (`{target_source}`)",
        f"- **Discretizzazione inversa**: `N = 16, M = 80`",
        f"- **Funzionale obiettivo**: `mode_a_deterministic_normalized_linf` ($\\eta = 0.0$, $\\lambda = 10^{{-4}}$, prior $\\widetilde{{\\boldsymbol{{\\theta}}}}_0 = (0.5, 0.5, 0.5, 0.5)$)",
        f"- **Intervallo fisico ammissibile di $\\beta$**: $[{grid_info['beta_min']:.2f},\\ {grid_info['beta_max']:.2f}]$",
        f"- **Griglia di scansione in $\\beta$**: `{grid_info['total_grid_points']}` punti (`{grid_info['num_uniform_base_points']}` uniformi + punti speciali richiesti `{grid_info['admissible_special_betas']}`)",
        f"- **Valutazioni FEM diagnostiche eseguite per le 3 scansioni**: `{checks_summary['diagnostic_scan_fem_evaluations']}` (`0` iterazioni/valutazioni aggiunte agli ottimizzatori CPSO)",
        f"- **Identità $J_{{\\mathrm{{total}}}} = J_{{\\mathrm{{data}}}} + J_{{\\mathrm{{reg}}}}$ verificata su tutti i punti**: `{checks_summary['j_total_equals_j_data_plus_j_reg_all_points']}` (errore max `{checks_summary['max_abs_j_decomposition_error']:.3e}`)",
        f"- **Coerenza coordinate fisiche/normalizzate (`ParameterCoordinateTransform`)**: `{checks_summary['physical_and_normalized_coordinates_consistent']}` (errore max round-trip `{checks_summary['max_abs_coordinate_roundtrip_error']:.3e}`)",
        "",
        "## 2. Escursione Quantitativa di $J_{\\mathrm{data}}$, $J_{\\mathrm{reg}}$ e $J_{\\mathrm{total}}$ lungo $\\beta$",
        "| Seed CPSO | $(\\widehat{\\alpha}_s, \\widehat{D}_s, \\widehat{E}_s)$ fissati | $\\widehat{\\beta}_s$ CPSO | Componente | Min su $\\beta$ | $\\beta_{\\arg\\min}$ | Max su $\\beta$ | Escursione $\\Delta J = \\max - \\min$ | Rapporto $\\max/\\min$ | Valore in $\\widehat{\\beta}_s$ | Valore in $\\beta^\\dagger = 0.45$ | Zona entro $+10\\%$ dal min |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for seed in sorted(stats_by_seed.keys()):
        st = stats_by_seed[seed]
        fp = st["fixed_parameters"]
        fixed_str = f"`({fp['alpha']:.4f}, {fp['D']:.5f}, {fp['E']:.4f})`"
        b_hat = fp["cpso_beta_hat"]

        jd = st["J_data"]
        jr = st["J_reg"]
        jt = st["J_total"]

        lines.append(
            f"| `{seed}` | {fixed_str} | `{b_hat:.4f}` | **$J_{{\\mathrm{{data}}}}$** | "
            f"`{jd['min']:.4e}` | `{jd['beta_at_min']:.4f}` | `{jd['max']:.4e}` | "
            f"`{jd['excursion_max_minus_min']:.4e}` | `{jd['ratio_max_over_min']:.2f}x` | "
            f"`{jd['at_cpso_beta_hat']:.4e}` | `{jd['at_beta_045']:.4e}` | "
            f"`[{jd['flat_zone_within_10pct_of_min_beta_interval'][0]:.3f}, {jd['flat_zone_within_10pct_of_min_beta_interval'][1]:.3f}]` |"
        )
        lines.append(
            f"| `{seed}` | {fixed_str} | `{b_hat:.4f}` | **$J_{{\\mathrm{{reg}}}}$** | "
            f"`{jr['min']:.4e}` | `{jr['beta_at_min']:.4f}` | `{jr['max']:.4e}` | "
            f"`{jr['excursion_max_minus_min']:.4e}` | `{jr['ratio_max_over_min']:.2f}x` | "
            f"`{jr['at_cpso_beta_hat']:.4e}` | `{jr['at_beta_045']:.4e}` | "
            f"— |"
        )
        lines.append(
            f"| `{seed}` | {fixed_str} | `{b_hat:.4f}` | **$J_{{\\mathrm{{total}}}}$** | "
            f"`{jt['min']:.4e}` | `{jt['beta_at_min']:.4f}` | `{jt['max']:.4e}` | "
            f"`{jt['excursion_max_minus_min']:.4e}` | `{jt['ratio_max_over_min']:.2f}x` | "
            f"`{jt['at_cpso_beta_hat']:.4e}` | `{jt['at_beta_045']:.4e}` | "
            f"`[{jt['flat_zone_within_10pct_of_min_beta_interval'][0]:.3f}, {jt['flat_zone_within_10pct_of_min_beta_interval'][1]:.3f}]` |"
        )

    lines.extend(
        [
            "",
            f"- **Riferimento diagnostico in $\\boldsymbol{{\\theta}}^\\dagger = (0.70, 0.45, 0.08, 2.50)$**: "
            f"$J_{{\\mathrm{{data}}}}(\\boldsymbol{{\\theta}}^\\dagger) = {float(bd_at_true.j_data):.6e}$, "
            f"$J_{{\\mathrm{{reg}}}}(\\boldsymbol{{\\theta}}^\\dagger) = {float(bd_at_true.j_reg):.6e}$, "
            f"$J_{{\\mathrm{{total}}}}(\\boldsymbol{{\\theta}}^\\dagger) = {float(bd_at_true.j_total):.6e}$.",
            "",
            "## 3. Interpretazione Fisica e Metodologica",
            "1. **Variazione del costo dati $J_{\\mathrm{data}}(\\beta \\mid \\widehat{\\alpha}_s, \\widehat{D}_s, \\widehat{E}_s)$**:",
            "   - Il pannello (a) e la tabella mostrano quanto varia il disaccordo sulle osservazioni di concentrazione $c$ quando $\\beta$ spazza l'intero intervallo fisico ammissibile tenendo bloccati $(\\widehat{\\alpha}_s, \\widehat{D}_s, \\widehat{E}_s)$.",
            "   - Inoltre, lo spostamento del punto di minimo $\\beta_{\\arg\\min J_{\\mathrm{data}}}$ da una curva all'altra quantifica l'accoppiamento di compensazione tra $(\\alpha, D, E)$ e $\\beta$: al decrescere di $(\\widehat{\\alpha}_s, \\widehat{D}_s)$ dal seed `20261017` al seed `20261019`, il valore di $\\beta$ che meglio accorda i dati si sposta verso valori più alti.",
            "2. **Variazione della regolarizzazione $J_{\\mathrm{reg}}(\\beta \\mid \\widehat{\\alpha}_s, \\widehat{D}_s, \\widehat{E}_s)$**:",
            "   - Il pannello (b) mostra la parabola convessa $\\frac{\\lambda}{2}(\\widetilde{\\beta}-0.5)^2$ centrata in $\\widetilde{\\beta}=0.5$ ($\\beta=0.45$), con escursione analitica $\\Delta J_{\\mathrm{reg}} = 1.25 \\times 10^{-5}$ sull'intero intervallo ammissibile.",
            "3. **Variazione del costo totale $J_{\\mathrm{total}} = J_{\\mathrm{data}} + J_{\\mathrm{reg}}$ e passo successivo raccomandato**:",
            f"   - Infrastruttura di profilo presente nel repository (`python/reteco/identifiability/profile_likelihood.py`): `{profile_module_available}`.",
            "   - Poiché le sezioni condizionali a $(\\alpha, D, E)$ fissati non escludono la compensazione mutua tra i parametri lungo una valle obliqua 4D, non si dichiara $\\beta$ non identificabile sulla sola base di queste sezioni 1D: il passo diagnostico successivo appropriato — qualora $J_{\\mathrm{data}}$ risulti debolmente sensibile o traslato per compensazione — consiste nel calcolare il **profilo ottimizzato (profile likelihood)** $\\beta \\mapsto \\min_{\\alpha, D, E} J(\\alpha, \\beta, D, E)$ riutilizzando `python/reteco/identifiability/profile_likelihood.py`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    out_dir = Path("artifacts/08_beta_landscape")
    out_dir.mkdir(parents=True, exist_ok=True)
    phase37_dir = Path("artifacts/07_cpso_multiseed")

    pre_phase37_hashes = hash_directory_files(phase37_dir)

    master = load_master_config("configs/scales_and_physics.yaml")
    with open("configs/twin_experiment_synthetic.yaml", "r", encoding="utf-8") as f:
        twin_cfg = yaml.safe_load(f)

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

    y_target_clean, verified_sha256, target_source = load_verified_clean_target(
        master=master,
        twin_cfg=twin_cfg,
        obs_op=obs_op,
        theta_dagger=theta_dagger,
        expected_sha256=EXPECTED_TARGET_SHA256,
    )
    print(f"[Fase 3.8] Target verificato: SHA-256={verified_sha256} ({target_source})")

    solutions = load_and_verify_phase37_solutions(
        phase37_dir / "cpso_multiseed_summary.json"
    )

    special_betas_to_include = [
        float(solutions[0]["beta_hat"]),
        0.45,
        float(solutions[1]["beta_hat"]),
        float(solutions[2]["beta_hat"]),
    ]
    beta_grid, grid_info = build_beta_scan_grid(
        transform=transform,
        special_betas=special_betas_to_include,
        num_uniform_points=BASE_UNIFORM_BETA_POINTS,
    )
    if not grid_info["all_admissible_specials_included"]:
        raise RuntimeError(
            f"Non tutti i valori speciali di beta ammissibili sono stati inclusi nella griglia: {grid_info}"
        )

    inv_cfg = twin_cfg["inverse_discretization"]
    n_inv = int(inv_cfg["subdivisions_N"])
    m_inv = int(inv_cfg["time_steps_M"])
    if (n_inv, m_inv) != (16, 80):
        raise RuntimeError(f"Discretizzazione inversa non conforme: {(n_inv, m_inv)} != (16, 80).")

    grid_inv = _core.TimeGrid.make_graded(
        float(inv_cfg["final_time_T"]),
        m_inv,
        float(inv_cfg["grading_r"]),
    )
    prob_inv = RetecoForwardProblem2D(
        subdivisions_N=n_inv,
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

    noise_cfg = twin_cfg["noise_and_regularization"]
    data_seed = int(noise_cfg["base_seed"])
    lambda_reg = float(noise_cfg["regularization_lambda"])
    prior_tilde_0 = np.asarray(noise_cfg["prior_theta_tilde_0"], dtype=np.float64).ravel()
    if (
        data_seed != FIXED_DATA_SEED
        or not np.isclose(lambda_reg, FIXED_LAMBDA_REG)
        or not np.allclose(prior_tilde_0, FIXED_PRIOR_TILDE)
    ):
        raise RuntimeError("Protocollo di regolarizzazione o seed dati alterato rispetto a Fase 3.7.")

    reg = TikhonovRegularizer(
        transform=transform,
        lambda_reg=lambda_reg,
        theta_tilde_0=prior_tilde_0,
    )
    sample_0 = generate_gaussian_noisy_observations(
        y_target_clean,
        eta_level=0.0,
        seed=data_seed,
        weighting_mode="mode_a_deterministic_normalized",
        reference_scale_type="linf",
    )
    obj_diag = InverseObjectiveFunctional.from_mode_a_deterministic(
        G_inv,
        sample_0.y_noisy,
        transform,
        reg,
        y_reference=sample_0.y_noisy,
        allow_observed_as_reference=True,
        scale_type="linf",
        max_evaluations=5000,
    )

    t0_scan = time.perf_counter()
    all_rows, by_seed, checks_summary = evaluate_conditional_beta_sections(
        solutions=solutions,
        beta_grid=beta_grid,
        forward_operator_inv=G_inv,
        obj_functional=obj_diag,
        transform=transform,
        beta_true_ref=0.45,
    )
    scan_wall_s = time.perf_counter() - t0_scan

    y_inv_true_1 = G_inv(theta_dagger)
    bd_at_true_1 = obj_diag.evaluate_from_prediction(
        theta_dagger, y_inv_true_1, record_in_history=False
    )
    y_inv_true_2 = G_inv(theta_dagger)
    bd_at_true_2 = obj_diag.evaluate_from_prediction(
        theta_dagger, y_inv_true_2, record_in_history=False
    )
    reproducibility_verified = bool(
        bd_at_true_1.j_total == bd_at_true_2.j_total
        and bd_at_true_1.j_data == bd_at_true_2.j_data
        and bd_at_true_1.j_reg == bd_at_true_2.j_reg
        and all(
            v["matches_phase37_within_tolerance"]
            for v in checks_summary["seed_optimum_reproduction_checks"].values()
        )
    )
    if not reproducibility_verified:
        raise RuntimeError(
            f"Controllo di riproducibilità delle valutazioni FEM fallito: "
            f"{checks_summary['seed_optimum_reproduction_checks']}"
        )

    post_phase37_hashes = hash_directory_files(phase37_dir)
    phase37_untouched = (pre_phase37_hashes == post_phase37_hashes)
    if not phase37_untouched:
        raise RuntimeError("ERRORE: I file di artifacts/07_cpso_multiseed/ sono stati modificati!")

    checks_summary["reproducibility_verified"] = reproducibility_verified
    checks_summary["original_cpso_phase37_artifacts_untouched"] = phase37_untouched
    checks_summary["reference_theta_dagger_fem_evaluations"] = 2
    checks_summary["scan_wall_time_seconds"] = scan_wall_s

    try:
        import reteco.identifiability.profile_likelihood as _pl_mod  # noqa: F401
        profile_module_available = True
    except Exception:
        profile_module_available = False

    stats_by_seed = summarize_section_landscape(by_seed, solutions)

    csv_path = out_dir / "beta_conditional_scans.csv"
    fig_base_path = out_dir / "fig_beta_conditional_landscape.png"
    report_md_path = out_dir / "beta_landscape_report.md"
    summary_json_path = out_dir / "beta_landscape_summary.json"

    write_scans_csv(all_rows, csv_path)
    png_path, pdf_path = plot_three_panel_beta_landscape(
        by_seed=by_seed,
        solutions=solutions,
        bd_at_theta_dagger=bd_at_true_1,
        out_path=fig_base_path,
        beta_true_ref=0.45,
    )

    report_md = build_landscape_markdown_report(
        grid_info=grid_info,
        stats_by_seed=stats_by_seed,
        checks_summary=checks_summary,
        bd_at_true=bd_at_true_1,
        target_sha256=verified_sha256,
        target_source=target_source,
        profile_module_available=profile_module_available,
    )
    report_md_path.write_text(report_md, encoding="utf-8")

    summary_payload = {
        "experiment_id": "RETECO-v1-PHASE3.8-BETA-CONDITIONAL-LANDSCAPE",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "forcing_protocol_version": FORCING_PROTOCOL_VERSION,
        "frozen_protocol": {
            "N_target": 32,
            "M_target": 160,
            "N_inv": 16,
            "M_inv": 80,
            "objective_mode": obj_diag.objective_mode,
            "lambda_reg": FIXED_LAMBDA_REG,
            "prior_theta_tilde_0": list(FIXED_PRIOR_TILDE),
            "data_seed": FIXED_DATA_SEED,
            "target_sha256": verified_sha256,
            "expected_target_sha256": EXPECTED_TARGET_SHA256,
            "target_hash_verified": True,
        },
        "beta_grid_specification": grid_info,
        "consistency_and_integrity_checks": checks_summary,
        "reference_evaluation_at_theta_dagger": {
            "theta_dagger": theta_dagger.tolist(),
            "theta_tilde_dagger": transform.to_normalized(theta_dagger).tolist(),
            "J_data_at_true": float(bd_at_true_1.j_data),
            "J_reg_at_true": float(bd_at_true_1.j_reg),
            "J_total_at_true": float(bd_at_true_1.j_total),
            "relative_residual_l2_at_true": float(bd_at_true_1.relative_residual_l2),
        },
        "section_statistics_by_seed": {str(k): v for k, v in stats_by_seed.items()},
        "profile_likelihood_infrastructure_available": profile_module_available,
        "artifacts_generated": {
            "scans_csv": str(csv_path),
            "summary_json": str(summary_json_path),
            "figure_png": str(png_path),
            "figure_pdf": str(pdf_path),
            "report_md": str(report_md_path),
        },
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("\n" + report_md)
    print(f"[Fase 3.8] Artefatti salvati in: {out_dir}")


if __name__ == "__main__":
    main()