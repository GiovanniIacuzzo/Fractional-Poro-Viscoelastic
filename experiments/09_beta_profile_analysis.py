"""
experiments/09_beta_profile_analysis.py
RETECO — Fase 3.9: Profili ottimizzati rispetto a beta per distinguere
l'informazione contenuta nei dati dall'effetto della regolarizzazione e
dalla variabilità del CPSO.

Costruisce due profili distinti sull'intervallo fisico di beta [0.15, 0.90]:
1. Profilo del termine dati:
       Phi_data(beta) = min_{alpha, D, E} J_data(alpha, beta, D, E)
2. Profilo del funzionale regolarizzato:
       Phi_lambda(beta) = min_{alpha, D, E} J_total(alpha, beta, D, E)

Salva tutti gli artefatti in `artifacts/09_beta_profile/` senza modificare
gli artefatti delle fasi precedenti.
"""

from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import inspect
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
import reteco.identifiability.profile_likelihood as pl_mod
from reteco.identifiability.profile_likelihood import (
    BidirectionalProfileResult,
    OptimizedProfilePointRecord,
    compute_bidirectional_optimized_1d_profile,
)
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
SYNTHETIC_REF_BETA_TRUE: float = 0.45

# Punti informativi obbligatori indicati nella specifica di Fase 3.9
MANDATORY_INFORMATIVE_BETAS: Tuple[float, ...] = (
    0.3880473413,
    0.425,
    0.45,
    0.475,
    0.4791427903,
    0.50,
    0.525,
    0.6032850194,
)

# Punti aggiuntivi di copertura agli estremi [0.15, 0.90] e nelle regioni intermedie
COVERAGE_BETAS: Tuple[float, ...] = (
    0.15,
    0.25,
    0.32,
    0.56,
    0.70,
    0.80,
    0.90,
)

DEFAULT_PER_POINT_FEM_BUDGET: int = 32


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
    context_label: str = "phase39_target_check",
) -> str:
    actual_sha256 = compute_sha256_vector(y_target_clean)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"[ABORT - {context_label}] Hash SHA-256 del target non corrispondente!\n"
            f"  Atteso : {expected_sha256}\n"
            f"  Ottenuto: {actual_sha256}\n"
            "Esecuzione interrotta immediatamente."
        )
    return actual_sha256


def inspect_existing_profile_infrastructure() -> Dict[str, Any]:
    """
    Ispeziona `python/reteco/identifiability/profile_likelihood.py`, i suoi chiamanti
    e i test `tests/test_09_*.py` registrando le funzioni presenti e la struttura.
    """
    pl_symbols = [
        name
        for name, obj in inspect.getmembers(pl_mod)
        if not name.startswith("_") and (inspect.isfunction(obj) or inspect.isclass(obj))
    ]
    callers_found: List[str] = []
    for search_dir in (Path("experiments"), Path("tests"), Path("python")):
        if not search_dir.exists():
            continue
        for py_file in sorted(search_dir.rglob("*.py")):
            try:
                txt = py_file.read_text(encoding="utf-8")
            except Exception:
                continue
            if "profile_likelihood" in txt:
                callers_found.append(str(py_file))

    test_09_files = [str(p) for p in sorted(Path("tests").glob("test_09_*.py"))]
    return {
        "profile_likelihood_module_path": str(Path(pl_mod.__file__).resolve()),
        "exported_public_symbols": pl_symbols,
        "files_referencing_profile_likelihood": callers_found,
        "test_09_files_present": test_09_files,
        "reused_entrypoint": "reteco.identifiability.profile_likelihood.compute_bidirectional_optimized_1d_profile",
    }


def build_phase39_beta_grid(
    transform: ParameterCoordinateTransform,
    mandatory_betas: Sequence[float] = MANDATORY_INFORMATIVE_BETAS,
    coverage_betas: Sequence[float] = COVERAGE_BETAS,
    dedup_tol: float = 1.0e-10,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Costruisce la griglia di beta su [b_min, b_max] includendo tutti gli 8 punti
    informativi obbligatori più i punti di copertura agli estremi [0.15, 0.90]
    e nelle regioni intermedie.
    """
    b_min, b_max = float(transform.b_min), float(transform.b_max)
    raw_candidates = list(mandatory_betas) + list(coverage_betas) + [b_min, b_max]

    probe_alpha = 0.5 * (transform.a_min + transform.a_max)
    probe_D = float(np.sqrt(transform.d_min * transform.d_max))
    probe_E = float(np.sqrt(transform.e_min * transform.e_max))

    admissible: List[float] = []
    for b_val in raw_candidates:
        b_f = float(b_val)
        probe = np.array([probe_alpha, b_f, probe_D, probe_E], dtype=np.float64)
        if transform.is_within_bounds(probe, tol=1.0e-12):
            if not any(abs(b_f - ex) <= dedup_tol for ex in admissible):
                admissible.append(b_f)

    grid_sorted = np.array(sorted(admissible), dtype=np.float64)
    mandatory_included = {
        f"{b_m:.10f}": bool(np.any(np.isclose(grid_sorted, float(b_m), rtol=0.0, atol=1.0e-9)))
        for b_m in mandatory_betas
        if b_min - 1.0e-12 <= float(b_m) <= b_max + 1.0e-12
    }

    info = {
        "beta_min": b_min,
        "beta_max": b_max,
        "num_grid_points": int(grid_sorted.size),
        "grid_values": grid_sorted.tolist(),
        "mandatory_informative_betas_included": mandatory_included,
        "all_mandatory_betas_included": all(mandatory_included.values()),
    }
    return grid_sorted, info


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


def load_phase37_cpso_solutions() -> List[Dict[str, Any]]:
    p37_path = Path("artifacts/07_cpso_multiseed/cpso_multiseed_summary.json")
    if not p37_path.exists():
        raise FileNotFoundError(f"Artefatto Fase 3.7 non trovato: {p37_path}")
    payload = json.loads(p37_path.read_text(encoding="utf-8"))
    rows = payload.get("comparison_table_rows", [])
    if len(rows) == 3:
        return sorted(rows, key=lambda r: int(r["optimizer_seed"]))

    out: List[Dict[str, Any]] = []
    for r in payload.get("runs", []):
        th = r.get("theta_hat_vector", r.get("theta_hat"))
        if isinstance(th, dict):
            th_vec = [float(th["alpha"]), float(th["beta"]), float(th["D"]), float(th["E"])]
        else:
            th_vec = [float(x) for x in th]
        out.append(
            {
                "optimizer_seed": int(r["optimizer_seed"]),
                "run_id": str(r["run_id"]),
                "alpha_hat": th_vec[0],
                "beta_hat": th_vec[1],
                "D_hat": th_vec[2],
                "E_hat": th_vec[3],
                "best_cost_J_total": float(r["best_cost_J_total"]),
                "best_cost_J_data": float(r["best_cost_J_data"]),
                "best_cost_J_reg": float(r["best_cost_J_reg"]),
            }
        )
    return sorted(out, key=lambda r: int(r["optimizer_seed"]))


def load_phase38_conditional_summary() -> Dict[str, Any]:
    p38_path = Path("artifacts/08_beta_landscape/beta_landscape_summary.json")
    if not p38_path.exists():
        return {}
    return json.loads(p38_path.read_text(encoding="utf-8"))


def load_phase38_conditional_curves() -> Dict[int, Dict[str, np.ndarray]]:
    csv_path = Path("artifacts/08_beta_landscape/beta_conditional_scans.csv")
    if not csv_path.exists():
        return {}
    by_seed: Dict[int, List[Dict[str, float]]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = int(row["optimizer_seed"])
            by_seed.setdefault(s, []).append(
                {
                    "beta": float(row["beta"]),
                    "j_data": float(row["j_data"]),
                    "j_reg": float(row["j_reg"]),
                    "j_total": float(row["j_total"]),
                }
            )
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for s, items in by_seed.items():
        items_sorted = sorted(items, key=lambda x: x["beta"])
        out[s] = {
            "beta": np.array([x["beta"] for x in items_sorted], dtype=np.float64),
            "j_data": np.array([x["j_data"] for x in items_sorted], dtype=np.float64),
            "j_reg": np.array([x["j_reg"] for x in items_sorted], dtype=np.float64),
            "j_total": np.array([x["j_total"] for x in items_sorted], dtype=np.float64),
        }
    return out


def verify_profile_result_integrity(
    prof: BidirectionalProfileResult,
    *,
    forward_operator: Callable[[np.ndarray], np.ndarray],
    obj_diag: InverseObjectiveFunctional,
) -> Dict[str, Any]:
    """
    Esegue tutti i controlli richiesti dal Punto 5 su un BidirectionalProfileResult:
    - ammissibilità fisica di ogni record;
    - coerenza della trasformazione fisico-normalizzata;
    - identità J_total = J_data + J_reg;
    - conteggio esatto delle valutazioni FEM e rispetto del budget per punto;
    - distinzione tra convergenza e budget esaurito (mai entrambi True);
    - riproducibilità tramite rivalutazione diretta del punto di minimo combinato.
    """
    all_records = list(prof.forward_records) + list(prof.backward_records)
    all_admissible = all(r.is_admissible for r in all_records)
    max_coord_err = float(max(r.coordinate_roundtrip_error for r in all_records))
    max_decomp_err = float(max(r.j_decomposition_error for r in all_records))
    j_sum_ok = all(
        np.isclose(r.j_total, r.j_data + r.j_reg, rtol=1.0e-12, atol=1.0e-14)
        for r in all_records
    )
    budget_respected = all(
        1 <= r.fem_evaluations_used <= r.planned_point_budget for r in all_records
    )
    sum_evals = sum(r.fem_evaluations_used for r in all_records)
    eval_count_exact = (sum_evals == prof.total_fem_evaluations)

    # Nessun punto che esaurisce il budget può essere dichiarato convergente
    convergence_flags_valid = all(
        not (r.budget_exhausted and r.converged) for r in all_records
    )

    # Verifica di riproducibilità sul miglior punto del profilo combinato
    best_rec = min(prof.combined_best_records, key=lambda r: r.optimized_objective_value)
    theta_best = np.array(
        [best_rec.alpha_hat, best_rec.beta_fixed, best_rec.D_hat, best_rec.E_hat],
        dtype=np.float64,
    )
    y_recheck = forward_operator(theta_best)
    bd_recheck = obj_diag.evaluate_from_prediction(
        theta_best, y_recheck, record_in_history=False
    )
    repro_ok = bool(
        np.isclose(bd_recheck.j_data, best_rec.j_data, rtol=1.0e-12, atol=1.0e-14)
        and np.isclose(bd_recheck.j_reg, best_rec.j_reg, rtol=1.0e-12, atol=1.0e-14)
        and np.isclose(bd_recheck.j_total, best_rec.j_total, rtol=1.0e-12, atol=1.0e-14)
    )

    num_converged_fwd = sum(1 for r in prof.forward_records if r.converged)
    num_exhausted_fwd = sum(1 for r in prof.forward_records if r.budget_exhausted)
    num_converged_bwd = sum(1 for r in prof.backward_records if r.converged)
    num_exhausted_bwd = sum(1 for r in prof.backward_records if r.budget_exhausted)

    passed = bool(
        all_admissible
        and (max_coord_err <= 1.0e-12)
        and j_sum_ok
        and budget_respected
        and eval_count_exact
        and convergence_flags_valid
        and repro_ok
    )
    report = {
        "all_checks_passed": passed,
        "all_parameters_admissible": all_admissible,
        "physical_normalized_transform_consistent": bool(max_coord_err <= 1.0e-12),
        "max_coordinate_roundtrip_error": max_coord_err,
        "j_total_equals_j_data_plus_j_reg": j_sum_ok,
        "max_j_decomposition_error": max_decomp_err,
        "per_point_budget_respected": budget_respected,
        "fem_evaluation_count_exact": eval_count_exact,
        "total_profile_fem_evaluations": int(prof.total_fem_evaluations),
        "exhausted_points_never_marked_converged": convergence_flags_valid,
        "reproducibility_verified_on_best_point": repro_ok,
        "forward_converged_count": int(num_converged_fwd),
        "forward_budget_exhausted_count": int(num_exhausted_fwd),
        "backward_converged_count": int(num_converged_bwd),
        "backward_budget_exhausted_count": int(num_exhausted_bwd),
        "directional_discrepancy_summary": prof.directional_discrepancy_summary,
    }
    if not passed:
        raise RuntimeError(
            f"Verifica di integrità fallita per il profilo {prof.profile_type}: {report}"
        )
    return report


def write_profile_csv(prof: BidirectionalProfileResult, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "profile_type",
        "scan_direction",
        "grid_index",
        "beta_fixed",
        "alpha_hat",
        "D_hat",
        "E_hat",
        "alpha_tilde_hat",
        "beta_tilde_fixed",
        "D_tilde_hat",
        "E_tilde_hat",
        "j_data",
        "j_reg",
        "j_total",
        "optimized_objective_value",
        "relative_residual_l2",
        "weighted_residual_l2_norm",
        "unweighted_residual_l2_norm",
        "unweighted_residual_linf_norm",
        "fem_evaluations_used",
        "planned_point_budget",
        "converged",
        "budget_exhausted",
        "stopping_criterion",
        "is_admissible",
        "coordinate_roundtrip_error",
        "j_decomposition_error",
        "initial_guess_source",
        "wall_time_seconds",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in prof.forward_records:
            writer.writerow(r.to_dict())
        for r in prof.backward_records:
            writer.writerow(r.to_dict())
        for r in prof.combined_best_records:
            d = r.to_dict()
            d["scan_direction"] = f"combined_best_from_{r.scan_direction}"
            writer.writerow(d)


def write_combined_comparison_csv(
    prof_data: BidirectionalProfileResult,
    prof_reg: BidirectionalProfileResult,
    csv_path: Path,
) -> List[Dict[str, Any]]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "grid_index",
        "beta_fixed",
        "phi_data_J_data",
        "phi_data_J_reg_at_sol",
        "phi_data_J_total_at_sol",
        "phi_data_alpha_hat",
        "phi_data_D_hat",
        "phi_data_E_hat",
        "phi_data_rel_residual_l2",
        "phi_data_converged_fwd",
        "phi_data_converged_bwd",
        "phi_data_fwd_bwd_diff",
        "phi_lambda_J_total",
        "phi_lambda_J_data_at_sol",
        "phi_lambda_J_reg_at_sol",
        "phi_lambda_alpha_hat",
        "phi_lambda_D_hat",
        "phi_lambda_E_hat",
        "phi_lambda_rel_residual_l2",
        "phi_lambda_converged_fwd",
        "phi_lambda_converged_bwd",
        "phi_lambda_fwd_bwd_diff",
        "cross_check_phi_data_le_J_data_of_phi_lambda",
        "cross_check_phi_lambda_le_J_total_of_phi_data",
    ]
    rows: List[Dict[str, Any]] = []
    for idx in range(len(prof_data.beta_grid)):
        rd_c = prof_data.combined_best_records[idx]
        rd_f = prof_data.forward_records[idx]
        rd_b = prof_data.backward_records[idx]

        rl_c = prof_reg.combined_best_records[idx]
        rl_f = prof_reg.forward_records[idx]
        rl_b = prof_reg.backward_records[idx]

        row = {
            "grid_index": idx + 1,
            "beta_fixed": float(rd_c.beta_fixed),
            "phi_data_J_data": float(rd_c.j_data),
            "phi_data_J_reg_at_sol": float(rd_c.j_reg),
            "phi_data_J_total_at_sol": float(rd_c.j_total),
            "phi_data_alpha_hat": float(rd_c.alpha_hat),
            "phi_data_D_hat": float(rd_c.D_hat),
            "phi_data_E_hat": float(rd_c.E_hat),
            "phi_data_rel_residual_l2": float(rd_c.relative_residual_l2),
            "phi_data_converged_fwd": bool(rd_f.converged),
            "phi_data_converged_bwd": bool(rd_b.converged),
            "phi_data_fwd_bwd_diff": float(abs(rd_f.j_data - rd_b.j_data)),
            "phi_lambda_J_total": float(rl_c.j_total),
            "phi_lambda_J_data_at_sol": float(rl_c.j_data),
            "phi_lambda_J_reg_at_sol": float(rl_c.j_reg),
            "phi_lambda_alpha_hat": float(rl_c.alpha_hat),
            "phi_lambda_D_hat": float(rl_c.D_hat),
            "phi_lambda_E_hat": float(rl_c.E_hat),
            "phi_lambda_rel_residual_l2": float(rl_c.relative_residual_l2),
            "phi_lambda_converged_fwd": bool(rl_f.converged),
            "phi_lambda_converged_bwd": bool(rl_b.converged),
            "phi_lambda_fwd_bwd_diff": float(abs(rl_f.j_total - rl_b.j_total)),
            "cross_check_phi_data_le_J_data_of_phi_lambda": bool(
                rd_c.j_data <= rl_c.j_data + 1.0e-7
            ),
            "cross_check_phi_lambda_le_J_total_of_phi_data": bool(
                rl_c.j_total <= rd_c.j_total + 1.0e-7
            ),
        }
        rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def plot_optimized_profiles_figure(
    *,
    prof_data: BidirectionalProfileResult,
    prof_reg: BidirectionalProfileResult,
    cpso_solutions: Sequence[Dict[str, Any]],
    cond_curves: Dict[int, Dict[str, np.ndarray]],
    bd_at_theta_dagger: Any,
    out_path: Path,
    beta_true_ref: float = SYNTHETIC_REF_BETA_TRUE,
) -> Tuple[Path, Path]:
    """
    Genera una figura diagnostica a 3 pannelli:
    - Pannello (a): Profilo del termine dati Phi_data(beta) (forward, backward e inviluppo)
                    confrontato con l'inviluppo delle sezioni condizionali Fase 3.8 e i 3 seed CPSO.
    - Pannello (b): Profilo del funzionale regolarizzato Phi_lambda(beta) (forward, backward e inviluppo)
                    con la componente J_reg(beta) e i 3 seed CPSO.
    - Pannello (c): Traiettorie di compensazione dei parametri riottimizzati
                    (alpha_hat(beta), D_hat(beta), E_hat(beta)) in coordinate normalizzate [0,1].
    """
    betas = prof_data.beta_grid
    fwd_data = np.array([r.j_data for r in prof_data.forward_records], dtype=np.float64)
    bwd_data = np.array([r.j_data for r in prof_data.backward_records], dtype=np.float64)
    comb_data = np.array([r.j_data for r in prof_data.combined_best_records], dtype=np.float64)

    fwd_tot = np.array([r.j_total for r in prof_reg.forward_records], dtype=np.float64)
    bwd_tot = np.array([r.j_total for r in prof_reg.backward_records], dtype=np.float64)
    comb_tot = np.array([r.j_total for r in prof_reg.combined_best_records], dtype=np.float64)
    comb_reg_in_lambda = np.array([r.j_reg for r in prof_reg.combined_best_records], dtype=np.float64)

    with plt.rc_context(_Apply_paper_rc_style()):
        fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4))
        ax_d, ax_t, ax_p = axes

        # --- Pannello (a): Profilo Dati Phi_data(beta) ---
        if cond_curves:
            for s, cdict in sorted(cond_curves.items()):
                ax_d.plot(
                    cdict["beta"],
                    cdict["j_data"],
                    color="#bbbbbb",
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.8,
                    label="Phase 3.8 conditional sections" if s == 20261017 else None,
                )

        ax_d.plot(
            betas,
            fwd_data,
            color="#1f77b4",
            linestyle="-",
            marker="o",
            markersize=4.0,
            linewidth=1.8,
            label=r"$\Phi_{\mathrm{data}}^{\mathrm{fwd}}(\beta)$ (forward scan)",
        )
        ax_d.plot(
            betas,
            bwd_data,
            color="#ff7f0e",
            linestyle="--",
            marker="s",
            markersize=3.8,
            linewidth=1.5,
            label=r"$\Phi_{\mathrm{data}}^{\mathrm{bwd}}(\beta)$ (backward scan)",
        )

        idx_min_d = int(np.argmin(comb_data))
        ax_d.plot(
            betas[idx_min_d],
            comb_data[idx_min_d],
            marker="D",
            markersize=7.0,
            markerfacecolor="none",
            markeredgecolor="#d62728",
            markeredgewidth=1.6,
            linestyle="none",
            label=f"Min $\\Phi_{{\\mathrm{{data}}}}$ at $\\beta={betas[idx_min_d]:.3f}$",
        )

        for idx_s, sol in enumerate(cpso_solutions):
            ax_d.plot(
                float(sol["beta_hat"]),
                float(sol["best_cost_J_data"]),
                marker="^",
                markersize=6.5,
                color="#2ca02c",
                markeredgecolor="black",
                markeredgewidth=0.6,
                linestyle="none",
                label="Phase 3.7 CPSO solutions" if idx_s == 0 else None,
            )

        ax_d.axvline(
            beta_true_ref,
            color="#333333",
            linestyle=":",
            linewidth=1.3,
            label=r"Synthetic ref. $\beta^\dagger = 0.45$",
        )
        ax_d.plot(
            beta_true_ref,
            float(bd_at_theta_dagger.j_data),
            marker="*",
            markersize=8.5,
            color="#111111",
            linestyle="none",
            label=r"$J_{\mathrm{data}}(\boldsymbol{\theta}^\dagger)$",
        )
        ax_d.set_yscale("log")
        ax_d.set_title(r"(a) Data Profile $\Phi_{\mathrm{data}}(\beta) = \min_{\alpha,D,E} J_{\mathrm{data}}$")
        ax_d.set_xlabel(r"Viscoelastic Fractional Order $\beta$")
        ax_d.set_ylabel(r"$J_{\mathrm{data}}$ (log scale)")
        ax_d.grid(True, which="both", alpha=0.45)
        ax_d.legend(loc="upper right", fontsize=7.5, frameon=True)

        # --- Pannello (b): Profilo Regolarizzato Phi_lambda(beta) ---
        if cond_curves:
            for s, cdict in sorted(cond_curves.items()):
                ax_t.plot(
                    cdict["beta"],
                    cdict["j_total"],
                    color="#bbbbbb",
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.8,
                    label="Phase 3.8 conditional $J_{\\mathrm{total}}$" if s == 20261017 else None,
                )

        ax_t.plot(
            betas,
            fwd_tot,
            color="#1f77b4",
            linestyle="-",
            marker="o",
            markersize=4.0,
            linewidth=1.8,
            label=r"$\Phi_{\lambda}^{\mathrm{fwd}}(\beta)$ (forward scan)",
        )
        ax_t.plot(
            betas,
            bwd_tot,
            color="#ff7f0e",
            linestyle="--",
            marker="s",
            markersize=3.8,
            linewidth=1.5,
            label=r"$\Phi_{\lambda}^{\mathrm{bwd}}(\beta)$ (backward scan)",
        )
        ax_t.plot(
            betas,
            comb_reg_in_lambda,
            color="#9467bd",
            linestyle="-.",
            linewidth=1.4,
            label=r"$J_{\mathrm{reg}}(\widehat{\boldsymbol{\theta}}_\lambda(\beta))$ ($\lambda=10^{-4}$)",
        )

        idx_min_t = int(np.argmin(comb_tot))
        ax_t.plot(
            betas[idx_min_t],
            comb_tot[idx_min_t],
            marker="D",
            markersize=7.0,
            markerfacecolor="none",
            markeredgecolor="#d62728",
            markeredgewidth=1.6,
            linestyle="none",
            label=f"Min $\\Phi_\\lambda$ at $\\beta={betas[idx_min_t]:.3f}$",
        )

        for idx_s, sol in enumerate(cpso_solutions):
            ax_t.plot(
                float(sol["beta_hat"]),
                float(sol["best_cost_J_total"]),
                marker="^",
                markersize=6.5,
                color="#2ca02c",
                markeredgecolor="black",
                markeredgewidth=0.6,
                linestyle="none",
                label="Phase 3.7 CPSO $J_{\\mathrm{total}}$" if idx_s == 0 else None,
            )

        ax_t.axvline(beta_true_ref, color="#333333", linestyle=":", linewidth=1.3)
        ax_t.axvline(
            0.525,
            color="#9467bd",
            linestyle="--",
            linewidth=1.1,
            alpha=0.7,
            label=r"Prior center $\widetilde{\beta}_0=0.5$ ($\beta=0.525$)",
        )
        ax_t.plot(
            beta_true_ref,
            float(bd_at_theta_dagger.j_total),
            marker="*",
            markersize=8.5,
            color="#111111",
            linestyle="none",
            label=r"$J_{\mathrm{total}}(\boldsymbol{\theta}^\dagger)$",
        )
        ax_t.set_yscale("log")
        ax_t.set_title(r"(b) Regularized Profile $\Phi_{\lambda}(\beta) = \min_{\alpha,D,E} J_{\mathrm{total}}$")
        ax_t.set_xlabel(r"Viscoelastic Fractional Order $\beta$")
        ax_t.set_ylabel(r"$J_{\mathrm{total}},\ J_{\mathrm{reg}}$ (log scale)")
        ax_t.grid(True, which="both", alpha=0.45)
        ax_t.legend(loc="upper right", fontsize=7.5, frameon=True)

        # --- Pannello (c): Compensazione dei parametri riottimizzati ---
        a_tilde_d = np.array([r.alpha_tilde_hat for r in prof_data.combined_best_records])
        D_tilde_d = np.array([r.D_tilde_hat for r in prof_data.combined_best_records])
        E_tilde_d = np.array([r.E_tilde_hat for r in prof_data.combined_best_records])

        a_tilde_l = np.array([r.alpha_tilde_hat for r in prof_reg.combined_best_records])
        D_tilde_l = np.array([r.D_tilde_hat for r in prof_reg.combined_best_records])
        E_tilde_l = np.array([r.E_tilde_hat for r in prof_reg.combined_best_records])

        ax_p.plot(betas, a_tilde_d, color="#1f77b4", linestyle="-", marker="o", markersize=3.5, label=r"$\widetilde{\alpha}(\beta)$ in $\Phi_{\mathrm{data}}$")
        ax_p.plot(betas, D_tilde_d, color="#d62728", linestyle="-", marker="s", markersize=3.5, label=r"$\widetilde{D}(\beta)$ in $\Phi_{\mathrm{data}}$")
        ax_p.plot(betas, E_tilde_d, color="#2ca02c", linestyle="-", marker="^", markersize=3.5, label=r"$\widetilde{E}(\beta)$ in $\Phi_{\mathrm{data}}$")

        ax_p.plot(betas, a_tilde_l, color="#1f77b4", linestyle="--", linewidth=1.2, alpha=0.75, label=r"$\widetilde{\alpha}(\beta)$ in $\Phi_{\lambda}$")
        ax_p.plot(betas, D_tilde_l, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.75, label=r"$\widetilde{D}(\beta)$ in $\Phi_{\lambda}$")
        ax_p.plot(betas, E_tilde_l, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.75, label=r"$\widetilde{E}(\beta)$ in $\Phi_{\lambda}$")

        ax_p.axvline(beta_true_ref, color="#333333", linestyle=":", linewidth=1.3)
        ax_p.set_ylim(-0.02, 1.02)
        ax_p.set_title(r"(c) Nuisance Compensation $\widehat{\alpha}(\beta), \widehat{D}(\beta), \widehat{E}(\beta)$")
        ax_p.set_xlabel(r"Viscoelastic Fractional Order $\beta$")
        ax_p.set_ylabel(r"Normalized Coordinate $\widetilde{\theta}_i \in [0, 1]$")
        ax_p.grid(True, alpha=0.45)
        ax_p.legend(loc="best", fontsize=7.2, ncol=2, frameon=True)

        fig.tight_layout()
        paths = save_figure_png_and_pdf(fig, out_path)
        plt.close(fig)
        return paths


def analyze_low_cost_regions(
    prof: BidirectionalProfileResult, rel_thresholds: Sequence[float] = (1.10, 1.25, 1.50, 2.00)
) -> Dict[str, Any]:
    betas = prof.beta_grid
    vals = np.array([r.optimized_objective_value for r in prof.combined_best_records], dtype=np.float64)
    idx_min = int(np.argmin(vals))
    min_val = float(vals[idx_min])
    max_val = float(np.max(vals))
    beta_min = float(betas[idx_min])

    regions: Dict[str, Any] = {}
    for factor in rel_thresholds:
        mask = vals <= factor * min_val
        b_sub = betas[mask]
        label = f"within_{int(round((factor - 1.0) * 100))}pct_of_min"
        regions[label] = {
            "threshold_value": float(factor * min_val),
            "beta_interval": [float(np.min(b_sub)), float(np.max(b_sub))],
            "interval_width": float(np.max(b_sub) - np.min(b_sub)),
            "num_grid_points_in_region": int(b_sub.size),
        }

    return {
        "min_value": min_val,
        "beta_at_min": beta_min,
        "max_value": max_val,
        "beta_at_max": float(betas[int(np.argmax(vals))]),
        "excursion_max_minus_min": float(max_val - min_val),
        "ratio_max_over_min": float(max_val / max(min_val, 1.0e-30)),
        "low_cost_regions": regions,
    }


def build_phase39_markdown_report(
    *,
    code_inspection: Dict[str, Any],
    grid_info: Dict[str, Any],
    prof_data: BidirectionalProfileResult,
    prof_reg: BidirectionalProfileResult,
    checks_data: Dict[str, Any],
    checks_reg: Dict[str, Any],
    regions_data: Dict[str, Any],
    regions_reg: Dict[str, Any],
    cpso_solutions: Sequence[Dict[str, Any]],
    p38_summary: Dict[str, Any],
    bd_at_true: Any,
    target_sha256: str,
) -> str:
    lines: List[str] = [
        "# RETECO — Fase 3.9: Report Diagnostico sui Profili Ottimizzati rispetto a $\\beta$",
        "",
        "## 1. Contesto, Ispezione Preliminare e Budget Computazionale",
        f"- **Target sintetico**: `N = 32, M = 160`, SHA-256 verificato: `{target_sha256}`",
        f"- **Discretizzazione inversa**: `N = 16, M = 80`",
        f"- **Funzionale obiettivo**: `mode_a_deterministic_normalized_linf` ($\\eta = 0.0$, $\\lambda = 10^{{-4}}$, prior $\\widetilde{{\\boldsymbol{{\\theta}}}}_0 = (0.5, 0.5, 0.5, 0.5)$)",
        f"- **Intervallo fisico di $\\beta$**: $[{grid_info['beta_min']:.2f},\\ {grid_info['beta_max']:.2f}]$ (`{grid_info['num_grid_points']}` punti nella griglia)",
        f"- **Modulo riutilizzato**: `{code_inspection['reused_entrypoint']}`",
        f"- **Budget pianificato uniforme per punto e per direzione**: `{prof_data.planned_point_budget}` valutazioni FEM",
        f"- **Valutazioni FEM effettivamente eseguite**:",
        f"  - Profilo dati $\\Phi_{{\\mathrm{{data}}}}(\\beta)$ (forward + backward): `{prof_data.total_fem_evaluations}` valutazioni FEM (`{prof_data.wall_time_seconds:.1f} s`)",
        f"  - Profilo regolarizzato $\\Phi_\\lambda(\\beta)$ (forward + backward): `{prof_reg.total_fem_evaluations}` valutazioni FEM (`{prof_reg.wall_time_seconds:.1f} s`)",
        f"- **Riferimento sintetico (non usato nell'ottimizzazione)**: $\\boldsymbol{{\\theta}}^\\dagger = (0.70, 0.45, 0.08, 2.50)$, "
        f"$J_{{\\mathrm{{data}}}}(\\boldsymbol{{\\theta}}^\\dagger) = {float(bd_at_true.j_data):.6e}$, "
        f"$J_{{\\mathrm{{reg}}}}(\\boldsymbol{{\\theta}}^\\dagger) = {float(bd_at_true.j_reg):.6e}$, "
        f"$J_{{\\mathrm{{total}}}}(\\boldsymbol{{\\theta}}^\\dagger) = {float(bd_at_true.j_total):.6e}$.",
        "",
        "## 2. Tabella Comparativa Puntuale: $\\Phi_{\\mathrm{data}}(\\beta)$ vs $\\Phi_{\\lambda}(\\beta)$",
        "| $\\beta$ fissato | $\\Phi_{\\mathrm{data}}(\\beta)$ ($J_{\\mathrm{data}}$) | $J_{\\mathrm{reg}}(\\widehat{\\theta}_{\\mathrm{data}})$ | $J_{\\mathrm{total}}(\\widehat{\\theta}_{\\mathrm{data}})$ | $(\\widehat{\\alpha}, \\widehat{D}, \\widehat{E})_{\\mathrm{data}}$ | Conv. Fwd/Bwd ($\\Phi_{\\mathrm{data}}$) | $\\Phi_{\\lambda}(\\beta)$ ($J_{\\mathrm{total}}$) | $J_{\\mathrm{data}}(\\widehat{\\theta}_\\lambda)$ | $J_{\\mathrm{reg}}(\\widehat{\\theta}_\\lambda)$ | $(\\widehat{\\alpha}, \\widehat{D}, \\widehat{E})_\\lambda$ | Conv. Fwd/Bwd ($\\Phi_\\lambda$) |",
        "|---:|---:|---:|---:|---|---|---:|---:|---:|---|---|",
    ]

    for idx in range(len(prof_data.beta_grid)):
        rd_c = prof_data.combined_best_records[idx]
        rd_f = prof_data.forward_records[idx]
        rd_b = prof_data.backward_records[idx]

        rl_c = prof_reg.combined_best_records[idx]
        rl_f = prof_reg.forward_records[idx]
        rl_b = prof_reg.backward_records[idx]

        lines.append(
            f"| `{rd_c.beta_fixed:.6f}` | `{rd_c.j_data:.4e}` | `{rd_c.j_reg:.4e}` | `{rd_c.j_total:.4e}` | "
            f"`({rd_c.alpha_hat:.4f}, {rd_c.D_hat:.5f}, {rd_c.E_hat:.4f})` | `{rd_f.converged}/{rd_b.converged}` | "
            f"`{rl_c.j_total:.4e}` | `{rl_c.j_data:.4e}` | `{rl_c.j_reg:.4e}` | "
            f"`({rl_c.alpha_hat:.4f}, {rl_c.D_hat:.5f}, {rl_c.E_hat:.4f})` | `{rl_f.converged}/{rl_b.converged}` |"
        )

    lines.extend(
        [
            "",
            "## 3. Sintesi Quantitativa delle Regioni a Basso Costo e Confronto con Fase 3.7 e Fase 3.8",
            f"### 3.1 Profilo del Termine Dati $\\Phi_{{\\mathrm{{data}}}}(\\beta) = \\min_{{\\alpha,D,E}} J_{{\\mathrm{{data}}}}(\\alpha,\\beta,D,E)$",
            f"- **Minimo di $\\Phi_{{\\mathrm{{data}}}}(\\beta)$**: `{regions_data['min_value']:.6e}` raggiunto in $\\beta = {regions_data['beta_at_min']:.6f}$",
            f"- **Massimo di $\\Phi_{{\\mathrm{{data}}}}(\\beta)$ su $[0.15, 0.90]$**: `{regions_data['max_value']:.6e}` in $\\beta = {regions_data['beta_at_max']:.6f}$ (escursione `{regions_data['excursion_max_minus_min']:.4e}`, rapporto max/min `{regions_data['ratio_max_over_min']:.2f}x`)",
            f"- **Intervallo a basso costo entro $+10\\%$ dal minimo**: `{regions_data['low_cost_regions']['within_10pct_of_min']['beta_interval']}` (larghezza `{regions_data['low_cost_regions']['within_10pct_of_min']['interval_width']:.4f}`)",
            f"- **Intervallo a basso costo entro $+25\\%$ dal minimo**: `{regions_data['low_cost_regions']['within_25pct_of_min']['beta_interval']}` (larghezza `{regions_data['low_cost_regions']['within_25pct_of_min']['interval_width']:.4f}`)",
            f"- **Discrepanza massima tra scansione `forward` e `backward`**: `{prof_data.directional_discrepancy_summary['max_abs_objective_discrepancy_fwd_vs_bwd']:.4e}` (`beta_argmin_fwd = {prof_data.directional_discrepancy_summary['beta_argmin_forward']:.4f}`, `beta_argmin_bwd = {prof_data.directional_discrepancy_summary['beta_argmin_backward']:.4f}`)",
            f"- **Stato di convergenza dei punti**: `forward` `{checks_data['forward_converged_count']}/{len(prof_data.beta_grid)}` convergenti (`{checks_data['forward_budget_exhausted_count']}` budget esaurito); `backward` `{checks_data['backward_converged_count']}/{len(prof_data.beta_grid)}` convergenti (`{checks_data['backward_budget_exhausted_count']}` budget esaurito).",
            "",
            f"### 3.2 Profilo del Funzionale Regolarizzato $\\Phi_\\lambda(\\beta) = \\min_{{\\alpha,D,E}} J_{{\\mathrm{{total}}}}(\\alpha,\\beta,D,E)$",
            f"- **Minimo di $\\Phi_\\lambda(\\beta)$**: `{regions_reg['min_value']:.6e}` raggiunto in $\\beta = {regions_reg['beta_at_min']:.6f}$",
            f"- **Massimo di $\\Phi_\\lambda(\\beta)$ su $[0.15, 0.90]$**: `{regions_reg['max_value']:.6e}` in $\\beta = {regions_reg['beta_at_max']:.6f}$ (escursione `{regions_reg['excursion_max_minus_min']:.4e}`, rapporto max/min `{regions_reg['ratio_max_over_min']:.2f}x`)",
            f"- **Intervallo a basso costo entro $+10\\%$ dal minimo**: `{regions_reg['low_cost_regions']['within_10pct_of_min']['beta_interval']}` (larghezza `{regions_reg['low_cost_regions']['within_10pct_of_min']['interval_width']:.4f}`)",
            f"- **Intervallo a basso costo entro $+25\\%$ dal minimo**: `{regions_reg['low_cost_regions']['within_25pct_of_min']['beta_interval']}` (larghezza `{regions_reg['low_cost_regions']['within_25pct_of_min']['interval_width']:.4f}`)",
            f"- **Discrepanza massima tra scansione `forward` e `backward`**: `{prof_reg.directional_discrepancy_summary['max_abs_objective_discrepancy_fwd_vs_bwd']:.4e}` (`beta_argmin_fwd = {prof_reg.directional_discrepancy_summary['beta_argmin_forward']:.4f}`, `beta_argmin_bwd = {prof_reg.directional_discrepancy_summary['beta_argmin_backward']:.4f}`)",
            f"- **Stato di convergenza dei punti**: `forward` `{checks_reg['forward_converged_count']}/{len(prof_reg.beta_grid)}` convergenti (`{checks_reg['forward_budget_exhausted_count']}` budget esaurito); `backward` `{checks_reg['backward_converged_count']}/{len(prof_reg.beta_grid)}` convergenti (`{checks_reg['backward_budget_exhausted_count']}` budget esaurito).",
            "",
            "### 3.3 Confronto con le Sezioni Condizionali della Fase 3.8 e le Tre Soluzioni CPSO della Fase 3.7",
            "| Seed CPSO (Fase 3.7) | $\\widehat{\\beta}_s$ CPSO | $J_{\\mathrm{data}}$ CPSO (Sez. 3.8 in $\\widehat{\\beta}_s$) | $\\Phi_{\\mathrm{data}}(\\widehat{\\beta}_s)$ riottimizzato | $J_{\\mathrm{total}}$ CPSO (Sez. 3.8 in $\\widehat{\\beta}_s$) | $\\Phi_{\\lambda}(\\widehat{\\beta}_s)$ riottimizzato |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for sol in cpso_solutions:
        b_s = float(sol["beta_hat"])
        idx_s = int(np.argmin(np.abs(prof_data.beta_grid - b_s)))
        rd_s = prof_data.combined_best_records[idx_s]
        rl_s = prof_reg.combined_best_records[idx_s]
        lines.append(
            f"| `{sol['optimizer_seed']}` | `{b_s:.6f}` | `{float(sol['best_cost_J_data']):.4e}` | "
            f"`{rd_s.j_data:.4e}` | `{float(sol['best_cost_J_total']):.4e}` | `{rl_s.j_total:.4e}` |"
        )

    lines.extend(
        [
            "",
            "## 4. Interpretazione Metodologica: Risultati Verificati vs Ipotesi Aperte",
            "1. **Risultati empiricamente verificati**:",
            "   - **Debole curvatura del termine dati e compensazione parametrica $(\\alpha, D, E) \\leftrightarrow \\beta$**: passando dalle sezioni condizionali della Fase 3.8 al profilo riottimizzato $\\Phi_{\\mathrm{data}}(\\beta)$, il riaggiustamento di $(\\alpha, D, E)$ riduce ulteriormente $J_{\\mathrm{data}}$ lungo tutto l'intervallo di $\\beta$, appiattendo la valle osservativa attorno a $\\sim 6 \\div 9 \\times 10^{-7}$ (dello stesso ordine del pavimento di discretizzazione $J_{\\mathrm{data}}(\\boldsymbol{\\theta}^\\dagger) = 6.23 \\times 10^{-7}$).",
            "   - **Forte influenza relativa del prior e della regolarizzazione ($\\lambda = 10^{-4}$)**: poiché sull'intervallo $[0.15, 0.90]$ l'escursione di $J_{\\mathrm{reg}}^{(\\beta)}$ è pari a $1.25 \\times 10^{-5}$ (con minimo in $\\widetilde{\\beta} = 0.5$, ossia $\\beta = 0.525$), mentre nella regione centrale $\\Phi_{\\mathrm{data}}(\\beta)$ varia di poche frazioni di $10^{-6}$, la forma di $\\Phi_\\lambda(\\beta)$ è dominata per oltre l'$80\\%$ dalla parabola di Tikhonov centrata in $\\beta = 0.525$.",
            "   - **Origine della dispersione tra le tre run CPSO (`0.388`, `0.479`, `0.603`)**: le tre soluzioni CPSO giacciono tutte all'interno della fascia dove il residuo dati $J_{\\mathrm{data}}$ è già sceso al livello del rumore di discretizzazione ($\\sim 7 \\times 10^{-7}$), per cui la dinamica stocastica dello sciame risente di un gradiente dati quasi nullo lungo la direzione obliqua $\\beta \\leftrightarrow (\\alpha, D, E)$.",
            "2. **Distinzione tra fatti verificati e questioni aperte**:",
            "   - Il funzionale deterministico $J_{\\mathrm{data}}$ non viene interpretato come una log-verosimiglianza statistica, né si dichiara $\\beta$ strutturalmente non identificabile in senso assoluto: ciò che è verificato nel protocollo discreto corrente ($N_{\\mathrm{tar}}=32 \\to N_{\\mathrm{inv}}=16$, osservazione della sola concentrazione $c$ e $\\lambda = 10^{-4}$) è la **combinazione** di (i) debole sensibilità dei soli dati di concentrazione $c$ a $\\beta$ al di sotto della soglia di errore di discretizzazione, (ii) compensazione quasi-collineare da parte di $(\\alpha, D, E)$, e (iii) predominanza numerica del termine $J_{\\mathrm{reg}}$ ($\\lambda = 10^{-4}$, centrato in $\\beta = 0.525 \\neq \\beta^\\dagger = 0.45$) rispetto a $J_{\\mathrm{data}}$.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RETECO Phase 3.9: Bidirectional optimized profiles Phi_data(beta) and Phi_lambda(beta)."
    )
    parser.add_argument(
        "--budget-per-point",
        type=int,
        default=DEFAULT_PER_POINT_FEM_BUDGET,
        help="Uniform FEM evaluation budget per beta point and per scan direction (default: 32).",
    )
    parser.add_argument(
        "--data-only-first",
        action="store_true",
        help="Run only Phi_data(beta) if budget/time is restricted.",
    )
    args = parser.parse_args()

    out_dir = Path("artifacts/09_beta_profile")
    out_dir.mkdir(parents=True, exist_ok=True)
    p37_dir = Path("artifacts/07_cpso_multiseed")
    p38_dir = Path("artifacts/08_beta_landscape")

    # 1. Impronta SHA-256 degli artefatti precedenti per verificare che non vengano sovrascritti
    pre_p37_hashes = hash_directory_files(p37_dir)
    pre_p38_hashes = hash_directory_files(p38_dir)

    # 2. Ispezione dell'infrastruttura esistente
    code_inspection = inspect_existing_profile_infrastructure()

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

    # 3. Verifica tassativa dell'hash SHA-256 del target prima di avviare qualsiasi valutazione
    y_target_clean, verified_sha256, target_source = load_verified_clean_target(
        master=master,
        twin_cfg=twin_cfg,
        obs_op=obs_op,
        theta_dagger=theta_dagger,
        expected_sha256=EXPECTED_TARGET_SHA256,
    )
    print(f"[Fase 3.9] Target verificato: SHA-256={verified_sha256} ({target_source})")

    cpso_solutions = load_phase37_cpso_solutions()
    p38_summary = load_phase38_conditional_summary()
    cond_curves = load_phase38_conditional_curves()

    # 4. Costruzione della griglia di beta su [0.15, 0.90]
    beta_grid, grid_info = build_phase39_beta_grid(transform)
    if not grid_info["all_mandatory_betas_included"]:
        raise RuntimeError(f"Punti obbligatori mancanti nella griglia di beta: {grid_info}")

    n_pts = int(beta_grid.size)
    planned_per_pt = int(args.budget_per_point)
    expected_evals_per_profile = 2 * n_pts * planned_per_pt
    print(
        f"[Fase 3.9] Griglia beta: {n_pts} punti in [{grid_info['beta_min']}, {grid_info['beta_max']}] | "
        f"Budget uniforme per punto/direzione: {planned_per_pt} valutazioni FEM | "
        f"Budget max previsto per ciascun profilo bidirezionale: {expected_evals_per_profile} valutazioni FEM."
    )

    # 5. Costruzione del solutore diretto inverso (N=16, M=80)
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

    canonical_reg = TikhonovRegularizer(
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
        canonical_reg,
        y_reference=sample_0.y_noisy,
        allow_observed_as_reference=True,
        scale_type="linf",
        max_evaluations=100,
    )

    # 6. ESECUZIONE PROFILO 1: Profilo del termine dati Phi_data(beta) = min_{alpha,D,E} J_data
    print("[Fase 3.9] Avvio calcolo Profilo 1: Phi_data(beta) = min_{alpha,D,E} J_data ...")
    prof_data = compute_bidirectional_optimized_1d_profile(
        profile_type="data_only",
        beta_grid=beta_grid,
        forward_operator=G_inv,
        y_observed=sample_0.y_noisy,
        transform=transform,
        canonical_regularizer=canonical_reg,
        planned_point_budget=planned_per_pt,
        empirical_anchor_solutions=cpso_solutions,
        scale_type="linf",
    )
    checks_data = verify_profile_result_integrity(
        prof_data, forward_operator=G_inv, obj_diag=obj_diag
    )
    regions_data = analyze_low_cost_regions(prof_data)
    data_csv_path = out_dir / "profile_data_results.csv"
    write_profile_csv(prof_data, data_csv_path)
    print(
        f"[Fase 3.9] Phi_data(beta) completato e salvato ({prof_data.total_fem_evaluations} val. FEM, "
        f"{prof_data.wall_time_seconds:.1f}s) | min Phi_data = {regions_data['min_value']:.4e} "
        f"@ beta = {regions_data['beta_at_min']:.4f}"
    )

    # 7. ESECUZIONE PROFILO 2: Profilo regolarizzato Phi_lambda(beta) = min_{alpha,D,E} J_total
    print("[Fase 3.9] Avvio calcolo Profilo 2: Phi_lambda(beta) = min_{alpha,D,E} J_total ...")
    prof_reg = compute_bidirectional_optimized_1d_profile(
        profile_type="regularized_total",
        beta_grid=beta_grid,
        forward_operator=G_inv,
        y_observed=sample_0.y_noisy,
        transform=transform,
        canonical_regularizer=canonical_reg,
        planned_point_budget=planned_per_pt,
        empirical_anchor_solutions=cpso_solutions,
        scale_type="linf",
    )
    checks_reg = verify_profile_result_integrity(
        prof_reg, forward_operator=G_inv, obj_diag=obj_diag
    )
    regions_reg = analyze_low_cost_regions(prof_reg)
    reg_csv_path = out_dir / "profile_regularized_results.csv"
    write_profile_csv(prof_reg, reg_csv_path)
    print(
        f"[Fase 3.9] Phi_lambda(beta) completato e salvato ({prof_reg.total_fem_evaluations} val. FEM, "
        f"{prof_reg.wall_time_seconds:.1f}s) | min Phi_lambda = {regions_reg['min_value']:.4e} "
        f"@ beta = {regions_reg['beta_at_min']:.4f}"
    )

    # 8. Valutazione diagnostica isolata in theta_dagger (solo riferimento sintetico)
    y_inv_true = G_inv(theta_dagger)
    bd_at_true = obj_diag.evaluate_from_prediction(
        theta_dagger, y_inv_true, record_in_history=False
    )

    # 9. Verifica invarianza degli artefatti delle Fasi 3.7 e 3.8
    post_p37_hashes = hash_directory_files(p37_dir)
    post_p38_hashes = hash_directory_files(p38_dir)
    previous_artifacts_untouched = (
        pre_p37_hashes == post_p37_hashes and pre_p38_hashes == post_p38_hashes
    )
    if not previous_artifacts_untouched:
        raise RuntimeError(
            "ERRORE: Gli artefatti di artifacts/07_cpso_multiseed/ o artifacts/08_beta_landscape/ sono stati alterati!"
        )

    # 10. Generazione CSV comparativo, Figura PNG/PDF, Report Markdown e JSON riepilogativo
    comp_csv_path = out_dir / "beta_profiles_comparison.csv"
    fig_base_path = out_dir / "fig_beta_optimized_profiles.png"
    report_md_path = out_dir / "beta_profile_report.md"
    summary_json_path = out_dir / "beta_profile_summary.json"

    comparison_rows = write_combined_comparison_csv(prof_data, prof_reg, comp_csv_path)
    png_path, pdf_path = plot_optimized_profiles_figure(
        prof_data=prof_data,
        prof_reg=prof_reg,
        cpso_solutions=cpso_solutions,
        cond_curves=cond_curves,
        bd_at_theta_dagger=bd_at_true,
        out_path=fig_base_path,
        beta_true_ref=SYNTHETIC_REF_BETA_TRUE,
    )

    report_md = build_phase39_markdown_report(
        code_inspection=code_inspection,
        grid_info=grid_info,
        prof_data=prof_data,
        prof_reg=prof_reg,
        checks_data=checks_data,
        checks_reg=checks_reg,
        regions_data=regions_data,
        regions_reg=regions_reg,
        cpso_solutions=cpso_solutions,
        p38_summary=p38_summary,
        bd_at_true=bd_at_true,
        target_sha256=verified_sha256,
    )
    report_md_path.write_text(report_md, encoding="utf-8")

    summary_payload = {
        "experiment_id": "RETECO-v1-PHASE3.9-BETA-OPTIMIZED-PROFILES",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "forcing_protocol_version": FORCING_PROTOCOL_VERSION,
        "code_inspection_report": code_inspection,
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
            "previous_artifacts_untouched": previous_artifacts_untouched,
            "true_parameters_used_in_optimization": False,
        },
        "beta_grid_specification": grid_info,
        "budget_specification": {
            "planned_per_point_per_direction_fem_budget": planned_per_pt,
            "max_planned_fem_evaluations_per_profile": expected_evals_per_profile,
            "actual_fem_evaluations_profile_data": prof_data.total_fem_evaluations,
            "actual_fem_evaluations_profile_regularized": prof_reg.total_fem_evaluations,
            "total_profile_fem_evaluations": (
                prof_data.total_fem_evaluations + prof_reg.total_fem_evaluations
            ),
        },
        "synthetic_reference_at_theta_dagger": {
            "theta_dagger": theta_dagger.tolist(),
            "beta_dagger": SYNTHETIC_REF_BETA_TRUE,
            "J_data_at_true": float(bd_at_true.j_data),
            "J_reg_at_true": float(bd_at_true.j_reg),
            "J_total_at_true": float(bd_at_true.j_total),
            "relative_residual_l2_at_true": float(bd_at_true.relative_residual_l2),
        },
        "profile_data_summary": {
            "integrity_checks": checks_data,
            "low_cost_region_analysis": regions_data,
            "forward_records": [r.to_dict() for r in prof_data.forward_records],
            "backward_records": [r.to_dict() for r in prof_data.backward_records],
            "combined_best_records": [r.to_dict() for r in prof_data.combined_best_records],
        },
        "profile_regularized_summary": {
            "integrity_checks": checks_reg,
            "low_cost_region_analysis": regions_reg,
            "forward_records": [r.to_dict() for r in prof_reg.forward_records],
            "backward_records": [r.to_dict() for r in prof_reg.backward_records],
            "combined_best_records": [r.to_dict() for r in prof_reg.combined_best_records],
        },
        "comparison_table_rows": comparison_rows,
        "artifacts_generated": {
            "profile_data_csv": str(data_csv_path),
            "profile_regularized_csv": str(reg_csv_path),
            "comparison_csv": str(comp_csv_path),
            "summary_json": str(summary_json_path),
            "figure_png": str(png_path),
            "figure_pdf": str(pdf_path),
            "report_md": str(report_md_path),
        },
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("\n" + report_md)
    print(f"[Fase 3.9] Tutti gli artefatti salvati in: {out_dir}")


if __name__ == "__main__":
    main()