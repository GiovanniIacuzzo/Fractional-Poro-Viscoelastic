"""
experiments/07_cpso_multiseed_variability.py
RETECO — Fase 3.7: Variabilità del CPSO a budget fissato (600 valutazioni FEM = 10 x 60)
nel caso senza rumore (eta = 0.0, Modalità A deterministica `linf`, lambda = 1.0e-4).

Protocollo congelato:
- Target sintetico: N = 32, M = 160, SHA-256 obbligatorio:
  1bca9e29c60d77c1450fd2fcea845d2320c9e44945dd7d3ccda52aa33ef261af
- Discretizzazione inversa: N = 16, M = 80
- Seed dei dati fissato: 20261016
- Seed CPSO confrontati: [20261017 (riutilizzato da Fase 3.6), 20261018 (nuovo), 20261019 (nuovo)]
- Budget per ciascuna run: 600 valutazioni FEM (10 particelle x 60 iterazioni)
- Valutazione in theta_dagger eseguita esclusivamente come riferimento diagnostico separato.

Salva tutti gli artefatti in `artifacts/07_cpso_multiseed/`.
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
from reteco.inverse.cpso import run_cpso_optimization
from reteco.inverse.diagnostics import (
    EvaluationHistoryRow,
    SingleRunDiagnosticRecord,
    extract_run_diagnostics_from_history,
    serialize_run_record,
    write_convergence_history_csv,
)

EXPECTED_TARGET_SHA256: str = (
    "1bca9e29c60d77c1450fd2fcea845d2320c9e44945dd7d3ccda52aa33ef261af"
)
FIXED_DATA_SEED: int = 20261016
CPSO_SEEDS: Tuple[int, int, int] = (20261017, 20261018, 20261019)
NUM_PARTICLES: int = 10
MAX_ITERATIONS: int = 60
PLANNED_FEM_BUDGET: int = NUM_PARTICLES * MAX_ITERATIONS  # 600
FIXED_LAMBDA_REG: float = 1.0e-4

REF_SEED_20261017: Dict[str, Any] = {
    "optimizer_seed": 20261017,
    "j_total": 6.322413978771646e-06,
    "j_data": 7.433555807819211e-07,
    "j_reg": 5.579058397989724e-06,
    "theta_hat": [
        0.7033132338355365,
        0.3880473413170004,
        0.07952391854002629,
        2.4787911063215984,
    ],
    "j_total_at_theta_dagger": 4.984481939398449e-06,
}


def compute_sha256_vector(vec: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(vec, dtype=np.float64).ravel())
    return hashlib.sha256(arr.tobytes()).hexdigest()


def verify_target_hash_or_abort(
    y_target_clean: np.ndarray,
    expected_sha256: str = EXPECTED_TARGET_SHA256,
    context_label: str = "target_verification",
) -> str:
    actual_sha256 = compute_sha256_vector(y_target_clean)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"[ABORT - {context_label}] Hash SHA-256 del target non corrispondente!\n"
            f"  Atteso : {expected_sha256}\n"
            f"  Ottenuto: {actual_sha256}\n"
            "Esecuzione interrotta per impedire la miscelazione di dataset differenti."
        )
    return actual_sha256


def load_or_generate_verified_clean_target(
    master: Any,
    twin_cfg: Dict[str, Any],
    obs_op: DiscreteObservationOperator,
    theta_dagger: np.ndarray,
    out_dir: Path,
    expected_sha256: str = EXPECTED_TARGET_SHA256,
) -> Tuple[np.ndarray, str, float, str]:
    """
    Riutilizza il vettore target pulito se già salvato su disco verificandone l'hash SHA-256;
    altrimenti lo rigenera con (N=32, M=160), verifica esattamente l'hash e lo salva in cache.
    """
    candidate_paths = [
        out_dir / "y_target_clean.npy",
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
            target_cache = out_dir / "y_target_clean.npy"
            if cpath.resolve() != target_cache.resolve():
                np.save(target_cache, y_loaded)
            return y_loaded, sha, 0.0, f"reused_saved_vector:{cpath}"

    tar_cfg = twin_cfg["target_discretization"]
    n_tar = int(tar_cfg["subdivisions_N"])
    m_tar = int(tar_cfg["time_steps_M"])
    if (n_tar, m_tar) != (32, 160):
        raise RuntimeError(
            f"Discretizzazione target non conforme: (N={n_tar}, M={m_tar}) != (32, 160)."
        )

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
    t0 = time.perf_counter()
    traj_tar = prob_tar.solve_trajectory(
        theta_dagger, c0=c0_tar, F_mech_func=F_tar, S_fluid_func=Sf_tar
    )
    y_target_clean = np.asarray(obs_op.apply(prob_tar, traj_tar), dtype=np.float64).ravel()
    wall_s = time.perf_counter() - t0

    sha = verify_target_hash_or_abort(
        y_target_clean,
        expected_sha256=expected_sha256,
        context_label="regenerated_N32_M160",
    )
    np.save(out_dir / "y_target_clean.npy", y_target_clean)
    return y_target_clean, sha, wall_s, "regenerated_and_hash_verified_N32_M160"


def verify_single_run_record_consistency(
    rec: SingleRunDiagnosticRecord,
    *,
    expected_budget: int = PLANNED_FEM_BUDGET,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-14,
) -> Dict[str, Any]:
    """
    Verifica tutti i controlli di coerenza richiesti dal Punto 4 della Fase 3.7 su un
    `SingleRunDiagnosticRecord` (sia riutilizzato da Fase 3.6 sia appena eseguito).
    """
    n_hist = len(rec.history)
    exact_600 = (
        n_hist == expected_budget
        and int(rec.optimizer_fem_evaluations) == expected_budget
        and int(rec.planned_fem_budget) == expected_budget
    )
    seq_indices = all(
        int(row.eval_index) == i + 1 for i, row in enumerate(rec.history)
    )

    decomp_errs = [
        abs(float(row.j_total) - (float(row.j_data) + float(row.j_reg)))
        for row in rec.history
    ]
    max_decomp_err = float(max(decomp_errs)) if decomp_errs else float("inf")
    j_sum_ok = all(
        np.isclose(
            float(row.j_total),
            float(row.j_data) + float(row.j_reg),
            rtol=rtol,
            atol=atol,
        )
        for row in rec.history
    ) and bool(
        np.isclose(
            float(rec.best_cost_J_total),
            float(rec.best_cost_J_data) + float(rec.best_cost_J_reg),
            rtol=rtol,
            atol=atol,
        )
    )

    j_totals = [float(row.j_total) for row in rec.history]
    best_so_far = [float(row.best_j_so_far) for row in rec.history]
    monotone_ok = all(
        best_so_far[k] <= best_so_far[k - 1] + atol for k in range(1, n_hist)
    ) and all(
        np.isclose(best_so_far[k], min(j_totals[: k + 1]), rtol=rtol, atol=atol)
        for k in range(n_hist)
    )

    min_pos = int(np.argmin(j_totals))
    min_row = rec.history[min_pos]
    min_matches_solution = bool(
        int(min_row.eval_index) == int(rec.best_eval_index)
        and np.isclose(float(min_row.j_total), float(rec.best_cost_J_total), rtol=rtol, atol=atol)
        and np.isclose(float(min_row.j_data), float(rec.best_cost_J_data), rtol=rtol, atol=atol)
        and np.isclose(float(min_row.j_reg), float(rec.best_cost_J_reg), rtol=rtol, atol=atol)
        and np.allclose(min_row.theta, rec.theta_hat, rtol=1.0e-10, atol=1.0e-12)
        and np.isclose(best_so_far[-1], float(rec.best_cost_J_total), rtol=rtol, atol=atol)
    )

    all_passed = bool(
        exact_600
        and seq_indices
        and j_sum_ok
        and monotone_ok
        and min_matches_solution
    )
    report = {
        "all_checks_passed": all_passed,
        "history_length_equals_600": exact_600,
        "sequential_eval_indices_1_to_600": seq_indices,
        "num_fem_evaluations": int(rec.optimizer_fem_evaluations),
        "history_length": int(n_hist),
        "j_total_equals_j_data_plus_j_reg": j_sum_ok,
        "max_abs_j_decomposition_error": max_decomp_err,
        "best_cost_monotone_non_increasing": monotone_ok,
        "historical_min_matches_returned_solution": min_matches_solution,
        "best_eval_index": int(rec.best_eval_index),
        "best_cost_J_total": float(rec.best_cost_J_total),
    }
    if not all_passed:
        raise RuntimeError(
            f"Controllo di coerenza fallito per run_id={rec.run_id} (seed={rec.optimizer_seed}): {report}"
        )
    return report


def load_reused_seed_20261017_record(
    phase36_summary_path: Path,
    expected_target_sha256: str = EXPECTED_TARGET_SHA256,
) -> SingleRunDiagnosticRecord:
    """
    Carica la run già eseguita con seed CPSO 20261017 da
    `artifacts/06_cpso_budget_diagnostic/cpso_budget_diagnostic_summary.json`
    senza rieseguirla, verificando l'hash del target, i metadati e i valori di riferimento.
    """
    if not phase36_summary_path.exists():
        raise FileNotFoundError(
            f"Artefatto Fase 3.6 non trovato in {phase36_summary_path}. "
            "Impossibile riutilizzare la run con seed 20261017."
        )

    payload = json.loads(phase36_summary_path.read_text(encoding="utf-8"))
    prov = payload.get("target_provenance", {})
    recorded_sha = str(prov.get("y_target_clean_sha256", ""))
    if recorded_sha != expected_target_sha256:
        raise RuntimeError(
            f"Hash del target in {phase36_summary_path} ({recorded_sha}) "
            f"diverso da quello atteso ({expected_target_sha256})."
        )

    full_rec = payload["diagnostic_600_full_record"]
    opt_seed = int(full_rec["optimizer_seed"])
    data_seed = int(full_rec["data_seed"])
    if opt_seed != 20261017 or data_seed != FIXED_DATA_SEED:
        raise RuntimeError(
            f"Seed inattesi nell'artefatto Fase 3.6: optimizer_seed={opt_seed}, data_seed={data_seed}."
        )

    conv = full_rec["convergence_history"]
    eval_indices = conv["eval_index"]
    j_totals = conv["j_total"]
    j_datas = conv["j_data"]
    j_regs = conv["j_reg"]
    best_j_so_fars = conv["best_j_so_far"]
    cand_thetas = conv["candidate_theta"]
    best_thetas = conv["best_theta_so_far"]

    history_rows: List[EvaluationHistoryRow] = []
    for i in range(len(eval_indices)):
        history_rows.append(
            EvaluationHistoryRow(
                eval_index=int(eval_indices[i]),
                theta=np.asarray(cand_thetas[i], dtype=np.float64).ravel(),
                j_total=float(j_totals[i]),
                j_data=float(j_datas[i]),
                j_reg=float(j_regs[i]),
                best_j_so_far=float(best_j_so_fars[i]),
                best_theta_so_far=np.asarray(best_thetas[i], dtype=np.float64).ravel(),
            )
        )

    theta_true_vec = np.asarray(full_rec["theta_true_vector"], dtype=np.float64).ravel()
    theta_hat_vec = np.asarray(full_rec["theta_hat_vector"], dtype=np.float64).ravel()
    theta_tilde_hat_vec = (
        np.asarray(full_rec["theta_tilde_hat_vector"], dtype=np.float64).ravel()
        if full_rec.get("theta_tilde_hat_vector") is not None
        else None
    )
    rel_errs_vec = np.asarray(
        full_rec["individual_relative_errors_pct_vector"], dtype=np.float64
    ).ravel()

    # Verifica esatta rispetto ai valori di riferimento certificati per il seed 20261017
    if not np.isclose(
        float(full_rec["best_cost_J_total"]), REF_SEED_20261017["j_total"], rtol=1e-12, atol=1e-15
    ):
        raise RuntimeError(
            f"J_total in seed 20261017 ({full_rec['best_cost_J_total']}) non coincide "
            f"col riferimento ({REF_SEED_20261017['j_total']})."
        )
    if not np.isclose(
        float(full_rec["best_cost_J_data"]), REF_SEED_20261017["j_data"], rtol=1e-12, atol=1e-15
    ):
        raise RuntimeError(
            f"J_data in seed 20261017 ({full_rec['best_cost_J_data']}) non coincide "
            f"col riferimento ({REF_SEED_20261017['j_data']})."
        )
    if not np.isclose(
        float(full_rec["best_cost_J_reg"]), REF_SEED_20261017["j_reg"], rtol=1e-12, atol=1e-15
    ):
        raise RuntimeError(
            f"J_reg in seed 20261017 ({full_rec['best_cost_J_reg']}) non coincide "
            f"col riferimento ({REF_SEED_20261017['j_reg']})."
        )
    if not np.allclose(
        theta_hat_vec, np.asarray(REF_SEED_20261017["theta_hat"], dtype=np.float64), rtol=1e-12, atol=1e-14
    ):
        raise RuntimeError(
            f"theta_hat in seed 20261017 ({theta_hat_vec}) non coincide "
            f"col riferimento ({REF_SEED_20261017['theta_hat']})."
        )

    rec = SingleRunDiagnosticRecord(
        run_id="cpso_600_seed_20261017",
        optimizer_seed=opt_seed,
        data_seed=data_seed,
        eta_level=float(full_rec["eta_level"]),
        theta_true=theta_true_vec,
        theta_hat=theta_hat_vec,
        individual_relative_errors_pct=rel_errs_vec,
        best_cost_J_total=float(full_rec["best_cost_J_total"]),
        best_cost_J_data=float(full_rec["best_cost_J_data"]),
        best_cost_J_reg=float(full_rec["best_cost_J_reg"]),
        best_eval_index=int(full_rec["best_eval_index"]),
        nominal_noise_std=float(full_rec["nominal_noise_std"]),
        effective_relative_noise_norm=float(full_rec["effective_relative_noise_norm"]),
        relative_residual_vs_noisy_data=float(full_rec["relative_residual_vs_noisy_data"]),
        relative_residual_vs_clean_target=float(full_rec["relative_residual_vs_clean_target"]),
        planned_fem_budget=int(full_rec["planned_fem_budget"]),
        optimizer_fem_evaluations=int(full_rec["optimizer_fem_evaluations"]),
        diagnostic_fem_evaluations=int(full_rec["diagnostic_fem_evaluations"]),
        termination_condition=str(full_rec["termination_condition"]),
        wall_time_seconds=float(full_rec["wall_time_seconds"]),
        history=history_rows,
        objective_mode=str(full_rec.get("objective_mode", "mode_a_deterministic_normalized_linf")),
        theta_tilde_hat=theta_tilde_hat_vec,
        mean_relative_error_pct=float(full_rec["mean_relative_error_pct"]),
        max_relative_error_pct=float(full_rec["max_relative_error_pct"]),
        unweighted_residual_l2_norm=float(full_rec["unweighted_residual_l2_norm"]),
        unweighted_residual_linf_norm=float(full_rec["unweighted_residual_linf_norm"]),
        weighted_residual_l2_norm=float(full_rec["weighted_residual_l2_norm"]),
        consistency_checks=dict(full_rec.get("consistency_checks", {})),
    )

    checks = verify_single_run_record_consistency(rec, expected_budget=PLANNED_FEM_BUDGET)
    checks["reused_from_phase36_without_rerun"] = True
    checks["source_artifact"] = str(phase36_summary_path)
    rec.consistency_checks.update(checks)
    return rec


def execute_single_cpso_600_run(
    *,
    optimizer_seed: int,
    data_seed: int,
    y_target_clean: np.ndarray,
    expected_target_sha256: str,
    forward_operator_inv: Callable[[np.ndarray], np.ndarray],
    transform: ParameterCoordinateTransform,
    regularizer: TikhonovRegularizer,
    theta_dagger: np.ndarray,
) -> SingleRunDiagnosticRecord:
    """
    Esegue una nuova ottimizzazione CPSO a 600 valutazioni FEM (10 particelle x 60 iterazioni),
    verificando l'hash del target prima dell'avvio e tutti i controlli di coerenza al termine.
    """
    verify_target_hash_or_abort(
        y_target_clean,
        expected_sha256=expected_target_sha256,
        context_label=f"pre_run_seed_{optimizer_seed}",
    )

    sample_0 = generate_gaussian_noisy_observations(
        y_target_clean,
        eta_level=0.0,
        seed=data_seed,
        weighting_mode="mode_a_deterministic_normalized",
        reference_scale_type="linf",
    )

    obj = InverseObjectiveFunctional.from_mode_a_deterministic(
        forward_operator_inv,
        sample_0.y_noisy,
        transform,
        regularizer,
        y_reference=sample_0.y_noisy,
        allow_observed_as_reference=True,
        scale_type="linf",
        max_evaluations=PLANNED_FEM_BUDGET,
    )
    if obj.max_evaluations != PLANNED_FEM_BUDGET:
        raise RuntimeError(
            f"max_evaluations disallineato: {obj.max_evaluations} != {PLANNED_FEM_BUDGET}."
        )

    t0 = time.perf_counter()
    opt_res = run_cpso_optimization(
        obj,
        seed=optimizer_seed,
        num_particles=NUM_PARTICLES,
        max_iterations=MAX_ITERATIONS,
    )
    wall_s = time.perf_counter() - t0

    rec = extract_run_diagnostics_from_history(
        run_id=f"cpso_600_seed_{optimizer_seed}",
        obj=obj,
        opt_result=opt_res,
        theta_true=theta_dagger,
        y_target_clean=y_target_clean,
        y_noisy=sample_0.y_noisy,
        nominal_noise_std=0.0,
        eta_level=0.0,
        data_seed=data_seed,
        optimizer_seed=optimizer_seed,
        planned_fem_budget=PLANNED_FEM_BUDGET,
        num_particles=NUM_PARTICLES,
        max_iterations=MAX_ITERATIONS,
        wall_time_seconds=wall_s,
        forward_operator_inv=forward_operator_inv,
        transform=transform,
        regularizer=regularizer,
    )
    checks = verify_single_run_record_consistency(rec, expected_budget=PLANNED_FEM_BUDGET)
    checks["reused_from_phase36_without_rerun"] = False
    rec.consistency_checks.update(checks)
    return rec


def plot_multiseed_convergence_figure(
    records: Sequence[SingleRunDiagnosticRecord],
    j_total_at_theta_dagger: float,
    out_dir: Path,
) -> Tuple[Path, Path]:
    """
    Genera la figura comparativa delle tre curve del miglior costo `best_j_so_far`
    rispetto alle valutazioni FEM (1..600) in scala y logaritmica, indicando il
    riferimento diagnostico separato J_total(theta_dagger).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "fig_cpso_multiseed_convergence.png"
    pdf_path = out_dir / "fig_cpso_multiseed_convergence.pdf"

    fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=300)
    palette = ["#0072B2", "#D55E00", "#009E73"]
    linestyles = ["-", "--", "-."]

    for idx, rec in enumerate(records):
        evals = np.array([r.eval_index for r in rec.history], dtype=np.int32)
        best_j = np.array([r.best_j_so_far for r in rec.history], dtype=np.float64)
        color = palette[idx % len(palette)]
        ls = linestyles[idx % len(linestyles)]
        label = (
            f"CPSO seed {rec.optimizer_seed} "
            f"(min $J_{{\\mathrm{{total}}}}={rec.best_cost_J_total:.3e}$ @ val. {rec.best_eval_index})"
        )
        ax.plot(evals, best_j, color=color, linestyle=ls, linewidth=2.1, label=label)
        ax.scatter(
            [rec.best_eval_index],
            [rec.best_cost_J_total],
            color=color,
            s=46,
            zorder=5,
            edgecolors="black",
            linewidths=0.6,
        )

    ax.axhline(
        j_total_at_theta_dagger,
        color="#333333",
        linestyle=":",
        linewidth=1.9,
        label=(
            f"Riferimento diagnostico fuori sciame $J_{{\\mathrm{{total}}}}(\\theta^\\dagger) = "
            f"{j_total_at_theta_dagger:.3e}$"
        ),
    )

    ax.set_yscale("log")
    ax.set_xlim(1, PLANNED_FEM_BUDGET)
    ax.set_xlabel("Valutazioni FEM del modello diretto ($N=16,\\ M=80$)", fontsize=11)
    ax.set_ylabel("Miglior costo cumulativo $J_{\\mathrm{total}}$ (scala log)", fontsize=11)
    ax.set_title(
        "RETECO Fase 3.7 — Convergenza CPSO su 3 seed a budget fissato (600 valutazioni, $\\eta=0$)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.55)
    ax.legend(loc="upper right", fontsize=9.2, frameon=True)
    fig.tight_layout()

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def write_multiseed_comparison_csv(
    records: Sequence[SingleRunDiagnosticRecord],
    target_sha256: str,
    j_total_at_true: float,
    j_data_at_true: float,
    j_reg_at_true: float,
    csv_path: Path,
) -> List[Dict[str, Any]]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "optimizer_seed",
        "data_seed",
        "reused_from_phase36",
        "target_sha256",
        "N_target",
        "M_target",
        "N_inv",
        "M_inv",
        "objective_mode",
        "lambda_reg",
        "planned_fem_budget",
        "optimizer_fem_evaluations",
        "best_eval_index",
        "best_cost_J_total",
        "best_cost_J_data",
        "best_cost_J_reg",
        "data_share_pct",
        "reg_share_pct",
        "J_total_at_theta_dagger",
        "J_data_at_theta_dagger",
        "J_reg_at_theta_dagger",
        "ratio_J_total_over_J_true",
        "diff_J_total_minus_J_true",
        "weighted_residual_l2_norm",
        "relative_residual_vs_clean_target",
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
        "mean_relative_error_pct",
        "max_relative_error_pct",
        "consistency_all_checks_passed",
        "wall_time_seconds",
    ]
    rows: List[Dict[str, Any]] = []
    for rec in records:
        data_share = (
            100.0 * rec.best_cost_J_data / rec.best_cost_J_total
            if rec.best_cost_J_total > 0.0
            else 0.0
        )
        reg_share = (
            100.0 * rec.best_cost_J_reg / rec.best_cost_J_total
            if rec.best_cost_J_total > 0.0
            else 0.0
        )
        row = {
            "run_id": rec.run_id,
            "optimizer_seed": rec.optimizer_seed,
            "data_seed": rec.data_seed,
            "reused_from_phase36": bool(
                rec.consistency_checks.get("reused_from_phase36_without_rerun", False)
            ),
            "target_sha256": target_sha256,
            "N_target": 32,
            "M_target": 160,
            "N_inv": 16,
            "M_inv": 80,
            "objective_mode": rec.objective_mode,
            "lambda_reg": FIXED_LAMBDA_REG,
            "planned_fem_budget": rec.planned_fem_budget,
            "optimizer_fem_evaluations": rec.optimizer_fem_evaluations,
            "best_eval_index": rec.best_eval_index,
            "best_cost_J_total": rec.best_cost_J_total,
            "best_cost_J_data": rec.best_cost_J_data,
            "best_cost_J_reg": rec.best_cost_J_reg,
            "data_share_pct": data_share,
            "reg_share_pct": reg_share,
            "J_total_at_theta_dagger": j_total_at_true,
            "J_data_at_theta_dagger": j_data_at_true,
            "J_reg_at_theta_dagger": j_reg_at_true,
            "ratio_J_total_over_J_true": float(rec.best_cost_J_total / j_total_at_true),
            "diff_J_total_minus_J_true": float(rec.best_cost_J_total - j_total_at_true),
            "weighted_residual_l2_norm": rec.weighted_residual_l2_norm,
            "relative_residual_vs_clean_target": rec.relative_residual_vs_clean_target,
            "alpha_true": float(rec.theta_true[0]),
            "beta_true": float(rec.theta_true[1]),
            "D_true": float(rec.theta_true[2]),
            "E_true": float(rec.theta_true[3]),
            "alpha_hat": float(rec.theta_hat[0]),
            "beta_hat": float(rec.theta_hat[1]),
            "D_hat": float(rec.theta_hat[2]),
            "E_hat": float(rec.theta_hat[3]),
            "err_rel_alpha_pct": float(rec.individual_relative_errors_pct[0]),
            "err_rel_beta_pct": float(rec.individual_relative_errors_pct[1]),
            "err_rel_D_pct": float(rec.individual_relative_errors_pct[2]),
            "err_rel_E_pct": float(rec.individual_relative_errors_pct[3]),
            "mean_relative_error_pct": rec.mean_relative_error_pct,
            "max_relative_error_pct": rec.max_relative_error_pct,
            "consistency_all_checks_passed": bool(
                rec.consistency_checks.get("all_checks_passed", False)
            ),
            "wall_time_seconds": rec.wall_time_seconds,
        }
        rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def compute_interseed_statistics(
    records: Sequence[SingleRunDiagnosticRecord],
    j_total_at_true: float,
) -> Dict[str, Any]:
    alphas = np.array([r.theta_hat[0] for r in records], dtype=np.float64)
    betas = np.array([r.theta_hat[1] for r in records], dtype=np.float64)
    Ds = np.array([r.theta_hat[2] for r in records], dtype=np.float64)
    Es = np.array([r.theta_hat[3] for r in records], dtype=np.float64)

    err_alpha = np.array([r.individual_relative_errors_pct[0] for r in records], dtype=np.float64)
    err_beta = np.array([r.individual_relative_errors_pct[1] for r in records], dtype=np.float64)
    err_D = np.array([r.individual_relative_errors_pct[2] for r in records], dtype=np.float64)
    err_E = np.array([r.individual_relative_errors_pct[3] for r in records], dtype=np.float64)

    j_totals = np.array([r.best_cost_J_total for r in records], dtype=np.float64)
    j_datas = np.array([r.best_cost_J_data for r in records], dtype=np.float64)
    j_regs = np.array([r.best_cost_J_reg for r in records], dtype=np.float64)
    rel_res = np.array([r.relative_residual_vs_clean_target for r in records], dtype=np.float64)
    best_evals = np.array([r.best_eval_index for r in records], dtype=np.float64)

    def _stat_block(arr: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(arr)),
            "std_ddof1": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "span_max_minus_min": float(np.max(arr) - np.min(arr)),
        }

    return {
        "parameters_hat": {
            "alpha": _stat_block(alphas),
            "beta": _stat_block(betas),
            "D": _stat_block(Ds),
            "E": _stat_block(Es),
        },
        "relative_errors_pct": {
            "alpha": _stat_block(err_alpha),
            "beta": _stat_block(err_beta),
            "D": _stat_block(err_D),
            "E": _stat_block(err_E),
        },
        "costs_and_residuals": {
            "J_total": _stat_block(j_totals),
            "J_data": _stat_block(j_datas),
            "J_reg": _stat_block(j_regs),
            "ratio_J_total_over_J_true": _stat_block(j_totals / j_total_at_true),
            "relative_residual_vs_clean_target": _stat_block(rel_res),
            "best_eval_index": _stat_block(best_evals),
        },
    }


def build_side_by_side_markdown_report(
    records: Sequence[SingleRunDiagnosticRecord],
    stats: Dict[str, Any],
    target_sha256: str,
    target_source: str,
    discretization_rel_gap: float,
    bd_at_true: Any,
    theta_dagger: np.ndarray,
) -> str:
    r0, r1, r2 = records[0], records[1], records[2]
    j_true_tot = float(bd_at_true.j_total)
    j_true_dat = float(bd_at_true.j_data)
    j_true_reg = float(bd_at_true.j_reg)
    res_true_rel = float(bd_at_true.relative_residual_l2)

    lines: List[str] = [
        "# RETECO — Fase 3.7: Report Comparativo Variabilità Multi-Seed CPSO a 600 Valutazioni",
        "",
        "## 1. Configurazione Congelata e Provenienza del Target",
        f"- **Target sintetico**: `N = 32, M = 160` (`T = 1.0, r = 1.5`)",
        f"- **Discretizzazione inversa**: `N = 16, M = 80` (`T = 1.0, r = 1.5`)",
        f"- **Hash SHA-256 del vettore target pulito (`y_target_clean`)**: `{target_sha256}`",
        f"- **Sorgente del target**: `{target_source}` (corrispondenza esatta verificata prima di ogni run)",
        f"- **Gap relativo di discretizzazione** $\\|G_{{h_{{\\mathrm{{tar}}}}}}(\\theta^\\dagger) - G_{{h_{{\\mathrm{{inv}}}}}}(\\theta^\\dagger)\\|_2 / \\|y_{{\\mathrm{{clean}}}}\\|_2$: `{discretization_rel_gap:.6e}`",
        f"- **Modalità obiettivo**: `{r0.objective_mode}` ($\\eta = 0.0$, $\\lambda = {FIXED_LAMBDA_REG:.1e}$)",
        f"- **Seed dati fissato**: `{FIXED_DATA_SEED}`",
        f"- **Seed CPSO confrontati**: `{r0.optimizer_seed}` (riutilizzato da Fase 3.6), `{r1.optimizer_seed}` (nuova esecuzione), `{r2.optimizer_seed}` (nuova esecuzione)",
        f"- **Budget per ciascuna run**: `{PLANNED_FEM_BUDGET}` valutazioni FEM (`{NUM_PARTICLES}` particelle $\\times$ `{MAX_ITERATIONS}` iterazioni)",
        "",
        "## 2. Risultati delle Tre Run Affiancati (Senza Selezione)",
        "| Metrica / Parametro | Riferimento in $\\theta^\\dagger$ (fuori sciame) | Run 1 (`seed = 20261017`, riutilizzata) | Run 2 (`seed = 20261018`, nuova) | Run 3 (`seed = 20261019`, nuova) | Media $\\pm$ Dev.Std. (3 seed) | Min $\\div$ Max (3 seed) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| **Valutazioni FEM ottimizzatore** | `0` (isolata) | `{r0.optimizer_fem_evaluations}` | `{r1.optimizer_fem_evaluations}` | `{r2.optimizer_fem_evaluations}` | `600.0 ± 0.0` | `600 ÷ 600` |",
        f"| **Indice miglior candidato (`best_eval_index`)** | — | `{r0.best_eval_index}` | `{r1.best_eval_index}` | `{r2.best_eval_index}` | `{stats['costs_and_residuals']['best_eval_index']['mean']:.1f} ± {stats['costs_and_residuals']['best_eval_index']['std_ddof1']:.1f}` | `{int(stats['costs_and_residuals']['best_eval_index']['min'])} ÷ {int(stats['costs_and_residuals']['best_eval_index']['max'])}` |",
        f"| **$J_{{\\mathrm{{total}}}}$** | `{j_true_tot:.6e}` | `{r0.best_cost_J_total:.6e}` | `{r1.best_cost_J_total:.6e}` | `{r2.best_cost_J_total:.6e}` | `{stats['costs_and_residuals']['J_total']['mean']:.4e} ± {stats['costs_and_residuals']['J_total']['std_ddof1']:.4e}` | `{stats['costs_and_residuals']['J_total']['min']:.4e} ÷ {stats['costs_and_residuals']['J_total']['max']:.4e}` |",
        f"| **$J_{{\\mathrm{{data}}}}$** | `{j_true_dat:.6e}` | `{r0.best_cost_J_data:.6e}` | `{r1.best_cost_J_data:.6e}` | `{r2.best_cost_J_data:.6e}` | `{stats['costs_and_residuals']['J_data']['mean']:.4e} ± {stats['costs_and_residuals']['J_data']['std_ddof1']:.4e}` | `{stats['costs_and_residuals']['J_data']['min']:.4e} ÷ {stats['costs_and_residuals']['J_data']['max']:.4e}` |",
        f"| **$J_{{\\mathrm{{reg}}}}$** | `{j_true_reg:.6e}` | `{r0.best_cost_J_reg:.6e}` | `{r1.best_cost_J_reg:.6e}` | `{r2.best_cost_J_reg:.6e}` | `{stats['costs_and_residuals']['J_reg']['mean']:.4e} ± {stats['costs_and_residuals']['J_reg']['std_ddof1']:.4e}` | `{stats['costs_and_residuals']['J_reg']['min']:.4e} ÷ {stats['costs_and_residuals']['J_reg']['max']:.4e}` |",
        f"| **Rapporto $J_{{\\mathrm{{total}}}} / J_{{\\mathrm{{total}}}}(\\theta^\\dagger)$** | `1.0000` | `{r0.best_cost_J_total / j_true_tot:.4f}` | `{r1.best_cost_J_total / j_true_tot:.4f}` | `{r2.best_cost_J_total / j_true_tot:.4f}` | `{stats['costs_and_residuals']['ratio_J_total_over_J_true']['mean']:.4f} ± {stats['costs_and_residuals']['ratio_J_total_over_J_true']['std_ddof1']:.4f}` | `{stats['costs_and_residuals']['ratio_J_total_over_J_true']['min']:.4f} ÷ {stats['costs_and_residuals']['ratio_J_total_over_J_true']['max']:.4f}` |",
        f"| **Residuo relativo $\\|G_{{h_{{\\mathrm{{inv}}}}}}(\\hat{{\\theta}})-y\\|_2/\\|y\\|_2$** | `{res_true_rel:.6e}` | `{r0.relative_residual_vs_clean_target:.6e}` | `{r1.relative_residual_vs_clean_target:.6e}` | `{r2.relative_residual_vs_clean_target:.6e}` | `{stats['costs_and_residuals']['relative_residual_vs_clean_target']['mean']:.4e} ± {stats['costs_and_residuals']['relative_residual_vs_clean_target']['std_ddof1']:.4e}` | `{stats['costs_and_residuals']['relative_residual_vs_clean_target']['min']:.4e} ÷ {stats['costs_and_residuals']['relative_residual_vs_clean_target']['max']:.4e}` |",
        f"| **$\\hat{{\\alpha}}$ (vero $\\alpha^\\dagger = {theta_dagger[0]:.2f}$)** | `{theta_dagger[0]:.6f}` | `{r0.theta_hat[0]:.6f}` | `{r1.theta_hat[0]:.6f}` | `{r2.theta_hat[0]:.6f}` | `{stats['parameters_hat']['alpha']['mean']:.6f} ± {stats['parameters_hat']['alpha']['std_ddof1']:.6f}` | `{stats['parameters_hat']['alpha']['min']:.6f} ÷ {stats['parameters_hat']['alpha']['max']:.6f}` |",
        f"| **Errore rel. $\\alpha$ [%]** | `0.00%` | `{r0.individual_relative_errors_pct[0]:.2f}%` | `{r1.individual_relative_errors_pct[0]:.2f}%` | `{r2.individual_relative_errors_pct[0]:.2f}%` | `{stats['relative_errors_pct']['alpha']['mean']:.2f}% ± {stats['relative_errors_pct']['alpha']['std_ddof1']:.2f}%` | `{stats['relative_errors_pct']['alpha']['min']:.2f}% ÷ {stats['relative_errors_pct']['alpha']['max']:.2f}%` |",
        f"| **$\\hat{{\\beta}}$ (vero $\\beta^\\dagger = {theta_dagger[1]:.2f}$)** | `{theta_dagger[1]:.6f}` | `{r0.theta_hat[1]:.6f}` | `{r1.theta_hat[1]:.6f}` | `{r2.theta_hat[1]:.6f}` | `{stats['parameters_hat']['beta']['mean']:.6f} ± {stats['parameters_hat']['beta']['std_ddof1']:.6f}` | `{stats['parameters_hat']['beta']['min']:.6f} ÷ {stats['parameters_hat']['beta']['max']:.6f}` |",
        f"| **Errore rel. $\\beta$ [%]** | `0.00%` | `{r0.individual_relative_errors_pct[1]:.2f}%` | `{r1.individual_relative_errors_pct[1]:.2f}%` | `{r2.individual_relative_errors_pct[1]:.2f}%` | `{stats['relative_errors_pct']['beta']['mean']:.2f}% ± {stats['relative_errors_pct']['beta']['std_ddof1']:.2f}%` | `{stats['relative_errors_pct']['beta']['min']:.2f}% ÷ {stats['relative_errors_pct']['beta']['max']:.2f}%` |",
        f"| **$\\hat{{D}}$ (vero $D^\\dagger = {theta_dagger[2]:.2f}$)** | `{theta_dagger[2]:.6f}` | `{r0.theta_hat[2]:.6f}` | `{r1.theta_hat[2]:.6f}` | `{r2.theta_hat[2]:.6f}` | `{stats['parameters_hat']['D']['mean']:.6f} ± {stats['parameters_hat']['D']['std_ddof1']:.6f}` | `{stats['parameters_hat']['D']['min']:.6f} ÷ {stats['parameters_hat']['D']['max']:.6f}` |",
        f"| **Errore rel. $D$ [%]** | `0.00%` | `{r0.individual_relative_errors_pct[2]:.2f}%` | `{r1.individual_relative_errors_pct[2]:.2f}%` | `{r2.individual_relative_errors_pct[2]:.2f}%` | `{stats['relative_errors_pct']['D']['mean']:.2f}% ± {stats['relative_errors_pct']['D']['std_ddof1']:.2f}%` | `{stats['relative_errors_pct']['D']['min']:.2f}% ÷ {stats['relative_errors_pct']['D']['max']:.2f}%` |",
        f"| **$\\hat{{E}}$ (vero $E^\\dagger = {theta_dagger[3]:.2f}$)** | `{theta_dagger[3]:.6f}` | `{r0.theta_hat[3]:.6f}` | `{r1.theta_hat[3]:.6f}` | `{r2.theta_hat[3]:.6f}` | `{stats['parameters_hat']['E']['mean']:.6f} ± {stats['parameters_hat']['E']['std_ddof1']:.6f}` | `{stats['parameters_hat']['E']['min']:.6f} ÷ {stats['parameters_hat']['E']['max']:.6f}` |",
        f"| **Errore rel. $E$ [%]** | `0.00%` | `{r0.individual_relative_errors_pct[3]:.2f}%` | `{r1.individual_relative_errors_pct[3]:.2f}%` | `{r2.individual_relative_errors_pct[3]:.2f}%` | `{stats['relative_errors_pct']['E']['mean']:.2f}% ± {stats['relative_errors_pct']['E']['std_ddof1']:.2f}%` | `{stats['relative_errors_pct']['E']['min']:.2f}% ÷ {stats['relative_errors_pct']['E']['max']:.2f}%` |",
        f"| **Controlli di coerenza (`all_checks_passed`)** | — | `{r0.consistency_checks['all_checks_passed']}` | `{r1.consistency_checks['all_checks_passed']}` | `{r2.consistency_checks['all_checks_passed']}` | `3/3 True` | `True` |",
        f"| **Tempo di esecuzione [s]** | — | `{r0.wall_time_seconds:.2f}` | `{r1.wall_time_seconds:.2f}` | `{r2.wall_time_seconds:.2f}` | — | — |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    out_dir = Path("artifacts/07_cpso_multiseed")
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # 1. Caricamento o rigenerazione del target (N=32, M=160) con verifica tassativa dell'hash SHA-256
    y_target_clean, verified_sha256, target_wall_s, target_source = (
        load_or_generate_verified_clean_target(
            master=master,
            twin_cfg=twin_cfg,
            obs_op=obs_op,
            theta_dagger=theta_dagger,
            out_dir=out_dir,
            expected_sha256=EXPECTED_TARGET_SHA256,
        )
    )
    print(f"[Fase 3.7] Target verificato: SHA-256={verified_sha256} ({target_source})")

    # 2. Costruzione del problema diretto inverso su (N=16, M=80)
    inv_cfg = twin_cfg["inverse_discretization"]
    n_inv = int(inv_cfg["subdivisions_N"])
    m_inv = int(inv_cfg["time_steps_M"])
    if (n_inv, m_inv) != (16, 80):
        raise RuntimeError(
            f"Discretizzazione inversa non conforme: (N={n_inv}, M={m_inv}) != (16, 80)."
        )

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
    if data_seed != FIXED_DATA_SEED or not np.isclose(lambda_reg, FIXED_LAMBDA_REG):
        raise RuntimeError(
            f"Configurazione rumore/regolarizzazione alterata: data_seed={data_seed}, lambda={lambda_reg}."
        )
    prior_theta_tilde_0 = np.asarray(noise_cfg["prior_theta_tilde_0"], dtype=np.float64).ravel()

    reg = TikhonovRegularizer(
        transform=transform,
        lambda_reg=lambda_reg,
        theta_tilde_0=prior_theta_tilde_0,
    )

    # 3. Riutilizzo certificato della run con seed 20261017 (senza riesecuzione)
    phase36_summary_path = Path(
        "artifacts/06_cpso_budget_diagnostic/cpso_budget_diagnostic_summary.json"
    )
    rec_20261017 = load_reused_seed_20261017_record(
        phase36_summary_path, expected_target_sha256=EXPECTED_TARGET_SHA256
    )
    print(
        f"[Fase 3.7] Run seed=20261017 riutilizzata e verificata: "
        f"J_total={rec_20261017.best_cost_J_total:.6e}, beta_hat={rec_20261017.theta_hat[1]:.6f}"
    )

    # 4. Esecuzione delle due nuove run con seed CPSO 20261018 e 20261019
    rec_20261018 = execute_single_cpso_600_run(
        optimizer_seed=20261018,
        data_seed=FIXED_DATA_SEED,
        y_target_clean=y_target_clean,
        expected_target_sha256=EXPECTED_TARGET_SHA256,
        forward_operator_inv=G_inv,
        transform=transform,
        regularizer=reg,
        theta_dagger=theta_dagger,
    )
    print(
        f"[Fase 3.7] Run seed=20261018 completata: "
        f"J_total={rec_20261018.best_cost_J_total:.6e}, beta_hat={rec_20261018.theta_hat[1]:.6f} "
        f"(err_beta={rec_20261018.individual_relative_errors_pct[1]:.2f}%)"
    )

    rec_20261019 = execute_single_cpso_600_run(
        optimizer_seed=20261019,
        data_seed=FIXED_DATA_SEED,
        y_target_clean=y_target_clean,
        expected_target_sha256=EXPECTED_TARGET_SHA256,
        forward_operator_inv=G_inv,
        transform=transform,
        regularizer=reg,
        theta_dagger=theta_dagger,
    )
    print(
        f"[Fase 3.7] Run seed=20261019 completata: "
        f"J_total={rec_20261019.best_cost_J_total:.6e}, beta_hat={rec_20261019.theta_hat[1]:.6f} "
        f"(err_beta={rec_20261019.individual_relative_errors_pct[1]:.2f}%)"
    )

    records: List[SingleRunDiagnosticRecord] = [rec_20261017, rec_20261018, rec_20261019]

    # 5. Verifica globale multi-run (Punto 4)
    seeds_tuple = tuple(int(r.optimizer_seed) for r in records)
    if len(set(seeds_tuple)) != 3 or seeds_tuple != CPSO_SEEDS:
        raise RuntimeError(f"I seed dell'ottimizzatore non sono distinti/conformi: {seeds_tuple}")

    # 6. Valutazione diagnostica separata nei parametri veri theta_dagger (fuori dallo sciame)
    sample_ref = generate_gaussian_noisy_observations(
        y_target_clean,
        eta_level=0.0,
        seed=FIXED_DATA_SEED,
        weighting_mode="mode_a_deterministic_normalized",
        reference_scale_type="linf",
    )
    obj_diag_ref = InverseObjectiveFunctional.from_mode_a_deterministic(
        G_inv,
        sample_ref.y_noisy,
        transform,
        reg,
        y_reference=sample_ref.y_noisy,
        allow_observed_as_reference=True,
        scale_type="linf",
        max_evaluations=10,
    )
    y_inv_at_true = G_inv(theta_dagger)
    bd_at_true = obj_diag_ref.evaluate_from_prediction(
        theta_dagger, y_inv_at_true, record_in_history=False
    )
    assert len(obj_diag_ref.history) == 0
    assert obj_diag_ref.num_fem_evaluations == 0
    if not np.isclose(
        float(bd_at_true.j_total), REF_SEED_20261017["j_total_at_theta_dagger"], rtol=1e-12, atol=1e-15
    ):
        raise RuntimeError(
            f"Costo diagnostico in theta_dagger ({bd_at_true.j_total}) diverso dal "
            f"riferimento ({REF_SEED_20261017['j_total_at_theta_dagger']})."
        )

    discretization_rel_gap = float(
        np.linalg.norm(y_target_clean - y_inv_at_true) / np.linalg.norm(y_target_clean)
    )

    # 7. Salvataggio artefatti CSV, JSON, Figura (PNG + PDF) e Report Markdown
    comp_csv_path = out_dir / "cpso_multiseed_comparison.csv"
    hist_csv_path = out_dir / "cpso_multiseed_convergence_history.csv"
    summary_json_path = out_dir / "cpso_multiseed_summary.json"
    report_md_path = out_dir / "cpso_multiseed_report.md"

    comparison_rows = write_multiseed_comparison_csv(
        records=records,
        target_sha256=verified_sha256,
        j_total_at_true=float(bd_at_true.j_total),
        j_data_at_true=float(bd_at_true.j_data),
        j_reg_at_true=float(bd_at_true.j_reg),
        csv_path=comp_csv_path,
    )
    write_convergence_history_csv(records, hist_csv_path)
    png_path, pdf_path = plot_multiseed_convergence_figure(
        records, float(bd_at_true.j_total), out_dir
    )

    stats = compute_interseed_statistics(records, float(bd_at_true.j_total))
    report_md = build_side_by_side_markdown_report(
        records=records,
        stats=stats,
        target_sha256=verified_sha256,
        target_source=target_source,
        discretization_rel_gap=discretization_rel_gap,
        bd_at_true=bd_at_true,
        theta_dagger=theta_dagger,
    )
    report_md_path.write_text(report_md, encoding="utf-8")

    global_verification = {
        "all_three_histories_have_600_evaluations": all(len(r.history) == 600 for r in records),
        "all_three_best_costs_monotone_non_increasing": all(
            bool(r.consistency_checks["best_cost_monotone_non_increasing"]) for r in records
        ),
        "all_three_historical_minima_match_returned_solution": all(
            bool(r.consistency_checks["historical_min_matches_returned_solution"]) for r in records
        ),
        "all_three_satisfy_j_total_equals_j_data_plus_j_reg": all(
            bool(r.consistency_checks["j_total_equals_j_data_plus_j_reg"]) for r in records
        ),
        "seeds_are_distinct": len(set(seeds_tuple)) == 3,
        "optimizer_seeds": list(seeds_tuple),
        "fixed_data_seed": FIXED_DATA_SEED,
        "identical_target_sha256_verified": verified_sha256 == EXPECTED_TARGET_SHA256,
        "target_sha256": verified_sha256,
    }
    if not all(
        [
            global_verification["all_three_histories_have_600_evaluations"],
            global_verification["all_three_best_costs_monotone_non_increasing"],
            global_verification["all_three_historical_minima_match_returned_solution"],
            global_verification["all_three_satisfy_j_total_equals_j_data_plus_j_reg"],
            global_verification["seeds_are_distinct"],
            global_verification["identical_target_sha256_verified"],
        ]
    ):
        raise RuntimeError(f"Verifica globale Fase 3.7 fallita: {global_verification}")

    summary_payload = {
        "experiment_id": "RETECO-v1-PHASE3.7-CPSO-MULTISEED-600",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "forcing_protocol_version": FORCING_PROTOCOL_VERSION,
        "frozen_configuration": {
            "N_target": 32,
            "M_target": 160,
            "N_inv": 16,
            "M_inv": 80,
            "objective_mode": records[0].objective_mode,
            "eta_level": 0.0,
            "lambda_reg": FIXED_LAMBDA_REG,
            "prior_theta_tilde_0": prior_theta_tilde_0.tolist(),
            "data_seed": FIXED_DATA_SEED,
            "optimizer_seeds": list(CPSO_SEEDS),
            "num_particles": NUM_PARTICLES,
            "max_iterations": MAX_ITERATIONS,
            "planned_fem_budget": PLANNED_FEM_BUDGET,
        },
        "target_provenance": {
            "target_sha256": verified_sha256,
            "expected_target_sha256": EXPECTED_TARGET_SHA256,
            "hash_match_verified": True,
            "target_source": target_source,
            "target_generation_wall_time_seconds": target_wall_s,
            "discretization_relative_gap_L2": discretization_rel_gap,
        },
        "reference_evaluation_at_theta_dagger": {
            "theta_dagger": theta_dagger.tolist(),
            "theta_tilde_dagger": transform.to_normalized(theta_dagger).tolist(),
            "J_data_at_true": float(bd_at_true.j_data),
            "J_reg_at_true": float(bd_at_true.j_reg),
            "J_total_at_true": float(bd_at_true.j_total),
            "weighted_residual_l2_at_true": float(bd_at_true.weighted_residual_l2_norm),
            "relative_residual_vs_clean_target_at_true": float(bd_at_true.relative_residual_l2),
            "recorded_in_optimizer_history": False,
        },
        "global_consistency_verification": global_verification,
        "interseed_statistics": stats,
        "comparison_table_rows": comparison_rows,
        "artifacts_generated": {
            "summary_json": str(summary_json_path),
            "comparison_csv": str(comp_csv_path),
            "convergence_history_csv": str(hist_csv_path),
            "figure_png": str(png_path),
            "figure_pdf": str(pdf_path),
            "report_md": str(report_md_path),
        },
        "runs": [serialize_run_record(r) for r in records],
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("\n" + report_md)
    print(f"[Fase 3.7] Tutti gli artefatti salvati in: {out_dir}")


if __name__ == "__main__":
    main()