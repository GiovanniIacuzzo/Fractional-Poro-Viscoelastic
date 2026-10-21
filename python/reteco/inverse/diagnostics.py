"""
python/reteco/inverse/diagnostics.py
Strumenti diagnostici rigorosi per gli esperimenti inversi sintetici (Twin Experiments):
- Invocazione automatica di `verify_history_and_best_candidate_consistency()` dopo l'ottimizzazione (Fase 3.6);
- Estrazione della storia delle valutazioni da `InverseObjectiveFunctional.history` senza
  valutazioni ridondanti del solutore diretto;
- Contabilizzazione separata tra valutazioni FEM dell'ottimizzatore e valutazioni
  diagnostiche supplementari;
- Generazione di artefatti riproducibili (JSON, CSV per run, CSV della storia, report Markdown)
  e delega dei grafici PNG/PDF a `reteco.visualization.twin_diagnostics_plots`.
"""

from __future__ import annotations
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

from ..visualization.benchmark_plots import compute_individual_relative_errors_pct
from ..visualization.twin_diagnostics_plots import (
    plot_multistart_convergence_v2,
    plot_multistart_objective_decomposition_v2,
    plot_multistart_parameter_recovery_v2,
    plot_noise_study_parameter_errors_v2,
    plot_noise_study_relative_residuals_v2,
)

PARAM_NAMES: tuple[str, str, str, str] = ("alpha", "beta", "D", "E")


def _get_field(record: Any, candidates: Sequence[str], default: Any = None) -> Any:
    """Estrae un campo da un record che può essere sia un dizionario sia un oggetto/dataclass."""
    if isinstance(record, dict):
        for key in candidates:
            if key in record and record[key] is not None:
                return record[key]
    else:
        for key in candidates:
            if hasattr(record, key):
                val = getattr(record, key)
                if val is not None:
                    return val
    return default


def _extract_theta_from_record(record: Any, transform: Any = None) -> Optional[np.ndarray]:
    """Estrae il vettore parametrico fisico (alpha, beta, D, E) da un record di history."""
    raw_theta = _get_field(
        record,
        ("theta", "theta_phys", "theta_vec", "theta_candidate", "parameters", "x"),
    )
    if raw_theta is not None:
        arr = np.asarray(raw_theta, dtype=np.float64).ravel()
        if arr.size == 4:
            return arr.copy()

    raw_tilde = _get_field(record, ("theta_tilde", "x_norm", "normalized_theta"))
    if raw_tilde is not None and transform is not None:
        arr_tilde = np.asarray(raw_tilde, dtype=np.float64).ravel()
        if arr_tilde.size == 4:
            if hasattr(transform, "from_normalized"):
                return np.asarray(transform.from_normalized(arr_tilde), dtype=np.float64).ravel()
            if hasattr(transform, "to_physical"):
                return np.asarray(transform.to_physical(arr_tilde), dtype=np.float64).ravel()

    return None


@dataclass(frozen=True)
class EvaluationHistoryRow:
    eval_index: int
    theta: np.ndarray
    j_total: float
    j_data: float
    j_reg: float
    best_j_so_far: float
    best_theta_so_far: np.ndarray


@dataclass(frozen=True)
class SingleRunDiagnosticRecord:
    run_id: str
    optimizer_seed: int
    data_seed: int
    eta_level: float
    theta_true: np.ndarray
    theta_hat: np.ndarray
    individual_relative_errors_pct: np.ndarray
    best_cost_J_total: float
    best_cost_J_data: float
    best_cost_J_reg: float
    best_eval_index: int
    nominal_noise_std: float
    effective_relative_noise_norm: float
    relative_residual_vs_noisy_data: float
    relative_residual_vs_clean_target: float
    planned_fem_budget: int
    optimizer_fem_evaluations: int
    diagnostic_fem_evaluations: int
    termination_condition: str
    wall_time_seconds: float
    history: List[EvaluationHistoryRow]
    objective_mode: str = "legacy_compatible"
    theta_tilde_hat: Optional[np.ndarray] = None
    mean_relative_error_pct: float = 0.0
    max_relative_error_pct: float = 0.0
    unweighted_residual_l2_norm: float = 0.0
    unweighted_residual_linf_norm: float = 0.0
    weighted_residual_l2_norm: float = 0.0
    consistency_checks: Dict[str, Any] = field(default_factory=dict)

    @property
    def history_records(self) -> List[EvaluationHistoryRow]:
        return self.history


def determine_verifiable_termination_condition(
    optimizer_fem_evaluations: int,
    planned_fem_budget: int,
    num_particles: int,
    max_iterations: int,
) -> str:
    """
    Riporta la condizione di arresto effettivamente verificabile senza dichiarare
    automaticamente convergente un'ottimizzazione che ha esaurito il budget.
    """
    expected_from_swarm = int(num_particles) * int(max_iterations)
    if optimizer_fem_evaluations >= planned_fem_budget:
        if planned_fem_budget == expected_from_swarm:
            return "budget_exhausted_max_iterations_reached"
        return "budget_exhausted_max_fem_evaluations_reached"
    return "stopped_before_budget_exhaustion"


def extract_run_diagnostics_from_history(
    *,
    run_id: str,
    obj: Any,
    opt_result: Any,
    theta_true: np.ndarray,
    y_target_clean: np.ndarray,
    y_noisy: np.ndarray,
    nominal_noise_std: float,
    eta_level: float,
    data_seed: int,
    optimizer_seed: int,
    planned_fem_budget: int,
    num_particles: int,
    max_iterations: int,
    wall_time_seconds: float,
    forward_operator_inv: Callable[[np.ndarray], np.ndarray],
    transform: Any = None,
    regularizer: Any = None,
    seed_reproducibility_verified: Optional[bool] = None,
) -> SingleRunDiagnosticRecord:
    """
    Estrae i valori di J_total, J_data, J_reg e la storia di convergenza direttamente da
    `obj.history`, invocando obbligatoriamente `verify_history_and_best_candidate_consistency`
    quando disponibile su `obj` (Fase 3.6 - Punto 1).
    """
    raw_history = list(getattr(obj, "history", []))
    if not raw_history:
        raise ValueError("InverseObjectiveFunctional.history è vuoto: impossibile estrarre la diagnostica.")

    consistency_report: Dict[str, Any] = {}
    if hasattr(obj, "verify_history_and_best_candidate_consistency"):
        consistency_report = dict(
            obj.verify_history_and_best_candidate_consistency(
                opt_result, raise_on_failure=True
            )
        )
    if seed_reproducibility_verified is not None:
        consistency_report["seed_reproducibility_verified"] = bool(
            seed_reproducibility_verified
        )

    theta_true_arr = np.asarray(theta_true, dtype=np.float64).ravel()
    theta_hat_arr = np.asarray(opt_result.theta_hat, dtype=np.float64).ravel()
    best_cost_reported = float(opt_result.best_cost_J)
    optimizer_evals = int(getattr(opt_result, "num_fem_evaluations", len(raw_history)))

    parsed_rows: List[EvaluationHistoryRow] = []
    running_best_j = float("inf")
    running_best_theta: Optional[np.ndarray] = None

    for idx, rec in enumerate(raw_history):
        eval_idx = int(_get_field(rec, ("eval_index", "evaluation_index", "eval_id", "call_index"), idx + 1))
        j_total = float(_get_field(rec, ("j_total", "cost_total", "J_total")))
        j_data = float(_get_field(rec, ("j_data", "cost_data", "J_data")))
        j_reg = float(_get_field(rec, ("j_reg", "cost_reg", "J_reg")))
        best_j_so_far = float(_get_field(rec, ("best_j_so_far", "best_cost_so_far"), min(running_best_j, j_total)))

        if not np.isclose(j_total, j_data + j_reg, rtol=1e-11, atol=1e-13):
            raise ValueError(
                f"Incoerenza additiva nel record {eval_idx}: "
                f"j_total={j_total:.15e} != j_data + j_reg = {j_data + j_reg:.15e}."
            )

        rec_theta = _extract_theta_from_record(rec, transform=transform)
        if rec_theta is None:
            if np.isclose(j_total, best_cost_reported, rtol=1e-12, atol=1e-14):
                rec_theta = theta_hat_arr.copy()
            elif running_best_theta is not None:
                rec_theta = running_best_theta.copy()
            else:
                rec_theta = np.full(4, np.nan, dtype=np.float64)

        if j_total < running_best_j or running_best_theta is None:
            running_best_j = j_total
            running_best_theta = rec_theta.copy()

        parsed_rows.append(
            EvaluationHistoryRow(
                eval_index=eval_idx,
                theta=rec_theta,
                j_total=j_total,
                j_data=j_data,
                j_reg=j_reg,
                best_j_so_far=best_j_so_far,
                best_theta_so_far=running_best_theta.copy(),
            )
        )

    matching_indices: List[int] = []
    for i, row in enumerate(parsed_rows):
        cost_match = np.isclose(row.j_total, best_cost_reported, rtol=1e-11, atol=1e-13)
        theta_match = np.all(np.isfinite(row.theta)) and np.allclose(
            row.theta, theta_hat_arr, rtol=1e-10, atol=1e-12
        )
        if cost_match and theta_match:
            matching_indices.append(i)

    if not matching_indices:
        for i, row in enumerate(parsed_rows):
            if np.isclose(row.j_total, best_cost_reported, rtol=1e-11, atol=1e-13):
                matching_indices.append(i)

    best_idx = matching_indices[0] if matching_indices else int(np.argmin([r.j_total for r in parsed_rows]))
    best_row = parsed_rows[best_idx]
    best_raw_rec = raw_history[best_idx]

    if not np.isclose(best_row.j_total, best_cost_reported, rtol=1e-10, atol=1e-12):
        raise ValueError(
            f"Il miglior costo riportato da CPSO ({best_cost_reported:.15e}) non coincide "
            f"con il miglior record in history ({best_row.j_total:.15e})."
        )

    if np.all(np.isfinite(best_row.theta)) and not np.allclose(
        best_row.theta, theta_hat_arr, rtol=1e-8, atol=1e-10
    ):
        raise ValueError(
            f"Il miglior candidato theta_hat ({theta_hat_arr}) non coincide con "
            f"il vettore memorizzato nel record ottimo ({best_row.theta})."
        )

    if regularizer is not None:
        reg_check = float(regularizer.evaluate(theta_hat_arr))
        if not np.isclose(best_row.j_reg, reg_check, rtol=1e-10, atol=1e-12):
            raise ValueError(
                f"Incoerenza in J_reg sul miglior candidato: history={best_row.j_reg:.15e} "
                f"vs regularizer.evaluate(theta_hat)={reg_check:.15e}."
            )

    final_rows: List[EvaluationHistoryRow] = []
    current_best_j = float("inf")
    current_best_theta = parsed_rows[0].theta.copy()
    for i, row in enumerate(parsed_rows):
        if i == best_idx or row.j_total < current_best_j:
            current_best_j = row.j_total
            current_best_theta = theta_hat_arr.copy() if i == best_idx else row.theta.copy()
        final_rows.append(
            EvaluationHistoryRow(
                eval_index=row.eval_index,
                theta=theta_hat_arr.copy() if i == best_idx else row.theta,
                j_total=row.j_total,
                j_data=row.j_data,
                j_reg=row.j_reg,
                best_j_so_far=row.best_j_so_far,
                best_theta_so_far=current_best_theta.copy(),
            )
        )

    evals_before_diag = len(getattr(obj, "history", []))
    y_hat = np.asarray(forward_operator_inv(theta_hat_arr), dtype=np.float64).ravel()
    evals_after_diag = len(getattr(obj, "history", []))
    if evals_after_diag != evals_before_diag:
        raise RuntimeError(
            "La valutazione forward diagnostica ha incrementato il contatore dell'ottimizzatore."
        )
    diagnostic_evals = 1

    clean_arr = np.asarray(y_target_clean, dtype=np.float64).ravel()
    noisy_arr = np.asarray(y_noisy, dtype=np.float64).ravel()
    clean_norm = float(np.linalg.norm(clean_arr)) or 1.0
    noisy_norm = float(np.linalg.norm(noisy_arr)) or 1.0

    residual_vec = y_hat - noisy_arr
    unweighted_l2 = float(
        _get_field(best_raw_rec, ("unweighted_residual_l2_norm",), np.linalg.norm(residual_vec))
    )
    unweighted_linf = float(
        _get_field(best_raw_rec, ("unweighted_residual_linf_norm",), np.max(np.abs(residual_vec)))
    )
    weighted_l2 = float(
        _get_field(
            best_raw_rec,
            ("weighted_residual_l2_norm",),
            np.sqrt(max(0.0, 2.0 * float(best_row.j_data))),
        )
    )

    rel_res_clean = float(np.linalg.norm(y_hat - clean_arr) / clean_norm)
    rel_res_noisy = float(np.linalg.norm(y_hat - noisy_arr) / noisy_norm)
    eff_rel_noise = float(np.linalg.norm(noisy_arr - clean_arr) / clean_norm)

    rel_errs_pct = np.asarray(
        compute_individual_relative_errors_pct(theta_hat_arr, theta_true_arr),
        dtype=np.float64,
    )
    mean_rel_err = float(np.mean(rel_errs_pct))
    max_rel_err = float(np.max(rel_errs_pct))

    theta_tilde_hat_arr: Optional[np.ndarray] = None
    raw_tilde_best = _get_field(best_raw_rec, ("theta_tilde",))
    if raw_tilde_best is not None:
        theta_tilde_hat_arr = np.asarray(raw_tilde_best, dtype=np.float64).ravel().copy()
    elif transform is not None and hasattr(transform, "to_normalized"):
        theta_tilde_hat_arr = np.asarray(
            transform.to_normalized(theta_hat_arr), dtype=np.float64
        ).ravel()

    term_cond = determine_verifiable_termination_condition(
        optimizer_fem_evaluations=optimizer_evals,
        planned_fem_budget=planned_fem_budget,
        num_particles=num_particles,
        max_iterations=max_iterations,
    )

    obj_mode_str = str(getattr(obj, "objective_mode", "legacy_compatible"))

    return SingleRunDiagnosticRecord(
        run_id=str(run_id),
        optimizer_seed=int(optimizer_seed),
        data_seed=int(data_seed),
        eta_level=float(eta_level),
        theta_true=theta_true_arr,
        theta_hat=theta_hat_arr,
        individual_relative_errors_pct=rel_errs_pct,
        best_cost_J_total=float(best_row.j_total),
        best_cost_J_data=float(best_row.j_data),
        best_cost_J_reg=float(best_row.j_reg),
        best_eval_index=int(best_row.eval_index),
        nominal_noise_std=float(nominal_noise_std),
        effective_relative_noise_norm=eff_rel_noise,
        relative_residual_vs_noisy_data=rel_res_noisy,
        relative_residual_vs_clean_target=rel_res_clean,
        planned_fem_budget=int(planned_fem_budget),
        optimizer_fem_evaluations=optimizer_evals,
        diagnostic_fem_evaluations=diagnostic_evals,
        termination_condition=term_cond,
        wall_time_seconds=float(wall_time_seconds),
        history=final_rows,
        objective_mode=obj_mode_str,
        theta_tilde_hat=theta_tilde_hat_arr,
        mean_relative_error_pct=mean_rel_err,
        max_relative_error_pct=max_rel_err,
        unweighted_residual_l2_norm=unweighted_l2,
        unweighted_residual_linf_norm=unweighted_linf,
        weighted_residual_l2_norm=weighted_l2,
        consistency_checks=consistency_report,
    )


def serialize_run_record(rec: SingleRunDiagnosticRecord) -> Dict[str, Any]:
    return {
        "run_id": rec.run_id,
        "objective_mode": rec.objective_mode,
        "optimizer_seed": rec.optimizer_seed,
        "data_seed": rec.data_seed,
        "eta_level": rec.eta_level,
        "theta_true": {
            name: float(rec.theta_true[i]) for i, name in enumerate(PARAM_NAMES)
        },
        "theta_true_vector": rec.theta_true.tolist(),
        "theta_hat": {
            name: float(rec.theta_hat[i]) for i, name in enumerate(PARAM_NAMES)
        },
        "theta_hat_vector": rec.theta_hat.tolist(),
        "theta_tilde_hat_vector": (
            rec.theta_tilde_hat.tolist() if rec.theta_tilde_hat is not None else None
        ),
        "individual_relative_errors_pct": {
            name: float(rec.individual_relative_errors_pct[i])
            for i, name in enumerate(PARAM_NAMES)
        },
        "individual_relative_errors_pct_vector": rec.individual_relative_errors_pct.tolist(),
        "mean_relative_error_pct": rec.mean_relative_error_pct,
        "max_relative_error_pct": rec.max_relative_error_pct,
        "best_cost_J_total": rec.best_cost_J_total,
        "best_cost_J_data": rec.best_cost_J_data,
        "best_cost_J_reg": rec.best_cost_J_reg,
        "best_eval_index": rec.best_eval_index,
        "unweighted_residual_l2_norm": rec.unweighted_residual_l2_norm,
        "unweighted_residual_linf_norm": rec.unweighted_residual_linf_norm,
        "weighted_residual_l2_norm": rec.weighted_residual_l2_norm,
        "nominal_noise_std": rec.nominal_noise_std,
        "effective_relative_noise_norm": rec.effective_relative_noise_norm,
        "relative_residual_vs_noisy_data": rec.relative_residual_vs_noisy_data,
        "relative_residual_vs_clean_target": rec.relative_residual_vs_clean_target,
        "planned_fem_budget": rec.planned_fem_budget,
        "optimizer_fem_evaluations": rec.optimizer_fem_evaluations,
        "diagnostic_fem_evaluations": rec.diagnostic_fem_evaluations,
        "termination_condition": rec.termination_condition,
        "wall_time_seconds": rec.wall_time_seconds,
        "consistency_checks": dict(rec.consistency_checks),
        "convergence_history": {
            "eval_index": [r.eval_index for r in rec.history],
            "j_total": [r.j_total for r in rec.history],
            "j_data": [r.j_data for r in rec.history],
            "j_reg": [r.j_reg for r in rec.history],
            "best_j_so_far": [r.best_j_so_far for r in rec.history],
            "candidate_theta": [r.theta.tolist() for r in rec.history],
            "best_theta_so_far": [r.best_theta_so_far.tolist() for r in rec.history],
        },
    }


def write_runs_summary_csv(
    records: Sequence[SingleRunDiagnosticRecord],
    csv_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "optimizer_seed",
        "data_seed",
        "eta_level",
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
        "best_eval_index",
        "nominal_noise_std",
        "effective_relative_noise_norm",
        "relative_residual_vs_noisy_data",
        "relative_residual_vs_clean_target",
        "planned_fem_budget",
        "optimizer_fem_evaluations",
        "diagnostic_fem_evaluations",
        "termination_condition",
        "wall_time_seconds",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "run_id": rec.run_id,
                    "optimizer_seed": rec.optimizer_seed,
                    "data_seed": rec.data_seed,
                    "eta_level": rec.eta_level,
                    "alpha_true": rec.theta_true[0],
                    "beta_true": rec.theta_true[1],
                    "D_true": rec.theta_true[2],
                    "E_true": rec.theta_true[3],
                    "alpha_hat": rec.theta_hat[0],
                    "beta_hat": rec.theta_hat[1],
                    "D_hat": rec.theta_hat[2],
                    "E_hat": rec.theta_hat[3],
                    "err_rel_alpha_pct": rec.individual_relative_errors_pct[0],
                    "err_rel_beta_pct": rec.individual_relative_errors_pct[1],
                    "err_rel_D_pct": rec.individual_relative_errors_pct[2],
                    "err_rel_E_pct": rec.individual_relative_errors_pct[3],
                    "best_cost_J_total": rec.best_cost_J_total,
                    "best_cost_J_data": rec.best_cost_J_data,
                    "best_cost_J_reg": rec.best_cost_J_reg,
                    "best_eval_index": rec.best_eval_index,
                    "nominal_noise_std": rec.nominal_noise_std,
                    "effective_relative_noise_norm": rec.effective_relative_noise_norm,
                    "relative_residual_vs_noisy_data": rec.relative_residual_vs_noisy_data,
                    "relative_residual_vs_clean_target": rec.relative_residual_vs_clean_target,
                    "planned_fem_budget": rec.planned_fem_budget,
                    "optimizer_fem_evaluations": rec.optimizer_fem_evaluations,
                    "diagnostic_fem_evaluations": rec.diagnostic_fem_evaluations,
                    "termination_condition": rec.termination_condition,
                    "wall_time_seconds": rec.wall_time_seconds,
                }
            )


def write_convergence_history_csv(
    records: Sequence[SingleRunDiagnosticRecord],
    csv_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "optimizer_seed",
        "eta_level",
        "eval_index",
        "j_total",
        "j_data",
        "j_reg",
        "best_j_so_far",
        "candidate_alpha",
        "candidate_beta",
        "candidate_D",
        "candidate_E",
        "best_alpha_so_far",
        "best_beta_so_far",
        "best_D_so_far",
        "best_E_so_far",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            for row in rec.history:
                writer.writerow(
                    {
                        "run_id": rec.run_id,
                        "optimizer_seed": rec.optimizer_seed,
                        "eta_level": rec.eta_level,
                        "eval_index": row.eval_index,
                        "j_total": row.j_total,
                        "j_data": row.j_data,
                        "j_reg": row.j_reg,
                        "best_j_so_far": row.best_j_so_far,
                        "candidate_alpha": row.theta[0],
                        "candidate_beta": row.theta[1],
                        "candidate_D": row.theta[2],
                        "candidate_E": row.theta[3],
                        "best_alpha_so_far": row.best_theta_so_far[0],
                        "best_beta_so_far": row.best_theta_so_far[1],
                        "best_D_so_far": row.best_theta_so_far[2],
                        "best_E_so_far": row.best_theta_so_far[3],
                    }
                )


def plot_multistart_parameter_recovery(
    records: Sequence[SingleRunDiagnosticRecord],
    bounds_dict: Optional[Dict[str, Tuple[float, float]]],
    out_path: Path,
) -> Tuple[Path, Path]:
    return plot_multistart_parameter_recovery_v2(records, bounds_dict, out_path)


def plot_multistart_convergence(
    records: Sequence[SingleRunDiagnosticRecord],
    out_path: Path,
) -> Tuple[Path, Path]:
    return plot_multistart_convergence_v2(records, out_path)


def plot_multistart_objective_decomposition(
    records: Sequence[SingleRunDiagnosticRecord],
    out_path: Path,
) -> Tuple[Path, Path]:
    return plot_multistart_objective_decomposition_v2(records, out_path)


def plot_noise_study_parameter_errors(
    records: Sequence[SingleRunDiagnosticRecord],
    out_path: Path,
) -> Tuple[Path, Path]:
    return plot_noise_study_parameter_errors_v2(records, out_path)


def plot_noise_study_relative_residuals(
    records: Sequence[SingleRunDiagnosticRecord],
    discretization_rel_gap: float,
    out_path: Path,
) -> Tuple[Path, Path]:
    return plot_noise_study_relative_residuals_v2(records, discretization_rel_gap, out_path)


def write_multistart_markdown_report(
    records: Sequence[SingleRunDiagnosticRecord],
    discretization_rel_gap: float,
    lambda_reg: float,
    prior_theta_tilde_0: Sequence[float],
    target_cfg: Dict[str, Any],
    inverse_cfg: Dict[str, Any],
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# Report Diagnostico: Multistart CPSO Senza Rumore (`03a`)",
        "",
        "## 1. Configurazione e Metodologia",
        f"- **Discretizzazione Target (`h_tar`)**: `N = {target_cfg.get('subdivisions_N')}`, `M = {target_cfg.get('time_steps_M')}`, `T = {target_cfg.get('final_time_T')}`, `r = {target_cfg.get('grading_r')}`.",
        f"- **Discretizzazione Inversione (`h_inv`)**: `N = {inverse_cfg.get('subdivisions_N')}`, `M = {inverse_cfg.get('time_steps_M')}`, `T = {inverse_cfg.get('final_time_T')}`, `r = {inverse_cfg.get('grading_r')}`.",
        f"- **Gap relativo di discretizzazione** `||G_{{h_tar}}(theta*) - G_{{h_inv}}(theta*)||_2 / ||y_clean||_2`: `{discretization_rel_gap:.6e}`.",
        f"- **Livello di rumore**: `eta = 0.0` (osservazioni prive di rumore; pesi adimensionali `W_ii = 1 / (sqrt(N_obs) * ||y_clean||_inf)`).",
        f"- **Regolarizzazione di Tikhonov**: `lambda = {lambda_reg:.2e}`, `theta_tilde_0 = {list(prior_theta_tilde_0)}`.",
        f"- **Numero di esecuzioni indipendenti (multistart)**: `{len(records)}` semi distinti dell'ottimizzatore CPSO (`10` particelle, `12` iterazioni).",
        "",
        "## 2. Contabilizzazione del Budget Computazionale",
        "- **Budget pianificato per esecuzione**: `120` valutazioni FEM del problema diretto su `h_inv`.",
        "- **Valutazioni diagnostiche fuori-budget**: `1` valutazione su `h_inv` per ciascun seed (calcolo del residuo finale non pesato su `theta_hat`), più `1` valutazione condivisa iniziale in `theta*` per misurare il gap di discretizzazione.",
        "- **Nota sulla terminazione**: Il completamento delle 120 valutazioni pianificate indica esclusivamente l'esaurimento del budget assegnato (`budget_exhausted_max_iterations_reached`) e **non** certifica la convergenza a un minimo locale o globale.",
        "",
        "## 3. Risultati per Seed (`Run 1`–`Run 5`)",
        "| Etichetta Figura | Run ID | Seed Ottimizzatore | $J_{\\mathrm{total}}$ | $J_{\\mathrm{data}}$ | $J_{\\mathrm{reg}}$ | Residuo Rel. Pulito | Err $\\alpha$ [%] | Err $\\beta$ [%] | Err $D$ [%] | Err $E$ [%] | Val. Ott. | Val. Diag. | Terminazione |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for idx, r in enumerate(records):
        errs = r.individual_relative_errors_pct
        lines.append(
            f"| `Run {idx + 1}` | `{r.run_id}` | `{r.optimizer_seed}` | `{r.best_cost_J_total:.4e}` | "
            f"`{r.best_cost_J_data:.4e}` | `{r.best_cost_J_reg:.4e}` | "
            f"`{r.relative_residual_vs_clean_target:.4e}` | "
            f"`{errs[0]:.2f}` | `{errs[1]:.2f}` | `{errs[2]:.2f}` | `{errs[3]:.2f}` | "
            f"`{r.optimizer_fem_evaluations}` | `{r.diagnostic_fem_evaluations}` | `{r.termination_condition}` |"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_noise_study_markdown_report(
    records: Sequence[SingleRunDiagnosticRecord],
    discretization_rel_gap: float,
    lambda_reg: float,
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# Report Diagnostico: Studio Esplorativo al Variare del Rumore (`03`)",
        "",
        "## 1. Avvertenza Metodologica sulla Comparabilità di $J$",
        "- Per $\\eta > 0$, i pesi osservativi sono definiti come $W_{ii} = 1 / (\\sqrt{N_{\\mathrm{obs}}}\\,\\sigma)$ con $\\sigma = \\eta \\Vert{}y_{\\mathrm{clean}}\\Vert{}_\\infty$, per cui $\\mathbf{W}^\\top \\mathbf{W} = \\boldsymbol{\\Sigma}^{-1} / N_{\\mathrm{obs}}$.",
        "- Poiché $J_{\\mathrm{data}}(\\theta) = \\frac{1}{2 N_{\\mathrm{obs}} \\sigma^2} \\Vert{}G_{h_{\\mathrm{inv}}}(\\theta) - y_{\\mathrm{noisy}}\\Vert{}_2^2$ dipende inversamente da $\\sigma^2 \\propto \\eta^2$, **il valore grezzo di $J$ non è confrontabile direttamente fra livelli diversi di rumore**.",
        "- Ciascun livello di rumore qui riportato comprende **una sola esecuzione CPSO** (budget 120 valutazioni FEM): i risultati hanno carattere esplorativo e non consentono inferenze statistiche sulla monotonicità rispetto a $\\eta$.",
        "",
        f"- **Gap relativo di discretizzazione (`h_tar` vs `h_inv`)**: `{discretization_rel_gap:.6e}`",
        f"- **Parametro di regolarizzazione**: `lambda = {lambda_reg:.2e}`",
        "",
        "## 2. Tabella Diagnostica per Livello di Rumore",
        "| $\\eta$ [%] | Seed Dati | Seed Ott. | $\\sigma_{\\mathrm{nom}}$ | Rumore Rel. Eff. | $J_{\\mathrm{total}}$ | $J_{\\mathrm{data}}$ | $J_{\\mathrm{reg}}$ | Res. Rel. Noisy | Res. Rel. Clean | Err $\\alpha$ [%] | Err $\\beta$ [%] | Err $D$ [%] | Err $E$ [%] | Val. Ott. | Val. Diag. |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        errs = r.individual_relative_errors_pct
        lines.append(
            f"| `{r.eta_level * 100:.1f}` | `{r.data_seed}` | `{r.optimizer_seed}` | "
            f"`{r.nominal_noise_std:.4e}` | `{r.effective_relative_noise_norm:.4e}` | "
            f"`{r.best_cost_J_total:.4e}` | `{r.best_cost_J_data:.4e}` | `{r.best_cost_J_reg:.4e}` | "
            f"`{r.relative_residual_vs_noisy_data:.4e}` | `{r.relative_residual_vs_clean_target:.4e}` | "
            f"`{errs[0]:.2f}` | `{errs[1]:.2f}` | `{errs[2]:.2f}` | `{errs[3]:.2f}` | "
            f"`{r.optimizer_fem_evaluations}` | `{r.diagnostic_fem_evaluations}` |"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")