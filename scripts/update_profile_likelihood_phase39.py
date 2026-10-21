"""
scripts/update_profile_likelihood_phase39.py
Estende in modo non distruttivo `python/reteco/identifiability/profile_likelihood.py`
aggiungendo `OptimizedProfilePointRecord`, `BidirectionalProfileResult` e
`compute_bidirectional_optimized_1d_profile` senza modificare alcuna funzione o classe
già presente nel file.
"""

from __future__ import annotations
from pathlib import Path

EXTENSION_MARKER = "# === RETECO PHASE 3.9 BIDIRECTIONAL OPTIMIZED PROFILE EXTENSION ==="

EXTENSION_CODE = '''

# === RETECO PHASE 3.9 BIDIRECTIONAL OPTIMIZED PROFILE EXTENSION ===
from dataclasses import asdict as _pl_asdict, dataclass as _pl_dataclass
import time as _pl_time
from typing import Any as _pl_Any, List as _pl_List, Optional as _pl_Optional, Sequence as _pl_Sequence, Tuple as _pl_Tuple
from scipy.optimize import minimize as _scipy_minimize

from ..config_schema import ParameterCoordinateTransform as _PLTransform
from ..inverse.objective_functional import (
    BudgetExhaustedException as _PLBudgetExhaustedException,
    InverseObjectiveFunctional as _PLInverseObjectiveFunctional,
)
from ..inverse.regularization import TikhonovRegularizer as _PLTikhonovRegularizer


@_pl_dataclass(frozen=True)
class OptimizedProfilePointRecord:
    profile_type: str
    scan_direction: str
    grid_index: int
    beta_fixed: float
    alpha_hat: float
    D_hat: float
    E_hat: float
    alpha_tilde_hat: float
    beta_tilde_fixed: float
    D_tilde_hat: float
    E_tilde_hat: float
    j_data: float
    j_reg: float
    j_total: float
    optimized_objective_value: float
    relative_residual_l2: float
    weighted_residual_l2_norm: float
    unweighted_residual_l2_norm: float
    unweighted_residual_linf_norm: float
    fem_evaluations_used: int
    planned_point_budget: int
    converged: bool
    budget_exhausted: bool
    stopping_criterion: str
    is_admissible: bool
    coordinate_roundtrip_error: float
    j_decomposition_error: float
    initial_guess_source: str
    wall_time_seconds: float

    def to_dict(self) -> Dict[str, _pl_Any]:
        return _pl_asdict(self)


@_pl_dataclass(frozen=True)
class BidirectionalProfileResult:
    profile_type: str
    beta_grid: np.ndarray
    planned_point_budget: int
    forward_records: _pl_List[OptimizedProfilePointRecord]
    backward_records: _pl_List[OptimizedProfilePointRecord]
    combined_best_records: _pl_List[OptimizedProfilePointRecord]
    total_fem_evaluations: int
    wall_time_seconds: float
    directional_discrepancy_summary: Dict[str, _pl_Any]


def _optimize_nuisance_at_fixed_beta(
    *,
    profile_type: str,
    scan_direction: str,
    grid_index: int,
    beta_fixed: float,
    u_init_candidates: _pl_Sequence[_pl_Tuple[str, np.ndarray]],
    forward_operator: Callable[[np.ndarray], np.ndarray],
    y_observed: np.ndarray,
    transform: _PLTransform,
    canonical_regularizer: _PLTikhonovRegularizer,
    planned_point_budget: int,
    scale_type: str = "linf",
    ftol: float = 1.0e-9,
    gtol: float = 1.0e-6,
    fd_eps: float = 1.0e-4,
) -> OptimizedProfilePointRecord:
    """
    Ottimizza i 3 parametri di disturbo (alpha, D, E) nello spazio normalizzato [0, 1]^3
    mantenendo beta = beta_fixed, con budget rigorosamente limitato a `planned_point_budget`
    valutazioni del solutore diretto tramite `InverseObjectiveFunctional`.
    """
    t0 = _pl_time.perf_counter()
    if planned_point_budget < 2:
        raise ValueError(f"planned_point_budget deve essere >= 2, ricevuto {planned_point_budget}.")

    # Calcolo di beta_tilde_fixed usando esclusivamente ParameterCoordinateTransform
    mid_alpha = 0.5 * (transform.a_min + transform.a_max)
    mid_D = float(np.sqrt(transform.d_min * transform.d_max))
    mid_E = float(np.sqrt(transform.e_min * transform.e_max))
    probe_phys = np.array([mid_alpha, float(beta_fixed), mid_D, mid_E], dtype=np.float64)
    if not transform.is_within_bounds(probe_phys, tol=1.0e-12):
        raise ValueError(f"Valore di beta={beta_fixed} fuori dai limiti ammissibili.")
    beta_tilde_fixed = float(transform.to_normalized(probe_phys)[1])

    # Costruzione del regolarizzatore attivo per l'ottimizzazione:
    # - Per "data_only" (Phi_data): lambda_active = 0.0 (minimizza solo J_data)
    # - Per "regularized_total" (Phi_lambda): lambda_active = canonical_regularizer.lambda_reg
    if profile_type == "data_only":
        active_reg = _PLTikhonovRegularizer(
            transform=transform,
            lambda_reg=0.0,
            theta_tilde_0=canonical_regularizer.theta_tilde_0.copy(),
        )
    elif profile_type == "regularized_total":
        active_reg = canonical_regularizer
    else:
        raise ValueError(f"profile_type non riconosciuto: {profile_type}")

    point_obj = _PLInverseObjectiveFunctional.from_mode_a_deterministic(
        forward_operator=forward_operator,
        y_observed=y_observed,
        transform=transform,
        regularizer=active_reg,
        y_reference=y_observed,
        allow_observed_as_reference=True,
        scale_type=scale_type,
        max_evaluations=int(planned_point_budget),
    )

    def u3_to_theta_phys(u3: np.ndarray) -> np.ndarray:
        u_clipped = np.clip(np.asarray(u3, dtype=np.float64).ravel(), 0.0, 1.0)
        tilde_4 = np.array(
            [u_clipped[0], beta_tilde_fixed, u_clipped[1], u_clipped[2]],
            dtype=np.float64,
        )
        th = transform.from_normalized(tilde_4)
        th[1] = float(beta_fixed)
        return th

    def objective_u3(u3: np.ndarray) -> float:
        th = u3_to_theta_phys(u3)
        return float(point_obj(th))

    # Valutazione dei candidati iniziali (warm-start / prior / CPSO empirico, mai theta_dagger)
    best_init_label = "prior_center"
    best_init_u = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    best_init_val = float("inf")

    budget_exhausted_flag = False
    stopping_reason = "not_started"
    scipy_converged = False

    try:
        for label, u_cand in u_init_candidates:
            if point_obj.num_fem_evaluations >= planned_point_budget:
                budget_exhausted_flag = True
                break
            u_c = np.clip(np.asarray(u_cand, dtype=np.float64).ravel(), 0.0, 1.0)
            val = objective_u3(u_c)
            if val < best_init_val:
                best_init_val = val
                best_init_u = u_c.copy()
                best_init_label = str(label)

        if point_obj.num_fem_evaluations < planned_point_budget:
            res = _scipy_minimize(
                objective_u3,
                best_init_u,
                method="L-BFGS-B",
                bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
                options={
                    "ftol": float(ftol),
                    "gtol": float(gtol),
                    "eps": float(fd_eps),
                    "maxfun": int(planned_point_budget),
                    "maxiter": int(planned_point_budget),
                },
            )
            msg_raw = (
                res.message.decode("utf-8", errors="replace")
                if isinstance(res.message, bytes)
                else str(res.message)
            )
            if point_obj.num_fem_evaluations >= planned_point_budget:
                budget_exhausted_flag = True
                scipy_converged = False
                stopping_reason = "budget_exhausted_max_fem_evaluations_reached"
            elif bool(res.success):
                budget_exhausted_flag = False
                scipy_converged = True
                stopping_reason = f"tolerance_converged:{msg_raw}"
            else:
                budget_exhausted_flag = False
                scipy_converged = False
                stopping_reason = f"optimizer_stopped_without_convergence:{msg_raw}"
        else:
            budget_exhausted_flag = True
            scipy_converged = False
            stopping_reason = "budget_exhausted_during_initial_candidates"

    except _PLBudgetExhaustedException:
        budget_exhausted_flag = True
        scipy_converged = False
        stopping_reason = "budget_exhausted_max_fem_evaluations_reached"

    if len(point_obj.history) == 0:
        raise RuntimeError(f"Nessuna valutazione registrata per beta={beta_fixed}.")

    # Estrazione del miglior punto visitato entro il budget assegnato
    opt_vals = [float(r.j_total) for r in point_obj.history]
    best_idx = int(np.argmin(opt_vals))
    best_rec = point_obj.history[best_idx]
    theta_best = best_rec.theta.copy()
    theta_best[1] = float(beta_fixed)

    # Verifica coerenza coordinate fisiche <-> normalizzate
    tilde_best = transform.to_normalized(theta_best)
    phys_roundtrip = transform.from_normalized(tilde_best)
    coord_err = float(np.max(np.abs(theta_best - phys_roundtrip)))
    is_adm = bool(
        transform.is_within_bounds(theta_best, tol=1.0e-12)
        and np.all(tilde_best >= -1.0e-12)
        and np.all(tilde_best <= 1.0 + 1.0e-12)
    )

    # Calcolo esatto di J_data, J_reg (con lambda canonico = 1e-4) e J_total = J_data + J_reg
    j_data_val = float(best_rec.j_data)
    reg_bd_canonical = canonical_regularizer.evaluate_breakdown(theta_best)
    j_reg_val = float(reg_bd_canonical.j_reg)
    j_total_val = float(j_data_val + j_reg_val)
    opt_obj_val = j_data_val if profile_type == "data_only" else j_total_val
    decomp_err = abs(j_total_val - (j_data_val + j_reg_val))

    wall_s = _pl_time.perf_counter() - t0
    return OptimizedProfilePointRecord(
        profile_type=str(profile_type),
        scan_direction=str(scan_direction),
        grid_index=int(grid_index),
        beta_fixed=float(beta_fixed),
        alpha_hat=float(theta_best[0]),
        D_hat=float(theta_best[2]),
        E_hat=float(theta_best[3]),
        alpha_tilde_hat=float(tilde_best[0]),
        beta_tilde_fixed=float(tilde_best[1]),
        D_tilde_hat=float(tilde_best[2]),
        E_tilde_hat=float(tilde_best[3]),
        j_data=j_data_val,
        j_reg=j_reg_val,
        j_total=j_total_val,
        optimized_objective_value=float(opt_obj_val),
        relative_residual_l2=float(best_rec.relative_residual_l2),
        weighted_residual_l2_norm=float(best_rec.weighted_residual_l2_norm),
        unweighted_residual_l2_norm=float(best_rec.unweighted_residual_l2_norm),
        unweighted_residual_linf_norm=float(best_rec.unweighted_residual_linf_norm),
        fem_evaluations_used=int(point_obj.num_fem_evaluations),
        planned_point_budget=int(planned_point_budget),
        converged=bool(scipy_converged and not budget_exhausted_flag),
        budget_exhausted=bool(budget_exhausted_flag),
        stopping_criterion=str(stopping_reason),
        is_admissible=is_adm,
        coordinate_roundtrip_error=coord_err,
        j_decomposition_error=float(decomp_err),
        initial_guess_source=best_init_label,
        wall_time_seconds=float(wall_s),
    )


def compute_bidirectional_optimized_1d_profile(
    *,
    profile_type: str,
    beta_grid: _pl_Sequence[float],
    forward_operator: Callable[[np.ndarray], np.ndarray],
    y_observed: np.ndarray,
    transform: _PLTransform,
    canonical_regularizer: _PLTikhonovRegularizer,
    planned_point_budget: int = 32,
    empirical_anchor_solutions: _pl_Optional[_pl_Sequence[Dict[str, _pl_Any]]] = None,
    scale_type: str = "linf",
    ftol: float = 1.0e-9,
    gtol: float = 1.0e-6,
    fd_eps: float = 1.0e-4,
) -> BidirectionalProfileResult:
    """
    Calcola un profilo 1D ottimizzato rispetto a beta (Phi_data oppure Phi_lambda)
    eseguendo la scansione in entrambe le direzioni (`forward` crescente e `backward`
    decrescente) con warm start dal punto adiacente e budget uniforme per punto.
    """
    t0_all = _pl_time.perf_counter()
    grid_arr = np.asarray(sorted(float(b) for b in beta_grid), dtype=np.float64)
    n_pts = int(grid_arr.size)

    u_prior = np.array(
        [
            float(canonical_regularizer.theta_tilde_0[0]),
            float(canonical_regularizer.theta_tilde_0[2]),
            float(canonical_regularizer.theta_tilde_0[3]),
        ],
        dtype=np.float64,
    )

    anchors_u: _pl_List[_pl_Tuple[float, str, np.ndarray]] = []
    if empirical_anchor_solutions:
        for sol in empirical_anchor_solutions:
            th_s = np.array(
                [
                    float(sol["alpha_hat"]),
                    float(sol["beta_hat"]),
                    float(sol["D_hat"]),
                    float(sol["E_hat"]),
                ],
                dtype=np.float64,
            )
            tilde_s = transform.to_normalized(th_s)
            u_s = np.array([tilde_s[0], tilde_s[2], tilde_s[3]], dtype=np.float64)
            anchors_u.append(
                (float(sol["beta_hat"]), f"cpso_seed_{sol['optimizer_seed']}", u_s)
            )

    def _build_candidates_for_beta(
        beta_val: float, warm_u: _pl_Optional[np.ndarray], warm_label: str
    ) -> _pl_List[_pl_Tuple[str, np.ndarray]]:
        cands: _pl_List[_pl_Tuple[str, np.ndarray]] = []
        if warm_u is not None:
            cands.append((warm_label, warm_u.copy()))
        if anchors_u:
            # Aggiungi l'ancora empirica CPSO più vicina in beta (mai theta_dagger!)
            nearest = min(anchors_u, key=lambda item: abs(item[0] - beta_val))
            cands.append((nearest[1], nearest[2].copy()))
        cands.append(("prior_center", u_prior.copy()))
        # Deduplicazione mantenendo l'ordine
        unique_cands: _pl_List[_pl_Tuple[str, np.ndarray]] = []
        for lbl, vec in cands:
            if not any(np.allclose(vec, u_ex, atol=1.0e-7) for _, u_ex in unique_cands):
                unique_cands.append((lbl, vec))
        return unique_cands[:2]

    # 1. Scansione FORWARD (beta crescente: grid_arr[0] -> grid_arr[-1])
    forward_records: _pl_List[OptimizedProfilePointRecord] = []
    warm_fwd: _pl_Optional[np.ndarray] = None
    for idx in range(n_pts):
        b_val = float(grid_arr[idx])
        cands_fwd = _build_candidates_for_beta(b_val, warm_fwd, "warm_start_forward")
        rec_f = _optimize_nuisance_at_fixed_beta(
            profile_type=profile_type,
            scan_direction="forward",
            grid_index=idx + 1,
            beta_fixed=b_val,
            u_init_candidates=cands_fwd,
            forward_operator=forward_operator,
            y_observed=y_observed,
            transform=transform,
            canonical_regularizer=canonical_regularizer,
            planned_point_budget=planned_point_budget,
            scale_type=scale_type,
            ftol=ftol,
            gtol=gtol,
            fd_eps=fd_eps,
        )
        forward_records.append(rec_f)
        warm_fwd = np.array(
            [rec_f.alpha_tilde_hat, rec_f.D_tilde_hat, rec_f.E_tilde_hat],
            dtype=np.float64,
        )

    # 2. Scansione BACKWARD (beta decrescente: grid_arr[-1] -> grid_arr[0])
    backward_records_rev: _pl_List[OptimizedProfilePointRecord] = []
    warm_bwd: _pl_Optional[np.ndarray] = None
    for idx in range(n_pts - 1, -1, -1):
        b_val = float(grid_arr[idx])
        cands_bwd = _build_candidates_for_beta(b_val, warm_bwd, "warm_start_backward")
        rec_b = _optimize_nuisance_at_fixed_beta(
            profile_type=profile_type,
            scan_direction="backward",
            grid_index=idx + 1,
            beta_fixed=b_val,
            u_init_candidates=cands_bwd,
            forward_operator=forward_operator,
            y_observed=y_observed,
            transform=transform,
            canonical_regularizer=canonical_regularizer,
            planned_point_budget=planned_point_budget,
            scale_type=scale_type,
            ftol=ftol,
            gtol=gtol,
            fd_eps=fd_eps,
        )
        backward_records_rev.append(rec_b)
        warm_bwd = np.array(
            [rec_b.alpha_tilde_hat, rec_b.D_tilde_hat, rec_b.E_tilde_hat],
            dtype=np.float64,
        )

    backward_records = list(reversed(backward_records_rev))

    # 3. Costruzione dell'inviluppo combinato puntuale e diagnostica dell'isteresi direzionale
    combined_best: _pl_List[OptimizedProfilePointRecord] = []
    obj_diffs: _pl_List[float] = []
    param_tilde_diffs: _pl_List[float] = []

    for rec_f, rec_b in zip(forward_records, backward_records):
        diff_obj = abs(rec_f.optimized_objective_value - rec_b.optimized_objective_value)
        obj_diffs.append(float(diff_obj))
        u_f = np.array([rec_f.alpha_tilde_hat, rec_f.D_tilde_hat, rec_f.E_tilde_hat])
        u_b = np.array([rec_b.alpha_tilde_hat, rec_b.D_tilde_hat, rec_b.E_tilde_hat])
        param_tilde_diffs.append(float(np.linalg.norm(u_f - u_b)))

        chosen = (
            rec_f
            if rec_f.optimized_objective_value <= rec_b.optimized_objective_value
            else rec_b
        )
        combined_best.append(chosen)

    fwd_vals = np.array([r.optimized_objective_value for r in forward_records])
    bwd_vals = np.array([r.optimized_objective_value for r in backward_records])
    comb_vals = np.array([r.optimized_objective_value for r in combined_best])

    beta_min_fwd = float(grid_arr[int(np.argmin(fwd_vals))])
    beta_min_bwd = float(grid_arr[int(np.argmin(bwd_vals))])
    beta_min_comb = float(grid_arr[int(np.argmin(comb_vals))])

    total_evals = sum(r.fem_evaluations_used for r in forward_records) + sum(
        r.fem_evaluations_used for r in backward_records
    )
    wall_total = _pl_time.perf_counter() - t0_all

    dir_summary = {
        "max_abs_objective_discrepancy_fwd_vs_bwd": float(max(obj_diffs)),
        "mean_abs_objective_discrepancy_fwd_vs_bwd": float(np.mean(obj_diffs)),
        "max_l2_nuisance_tilde_discrepancy_fwd_vs_bwd": float(max(param_tilde_diffs)),
        "mean_l2_nuisance_tilde_discrepancy_fwd_vs_bwd": float(np.mean(param_tilde_diffs)),
        "beta_argmin_forward": beta_min_fwd,
        "beta_argmin_backward": beta_min_bwd,
        "beta_argmin_combined": beta_min_comb,
        "argmin_shift_abs_fwd_vs_bwd": abs(beta_min_fwd - beta_min_bwd),
        "conclusions_direction_independent": bool(abs(beta_min_fwd - beta_min_bwd) <= 0.05),
    }

    return BidirectionalProfileResult(
        profile_type=str(profile_type),
        beta_grid=grid_arr,
        planned_point_budget=int(planned_point_budget),
        forward_records=forward_records,
        backward_records=backward_records,
        combined_best_records=combined_best,
        total_fem_evaluations=int(total_evals),
        wall_time_seconds=float(wall_total),
        directional_discrepancy_summary=dir_summary,
    )
'''


def main() -> None:
    pl_path = Path("python/reteco/identifiability/profile_likelihood.py")
    if not pl_path.exists():
        raise FileNotFoundError(f"File non trovato: {pl_path}")

    current_text = pl_path.read_text(encoding="utf-8")
    if EXTENSION_MARKER in current_text:
        print(f"[update_profile_likelihood] Estensione Fase 3.9 già presente in {pl_path}.")
        return

    updated_text = current_text.rstrip() + "\n\n" + EXTENSION_CODE.lstrip()
    pl_path.write_text(updated_text, encoding="utf-8")
    print(f"[update_profile_likelihood] Estensione Fase 3.9 aggiunta con successo a {pl_path}.")


if __name__ == "__main__":
    main()