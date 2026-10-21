"""
experiments/10_beta_mesh_sensitivity.py
RETECO — Fase 3.10: Sensibilità del profilo dati Phi_data(beta) alla discretizzazione inversa.

Confronta il profilo del solo termine dati:
    Phi_data(beta) = min_{alpha, D, E} J_data(alpha, beta, D, E)
sulla griglia locale comune di 9 valori di beta:
    [0.425, 0.4375, 0.4500, 0.4625, 0.4750, 0.4791427903, 0.4875, 0.5000, 0.5250]
tra due discretizzazioni inverse:
    - Riferimento: N = 16, M = 80
    - Raffinata  : N = 24, M = 120
mantenendo fisso il target sintetico generato a (N = 32, M = 160) con SHA-256:
    1bca9e29c60d77c1450fd2fcea845d2320c9e44945dd7d3ccda52aa33ef261af

Salva tutti gli artefatti in `artifacts/10_beta_mesh_sensitivity/` senza sovrascrivere
gli artefatti delle fasi precedenti.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
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
from reteco.identifiability.profile_likelihood import (
    BidirectionalProfileResult,
    OptimizedProfilePointRecord,
    _optimize_nuisance_at_fixed_beta,
    compute_bidirectional_optimized_1d_profile,
)
from reteco.visualization.twin_diagnostics_plots import (
    _Apply_paper_rc_style,
    save_figure_png_and_pdf,
)

EXPECTED_TARGET_SHA256: str = (
    "1bca9e29c60d77c1450fd2fcea845d2320c9e44945dd7d3ccda52aa33ef261af"
)
EXPECTED_N_OBS: int = 180
FIXED_DATA_SEED: int = 20261016
FIXED_LAMBDA_REG: float = 1.0e-4
FIXED_PRIOR_TILDE: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5)
SYNTHETIC_REF_BETA_TRUE: float = 0.45

LOCAL_BETA_GRID: Tuple[float, ...] = (
    0.4250,
    0.4375,
    0.4500,
    0.4625,
    0.4750,
    0.4791427903,
    0.4875,
    0.5000,
    0.5250,
)

CENTRAL_MIN_REGION_BOUNDS: Tuple[float, float] = (0.4375, 0.5000)
INITIAL_PER_POINT_BUDGET: int = 32
SUPPLEMENTARY_PER_POINT_BUDGET: int = 64


@dataclass(frozen=True)
class SupplementaryPointVerification:
    resolution_label: str
    N_inv: int
    M_inv: int
    scan_direction: str
    grid_index: int
    beta_fixed: float
    primary_budget: int
    primary_fem_evaluations: int
    primary_j_data: float
    primary_converged: bool
    primary_stopping_criterion: str
    supplementary_budget: int
    supplementary_fem_evaluations: int
    supplementary_j_data: float
    supplementary_alpha_hat: float
    supplementary_D_hat: float
    supplementary_E_hat: float
    supplementary_relative_residual_l2: float
    supplementary_converged: bool
    supplementary_budget_exhausted: bool
    supplementary_stopping_criterion: str
    j_data_improvement: float
    wall_time_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
    context_label: str = "phase310_target_check",
) -> str:
    actual_sha256 = compute_sha256_vector(y_target_clean)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"[ABORT - {context_label}] Hash SHA-256 del target non corrispondente!\n"
            f"  Atteso : {expected_sha256}\n"
            f"  Ottenuto: {actual_sha256}\n"
            "Esecuzione interrotta immediatamente."
        )
    if int(y_target_clean.size) != EXPECTED_N_OBS:
        raise RuntimeError(
            f"[ABORT - {context_label}] Numero di osservazioni {y_target_clean.size} != {EXPECTED_N_OBS}."
        )
    return actual_sha256


def load_saved_clean_target_without_regeneration(
    master: Any,
    twin_cfg: Dict[str, Any],
    obs_op: DiscreteObservationOperator,
    theta_dagger: np.ndarray,
    out_dir: Path,
    expected_sha256: str = EXPECTED_TARGET_SHA256,
) -> Tuple[np.ndarray, str, str]:
    """
    Carica il vettore target pulito già salvato su disco senza rigenerarlo tra le risoluzioni,
    verificando l'hash SHA-256 e la dimensione (180 osservazioni).
    """
    candidate_paths = [
        Path("artifacts/07_cpso_multiseed/y_target_clean.npy"),
        Path("artifacts/09_beta_profile/y_target_clean.npy"),
        Path("artifacts/08_beta_landscape/y_target_clean.npy"),
        Path("artifacts/06_cpso_budget_diagnostic/y_target_clean.npy"),
        Path("artifacts/05_pilot_inverse/y_target_clean.npy"),
        out_dir / "y_target_clean.npy",
    ]
    for cpath in candidate_paths:
        if cpath.exists():
            y_loaded = np.asarray(np.load(cpath), dtype=np.float64).ravel()
            sha = verify_target_hash_or_abort(
                y_loaded,
                expected_sha256=expected_sha256,
                context_label=f"reused_from_{cpath}",
            )
            return y_loaded, sha, f"reused_saved_target:{cpath}"

    # Fallback unico se nessun file .npy è stato ancora salvato su disco:
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
        context_label="single_target_reconstruction_N32_M160",
    )
    np.save(out_dir / "y_target_clean.npy", y_target_clean)
    return y_target_clean, sha, "cached_once_N32_M160_and_frozen"


def _resolve_attr_or_call(obj: Any, attr_name: str, default: Any = None) -> Any:
    if not hasattr(obj, attr_name):
        return default
    val = getattr(obj, attr_name)
    return val() if callable(val) else val


def build_and_verify_inverse_forward_problem(
    *,
    subdivisions_N: int,
    time_steps_M: int,
    master: Any,
    twin_cfg: Dict[str, Any],
    obs_op: DiscreteObservationOperator,
) -> Tuple[RetecoForwardProblem2D, Callable[[np.ndarray], np.ndarray], Dict[str, Any]]:
    """
    Costruisce il problema diretto per la discretizzazione inversa (N, M) e verifica che
    i parametri di discretizzazione, la fisica, le forzanti e l'operatore osservativo
    corrispondano esattamente al protocollo congelato.
    """
    inv_cfg = twin_cfg["inverse_discretization"]
    final_T = float(inv_cfg["final_time_T"])
    grading_r = float(inv_cfg["grading_r"])
    fe_u = int(inv_cfg.get("fe_deg_disp", 2))
    fe_p = int(inv_cfg.get("fe_deg_press", 1))
    fe_c = int(inv_cfg.get("fe_deg_tracer", 1))

    time_grid = _core.TimeGrid.make_graded(final_T, int(time_steps_M), grading_r)
    prob = RetecoForwardProblem2D(
        subdivisions_N=int(subdivisions_N),
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=time_grid,
        fe_deg_disp=fe_u,
        fe_deg_press=fe_p,
        fe_deg_tracer=fe_c,
    )
    c0_vec, F_mech_cb, S_fluid_cb = build_standard_driving_Callbacks(
        prob, twin_cfg["driving_protocol"]
    )

    actual_N = int(_resolve_attr_or_call(prob, "subdivisions_N", subdivisions_N))
    raw_M = _resolve_attr_or_call(prob.time_grid, "num_steps", None)
    if raw_M is None:
        raw_M = _resolve_attr_or_call(prob.time_grid, "M", None)
    if raw_M is None:
        times_seq = _resolve_attr_or_call(prob.time_grid, "times", None)
        raw_M = (len(times_seq) - 1) if times_seq is not None else time_steps_M
    actual_M = int(raw_M)

    if actual_N != int(subdivisions_N) or actual_M != int(time_steps_M):
        raise RuntimeError(
            f"Disallineamento configurazione solver: atteso (N={subdivisions_N}, M={time_steps_M}), "
            f"ottenuto (N={actual_N}, M={actual_M})."
        )

    n_dofs_d = int(
        _resolve_attr_or_call(
            prob, "num_disp_dofs", _resolve_attr_or_call(prob, "n_dofs_disp", 0)
        )
    )
    n_dofs_p = int(
        _resolve_attr_or_call(
            prob, "num_press_dofs", _resolve_attr_or_call(prob, "n_dofs_press", 0)
        )
    )
    n_dofs_c = int(
        _resolve_attr_or_call(
            prob, "num_tracer_dofs", _resolve_attr_or_call(prob, "n_dofs_tracer", 0)
        )
    )

    def forward_op(theta: np.ndarray) -> np.ndarray:
        traj = prob.solve_trajectory(
            theta, c0=c0_vec, F_mech_func=F_mech_cb, S_fluid_func=S_fluid_cb
        )
        y_sim = np.asarray(obs_op.apply(prob, traj), dtype=np.float64).ravel()
        if int(y_sim.size) != EXPECTED_N_OBS:
            raise RuntimeError(
                f"Operatore osservativo a (N={subdivisions_N}, M={time_steps_M}) ha restituito "
                f"{y_sim.size} osservazioni anziché {EXPECTED_N_OBS}."
            )
        return y_sim

    solver_audit = {
        "subdivisions_N": actual_N,
        "time_steps_M": actual_M,
        "final_time_T": final_T,
        "grading_r": grading_r,
        "fe_degrees": {"disp": fe_u, "press": fe_p, "tracer": fe_c},
        "n_dofs_disp": n_dofs_d,
        "n_dofs_press": n_dofs_p,
        "n_dofs_tracer": n_dofs_c,
        "forcing_protocol_version": FORCING_PROTOCOL_VERSION,
        "expected_n_obs": EXPECTED_N_OBS,
    }
    return prob, forward_op, solver_audit

def load_phase37_cpso_anchors() -> List[Dict[str, Any]]:
    p37_path = Path("artifacts/07_cpso_multiseed/cpso_multiseed_summary.json")
    if not p37_path.exists():
        return []
    payload = json.loads(p37_path.read_text(encoding="utf-8"))
    rows = payload.get("comparison_table_rows", [])
    if len(rows) == 3:
        return sorted(rows, key=lambda r: int(r["optimizer_seed"]))
    return []


def build_or_reuse_n16_m80_profile_on_local_grid(
    *,
    beta_grid: Sequence[float],
    forward_operator_16: Callable[[np.ndarray], np.ndarray],
    y_observed: np.ndarray,
    transform: ParameterCoordinateTransform,
    canonical_regularizer: TikhonovRegularizer,
    planned_point_budget: int,
    cpso_anchors: Sequence[Dict[str, Any]],
    phase39_summary_path: Path = Path("artifacts/09_beta_profile/beta_profile_summary.json"),
) -> Tuple[BidirectionalProfileResult, Dict[float, bool], Dict[str, Any]]:
    """
    Costruisce il profilo dati Phi_data(beta) sulla griglia locale di 9 punti per N=16, M=80.
    Riutilizza i punti già calcolati nella Fase 3.9 SOLO SE target SHA-256, discretizzazione
    (N=16, M=80), modalità del funzionale, trasformazioni e budget per punto (32) sono identici;
    calcola i punti nuovi (0.4375, 0.4625, 0.4875) con warm start forward e backward.
    """
    t0 = time.perf_counter()
    grid_arr = np.asarray(sorted(float(b) for b in beta_grid), dtype=np.float64)
    reused_flags: Dict[float, bool] = {float(b): False for b in grid_arr}

    can_reuse_p39 = False
    p39_fwd_by_beta: Dict[float, OptimizedProfilePointRecord] = {}
    p39_bwd_by_beta: Dict[float, OptimizedProfilePointRecord] = {}
    reuse_audit_reason = "phase39_artifact_not_found"

    if phase39_summary_path.exists():
        p39 = json.loads(phase39_summary_path.read_text(encoding="utf-8"))
        fp = p39.get("frozen_protocol", {})
        bs = p39.get("budget_specification", {})
        protocol_matches = (
            str(fp.get("target_sha256", "")) == EXPECTED_TARGET_SHA256
            and int(fp.get("N_inv", -1)) == 16
            and int(fp.get("M_inv", -1)) == 80
            and str(fp.get("objective_mode", "")) == "mode_a_deterministic_normalized_linf"
            and int(bs.get("planned_per_point_per_direction_fem_budget", -1)) == int(planned_point_budget)
            and bool(fp.get("true_parameters_used_in_optimization", True)) is False
        )
        if protocol_matches:
            can_reuse_p39 = True
            reuse_audit_reason = "verified_exact_protocol_match_with_phase39"
            p_data = p39.get("profile_data_summary", {})
            for d in p_data.get("forward_records", []):
                rec = OptimizedProfilePointRecord(**d)
                p39_fwd_by_beta[float(rec.beta_fixed)] = rec
            for d in p_data.get("backward_records", []):
                rec = OptimizedProfilePointRecord(**d)
                p39_bwd_by_beta[float(rec.beta_fixed)] = rec
        else:
            reuse_audit_reason = f"protocol_mismatch_in_phase39:{fp}"

    def _find_matching_p39(
        lookup: Dict[float, OptimizedProfilePointRecord], b_target: float, tol: float = 1.0e-9
    ) -> Optional[OptimizedProfilePointRecord]:
        for b_k, rec in lookup.items():
            if abs(b_k - b_target) <= tol:
                return rec
        return None

    if not can_reuse_p39:
        prof_full = compute_bidirectional_optimized_1d_profile(
            profile_type="data_only",
            beta_grid=grid_arr,
            forward_operator=forward_operator_16,
            y_observed=y_observed,
            transform=transform,
            canonical_regularizer=canonical_regularizer,
            planned_point_budget=planned_point_budget,
            empirical_anchor_solutions=cpso_anchors,
            scale_type="linf",
        )
        audit_info = {
            "phase39_reused": False,
            "reuse_audit_reason": reuse_audit_reason,
            "reused_betas": [],
            "newly_computed_betas": grid_arr.tolist(),
            "newly_executed_fem_evaluations_n16": prof_full.total_fem_evaluations,
        }
        return prof_full, reused_flags, audit_info

    u_prior = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    anchors_u: List[Tuple[float, str, np.ndarray]] = []
    for sol in cpso_anchors:
        th_s = np.array(
            [float(sol["alpha_hat"]), float(sol["beta_hat"]), float(sol["D_hat"]), float(sol["E_hat"])],
            dtype=np.float64,
        )
        tilde_s = transform.to_normalized(th_s)
        anchors_u.append(
            (
                float(sol["beta_hat"]),
                f"cpso_seed_{sol['optimizer_seed']}",
                np.array([tilde_s[0], tilde_s[2], tilde_s[3]], dtype=np.float64),
            )
        )

    def _candidates(beta_val: float, warm_u: Optional[np.ndarray], warm_label: str):
        cands: List[Tuple[str, np.ndarray]] = []
        if warm_u is not None:
            cands.append((warm_label, warm_u.copy()))
        if anchors_u:
            nearest = min(anchors_u, key=lambda item: abs(item[0] - beta_val))
            cands.append((nearest[1], nearest[2].copy()))
        cands.append(("prior_center", u_prior.copy()))
        unique: List[Tuple[str, np.ndarray]] = []
        for lbl, vec in cands:
            if not any(np.allclose(vec, ex, atol=1.0e-7) for _, ex in unique):
                unique.append((lbl, vec))
        return unique[:2]

    forward_records: List[OptimizedProfilePointRecord] = []
    warm_fwd: Optional[np.ndarray] = None
    new_fem_evals = 0
    reused_betas_list: List[float] = []
    new_betas_list: List[float] = []

    for idx, b_val in enumerate(grid_arr):
        b_f = float(b_val)
        rec_p39 = _find_matching_p39(p39_fwd_by_beta, b_f)
        if rec_p39 is not None:
            rec_f = OptimizedProfilePointRecord(
                **{
                    **rec_p39.to_dict(),
                    "grid_index": idx + 1,
                    "initial_guess_source": f"reused_phase39:{rec_p39.initial_guess_source}",
                }
            )
            reused_flags[b_f] = True
            if b_f not in reused_betas_list:
                reused_betas_list.append(b_f)
        else:
            cands_f = _candidates(b_f, warm_fwd, "warm_start_forward")
            rec_f = _optimize_nuisance_at_fixed_beta(
                profile_type="data_only",
                scan_direction="forward",
                grid_index=idx + 1,
                beta_fixed=b_f,
                u_init_candidates=cands_f,
                forward_operator=forward_operator_16,
                y_observed=y_observed,
                transform=transform,
                canonical_regularizer=canonical_regularizer,
                planned_point_budget=planned_point_budget,
                scale_type="linf",
            )
            new_fem_evals += int(rec_f.fem_evaluations_used)
            if b_f not in new_betas_list:
                new_betas_list.append(b_f)
        forward_records.append(rec_f)
        warm_fwd = np.array([rec_f.alpha_tilde_hat, rec_f.D_tilde_hat, rec_f.E_tilde_hat], dtype=np.float64)

    backward_records_rev: List[OptimizedProfilePointRecord] = []
    warm_bwd: Optional[np.ndarray] = None
    for idx in range(len(grid_arr) - 1, -1, -1):
        b_f = float(grid_arr[idx])
        rec_p39_b = _find_matching_p39(p39_bwd_by_beta, b_f)
        if rec_p39_b is not None:
            rec_b = OptimizedProfilePointRecord(
                **{
                    **rec_p39_b.to_dict(),
                    "grid_index": idx + 1,
                    "initial_guess_source": f"reused_phase39:{rec_p39_b.initial_guess_source}",
                }
            )
        else:
            cands_b = _candidates(b_f, warm_bwd, "warm_start_backward")
            rec_b = _optimize_nuisance_at_fixed_beta(
                profile_type="data_only",
                scan_direction="backward",
                grid_index=idx + 1,
                beta_fixed=b_f,
                u_init_candidates=cands_b,
                forward_operator=forward_operator_16,
                y_observed=y_observed,
                transform=transform,
                canonical_regularizer=canonical_regularizer,
                planned_point_budget=planned_point_budget,
                scale_type="linf",
            )
            new_fem_evals += int(rec_b.fem_evaluations_used)
        backward_records_rev.append(rec_b)
        warm_bwd = np.array([rec_b.alpha_tilde_hat, rec_b.D_tilde_hat, rec_b.E_tilde_hat], dtype=np.float64)

    backward_records = list(reversed(backward_records_rev))

    combined_best: List[OptimizedProfilePointRecord] = []
    obj_diffs: List[float] = []
    param_diffs: List[float] = []
    for rf, rb in zip(forward_records, backward_records):
        obj_diffs.append(float(abs(rf.j_data - rb.j_data)))
        uf = np.array([rf.alpha_tilde_hat, rf.D_tilde_hat, rf.E_tilde_hat])
        ub = np.array([rb.alpha_tilde_hat, rb.D_tilde_hat, rb.E_tilde_hat])
        param_diffs.append(float(np.linalg.norm(uf - ub)))
        combined_best.append(rf if rf.j_data <= rb.j_data else rb)

    fwd_vals = np.array([r.j_data for r in forward_records])
    bwd_vals = np.array([r.j_data for r in backward_records])
    comb_vals = np.array([r.j_data for r in combined_best])

    beta_min_fwd = float(grid_arr[int(np.argmin(fwd_vals))])
    beta_min_bwd = float(grid_arr[int(np.argmin(bwd_vals))])
    beta_min_comb = float(grid_arr[int(np.argmin(comb_vals))])

    total_evals_accounting = sum(r.fem_evaluations_used for r in forward_records) + sum(
        r.fem_evaluations_used for r in backward_records
    )
    wall_s = time.perf_counter() - t0

    dir_summary = {
        "max_abs_objective_discrepancy_fwd_vs_bwd": float(max(obj_diffs)),
        "mean_abs_objective_discrepancy_fwd_vs_bwd": float(np.mean(obj_diffs)),
        "max_l2_nuisance_tilde_discrepancy_fwd_vs_bwd": float(max(param_diffs)),
        "mean_l2_nuisance_tilde_discrepancy_fwd_vs_bwd": float(np.mean(param_diffs)),
        "beta_argmin_forward": beta_min_fwd,
        "beta_argmin_backward": beta_min_bwd,
        "beta_argmin_combined": beta_min_comb,
        "argmin_shift_abs_fwd_vs_bwd": abs(beta_min_fwd - beta_min_bwd),
        "conclusions_direction_independent": bool(abs(beta_min_fwd - beta_min_bwd) <= 0.03),
    }

    prof_res = BidirectionalProfileResult(
        profile_type="data_only",
        beta_grid=grid_arr,
        planned_point_budget=int(planned_point_budget),
        forward_records=forward_records,
        backward_records=backward_records,
        combined_best_records=combined_best,
        total_fem_evaluations=int(total_evals_accounting),
        wall_time_seconds=float(wall_s),
        directional_discrepancy_summary=dir_summary,
    )
    audit_info = {
        "phase39_reused": True,
        "reuse_audit_reason": reuse_audit_reason,
        "reused_betas": reused_betas_list,
        "newly_computed_betas": new_betas_list,
        "newly_executed_fem_evaluations_n16": int(new_fem_evals),
        "total_profile_fem_evaluations_including_reused": int(total_evals_accounting),
    }
    return prof_res, reused_flags, audit_info


def run_supplementary_verifications_for_nonconverged_central_points(
    *,
    resolution_label: str,
    N_inv: int,
    M_inv: int,
    prof: BidirectionalProfileResult,
    forward_operator: Callable[[np.ndarray], np.ndarray],
    y_observed: np.ndarray,
    transform: ParameterCoordinateTransform,
    canonical_regularizer: TikhonovRegularizer,
    supplementary_budget: int = SUPPLEMENTARY_PER_POINT_BUDGET,
    central_bounds: Tuple[float, float] = CENTRAL_MIN_REGION_BOUNDS,
) -> List[SupplementaryPointVerification]:
    verifications: List[SupplementaryPointVerification] = []
    comb_vals = np.array([r.j_data for r in prof.combined_best_records], dtype=np.float64)
    min_beta_sampled = float(prof.beta_grid[int(np.argmin(comb_vals))])

    for direction_label, records in (
        ("forward", prof.forward_records),
        ("backward", prof.backward_records),
    ):
        for rec in records:
            b_val = float(rec.beta_fixed)
            in_central_region = (
                central_bounds[0] - 1.0e-9 <= b_val <= central_bounds[1] + 1.0e-9
            ) or (abs(b_val - min_beta_sampled) <= 0.026)

            if in_central_region and (not rec.converged or rec.budget_exhausted):
                u_start = np.array(
                    [rec.alpha_tilde_hat, rec.D_tilde_hat, rec.E_tilde_hat],
                    dtype=np.float64,
                )
                u_prior = np.array([0.5, 0.5, 0.5], dtype=np.float64)
                cands = [
                    (f"warm_continuation_from_{direction_label}_32evals", u_start),
                    ("prior_center", u_prior),
                ]
                supp_rec = _optimize_nuisance_at_fixed_beta(
                    profile_type="data_only",
                    scan_direction=f"{direction_label}_supplementary",
                    grid_index=rec.grid_index,
                    beta_fixed=b_val,
                    u_init_candidates=cands,
                    forward_operator=forward_operator,
                    y_observed=y_observed,
                    transform=transform,
                    canonical_regularizer=canonical_regularizer,
                    planned_point_budget=int(supplementary_budget),
                    scale_type="linf",
                )
                verifications.append(
                    SupplementaryPointVerification(
                        resolution_label=resolution_label,
                        N_inv=int(N_inv),
                        M_inv=int(M_inv),
                        scan_direction=direction_label,
                        grid_index=int(rec.grid_index),
                        beta_fixed=b_val,
                        primary_budget=int(rec.planned_point_budget),
                        primary_fem_evaluations=int(rec.fem_evaluations_used),
                        primary_j_data=float(rec.j_data),
                        primary_converged=bool(rec.converged),
                        primary_stopping_criterion=str(rec.stopping_criterion),
                        supplementary_budget=int(supplementary_budget),
                        supplementary_fem_evaluations=int(supp_rec.fem_evaluations_used),
                        supplementary_j_data=float(supp_rec.j_data),
                        supplementary_alpha_hat=float(supp_rec.alpha_hat),
                        supplementary_D_hat=float(supp_rec.D_hat),
                        supplementary_E_hat=float(supp_rec.E_hat),
                        supplementary_relative_residual_l2=float(supp_rec.relative_residual_l2),
                        supplementary_converged=bool(supp_rec.converged),
                        supplementary_budget_exhausted=bool(supp_rec.budget_exhausted),
                        supplementary_stopping_criterion=str(supp_rec.stopping_criterion),
                        j_data_improvement=float(rec.j_data - supp_rec.j_data),
                        wall_time_seconds=float(supp_rec.wall_time_seconds),
                    )
                )
    return verifications


def verify_profile_consistency(
    prof: BidirectionalProfileResult,
    forward_operator: Callable[[np.ndarray], np.ndarray],
    obj_diag: InverseObjectiveFunctional,
) -> Dict[str, Any]:
    all_recs = list(prof.forward_records) + list(prof.backward_records)
    all_admissible = all(r.is_admissible for r in all_recs)
    max_coord_err = float(max(r.coordinate_roundtrip_error for r in all_recs))
    max_decomp_err = float(max(r.j_decomposition_error for r in all_recs))
    j_sum_ok = all(
        np.isclose(r.j_total, r.j_data + r.j_reg, rtol=1.0e-12, atol=1.0e-14)
        for r in all_recs
    )
    budget_ok = all(1 <= r.fem_evaluations_used <= r.planned_point_budget for r in all_recs)
    exhausted_not_conv = all(not (r.budget_exhausted and r.converged) for r in all_recs)

    best_rec = min(prof.combined_best_records, key=lambda r: r.j_data)
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
        and np.isclose(
            bd_recheck.relative_residual_l2,
            best_rec.relative_residual_l2,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
    )

    passed = bool(
        all_admissible
        and (max_coord_err <= 1.0e-12)
        and j_sum_ok
        and budget_ok
        and exhausted_not_conv
        and repro_ok
    )
    if not passed:
        raise RuntimeError(f"Controllo di coerenza fallito sul profilo: {prof.profile_type}")

    return {
        "all_checks_passed": passed,
        "all_parameters_admissible": all_admissible,
        "physical_normalized_transform_consistent": bool(max_coord_err <= 1.0e-12),
        "max_coordinate_roundtrip_error": max_coord_err,
        "j_total_equals_j_data_plus_j_reg": j_sum_ok,
        "max_j_decomposition_error": max_decomp_err,
        "per_point_budget_respected": budget_ok,
        "exhausted_points_never_marked_converged": exhausted_not_conv,
        "reproducibility_verified_on_best_point": repro_ok,
        "forward_converged_count": int(sum(1 for r in prof.forward_records if r.converged)),
        "forward_budget_exhausted_count": int(
            sum(1 for r in prof.forward_records if r.budget_exhausted)
        ),
        "backward_converged_count": int(sum(1 for r in prof.backward_records if r.converged)),
        "backward_budget_exhausted_count": int(
            sum(1 for r in prof.backward_records if r.budget_exhausted)
        ),
    }


def write_single_resolution_csv(
    prof: BidirectionalProfileResult,
    reused_flags: Dict[float, bool],
    N_inv: int,
    M_inv: int,
    csv_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "N_inv",
        "M_inv",
        "scan_direction",
        "grid_index",
        "beta_fixed",
        "reused_from_phase39",
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
        "relative_residual_l2",
        "weighted_residual_l2_norm",
        "unweighted_residual_l2_norm",
        "unweighted_residual_linf_norm",
        "fem_evaluations_used",
        "planned_point_budget",
        "converged",
        "budget_exhausted",
        "stopping_criterion",
        "initial_guess_source",
        "is_admissible",
        "wall_time_seconds",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seq_label, seq in (
            ("forward", prof.forward_records),
            ("backward", prof.backward_records),
            ("combined_best", prof.combined_best_records),
        ):
            for r in seq:
                d = r.to_dict()
                b_f = float(r.beta_fixed)
                reused = bool(
                    any(abs(b_f - k) <= 1.0e-9 and v for k, v in reused_flags.items())
                )
                row = {k: d.get(k, "") for k in fieldnames}
                row["N_inv"] = int(N_inv)
                row["M_inv"] = int(M_inv)
                row["scan_direction"] = (
                    seq_label
                    if seq_label != "combined_best"
                    else f"combined_best_{r.scan_direction}"
                )
                row["reused_from_phase39"] = reused
                writer.writerow(row)


def write_mesh_comparison_csv(
    prof_16: BidirectionalProfileResult,
    reused_16: Dict[float, bool],
    prof_24: BidirectionalProfileResult,
    csv_path: Path,
) -> List[Dict[str, Any]]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "grid_index",
        "beta_fixed",
        "n16_reused_from_phase39",
        "n16_j_data_combined",
        "n16_j_data_fwd",
        "n16_j_data_bwd",
        "n16_fwd_bwd_abs_diff",
        "n16_rel_residual_l2",
        "n16_alpha_hat",
        "n16_D_hat",
        "n16_E_hat",
        "n16_fem_evals_fwd",
        "n16_fem_evals_bwd",
        "n16_converged_fwd",
        "n16_converged_bwd",
        "n16_init_source_fwd",
        "n16_init_source_bwd",
        "n24_j_data_combined",
        "n24_j_data_fwd",
        "n24_j_data_bwd",
        "n24_fwd_bwd_abs_diff",
        "n24_rel_residual_l2",
        "n24_alpha_hat",
        "n24_D_hat",
        "n24_E_hat",
        "n24_fem_evals_fwd",
        "n24_fem_evals_bwd",
        "n24_converged_fwd",
        "n24_converged_bwd",
        "n24_init_source_fwd",
        "n24_init_source_bwd",
        "ratio_j_data_n16_over_n24",
        "diff_alpha_hat_n24_minus_n16",
        "diff_D_hat_n24_minus_n16",
        "diff_E_hat_n24_minus_n16",
    ]
    rows: List[Dict[str, Any]] = []
    for idx in range(len(prof_16.beta_grid)):
        r16_c = prof_16.combined_best_records[idx]
        r16_f = prof_16.forward_records[idx]
        r16_b = prof_16.backward_records[idx]

        r24_c = prof_24.combined_best_records[idx]
        r24_f = prof_24.forward_records[idx]
        r24_b = prof_24.backward_records[idx]

        b_val = float(r16_c.beta_fixed)
        reused = bool(any(abs(b_val - k) <= 1.0e-9 and v for k, v in reused_16.items()))

        row = {
            "grid_index": idx + 1,
            "beta_fixed": b_val,
            "n16_reused_from_phase39": reused,
            "n16_j_data_combined": float(r16_c.j_data),
            "n16_j_data_fwd": float(r16_f.j_data),
            "n16_j_data_bwd": float(r16_b.j_data),
            "n16_fwd_bwd_abs_diff": float(abs(r16_f.j_data - r16_b.j_data)),
            "n16_rel_residual_l2": float(r16_c.relative_residual_l2),
            "n16_alpha_hat": float(r16_c.alpha_hat),
            "n16_D_hat": float(r16_c.D_hat),
            "n16_E_hat": float(r16_c.E_hat),
            "n16_fem_evals_fwd": int(r16_f.fem_evaluations_used),
            "n16_fem_evals_bwd": int(r16_b.fem_evaluations_used),
            "n16_converged_fwd": bool(r16_f.converged),
            "n16_converged_bwd": bool(r16_b.converged),
            "n16_init_source_fwd": str(r16_f.initial_guess_source),
            "n16_init_source_bwd": str(r16_b.initial_guess_source),
            "n24_j_data_combined": float(r24_c.j_data),
            "n24_j_data_fwd": float(r24_f.j_data),
            "n24_j_data_bwd": float(r24_b.j_data),
            "n24_fwd_bwd_abs_diff": float(abs(r24_f.j_data - r24_b.j_data)),
            "n24_rel_residual_l2": float(r24_c.relative_residual_l2),
            "n24_alpha_hat": float(r24_c.alpha_hat),
            "n24_D_hat": float(r24_c.D_hat),
            "n24_E_hat": float(r24_c.E_hat),
            "n24_fem_evals_fwd": int(r24_f.fem_evaluations_used),
            "n24_fem_evals_bwd": int(r24_b.fem_evaluations_used),
            "n24_converged_fwd": bool(r24_f.converged),
            "n24_converged_bwd": bool(r24_b.converged),
            "n24_init_source_fwd": str(r24_f.initial_guess_source),
            "n24_init_source_bwd": str(r24_b.initial_guess_source),
            "ratio_j_data_n16_over_n24": float(r16_c.j_data / max(r24_c.j_data, 1.0e-30)),
            "diff_alpha_hat_n24_minus_n16": float(r24_c.alpha_hat - r16_c.alpha_hat),
            "diff_D_hat_n24_minus_n16": float(r24_c.D_hat - r16_c.D_hat),
            "diff_E_hat_n24_minus_n16": float(r24_c.E_hat - r16_c.E_hat),
        }
        rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def write_supplementary_verifications_csv(
    verifications: Sequence[SupplementaryPointVerification], csv_path: Path
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "resolution_label",
        "N_inv",
        "M_inv",
        "scan_direction",
        "grid_index",
        "beta_fixed",
        "primary_budget",
        "primary_fem_evaluations",
        "primary_j_data",
        "primary_converged",
        "primary_stopping_criterion",
        "supplementary_budget",
        "supplementary_fem_evaluations",
        "supplementary_j_data",
        "supplementary_alpha_hat",
        "supplementary_D_hat",
        "supplementary_E_hat",
        "supplementary_relative_residual_l2",
        "supplementary_converged",
        "supplementary_budget_exhausted",
        "supplementary_stopping_criterion",
        "j_data_improvement",
        "wall_time_seconds",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for v in verifications:
            writer.writerow(v.to_dict())


def plot_beta_mesh_sensitivity_figure(
    *,
    prof_16: BidirectionalProfileResult,
    prof_24: BidirectionalProfileResult,
    bd_true_16: Any,
    bd_true_24: Any,
    supplementary_verifications: Sequence[SupplementaryPointVerification],
    out_path: Path,
    beta_true_ref: float = SYNTHETIC_REF_BETA_TRUE,
) -> Tuple[Path, Path]:
    betas = prof_16.beta_grid
    j16_comb = np.array([r.j_data for r in prof_16.combined_best_records], dtype=np.float64)
    j16_fwd = np.array([r.j_data for r in prof_16.forward_records], dtype=np.float64)
    j16_bwd = np.array([r.j_data for r in prof_16.backward_records], dtype=np.float64)

    j24_comb = np.array([r.j_data for r in prof_24.combined_best_records], dtype=np.float64)
    j24_fwd = np.array([r.j_data for r in prof_24.forward_records], dtype=np.float64)
    j24_bwd = np.array([r.j_data for r in prof_24.backward_records], dtype=np.float64)

    with plt.rc_context(_Apply_paper_rc_style()):
        fig, (ax_prof, ax_nuis) = plt.subplots(1, 2, figsize=(12.4, 4.8))

        ax_prof.plot(
            betas,
            j16_comb,
            color="#1f77b4",
            linestyle="-",
            marker="o",
            markersize=5.2,
            linewidth=2.0,
            label=r"$\Phi_{\mathrm{data}}(\beta)$ at $N=16, M=80$ (combined)",
            zorder=4,
        )
        ax_prof.fill_between(
            betas,
            np.minimum(j16_fwd, j16_bwd),
            np.maximum(j16_fwd, j16_bwd),
            color="#1f77b4",
            alpha=0.16,
            label=r"$N=16$ fwd/bwd directional band",
        )

        ax_prof.plot(
            betas,
            j24_comb,
            color="#d62728",
            linestyle="-",
            marker="s",
            markersize=5.2,
            linewidth=2.0,
            label=r"$\Phi_{\mathrm{data}}(\beta)$ at $N=24, M=120$ (combined)",
            zorder=4,
        )
        ax_prof.fill_between(
            betas,
            np.minimum(j24_fwd, j24_bwd),
            np.maximum(j24_fwd, j24_bwd),
            color="#d62728",
            alpha=0.16,
            label=r"$N=24$ fwd/bwd directional band",
        )

        idx_min_16 = int(np.argmin(j16_comb))
        idx_min_24 = int(np.argmin(j24_comb))
        ax_prof.plot(
            betas[idx_min_16],
            j16_comb[idx_min_16],
            marker="D",
            markersize=8.0,
            markerfacecolor="none",
            markeredgecolor="#08306b",
            markeredgewidth=1.8,
            linestyle="none",
            zorder=6,
            label=f"Sampled min $N=16$: $\\beta={betas[idx_min_16]:.4f}$",
        )
        ax_prof.plot(
            betas[idx_min_24],
            j24_comb[idx_min_24],
            marker="D",
            markersize=8.0,
            markerfacecolor="none",
            markeredgecolor="#67000d",
            markeredgewidth=1.8,
            linestyle="none",
            zorder=6,
            label=f"Sampled min $N=24$: $\\beta={betas[idx_min_24]:.4f}$",
        )

        nonconv_betas: List[float] = []
        nonconv_vals: List[float] = []
        for r in prof_16.combined_best_records:
            if not r.converged or r.budget_exhausted:
                nonconv_betas.append(float(r.beta_fixed))
                nonconv_vals.append(float(r.j_data))
        for r in prof_24.combined_best_records:
            if not r.converged or r.budget_exhausted:
                nonconv_betas.append(float(r.beta_fixed))
                nonconv_vals.append(float(r.j_data))

        if nonconv_betas:
            ax_prof.scatter(
                nonconv_betas,
                nonconv_vals,
                marker="x",
                s=72,
                color="#111111",
                linewidths=1.8,
                zorder=7,
                label="Budget exhausted (32 evals, non-converged)",
            )

        if supplementary_verifications:
            supp_b = [v.beta_fixed for v in supplementary_verifications]
            supp_j = [v.supplementary_j_data for v in supplementary_verifications]
            ax_prof.scatter(
                supp_b,
                supp_j,
                marker="P",
                s=58,
                color="#2ca02c",
                edgecolors="black",
                linewidths=0.6,
                zorder=7,
                label="Supplementary check (64 evals)",
            )

        ax_prof.axvline(
            beta_true_ref,
            color="#333333",
            linestyle=":",
            linewidth=1.4,
            label=r"Synthetic reference $\beta^\dagger = 0.45$",
        )
        ax_prof.plot(
            beta_true_ref,
            float(bd_true_16.j_data),
            marker="*",
            markersize=9.0,
            color="#1f77b4",
            markeredgecolor="black",
            markeredgewidth=0.5,
            linestyle="none",
            zorder=6,
            label=f"$J_{{\\mathrm{{data}}}}(\\boldsymbol{{\\theta}}^\\dagger)$ at $N=16$ ({float(bd_true_16.j_data):.2e})",
        )
        ax_prof.plot(
            beta_true_ref,
            float(bd_true_24.j_data),
            marker="*",
            markersize=9.0,
            color="#d62728",
            markeredgecolor="black",
            markeredgewidth=0.5,
            linestyle="none",
            zorder=6,
            label=f"$J_{{\\mathrm{{data}}}}(\\boldsymbol{{\\theta}}^\\dagger)$ at $N=24$ ({float(bd_true_24.j_data):.2e})",
        )

        ax_prof.set_yscale("log")
        ax_prof.set_title(
            r"(a) Data Profile $\Phi_{\mathrm{data}}(\beta)$ vs Inverse Discretization ($N=16$ vs $N=24$)"
        )
        ax_prof.set_xlabel(r"Viscoelastic Fractional Order $\beta$")
        ax_prof.set_ylabel(r"Optimized Data Cost $\Phi_{\mathrm{data}}(\beta)$ (log scale)")
        ax_prof.grid(True, which="both", alpha=0.45)
        ax_prof.legend(loc="best", fontsize=7.2, frameon=True)

        err_a_16 = np.array([100.0 * (r.alpha_hat - 0.70) / 0.70 for r in prof_16.combined_best_records])
        err_D_16 = np.array([100.0 * (r.D_hat - 0.08) / 0.08 for r in prof_16.combined_best_records])
        err_E_16 = np.array([100.0 * (r.E_hat - 2.50) / 2.50 for r in prof_16.combined_best_records])

        err_a_24 = np.array([100.0 * (r.alpha_hat - 0.70) / 0.70 for r in prof_24.combined_best_records])
        err_D_24 = np.array([100.0 * (r.D_hat - 0.08) / 0.08 for r in prof_24.combined_best_records])
        err_E_24 = np.array([100.0 * (r.E_hat - 2.50) / 2.50 for r in prof_24.combined_best_records])

        ax_nuis.plot(betas, err_a_16, color="#1f77b4", linestyle="--", marker="o", markersize=3.8, label=r"$\Delta\alpha/\alpha^\dagger$ [%] ($N=16$)")
        ax_nuis.plot(betas, err_D_16, color="#ff7f0e", linestyle="--", marker="s", markersize=3.8, label=r"$\Delta D/D^\dagger$ [%] ($N=16$)")
        ax_nuis.plot(betas, err_E_16, color="#2ca02c", linestyle="--", marker="^", markersize=3.8, label=r"$\Delta E/E^\dagger$ [%] ($N=16$)")

        ax_nuis.plot(betas, err_a_24, color="#1f77b4", linestyle="-", marker="o", markersize=4.2, linewidth=1.8, label=r"$\Delta\alpha/\alpha^\dagger$ [%] ($N=24$)")
        ax_nuis.plot(betas, err_D_24, color="#ff7f0e", linestyle="-", marker="s", markersize=4.2, linewidth=1.8, label=r"$\Delta D/D^\dagger$ [%] ($N=24$)")
        ax_nuis.plot(betas, err_E_24, color="#2ca02c", linestyle="-", marker="^", markersize=4.2, linewidth=1.8, label=r"$\Delta E/E^\dagger$ [%] ($N=24$)")

        ax_nuis.axhline(0.0, color="#555555", linestyle="-", linewidth=0.9, alpha=0.7)
        ax_nuis.axvline(beta_true_ref, color="#333333", linestyle=":", linewidth=1.4)
        ax_nuis.set_title(r"(b) Nuisance Parameter Shifts Relative to $\boldsymbol{\theta}^\dagger$ ($N=16$ vs $N=24$)")
        ax_nuis.set_xlabel(r"Viscoelastic Fractional Order $\beta$")
        ax_nuis.set_ylabel(r"Signed Relative Deviation $100 \cdot (\widehat{\theta}_i - \theta_i^\dagger)/\theta_i^\dagger$ [%]")
        ax_nuis.grid(True, alpha=0.45)
        ax_nuis.legend(loc="best", fontsize=7.2, ncol=2, frameon=True)

        fig.tight_layout()
        paths = save_figure_png_and_pdf(fig, out_path)
        plt.close(fig)
        return paths


def summarize_mesh_sensitivity_findings(
    prof_16: BidirectionalProfileResult,
    prof_24: BidirectionalProfileResult,
    bd_true_16: Any,
    bd_true_24: Any,
    supplementary: Sequence[SupplementaryPointVerification],
) -> Dict[str, Any]:
    betas = prof_16.beta_grid
    j16 = np.array([r.j_data for r in prof_16.combined_best_records], dtype=np.float64)
    j24 = np.array([r.j_data for r in prof_24.combined_best_records], dtype=np.float64)

    idx_min_16 = int(np.argmin(j16))
    idx_min_24 = int(np.argmin(j24))
    beta_min_16 = float(betas[idx_min_16])
    beta_min_24 = float(betas[idx_min_24])
    min_shift = abs(beta_min_24 - beta_min_16)

    reg_16 = betas[j16 <= 1.15 * j16[idx_min_16]]
    reg_24 = betas[j24 <= 1.15 * j24[idx_min_24]]

    da = np.abs([r24.alpha_hat - r16.alpha_hat for r16, r24 in zip(prof_16.combined_best_records, prof_24.combined_best_records)])
    dD = np.abs([r24.D_hat - r16.D_hat for r16, r24 in zip(prof_16.combined_best_records, prof_24.combined_best_records)])
    dE = np.abs([r24.E_hat - r16.E_hat for r16, r24 in zip(prof_16.combined_best_records, prof_24.combined_best_records)])

    minimum_region_stable = bool(min_shift <= 0.026)
    recommendation = (
        "proceed_to_targeted_lambda_sensitivity"
        if minimum_region_stable
        else "clarify_discretization_dependence_first"
    )

    return {
        "sampled_min_beta_N16_M80": beta_min_16,
        "sampled_min_j_data_N16_M80": float(j16[idx_min_16]),
        "sampled_min_beta_N24_M120": beta_min_24,
        "sampled_min_j_data_N24_M120": float(j24[idx_min_24]),
        "sampled_min_beta_shift_abs": float(min_shift),
        "j_data_at_theta_dagger_N16_M80": float(bd_true_16.j_data),
        "j_data_at_theta_dagger_N24_M120": float(bd_true_24.j_data),
        "rel_residual_at_theta_dagger_N16_M80": float(bd_true_16.relative_residual_l2),
        "rel_residual_at_theta_dagger_N24_M120": float(bd_true_24.relative_residual_l2),
        "true_point_j_data_reduction_factor_N16_to_N24": float(
            float(bd_true_16.j_data) / max(float(bd_true_24.j_data), 1.0e-30)
        ),
        "profile_min_j_data_reduction_factor_N16_to_N24": float(
            float(j16[idx_min_16]) / max(float(j24[idx_min_24]), 1.0e-30)
        ),
        "low_cost_15pct_interval_N16_M80": [float(np.min(reg_16)), float(np.max(reg_16))],
        "low_cost_15pct_interval_N24_M120": [float(np.min(reg_24)), float(np.max(reg_24))],
        "mean_abs_shift_alpha_hat_N16_vs_N24": float(np.mean(da)),
        "mean_abs_shift_D_hat_N16_vs_N24": float(np.mean(dD)),
        "mean_abs_shift_E_hat_N16_vs_N24": float(np.mean(dE)),
        "num_supplementary_verifications_triggered": len(supplementary),
        "minimum_region_and_shape_stable": minimum_region_stable,
        "recommended_next_step": recommendation,
    }


def build_phase310_markdown_report(
    *,
    solver_audit_16: Dict[str, Any],
    solver_audit_24: Dict[str, Any],
    reuse_audit_16: Dict[str, Any],
    checks_16: Dict[str, Any],
    checks_24: Dict[str, Any],
    prof_16: BidirectionalProfileResult,
    prof_24: BidirectionalProfileResult,
    comparison_rows: Sequence[Dict[str, Any]],
    supplementary: Sequence[SupplementaryPointVerification],
    findings: Dict[str, Any],
    target_sha256: str,
    target_source: str,
) -> str:
    lines: List[str] = [
        "# RETECO — Fase 3.10: Sensibilità del Profilo Dati $\\Phi_{\\mathrm{data}}(\\beta)$ alla Discretizzazione Inversa",
        "",
        "## 1. Protocollo Congelato e Verifica delle Risoluzioni del Solver",
        f"- **Target sintetico fisso** (`N = 32, M = 160`): SHA-256 `{target_sha256}` (`{target_source}`, `180` osservazioni pulite, mai rigenerato tra le risoluzioni)",
        f"- **Funzionale ottimizzato**: esclusivamente $\\Phi_{{\\mathrm{{data}}}}(\\beta) = \\min_{{\\alpha,D,E}} J_{{\\mathrm{{data}}}}(\\alpha,\\beta,D,E)$ in modalità `mode_a_deterministic_normalized_linf` (`lambda = 0` nell'ottimizzazione)",
        f"- **Risoluzione inversa di riferimento**: `N = {solver_audit_16['subdivisions_N']}, M = {solver_audit_16['time_steps_M']}` (`DOFs: d={solver_audit_16['n_dofs_disp']}, p={solver_audit_16['n_dofs_press']}, c={solver_audit_16['n_dofs_tracer']}`)",
        f"- **Risoluzione inversa raffinata**: `N = {solver_audit_24['subdivisions_N']}, M = {solver_audit_24['time_steps_M']}` (`DOFs: d={solver_audit_24['n_dofs_disp']}, p={solver_audit_24['n_dofs_press']}, c={solver_audit_24['n_dofs_tracer']}`)",
        f"- **Riutilizzo certificato da Fase 3.9 per `N=16, M=80`**: `{reuse_audit_16['phase39_reused']}` (`{len(reuse_audit_16['reused_betas'])}` punti riutilizzati `{reuse_audit_16['reused_betas']}`, `{len(reuse_audit_16['newly_computed_betas'])}` punti nuovi calcolati `{reuse_audit_16['newly_computed_betas']}`)",
        "",
        "## 2. Valutazione Diagnostica Separata nei Parametri Veri $\\boldsymbol{\\theta}^\\dagger = (0.70, 0.45, 0.08, 2.50)$",
        "| Discretizzazione Inversa | Costo Dati nel Punto Vero $J_{\\mathrm{data}}(\\boldsymbol{\\theta}^\\dagger)$ | Residuo Relativo $\\|G_{N,M}(\\boldsymbol{\\theta}^\\dagger)-y_{\\mathrm{clean}}\\|_2 / \\|y_{\\mathrm{clean}}\\|_2$ | Minimo del Profilo Ottimizzato $\\min_\\beta \\Phi_{\\mathrm{data}}(\\beta)$ | $\\beta$ al Minimo Campionato |",
        "|---|---:|---:|---:|---:|",
        f"| `N = 16, M = 80` (riferimento) | `{findings['j_data_at_theta_dagger_N16_M80']:.6e}` | `{findings['rel_residual_at_theta_dagger_N16_M80']:.6e}` | `{findings['sampled_min_j_data_N16_M80']:.6e}` | `{findings['sampled_min_beta_N16_M80']:.6f}` |",
        f"| `N = 24, M = 120` (raffinata) | `{findings['j_data_at_theta_dagger_N24_M120']:.6e}` | `{findings['rel_residual_at_theta_dagger_N24_M120']:.6e}` | `{findings['sampled_min_j_data_N24_M120']:.6e}` | `{findings['sampled_min_beta_N24_M120']:.6f}` |",
        "",
        "> **Nota metodologica**: $J_{\\mathrm{data}}(\\boldsymbol{\\theta}^\\dagger)$ documenta esclusivamente come cambia il costo dati valutato nel parametro vero al variare della discretizzazione inversa; non rappresenta un limite inferiore per $\\Phi_{\\mathrm{data}}(\\beta)$ poiché la sotto-ottimizzazione rispetto a $(\\alpha, D, E)$ compensa la proiezione dello scarto di discretizzazione.",
        "",
        "## 3. Confronto Puntuale sulla Griglia Locale Comune di $\\beta$",
        "| $\\beta$ | $N=16$ $\\Phi_{\\mathrm{data}}(\\beta)$ | $N=16$ $|\\text{fwd}-\\text{bwd}|$ | $N=16$ $(\\widehat{\\alpha}, \\widehat{D}, \\widehat{E})$ | $N=16$ Conv. (F/B) | $N=24$ $\\Phi_{\\mathrm{data}}(\\beta)$ | $N=24$ $|\\text{fwd}-\\text{bwd}|$ | $N=24$ $(\\widehat{\\alpha}, \\widehat{D}, \\widehat{E})$ | $N=24$ Conv. (F/B) | Rapporto $J_{16}/J_{24}$ |",
        "|---:|---:|---:|---|---|---:|---:|---|---|---:|",
    ]

    for row in comparison_rows:
        lines.append(
            f"| `{row['beta_fixed']:.6f}` | `{row['n16_j_data_combined']:.4e}` | `{row['n16_fwd_bwd_abs_diff']:.2e}` | "
            f"`({row['n16_alpha_hat']:.4f}, {row['n16_D_hat']:.5f}, {row['n16_E_hat']:.4f})` | "
            f"`{row['n16_converged_fwd']}/{row['n16_converged_bwd']}` | "
            f"`{row['n24_j_data_combined']:.4e}` | `{row['n24_fwd_bwd_abs_diff']:.2e}` | "
            f"`({row['n24_alpha_hat']:.4f}, {row['n24_D_hat']:.5f}, {row['n24_E_hat']:.4f})` | "
            f"`{row['n24_converged_fwd']}/{row['n24_converged_bwd']}` | `{row['ratio_j_data_n16_over_n24']:.2f}x` |"
        )

    lines.append("")
    lines.append("## 4. Verifiche Supplementari a Budget Esteso (64 Valutazioni FEM) per Punti Centrali Non Convergenti")
    if not supplementary:
        lines.append("- Tutti i punti centrali della regione di minimo hanno soddisfatto il criterio di convergenza entro il budget primario di 32 valutazioni FEM.")
    else:
        lines.append(
            "| Risoluzione | Direzione | $\\beta$ | $J_{\\mathrm{data}}$ Primario (32 val.) | $J_{\\mathrm{data}}$ Supplementare (64 val.) | Miglioramento $\\Delta J_{\\mathrm{data}}$ | Val. FEM Usate | Conv. Suppl. | $(\\widehat{\\alpha}, \\widehat{D}, \\widehat{E})$ Suppl. |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---|---|")
        for v in supplementary:
            lines.append(
                f"| `{v.resolution_label}` | `{v.scan_direction}` | `{v.beta_fixed:.6f}` | "
                f"`{v.primary_j_data:.4e}` | `{v.supplementary_j_data:.4e}` | `{v.j_data_improvement:.2e}` | "
                f"`{v.supplementary_fem_evaluations}/{v.supplementary_budget}` | `{v.supplementary_converged}` | "
                f"`({v.supplementary_alpha_hat:.4f}, {v.supplementary_D_hat:.5f}, {v.supplementary_E_hat:.4f})` |"
            )

    lines.extend(
        [
            "",
            "## 5. Interpretazione strutturata e Raccomandazione sul Passo Successivo",
            f"1. **Stabilità della posizione del minimo rispetto alla discretizzazione**:",
            f"   - Minimo campionato a `N=16, M=80`: $\\beta = {findings['sampled_min_beta_N16_M80']:.6f}$ (fascia entro $+15\\%$: `{findings['low_cost_15pct_interval_N16_M80']}`).",
            f"   - Minimo campionato a `N=24, M=120`: $\\beta = {findings['sampled_min_beta_N24_M120']:.6f}$ (fascia entro $+15\\%$: `{findings['low_cost_15pct_interval_N24_M120']}`).",
            f"   - Spostamento assoluto del minimo campionato: `|{findings['sampled_min_beta_N24_M120']:.4f} - {findings['sampled_min_beta_N16_M80']:.4f}| = {findings['sampled_min_beta_shift_abs']:.4f}`.",
            f"2. **Variazione assoluta del costo dati**:",
            f"   - Passando da `(N=16, M=80)` a `(N=24, M=120)`, il costo dati nel punto vero $J_{{\\mathrm{{data}}}}(\\boldsymbol{{\\theta}}^\\dagger)$ scende di un fattore `{findings['true_point_j_data_reduction_factor_N16_to_N24']:.2f}x` e il minimo del profilo $\\Phi_{{\\mathrm{{data}}}}(\\beta)$ scende di un fattore `{findings['profile_min_j_data_reduction_factor_N16_to_N24']:.2f}x`.",
            f"3. **Variazione delle stime di $\\alpha, D$ ed $E$**:",
            f"   - Scostamento medio assoluto tra `N=16` e `N=24`: $|\\Delta\\widehat{{\\alpha}}| = {findings['mean_abs_shift_alpha_hat_N16_vs_N24']:.5f}$, $|\\Delta\\widehat{{D}}| = {findings['mean_abs_shift_D_hat_N16_vs_N24']:.6f}$, $|\\Delta\\widehat{{E}}| = {findings['mean_abs_shift_E_hat_N16_vs_N24']:.5f}$.",
            f"4. **Raccomandazione motivata**:",
            f"   - Esito stabilità: `minimum_region_and_shape_stable = {findings['minimum_region_and_shape_stable']}` $\\rightarrow$ **`{findings['recommended_next_step']}`**.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RETECO Phase 3.10: Sensitivity of Phi_data(beta) to inverse discretization (N=16,M=80 vs N=24,M=120)."
    )
    parser.add_argument(
        "--budget-per-point",
        type=int,
        default=INITIAL_PER_POINT_BUDGET,
        help="Initial FEM evaluation budget per point and direction (default: 32).",
    )
    parser.add_argument(
        "--supplementary-budget",
        type=int,
        default=SUPPLEMENTARY_PER_POINT_BUDGET,
        help="Extended FEM budget for isolated re-verification of non-converged central points (default: 64).",
    )
    args = parser.parse_args()

    out_dir = Path("artifacts/10_beta_mesh_sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)

    prev_dirs = [
        Path("artifacts/07_cpso_multiseed"),
        Path("artifacts/08_beta_landscape"),
        Path("artifacts/09_beta_profile"),
    ]
    pre_hashes = {str(d): hash_directory_files(d) for d in prev_dirs}

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

    y_target_clean, verified_sha256, target_source = (
        load_saved_clean_target_without_regeneration(
            master=master,
            twin_cfg=twin_cfg,
            obs_op=obs_op,
            theta_dagger=theta_dagger,
            out_dir=out_dir,
            expected_sha256=EXPECTED_TARGET_SHA256,
        )
    )
    print(f"[Fase 3.10] Target caricato e verificato: SHA-256={verified_sha256} ({target_source})")

    _, G_inv_16, solver_audit_16 = build_and_verify_inverse_forward_problem(
        subdivisions_N=16,
        time_steps_M=80,
        master=master,
        twin_cfg=twin_cfg,
        obs_op=obs_op,
    )
    _, G_inv_24, solver_audit_24 = build_and_verify_inverse_forward_problem(
        subdivisions_N=24,
        time_steps_M=120,
        master=master,
        twin_cfg=twin_cfg,
        obs_op=obs_op,
    )
    print(
        f"[Fase 3.10] Solver verificati: "
        f"Rif=(N={solver_audit_16['subdivisions_N']}, M={solver_audit_16['time_steps_M']}) | "
        f"Fine=(N={solver_audit_24['subdivisions_N']}, M={solver_audit_24['time_steps_M']})"
    )

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

    obj_diag_16 = InverseObjectiveFunctional.from_mode_a_deterministic(
        G_inv_16,
        sample_0.y_noisy,
        transform,
        canonical_reg,
        y_reference=sample_0.y_noisy,
        allow_observed_as_reference=True,
        scale_type="linf",
        max_evaluations=100,
    )
    obj_diag_24 = InverseObjectiveFunctional.from_mode_a_deterministic(
        G_inv_24,
        sample_0.y_noisy,
        transform,
        canonical_reg,
        y_reference=sample_0.y_noisy,
        allow_observed_as_reference=True,
        scale_type="linf",
        max_evaluations=100,
    )

    cpso_anchors = load_phase37_cpso_anchors()
    planned_budget = int(args.budget_per_point)
    supp_budget = int(args.supplementary_budget)

    print("[Fase 3.10] Calcolo/riutilizzo profilo Phi_data(beta) a (N=16, M=80) ...")
    prof_16, reused_16, reuse_audit_16 = build_or_reuse_n16_m80_profile_on_local_grid(
        beta_grid=LOCAL_BETA_GRID,
        forward_operator_16=G_inv_16,
        y_observed=sample_0.y_noisy,
        transform=transform,
        canonical_regularizer=canonical_reg,
        planned_point_budget=planned_budget,
        cpso_anchors=cpso_anchors,
    )
    checks_16 = verify_profile_consistency(prof_16, G_inv_16, obj_diag_16)

    print("[Fase 3.10] Calcolo profilo Phi_data(beta) a (N=24, M=120) ...")
    prof_24 = compute_bidirectional_optimized_1d_profile(
        profile_type="data_only",
        beta_grid=LOCAL_BETA_GRID,
        forward_operator=G_inv_24,
        y_observed=sample_0.y_noisy,
        transform=transform,
        canonical_regularizer=canonical_reg,
        planned_point_budget=planned_budget,
        empirical_anchor_solutions=cpso_anchors,
        scale_type="linf",
    )
    checks_24 = verify_profile_consistency(prof_24, G_inv_24, obj_diag_24)

    supp_16 = run_supplementary_verifications_for_nonconverged_central_points(
        resolution_label="N16_M80",
        N_inv=16,
        M_inv=80,
        prof=prof_16,
        forward_operator=G_inv_16,
        y_observed=sample_0.y_noisy,
        transform=transform,
        canonical_regularizer=canonical_reg,
        supplementary_budget=supp_budget,
    )
    supp_24 = run_supplementary_verifications_for_nonconverged_central_points(
        resolution_label="N24_M120",
        N_inv=24,
        M_inv=120,
        prof=prof_24,
        forward_operator=G_inv_24,
        y_observed=sample_0.y_noisy,
        transform=transform,
        canonical_regularizer=canonical_reg,
        supplementary_budget=supp_budget,
    )
    all_supplementary = supp_16 + supp_24

    y_true_16 = G_inv_16(theta_dagger)
    bd_true_16 = obj_diag_16.evaluate_from_prediction(
        theta_dagger, y_true_16, record_in_history=False
    )
    y_true_24 = G_inv_24(theta_dagger)
    bd_true_24 = obj_diag_24.evaluate_from_prediction(
        theta_dagger, y_true_24, record_in_history=False
    )

    post_hashes = {str(d): hash_directory_files(d) for d in prev_dirs}
    previous_artifacts_untouched = (pre_hashes == post_hashes)
    if not previous_artifacts_untouched:
        raise RuntimeError(
            "ERRORE: Uno o più file nelle cartelle degli artefatti precedenti sono stati modificati!"
        )

    csv_16_path = out_dir / "profile_data_N16_M80.csv"
    csv_24_path = out_dir / "profile_data_N24_M120.csv"
    comp_csv_path = out_dir / "beta_mesh_sensitivity_comparison.csv"
    supp_csv_path = out_dir / "supplementary_point_verifications.csv"
    fig_base_path = out_dir / "fig_beta_mesh_sensitivity.png"
    report_md_path = out_dir / "beta_mesh_sensitivity_report.md"
    summary_json_path = out_dir / "beta_mesh_sensitivity_summary.json"

    write_single_resolution_csv(prof_16, reused_16, 16, 80, csv_16_path)
    write_single_resolution_csv(
        prof_24, {float(b): False for b in LOCAL_BETA_GRID}, 24, 120, csv_24_path
    )
    comparison_rows = write_mesh_comparison_csv(prof_16, reused_16, prof_24, comp_csv_path)
    write_supplementary_verifications_csv(all_supplementary, supp_csv_path)

    png_path, pdf_path = plot_beta_mesh_sensitivity_figure(
        prof_16=prof_16,
        prof_24=prof_24,
        bd_true_16=bd_true_16,
        bd_true_24=bd_true_24,
        supplementary_verifications=all_supplementary,
        out_path=fig_base_path,
        beta_true_ref=SYNTHETIC_REF_BETA_TRUE,
    )

    findings = summarize_mesh_sensitivity_findings(
        prof_16=prof_16,
        prof_24=prof_24,
        bd_true_16=bd_true_16,
        bd_true_24=bd_true_24,
        supplementary=all_supplementary,
    )

    report_md = build_phase310_markdown_report(
        solver_audit_16=solver_audit_16,
        solver_audit_24=solver_audit_24,
        reuse_audit_16=reuse_audit_16,
        checks_16=checks_16,
        checks_24=checks_24,
        prof_16=prof_16,
        prof_24=prof_24,
        comparison_rows=comparison_rows,
        supplementary=all_supplementary,
        findings=findings,
        target_sha256=verified_sha256,
        target_source=target_source,
    )
    report_md_path.write_text(report_md, encoding="utf-8")

    summary_payload = {
        "experiment_id": "RETECO-v1-PHASE3.10-BETA-MESH-SENSITIVITY",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "forcing_protocol_version": FORCING_PROTOCOL_VERSION,
        "frozen_protocol": {
            "target_sha256": verified_sha256,
            "expected_target_sha256": EXPECTED_TARGET_SHA256,
            "target_hash_verified": True,
            "target_source": target_source,
            "target_regenerated_during_run": False,
            "n_observations": EXPECTED_N_OBS,
            "objective_optimized": "Phi_data(beta) = min_{alpha,D,E} J_data(alpha,beta,D,E)",
            "regularization_included_in_profile_optimization": False,
            "local_beta_grid": list(LOCAL_BETA_GRID),
            "initial_per_point_per_direction_budget": planned_budget,
            "supplementary_per_point_budget": supp_budget,
            "true_parameters_used_in_optimization": False,
            "previous_artifacts_untouched": previous_artifacts_untouched,
        },
        "solver_configurations_verified": {
            "reference_N16_M80": solver_audit_16,
            "refined_N24_M120": solver_audit_24,
        },
        "phase39_reuse_audit_N16_M80": reuse_audit_16,
        "diagnostic_evaluation_at_theta_dagger": {
            "theta_dagger": theta_dagger.tolist(),
            "N16_M80": {
                "J_data_at_theta_dagger": float(bd_true_16.j_data),
                "relative_residual_l2_at_theta_dagger": float(bd_true_16.relative_residual_l2),
                "weighted_residual_l2_at_theta_dagger": float(bd_true_16.weighted_residual_l2_norm),
            },
            "N24_M120": {
                "J_data_at_theta_dagger": float(bd_true_24.j_data),
                "relative_residual_l2_at_theta_dagger": float(bd_true_24.relative_residual_l2),
                "weighted_residual_l2_at_theta_dagger": float(bd_true_24.weighted_residual_l2_norm),
            },
        },
        "profile_N16_M80": {
            "integrity_checks": checks_16,
            "directional_discrepancy_summary": prof_16.directional_discrepancy_summary,
            "forward_records": [r.to_dict() for r in prof_16.forward_records],
            "backward_records": [r.to_dict() for r in prof_16.backward_records],
            "combined_best_records": [r.to_dict() for r in prof_16.combined_best_records],
        },
        "profile_N24_M120": {
            "integrity_checks": checks_24,
            "directional_discrepancy_summary": prof_24.directional_discrepancy_summary,
            "forward_records": [r.to_dict() for r in prof_24.forward_records],
            "backward_records": [r.to_dict() for r in prof_24.backward_records],
            "combined_best_records": [r.to_dict() for r in prof_24.combined_best_records],
        },
        "supplementary_convergence_verifications": [
            v.to_dict() for v in all_supplementary
        ],
        "mesh_sensitivity_findings": findings,
        "comparison_table_rows": comparison_rows,
        "artifacts_generated": {
            "profile_n16_csv": str(csv_16_path),
            "profile_n24_csv": str(csv_24_path),
            "comparison_csv": str(comp_csv_path),
            "supplementary_csv": str(supp_csv_path),
            "summary_json": str(summary_json_path),
            "figure_png": str(png_path),
            "figure_pdf": str(pdf_path),
            "report_md": str(report_md_path),
        },
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("\n" + report_md)
    print(f"[Fase 3.10] Tutti gli artefatti salvati in: {out_dir}")


if __name__ == "__main__":
    main()