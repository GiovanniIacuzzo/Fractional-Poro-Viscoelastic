"""
python/reteco/identifiability/profile_likelihood.py
Analisi pratica dell'identificabilità parametrica sul funzionale di costo:
1) `compute_one_at_a_time_cost_slices`: sezioni 1D del funzionale mantenendo fissi gli altri
   tre parametri al valore di riferimento theta_dagger (One-at-a-Time Cost Slices);
2) `compute_optimized_objective_profiles`: profili dell'obiettivo ottimizzati (Optimized
   Objective Profiles) con:
   - limite cumulativo RIGIDO per punto (`nfev <= max_evals_per_point`) condiviso tra
     valutazione OAT, sonde warm-start, ottimizzazione primaria, recupero derivative-free
     e scansione bidirezionale;
   - contabilizzazione esatta e verificata (`setup_reference_evaluations + sum(nfev) == total_forward_evaluations`);
   - recupero robusto a due stadi (L-BFGS-B con riserva di budget + Nelder-Mead vincolato
     su [0,1]^3 in caso di ABNORMAL_TERMINATION_IN_LNSRCH);
   - nessuna impostazione artificiale di `success=True`.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

from ..config_schema import ParameterCoordinateTransform
from .sensitivity_jacobian import PARAMETER_NAMES, ObservationWeightingSpec
from .svd_conditioning import sanitize_for_json


@dataclass(frozen=True)
class CostSlice1D:
    parameter_name: str
    parameter_index: int
    grid_normalized: np.ndarray                # (M,) in [0, 1]
    grid_physical: np.ndarray                  # (M,) in unità del parametro
    cost_values: np.ndarray                    # (M,) J_obj(theta_j; theta_{-j}^dagger)
    reference_normalized: float
    reference_physical: float
    reference_cost: float
    min_slice_cost: float
    min_slice_physical: float
    is_flat_within_tol: bool
    num_evaluations: int


@dataclass(frozen=True)
class OneAtATimeCostSlicesReport:
    analysis_kind: str                         # "one_at_a_time_cost_slices"
    slices: Dict[str, CostSlice1D]
    setup_reference_evaluations: int
    total_slice_point_evaluations: int
    total_forward_evaluations: int
    methodology_note: str = (
        "One-at-a-Time (OAT) 1D Cost Slices: each parameter is varied across its admissible "
        "interval while holding the other three parameters fixed at theta_dagger. "
        "These curves do NOT account for multi-parameter trade-offs and are NOT profile likelihoods."
    )

    def to_json_dict(self) -> Dict[str, Any]:
        return sanitize_for_json(
            {
                "analysis_kind": self.analysis_kind,
                "methodology_note": self.methodology_note,
                "setup_reference_evaluations": int(self.setup_reference_evaluations),
                "total_slice_point_evaluations": int(self.total_slice_point_evaluations),
                "total_forward_evaluations": int(self.total_forward_evaluations),
                "slices_by_parameter": {
                    k: {
                        "parameter_name": s.parameter_name,
                        "parameter_index": s.parameter_index,
                        "grid_normalized": s.grid_normalized.tolist(),
                        "grid_physical": s.grid_physical.tolist(),
                        "cost_values": s.cost_values.tolist(),
                        "reference_normalized": s.reference_normalized,
                        "reference_physical": s.reference_physical,
                        "reference_cost": s.reference_cost,
                        "min_slice_cost": s.min_slice_cost,
                        "min_slice_physical": s.min_slice_physical,
                        "max_slice_cost": float(np.max(s.cost_values)),
                        "is_flat_within_tol": s.is_flat_within_tol,
                        "num_evaluations": s.num_evaluations,
                    }
                    for k, s in self.slices.items()
                },
            }
        )


@dataclass(frozen=True)
class OptimizedProfile1D:
    parameter_name: str
    parameter_index: int
    grid_normalized: np.ndarray                # (M,)
    grid_physical: np.ndarray                  # (M,)
    profile_cost_values: np.ndarray            # (M,) min_{theta_{-j}} J_obj(theta_j, theta_{-j})
    oat_slice_cost_values: np.ndarray          # (M,) J_obj(theta_j, theta_{-j}^dagger)
    optimized_thetas_normalized: np.ndarray    # (M, 4)
    optimized_thetas_physical: np.ndarray      # (M, 4)
    nuisance_hit_bounds_mask: List[bool]
    nuisance_at_bounds_names: List[List[str]]
    evaluations_per_point: List[int]           # nfev cumulative per punto (sempre <= max_evals_per_point)
    iterations_per_point: List[int]            # nit cumulative per punto
    converged_flags: List[bool]                # True solo se convergenza reale certificata
    optimizer_status_codes: List[int]
    optimizer_messages: List[str]
    termination_categories: List[str]
    incomplete_points_mask: List[bool]
    pointwise_optimization_records: List[Dict[str, Any]]
    reference_normalized: float
    reference_physical: float
    profile_min_cost: float
    profile_argmin_physical: float
    profile_argmin_normalized: float
    profile_minimum_at_grid_boundary: bool
    is_flat_within_tol: bool
    num_converged_points: int
    num_non_converged_points: int
    num_incomplete_points: int
    all_points_converged: bool
    max_point_evaluations_observed: int
    total_evaluations_for_parameter: int


@dataclass(frozen=True)
class OptimizedObjectiveProfilesReport:
    analysis_kind: str                         # "optimized_objective_profiles"
    is_statistical_profile_likelihood: bool
    profiles: Dict[str, OptimizedProfile1D]
    setup_reference_evaluations: int           # Valutazioni preliminari in theta_ref (1)
    total_point_evaluations: int               # Somma esatta dei nfev di tutti i punti dei 4 profili
    total_forward_evaluations: int             # setup_reference_evaluations + total_point_evaluations
    max_total_budget: int
    max_evals_per_point: int
    max_iter_per_point: int
    bidirectional_scan: bool
    budget_exhausted: bool
    per_point_budget_respected: bool
    accounting_consistency_verified: bool
    total_grid_points: int
    total_converged_points: int
    total_non_converged_points: int
    total_incomplete_points: int
    all_profiles_complete_and_converged: bool
    optimizer_method: str
    methodology_note: str

    def to_json_dict(self) -> Dict[str, Any]:
        return sanitize_for_json(
            {
                "analysis_kind": self.analysis_kind,
                "is_statistical_profile_likelihood": self.is_statistical_profile_likelihood,
                "methodology_note": self.methodology_note,
                "optimizer_method": self.optimizer_method,
                "bidirectional_scan": bool(self.bidirectional_scan),
                "max_evals_per_point": int(self.max_evals_per_point),
                "max_iter_per_point": int(self.max_iter_per_point),
                "max_total_budget": int(self.max_total_budget),
                "evaluation_accounting": {
                    "setup_reference_evaluations": int(self.setup_reference_evaluations),
                    "total_point_evaluations": int(self.total_point_evaluations),
                    "evaluations_by_parameter": {
                        k: int(p.total_evaluations_for_parameter)
                        for k, p in self.profiles.items()
                    },
                    "max_point_evaluations_observed_by_parameter": {
                        k: int(p.max_point_evaluations_observed)
                        for k, p in self.profiles.items()
                    },
                    "total_forward_evaluations": int(self.total_forward_evaluations),
                    "per_point_budget_respected": bool(self.per_point_budget_respected),
                    "accounting_consistency_verified": bool(self.accounting_consistency_verified),
                    "budget_exhausted": bool(self.budget_exhausted),
                },
                "setup_reference_evaluations": int(self.setup_reference_evaluations),
                "total_point_evaluations": int(self.total_point_evaluations),
                "total_forward_evaluations": int(self.total_forward_evaluations),
                "budget_exhausted": bool(self.budget_exhausted),
                "per_point_budget_respected": bool(self.per_point_budget_respected),
                "accounting_consistency_verified": bool(self.accounting_consistency_verified),
                "convergence_summary": {
                    "total_grid_points": int(self.total_grid_points),
                    "total_converged_points": int(self.total_converged_points),
                    "total_non_converged_points": int(self.total_non_converged_points),
                    "total_incomplete_points": int(self.total_incomplete_points),
                    "all_profiles_complete_and_converged": bool(
                        self.all_profiles_complete_and_converged
                    ),
                    "converged_by_parameter": {
                        k: f"{p.num_converged_points}/{len(p.grid_normalized)}"
                        for k, p in self.profiles.items()
                    },
                },
                "profiles_by_parameter": {
                    k: {
                        "parameter_name": p.parameter_name,
                        "parameter_index": p.parameter_index,
                        "grid_normalized": p.grid_normalized.tolist(),
                        "grid_physical": p.grid_physical.tolist(),
                        "profile_cost_values": p.profile_cost_values.tolist(),
                        "oat_slice_cost_values": p.oat_slice_cost_values.tolist(),
                        "optimized_thetas_normalized": p.optimized_thetas_normalized.tolist(),
                        "optimized_thetas_physical": p.optimized_thetas_physical.tolist(),
                        "nuisance_hit_bounds_mask": list(p.nuisance_hit_bounds_mask),
                        "nuisance_at_bounds_names": list(p.nuisance_at_bounds_names),
                        "num_points_with_nuisance_at_bounds": int(sum(p.nuisance_hit_bounds_mask)),
                        "evaluations_per_point": list(p.evaluations_per_point),
                        "max_point_evaluations_observed": int(p.max_point_evaluations_observed),
                        "iterations_per_point": list(p.iterations_per_point),
                        "converged_flags": list(p.converged_flags),
                        "optimizer_status_codes": list(p.optimizer_status_codes),
                        "optimizer_messages": list(p.optimizer_messages),
                        "termination_categories": list(p.termination_categories),
                        "incomplete_points_mask": list(p.incomplete_points_mask),
                        "num_converged_points": int(p.num_converged_points),
                        "num_non_converged_points": int(p.num_non_converged_points),
                        "num_incomplete_points": int(p.num_incomplete_points),
                        "all_points_converged": bool(p.all_points_converged),
                        "reference_normalized": p.reference_normalized,
                        "reference_physical": p.reference_physical,
                        "profile_min_cost": p.profile_min_cost,
                        "profile_argmin_physical": p.profile_argmin_physical,
                        "profile_argmin_normalized": p.profile_argmin_normalized,
                        "profile_minimum_at_grid_boundary": p.profile_minimum_at_grid_boundary,
                        "is_flat_within_tol": p.is_flat_within_tol,
                        "total_evaluations_for_parameter": p.total_evaluations_for_parameter,
                        "pointwise_optimization_records": p.pointwise_optimization_records,
                    }
                    for k, p in self.profiles.items()
                },
            }
        )


def build_misfit_objective_from_observations(
    forward_obs_callable: Callable[[np.ndarray], np.ndarray],
    y_target: np.ndarray,
    weighting_spec: Optional[ObservationWeightingSpec] = None,
) -> Callable[[np.ndarray], float]:
    """
    Costruisce il funzionale di discrepanza coerente con la pesatura del Jacobiano:
        J_obj(theta) = 0.5 * || W^{1/2} (G(theta) - y_target) ||_2^2.
    """
    y_ref = np.asarray(y_target, dtype=np.float64)
    w_spec = weighting_spec or ObservationWeightingSpec()

    def _objective(theta_phys: np.ndarray) -> float:
        y_sim = np.asarray(forward_obs_callable(theta_phys), dtype=np.float64)
        res = w_spec.apply_to_residual(y_sim - y_ref)
        return 0.5 * float(np.dot(res, res))

    return _objective


def _build_normalized_grid_including_ref(
    t_ref: float, num_points: int, margin: float = 0.04
) -> np.ndarray:
    if num_points < 3:
        raise ValueError("num_points deve essere >= 3.")
    low = min(margin, 0.5 * t_ref)
    high = max(1.0 - margin, 1.0 - 0.5 * (1.0 - t_ref))
    grid = np.linspace(low, high, num_points, dtype=np.float64)
    closest_idx = int(np.argmin(np.abs(grid - t_ref)))
    grid[closest_idx] = float(t_ref)
    return np.sort(grid)


def compute_one_at_a_time_cost_slices(
    objective_callable: Callable[[np.ndarray], float],
    theta_ref: np.ndarray,
    transform: ParameterCoordinateTransform,
    num_points: int = 21,
    flat_tol: float = 1.0e-12,
) -> OneAtATimeCostSlicesReport:
    """
    Calcola le sezioni 1D del funzionale di costo (One-at-a-Time Cost Slices)
    variando un parametro alla volta e mantenendo gli altri 3 fissi a `theta_ref`.
    """
    theta_phys_0 = np.asarray(theta_ref, dtype=np.float64)
    theta_tilde_0 = np.asarray(transform.to_normalized(theta_phys_0), dtype=np.float64)
    ref_cost = float(objective_callable(theta_phys_0))
    setup_evals = 1
    slice_evals = 0

    slices_dict: Dict[str, CostSlice1D] = {}
    for j, pname in enumerate(PARAMETER_NAMES):
        t_ref_j = float(theta_tilde_0[j])
        grid_norm = _build_normalized_grid_including_ref(t_ref_j, num_points=num_points)
        grid_phys = np.zeros_like(grid_norm)
        costs = np.zeros_like(grid_norm)
        n_ev_param = 0

        for k, val_norm in enumerate(grid_norm):
            t_vec = theta_tilde_0.copy()
            t_vec[j] = float(val_norm)
            p_vec = transform.from_normalized(t_vec)
            grid_phys[k] = float(p_vec[j])
            if abs(val_norm - t_ref_j) < 1.0e-15:
                costs[k] = ref_cost
            else:
                costs[k] = float(objective_callable(p_vec))
                n_ev_param += 1
                slice_evals += 1

        min_idx = int(np.argmin(costs))
        is_flat = bool((np.max(costs) - np.min(costs)) <= flat_tol)

        slices_dict[pname] = CostSlice1D(
            parameter_name=pname,
            parameter_index=j,
            grid_normalized=grid_norm,
            grid_physical=grid_phys,
            cost_values=costs,
            reference_normalized=t_ref_j,
            reference_physical=float(theta_phys_0[j]),
            reference_cost=ref_cost,
            min_slice_cost=float(costs[min_idx]),
            min_slice_physical=float(grid_phys[min_idx]),
            is_flat_within_tol=is_flat,
            num_evaluations=n_ev_param,
        )

    return OneAtATimeCostSlicesReport(
        analysis_kind="one_at_a_time_cost_slices",
        slices=slices_dict,
        setup_reference_evaluations=setup_evals,
        total_slice_point_evaluations=slice_evals,
        total_forward_evaluations=setup_evals + slice_evals,
    )


compute_cost_slices = compute_one_at_a_time_cost_slices


class _GlobalBudgetExhausted(Exception):
    """Sollevata esclusivamente quando viene raggiunto `max_total_evaluations`."""


class _PointBudgetExhausted(Exception):
    """Sollevata quando il singolo punto del profilo raggiunge il tetto cumulativo `max_evals_per_point`."""


class _Stage1SubBudgetReached(Exception):
    """Sollevata quando lo Stadio 1 (L-BFGS-B) raggiunge la propria quota per cedere il passo al recupero."""


def _build_local_bounded_simplex(
    u_center: np.ndarray,
    radius: float = 0.003,
) -> np.ndarray:
    """
    Costruisce un simplesso iniziale (4, 3) contenuto in [0, 1]^3 attorno a `u_center`
    per il recupero derivative-free con Nelder-Mead vincolato.
    """
    u0 = np.clip(np.asarray(u_center, dtype=np.float64), 0.0, 1.0)
    simplex = np.tile(u0, (4, 1))
    for d in range(3):
        step = radius if (u0[d] + radius <= 1.0) else -radius
        if u0[d] + step < 0.0 or u0[d] + step > 1.0:
            step = 0.5 * (1.0 - u0[d]) if u0[d] < 0.5 else -0.5 * u0[d]
        if abs(step) < 1.0e-5:
            step = 1.0e-3 if u0[d] < 0.5 else -1.0e-3
        simplex[d + 1, d] = np.clip(u0[d] + step, 0.0, 1.0)
    return simplex


def compute_optimized_objective_profiles(
    objective_callable: Callable[[np.ndarray], float],
    theta_ref: np.ndarray,
    transform: ParameterCoordinateTransform,
    num_grid_points: int = 9,
    max_evals_per_point: int = 160,
    max_iter_per_point: int = 60,
    max_total_evaluations: int = 4500,
    optimizer_method: str = "L-BFGS-B",
    ftol: float = 1.0e-7,
    gtol: float = 1.0e-5,
    fd_eps: float = 1.0e-7,
    boundary_tol: float = 1.0e-3,
    flat_tol: float = 1.0e-12,
    bidirectional_scan: bool = True,
    is_statistical_profile_likelihood: bool = False,
) -> OptimizedObjectiveProfilesReport:
    """
    Calcola i profili ottimizzati del funzionale obiettivo:
        P_j(theta_{j,k}) = min_{theta_{-j} in Theta_{-j}} J_obj(theta_{j,k}, theta_{-j})

    Garanzie numeriche e contabili:
    1) Ogni punto `k_idx` dispone di un contatore cumulativo condiviso `evals_per_pt[k_idx]`
       (suddiviso in `oat`, `warm_probes`, `primary_opt`, `recovery_opt`).
       Prima di OGNI valutazione viene verificato `evals_per_pt[k_idx] < max_evals_per_point`
       e `total_evals < max_total_evaluations`: è impossibile superare `max_evals_per_point`.
    2) Lo Stadio 1 (`L-BFGS-B` con `fd_eps=1e-7`, `maxls=8`) utilizza al massimo ~55% del
       budget del punto, così che in caso di arresto anomalo del line-search resti sempre
       budget locale sufficiente per lo Stadio 2 di recupero derivative-free (`Nelder-Mead`
       vincolato su `[0, 1]^3` con simplesso locale attorno al miglior candidato).
    3) La convergenza (`converged_flags[k_idx] = True`) è assegnata SOLO se l'ottimizzatore
       certifica `res.success == True` senza aver esaurito il budget locale o globale.
    """
    if max_evals_per_point < 6:
        raise ValueError("max_evals_per_point deve essere >= 6.")
    if max_iter_per_point < 2:
        raise ValueError("max_iter_per_point deve essere >= 2.")
    if max_total_evaluations < 1:
        raise ValueError("max_total_evaluations deve essere >= 1.")

    raw_call_counter = 0

    def _counted_objective(theta_phys: np.ndarray) -> float:
        nonlocal raw_call_counter
        raw_call_counter += 1
        return float(objective_callable(theta_phys))

    theta_phys_0 = np.asarray(theta_ref, dtype=np.float64)
    theta_tilde_0 = np.asarray(transform.to_normalized(theta_phys_0), dtype=np.float64)

    ref_cost = _counted_objective(theta_phys_0)
    setup_reference_evals = 1
    total_evals = 1
    budget_exhausted = False

    profiles_dict: Dict[str, OptimizedProfile1D] = {}

    for j, pname in enumerate(PARAMETER_NAMES):
        nuisance_indices = [idx for idx in range(4) if idx != j]
        nuisance_names = [PARAMETER_NAMES[idx] for idx in nuisance_indices]
        t_ref_j = float(theta_tilde_0[j])

        grid_norm = _build_normalized_grid_including_ref(t_ref_j, num_points=num_grid_points)
        m_pts = int(grid_norm.size)
        grid_phys = np.zeros(m_pts, dtype=np.float64)
        for k in range(m_pts):
            t_tmp = theta_tilde_0.copy()
            t_tmp[j] = float(grid_norm[k])
            grid_phys[k] = float(transform.from_normalized(t_tmp)[j])

        prof_costs = np.full(m_pts, np.nan, dtype=np.float64)
        oat_costs = np.full(m_pts, np.nan, dtype=np.float64)
        opt_norm_mat = np.zeros((m_pts, 4), dtype=np.float64)
        opt_phys_mat = np.zeros((m_pts, 4), dtype=np.float64)
        best_u_per_pt: List[np.ndarray] = [
            theta_tilde_0[nuisance_indices].copy() for _ in range(m_pts)
        ]

        hit_bounds_mask: List[bool] = [False] * m_pts
        at_bounds_names: List[List[str]] = [[] for _ in range(m_pts)]

        nfev_oat: List[int] = [0] * m_pts
        nfev_warm: List[int] = [0] * m_pts
        nfev_prim: List[int] = [0] * m_pts
        nfev_rec: List[int] = [0] * m_pts
        evals_per_pt: List[int] = [0] * m_pts
        iters_per_pt: List[int] = [0] * m_pts

        conv_flags: List[bool] = [False] * m_pts
        status_codes: List[int] = [-1] * m_pts
        messages: List[str] = ["Not evaluated"] * m_pts
        term_categories: List[str] = ["not_evaluated"] * m_pts
        incomplete_mask: List[bool] = [True] * m_pts

        ref_idx = int(np.argmin(np.abs(grid_norm - t_ref_j)))

        def _detect_bounds(u_vec: np.ndarray) -> Tuple[bool, List[str]]:
            b_names: List[str] = []
            for loc_i, u_val in enumerate(u_vec):
                if u_val <= boundary_tol or u_val >= (1.0 - boundary_tol):
                    b_names.append(nuisance_names[loc_i])
            return (len(b_names) > 0), b_names

        def _rem_point_budget(k_idx: int, cap_override: Optional[int] = None) -> int:
            pt_limit = max_evals_per_point if cap_override is None else min(max_evals_per_point, cap_override)
            rem_pt = pt_limit - evals_per_pt[k_idx]
            rem_gl = max_total_evaluations - total_evals
            return max(0, min(rem_pt, rem_gl))

        def _commit_best_candidate_to_point(k_idx: int) -> None:
            val_norm = float(grid_norm[k_idx])
            u_b = np.clip(best_u_per_pt[k_idx], 0.0, 1.0)
            t_full = np.zeros(4, dtype=np.float64)
            t_full[j] = val_norm
            t_full[nuisance_indices] = u_b
            opt_norm_mat[k_idx] = t_full
            opt_phys_mat[k_idx] = transform.from_normalized(t_full)
            hb, bn = _detect_bounds(u_b)
            hit_bounds_mask[k_idx] = hb
            at_bounds_names[k_idx] = bn

        def _eval_for_point(
            k_idx: int,
            u_3d: np.ndarray,
            bucket: str,
            point_cap_override: Optional[int] = None,
        ) -> float:
            nonlocal total_evals, budget_exhausted
            if total_evals >= max_total_evaluations:
                budget_exhausted = True
                raise _GlobalBudgetExhausted()
            pt_cap = (
                max_evals_per_point
                if point_cap_override is None
                else min(max_evals_per_point, point_cap_override)
            )
            if evals_per_pt[k_idx] >= pt_cap:
                raise _PointBudgetExhausted()

            u_clipped = np.clip(np.asarray(u_3d, dtype=np.float64), 0.0, 1.0)
            t_full = np.zeros(4, dtype=np.float64)
            t_full[j] = float(grid_norm[k_idx])
            t_full[nuisance_indices] = u_clipped
            p_full = transform.from_normalized(t_full)

            val_c = _counted_objective(p_full)
            total_evals += 1
            evals_per_pt[k_idx] += 1
            if bucket == "oat":
                nfev_oat[k_idx] += 1
            elif bucket == "warm":
                nfev_warm[k_idx] += 1
            elif bucket == "prim":
                nfev_prim[k_idx] += 1
            elif bucket == "rec":
                nfev_rec[k_idx] += 1
            else:
                raise ValueError(f"Bucket contatore non valido: {bucket}")

            if (not np.isfinite(prof_costs[k_idx])) or (val_c < prof_costs[k_idx]):
                prof_costs[k_idx] = val_c
                best_u_per_pt[k_idx] = u_clipped.copy()
                _commit_best_candidate_to_point(k_idx)

            return val_c

        def _run_recovery_stage(
            k_idx: int,
            u_start: np.ndarray,
            simplex_radius: float = 0.003,
            point_cap_override: Optional[int] = None,
        ) -> Tuple[bool, int, str, int]:
            """
            Esegue il tentativo di recupero derivative-free vincolato su [0,1]^3
            utilizzando esclusivamente il budget cumulativo residuo del punto `k_idx`.
            """
            rem_b = _rem_point_budget(k_idx, cap_override=point_cap_override)
            if rem_b < 8:
                return False, 1, "Insufficient remaining point budget for recovery stage.", 0

            u0 = np.clip(np.asarray(u_start, dtype=np.float64), 0.0, 1.0)
            init_simplex = _build_local_bounded_simplex(u0, radius=simplex_radius)
            max_iter_rec = max(6, int(max_iter_per_point))

            def _sub_rec(u_3d: np.ndarray) -> float:
                return _eval_for_point(
                    k_idx, u_3d, bucket="rec", point_cap_override=point_cap_override
                )

            try:
                res_nm = minimize(
                    _sub_rec,
                    x0=u0,
                    method="Nelder-Mead",
                    bounds=[(0.0, 1.0)] * 3,
                    options={
                        "initial_simplex": init_simplex,
                        "maxfev": int(rem_b + 2),
                        "maxiter": int(max_iter_rec),
                        "fatol": 5.0e-5,
                        "xatol": 1.5e-3,
                        "adaptive": True,
                    },
                )
                nit_used = int(getattr(res_nm, "nit", 0))
                st_code = int(getattr(res_nm, "status", 0 if res_nm.success else 1))
                msg_nm = str(getattr(res_nm, "message", ""))
                conv_ok = bool(res_nm.success) and (st_code == 0)
                return conv_ok, st_code, f"Recovery Nelder-Mead: {msg_nm}", nit_used
            except _PointBudgetExhausted:
                return (
                    False,
                    1,
                    f"Point evaluation budget ({max_evals_per_point}) reached during recovery.",
                    0,
                )

        def _optimize_at_grid_index(
            k_idx: int,
            candidate_warm_list: List[np.ndarray],
            is_reverse_pass: bool = False,
        ) -> np.ndarray:
            nonlocal budget_exhausted
            val_norm = float(grid_norm[k_idx])
            u_oat = theta_tilde_0[nuisance_indices].copy()

            # Riserviamo 1 valutazione per la sonda inward se bidirectional_scan è attivo
            # e max_evals_per_point >= 40
            reserve_inward = 1 if (bidirectional_scan and (not is_reverse_pass) and max_evals_per_point >= 40) else 0
            pass1_cap = max_evals_per_point - reserve_inward

            # 1. Valutazione OAT se non ancora effettuata per il punto k_idx
            if not np.isfinite(oat_costs[k_idx]):
                if abs(val_norm - t_ref_j) < 1.0e-14:
                    oat_costs[k_idx] = ref_cost
                    prof_costs[k_idx] = ref_cost
                    best_u_per_pt[k_idx] = u_oat.copy()
                    _commit_best_candidate_to_point(k_idx)
                else:
                    try:
                        c_oat_val = _eval_for_point(
                            k_idx, u_oat, bucket="oat", point_cap_override=pass1_cap
                        )
                        oat_costs[k_idx] = c_oat_val
                    except _GlobalBudgetExhausted:
                        incomplete_mask[k_idx] = True
                        status_codes[k_idx] = -2
                        messages[k_idx] = "Global evaluation budget exhausted before OAT evaluation."
                        term_categories[k_idx] = "failed_global_budget_exhausted"
                        return u_oat
                    except _PointBudgetExhausted:
                        incomplete_mask[k_idx] = False
                        status_codes[k_idx] = 1
                        messages[k_idx] = "Point budget exhausted at OAT evaluation."
                        term_categories[k_idx] = "failed_local_eval_limit"
                        return best_u_per_pt[k_idx].copy()

            # 2. Punto di riferimento esatto theta_j = theta_j^dagger con J_obj == 0
            if abs(val_norm - t_ref_j) < 1.0e-14 and float(oat_costs[k_idx]) <= 1.0e-14:
                conv_flags[k_idx] = True
                status_codes[k_idx] = 0
                messages[k_idx] = (
                    "Exact reference point theta_dagger verified analytically (J_obj = 0 <= 1e-14)."
                )
                term_categories[k_idx] = "exact_reference_zero_cost"
                incomplete_mask[k_idx] = False
                return best_u_per_pt[k_idx].copy()

            # 3. Passaggio inverso (Inward):
            if is_reverse_pass:
                rem_b = _rem_point_budget(k_idx)
                if rem_b < 1:
                    return best_u_per_pt[k_idx].copy()

                improved_by_probe = False
                for cand_u in candidate_warm_list:
                    u_c = np.clip(np.asarray(cand_u, dtype=np.float64), 0.0, 1.0)
                    if np.linalg.norm(u_c - best_u_per_pt[k_idx]) <= 1.5e-3:
                        continue
                    if _rem_point_budget(k_idx) < 1:
                        break
                    prev_best = float(prof_costs[k_idx])
                    try:
                        c_probe = _eval_for_point(k_idx, u_c, bucket="warm")
                        if c_probe < prev_best - 1.0e-6:
                            improved_by_probe = True
                    except (_GlobalBudgetExhausted, _PointBudgetExhausted):
                        break

                if conv_flags[k_idx] and (not improved_by_probe):
                    return best_u_per_pt[k_idx].copy()

                if _rem_point_budget(k_idx) >= 12 and (
                    (not conv_flags[k_idx]) or improved_by_probe
                ):
                    try:
                        conv_rec, st_rec, msg_rec, nit_rec = _run_recovery_stage(
                            k_idx, best_u_per_pt[k_idx].copy(), simplex_radius=0.003
                        )
                        iters_per_pt[k_idx] += nit_rec
                        if conv_rec:
                            conv_flags[k_idx] = True
                            status_codes[k_idx] = st_rec
                            messages[k_idx] = f"Inward pass {msg_rec}"
                            term_categories[k_idx] = "converged_optimality"
                            incomplete_mask[k_idx] = False
                    except _GlobalBudgetExhausted:
                        if not conv_flags[k_idx]:
                            incomplete_mask[k_idx] = True
                            status_codes[k_idx] = -2
                            messages[k_idx] = "Global budget exhausted during inward pass."
                            term_categories[k_idx] = "failed_global_budget_exhausted"

                return best_u_per_pt[k_idx].copy()

            # 4. Passaggio Outward: Valutazione dei candidati warm-start (adiacente + secante)
            for cand_u in candidate_warm_list:
                u_c = np.clip(np.asarray(cand_u, dtype=np.float64), 0.0, 1.0)
                if np.linalg.norm(u_c - best_u_per_pt[k_idx]) <= 1.0e-7:
                    continue
                if _rem_point_budget(k_idx, cap_override=pass1_cap) < 2:
                    break
                try:
                    _eval_for_point(k_idx, u_c, bucket="warm", point_cap_override=pass1_cap)
                except _GlobalBudgetExhausted:
                    incomplete_mask[k_idx] = True
                    status_codes[k_idx] = -2
                    messages[k_idx] = "Global evaluation budget exhausted during warm-start probe."
                    term_categories[k_idx] = "failed_global_budget_exhausted"
                    return best_u_per_pt[k_idx].copy()
                except _PointBudgetExhausted:
                    break

            if float(prof_costs[k_idx]) <= 1.0e-15:
                conv_flags[k_idx] = True
                status_codes[k_idx] = 0
                messages[k_idx] = "Zero objective value (<= 1e-15) achieved at initial/warm point."
                term_categories[k_idx] = "converged_optimality"
                incomplete_mask[k_idx] = False
                return best_u_per_pt[k_idx].copy()

            rem_before_opt = _rem_point_budget(k_idx, cap_override=pass1_cap)
            if rem_before_opt < 4:
                if total_evals >= max_total_evaluations:
                    incomplete_mask[k_idx] = True
                    status_codes[k_idx] = -2
                    messages[k_idx] = "Global evaluation budget exhausted prior to local optimization."
                    term_categories[k_idx] = "failed_global_budget_exhausted"
                else:
                    incomplete_mask[k_idx] = False
                    status_codes[k_idx] = 1
                    messages[k_idx] = f"Point evaluation budget ({max_evals_per_point}) exhausted prior to optimization."
                    term_categories[k_idx] = "failed_local_eval_limit"
                return best_u_per_pt[k_idx].copy()

            # 5. Stadio 1: Ottimizzazione primaria L-BFGS-B con riserva di budget per il recupero
            if rem_before_opt >= 44:
                stage1_eval_cap = max(20, int(0.55 * rem_before_opt))
            else:
                stage1_eval_cap = rem_before_opt

            stage1_evals_used = 0
            start_u = best_u_per_pt[k_idx].copy()

            def _sub_prim(u_3d: np.ndarray) -> float:
                nonlocal stage1_evals_used
                if stage1_evals_used >= stage1_eval_cap:
                    raise _Stage1SubBudgetReached()
                val = _eval_for_point(
                    k_idx, u_3d, bucket="prim", point_cap_override=pass1_cap
                )
                stage1_evals_used += 1
                return val

            raw_success = False
            status_point = 1
            msg_point = ""
            nit_point = 0
            aborted_by_global = False
            aborted_by_point_cap = False
            hit_stage1_subcap = False

            try:
                if optimizer_method.upper() == "L-BFGS-B":
                    opts = {
                        "maxfun": int(stage1_eval_cap),
                        "maxiter": int(max_iter_per_point),
                        "ftol": float(ftol),
                        "gtol": float(gtol),
                        "eps": float(fd_eps),
                        "maxls": 8,
                    }
                else:
                    opts = {
                        "maxfev": int(stage1_eval_cap),
                        "maxiter": int(max_iter_per_point),
                        "ftol": float(ftol),
                    }

                res = minimize(
                    _sub_prim,
                    x0=start_u,
                    method=optimizer_method,
                    bounds=[(0.0, 1.0)] * 3,
                    options=opts,
                )
                raw_success = bool(res.success)
                status_point = int(getattr(res, "status", 0 if raw_success else 1))
                msg_raw = getattr(res, "message", "")
                msg_point = (
                    msg_raw.decode("utf-8", errors="ignore")
                    if isinstance(msg_raw, bytes)
                    else str(msg_raw)
                )
                nit_point = int(getattr(res, "nit", 0))
            except _Stage1SubBudgetReached:
                hit_stage1_subcap = True
                raw_success = False
                status_point = 1
                msg_point = f"Stage 1 sub-budget ({stage1_eval_cap} evals) reached; switching to recovery."
            except _PointBudgetExhausted:
                aborted_by_point_cap = True
                raw_success = False
                status_point = 1
                msg_point = f"Point evaluation budget ({max_evals_per_point}) reached in Stage 1."
            except _GlobalBudgetExhausted:
                aborted_by_global = True
                raw_success = False
                status_point = -2
                msg_point = "Optimization interrupted: global evaluation budget exhausted."

            stage1_genuinely_converged = (
                bool(raw_success)
                and (status_point == 0)
                and (not hit_stage1_subcap)
                and (not aborted_by_point_cap)
                and (not aborted_by_global)
                and (nit_point < max_iter_per_point)
            )

            # 6. Stadio 2: Recupero derivative-free (Nelder-Mead vincolato su [0,1]^3)
            final_converged = stage1_genuinely_converged
            rem_for_rec = _rem_point_budget(k_idx, cap_override=pass1_cap)
            if (not final_converged) and (not aborted_by_global) and (rem_for_rec >= 10):
                try:
                    rad = 0.003 if (prof_costs[k_idx] < oat_costs[k_idx] - 1.0e-7) else 0.012
                    conv_rec, st_rec, msg_rec, nit_rec = _run_recovery_stage(
                        k_idx,
                        best_u_per_pt[k_idx].copy(),
                        simplex_radius=rad,
                        point_cap_override=pass1_cap,
                    )
                    nit_point += nit_rec
                    if conv_rec:
                        final_converged = True
                        status_point = st_rec
                        msg_point = f"Stage 1 ({msg_point}) -> {msg_rec}"
                    else:
                        msg_point = f"Stage 1 ({msg_point}); {msg_rec}"
                        if evals_per_pt[k_idx] >= pass1_cap:
                            aborted_by_point_cap = True
                except _GlobalBudgetExhausted:
                    aborted_by_global = True
                    final_converged = False
                    status_point = -2
                    msg_point = "Optimization interrupted during recovery: global budget exhausted."

            if aborted_by_global:
                t_cat = "failed_global_budget_exhausted"
            elif final_converged:
                t_cat = "converged_optimality"
            elif (
                aborted_by_point_cap
                or (evals_per_pt[k_idx] >= pass1_cap)
                or (nit_point >= max_iter_per_point)
                or (status_point == 1)
            ):
                t_cat = "failed_local_eval_limit"
            else:
                t_cat = "failed_other"

            conv_flags[k_idx] = bool(final_converged)
            status_codes[k_idx] = int(status_point)
            messages[k_idx] = str(msg_point)
            term_categories[k_idx] = t_cat
            incomplete_mask[k_idx] = bool(aborted_by_global)
            iters_per_pt[k_idx] += nit_point

            return best_u_per_pt[k_idx].copy()

        # --- PASSAGGIO 1: Scansione Outward dal riferimento verso i bordi ---
        solved_up: List[int] = []
        for k_idx in range(ref_idx, m_pts):
            cands: List[np.ndarray] = []
            if len(solved_up) >= 1:
                k_p1 = solved_up[-1]
                u_p1 = best_u_per_pt[k_p1].copy()
                cands.append(u_p1)
                if len(solved_up) >= 2:
                    k_p2 = solved_up[-2]
                    u_p2 = best_u_per_pt[k_p2].copy()
                    dg = float(grid_norm[k_p1] - grid_norm[k_p2])
                    if abs(dg) > 1.0e-12:
                        ratio = float(grid_norm[k_idx] - grid_norm[k_p1]) / dg
                        u_sec = np.clip(u_p1 + ratio * (u_p1 - u_p2), 0.0, 1.0)
                        cands.append(u_sec)
            _optimize_at_grid_index(k_idx, cands, is_reverse_pass=False)
            solved_up.append(k_idx)

        solved_down: List[int] = [ref_idx]
        for k_idx in range(ref_idx - 1, -1, -1):
            cands = []
            if len(solved_down) >= 1:
                k_p1 = solved_down[-1]
                u_p1 = best_u_per_pt[k_p1].copy()
                cands.append(u_p1)
                if len(solved_down) >= 2:
                    k_p2 = solved_down[-2]
                    u_p2 = best_u_per_pt[k_p2].copy()
                    dg = float(grid_norm[k_p1] - grid_norm[k_p2])
                    if abs(dg) > 1.0e-12:
                        ratio = float(grid_norm[k_idx] - grid_norm[k_p1]) / dg
                        u_sec = np.clip(u_p1 + ratio * (u_p1 - u_p2), 0.0, 1.0)
                        cands.append(u_sec)
            _optimize_at_grid_index(k_idx, cands, is_reverse_pass=False)
            solved_down.append(k_idx)

        # --- PASSAGGIO 2: Scansione Inward (Reverse) dai bordi verso il riferimento ---
        if bidirectional_scan and (not budget_exhausted):
            for k_idx in range(m_pts - 2, ref_idx, -1):
                if budget_exhausted:
                    break
                inward_u = best_u_per_pt[k_idx + 1].copy()
                _optimize_at_grid_index(k_idx, [inward_u], is_reverse_pass=True)

            for k_idx in range(1, ref_idx):
                if budget_exhausted:
                    break
                inward_u = best_u_per_pt[k_idx - 1].copy()
                _optimize_at_grid_index(k_idx, [inward_u], is_reverse_pass=True)

        valid_mask = np.isfinite(prof_costs)
        if np.any(valid_mask):
            min_idx = int(np.nanargmin(prof_costs))
            min_cost = float(prof_costs[min_idx])
            argmin_phys = float(grid_phys[min_idx])
            argmin_norm = float(grid_norm[min_idx])
            is_flat = bool((np.nanmax(prof_costs) - np.nanmin(prof_costs)) <= flat_tol)
            min_at_bdy = bool((not is_flat) and (min_idx == 0 or min_idx == (m_pts - 1)))
        else:
            min_cost = float("nan")
            argmin_phys = float("nan")
            argmin_norm = float("nan")
            is_flat = False
            min_at_bdy = False

        n_conv = int(sum(conv_flags))
        n_incomp = int(sum(incomplete_mask))
        n_non_conv = int(m_pts - n_conv)
        max_pt_ev = int(max(evals_per_pt)) if evals_per_pt else 0

        point_records: List[Dict[str, Any]] = []
        for k_idx in range(m_pts):
            breakdown_sum = (
                nfev_oat[k_idx]
                + nfev_warm[k_idx]
                + nfev_prim[k_idx]
                + nfev_rec[k_idx]
            )
            if breakdown_sum != evals_per_pt[k_idx]:
                raise RuntimeError(
                    f"Incoerenza interna contatori per {pname}[{k_idx}]: "
                    f"breakdown_sum={breakdown_sum} != nfev={evals_per_pt[k_idx]}"
                )
            if evals_per_pt[k_idx] > max_evals_per_point:
                raise RuntimeError(
                    f"Violazione del limite per punto su {pname}[{k_idx}]: "
                    f"nfev={evals_per_pt[k_idx]} > max_evals_per_point={max_evals_per_point}"
                )

            point_records.append(
                {
                    "grid_index": int(k_idx),
                    "grid_normalized": float(grid_norm[k_idx]),
                    "grid_physical": float(grid_phys[k_idx]),
                    "oat_cost": float(oat_costs[k_idx]),
                    "profile_cost": float(prof_costs[k_idx]),
                    "success": bool(conv_flags[k_idx]),
                    "status": int(status_codes[k_idx]),
                    "message": str(messages[k_idx]),
                    "nit": int(iters_per_pt[k_idx]),
                    "nfev": int(evals_per_pt[k_idx]),
                    "nfev_breakdown": {
                        "oat": int(nfev_oat[k_idx]),
                        "warm_probes": int(nfev_warm[k_idx]),
                        "primary_opt": int(nfev_prim[k_idx]),
                        "recovery_opt": int(nfev_rec[k_idx]),
                    },
                    "termination_category": str(term_categories[k_idx]),
                    "incomplete_due_to_global_budget": bool(incomplete_mask[k_idx]),
                    "hit_bounds": bool(hit_bounds_mask[k_idx]),
                    "nuisance_at_bounds": list(at_bounds_names[k_idx]),
                    "optimized_theta_normalized": opt_norm_mat[k_idx].tolist(),
                    "optimized_theta_physical": opt_phys_mat[k_idx].tolist(),
                }
            )

        profiles_dict[pname] = OptimizedProfile1D(
            parameter_name=pname,
            parameter_index=j,
            grid_normalized=grid_norm,
            grid_physical=grid_phys,
            profile_cost_values=prof_costs,
            oat_slice_cost_values=oat_costs,
            optimized_thetas_normalized=opt_norm_mat,
            optimized_thetas_physical=opt_phys_mat,
            nuisance_hit_bounds_mask=hit_bounds_mask,
            nuisance_at_bounds_names=at_bounds_names,
            evaluations_per_point=evals_per_pt,
            iterations_per_point=iters_per_pt,
            converged_flags=conv_flags,
            optimizer_status_codes=status_codes,
            optimizer_messages=messages,
            termination_categories=term_categories,
            incomplete_points_mask=incomplete_mask,
            pointwise_optimization_records=point_records,
            reference_normalized=t_ref_j,
            reference_physical=float(theta_phys_0[j]),
            profile_min_cost=min_cost,
            profile_argmin_physical=argmin_phys,
            profile_argmin_normalized=argmin_norm,
            profile_minimum_at_grid_boundary=min_at_bdy,
            is_flat_within_tol=is_flat,
            num_converged_points=n_conv,
            num_non_converged_points=n_non_conv,
            num_incomplete_points=n_incomp,
            all_points_converged=bool(n_conv == m_pts and n_incomp == 0),
            max_point_evaluations_observed=max_pt_ev,
            total_evaluations_for_parameter=int(sum(evals_per_pt)),
        )

    tot_pts = int(sum(len(p.grid_normalized) for p in profiles_dict.values()))
    tot_conv = int(sum(p.num_converged_points for p in profiles_dict.values()))
    tot_non_conv = int(sum(p.num_non_converged_points for p in profiles_dict.values()))
    tot_incomp = int(sum(p.num_incomplete_points for p in profiles_dict.values()))
    tot_pt_evals = int(sum(p.total_evaluations_for_parameter for p in profiles_dict.values()))

    per_point_ok = all(
        p.max_point_evaluations_observed <= max_evals_per_point for p in profiles_dict.values()
    )
    accounting_ok = bool(
        (setup_reference_evals + tot_pt_evals == total_evals)
        and (total_evals == raw_call_counter)
        and (total_evals <= max_total_evaluations)
    )
    if not accounting_ok:
        raise RuntimeError(
            f"Errore di contabilizzazione valutazioni: setup={setup_reference_evals} + "
            f"points={tot_pt_evals} vs total_evals={total_evals} vs raw_calls={raw_call_counter}"
        )

    all_ok = bool(
        tot_conv == tot_pts
        and tot_incomp == 0
        and (not budget_exhausted)
        and per_point_ok
        and accounting_ok
    )

    note = (
        "Optimized Objective Profiles P_j(theta_j) = min_{theta_{-j} in Theta_{-j}} J_obj(theta_j, theta_{-j}) "
        f"computed with strict per-point budget cap (nfev <= {max_evals_per_point}) and two-stage "
        f"L-BFGS-B + bounded Nelder-Mead recovery ({tot_conv}/{tot_pts} points genuinely converged; "
        f"total evaluations = {total_evals} = {setup_reference_evals} setup + {tot_pt_evals} point evals). "
        + (
            "Calibrated statistical profile likelihood under declared noise covariance."
            if is_statistical_profile_likelihood
            else "Deterministic objective profile in dimensionless observation space (NO statistical noise model assumed; "
            "do NOT interpret as calibrated statistical profile likelihood or chi-square confidence intervals)."
        )
    )

    return OptimizedObjectiveProfilesReport(
        analysis_kind="optimized_objective_profiles",
        is_statistical_profile_likelihood=bool(is_statistical_profile_likelihood),
        profiles=profiles_dict,
        setup_reference_evaluations=setup_reference_evals,
        total_point_evaluations=tot_pt_evals,
        total_forward_evaluations=total_evals,
        max_total_budget=int(max_total_evaluations),
        max_evals_per_point=int(max_evals_per_point),
        max_iter_per_point=int(max_iter_per_point),
        bidirectional_scan=bool(bidirectional_scan),
        budget_exhausted=budget_exhausted,
        per_point_budget_respected=per_point_ok,
        accounting_consistency_verified=accounting_ok,
        total_grid_points=tot_pts,
        total_converged_points=tot_conv,
        total_non_converged_points=tot_non_conv,
        total_incomplete_points=tot_incomp,
        all_profiles_complete_and_converged=all_ok,
        optimizer_method=optimizer_method,
        methodology_note=note,
    )

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
