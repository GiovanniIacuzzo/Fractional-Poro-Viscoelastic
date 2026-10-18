"""
python/reteco/identifiability/profile_likelihood.py
Analisi pratica dell'identificabilità parametrica sul funzionale di costo:
1) `compute_one_at_a_time_cost_slices`: sezioni 1D del funzionale mantenendo fissi gli altri
   tre parametri al valore di riferimento theta_dagger (etichettate esplicitamente come
   One-at-a-Time Cost Slices e MAI come profile likelihood);
2) `compute_optimized_objective_profiles`: profili dell'obiettivo in cui, per ogni valore
   fissato di theta_j sulla griglia, i tre parametri di disturbo theta_{-j} vengono
   ottimizzati nel dominio ammissibile Theta_{-j} con budget computazionale controllato
   e rilevamento esplicito dei minimi vincolati ai bordi.
"""

from __future__ import annotations
from dataclasses import dataclass, field
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
    oat_slice_cost_values: np.ndarray          # (M,) J_obj(theta_j, theta_{-j}^dagger) sulla stessa griglia
    optimized_thetas_normalized: np.ndarray    # (M, 4)
    optimized_thetas_physical: np.ndarray      # (M, 4)
    nuisance_hit_bounds_mask: List[bool]       # True se almeno un parametro di disturbo tocca [0, 1]
    nuisance_at_bounds_names: List[List[str]]
    evaluations_per_point: List[int]
    converged_flags: List[bool]
    reference_normalized: float
    reference_physical: float
    profile_min_cost: float
    profile_argmin_physical: float
    profile_argmin_normalized: float
    profile_minimum_at_grid_boundary: bool
    is_flat_within_tol: bool
    total_evaluations_for_parameter: int


@dataclass(frozen=True)
class OptimizedObjectiveProfilesReport:
    analysis_kind: str                         # "optimized_objective_profiles"
    is_statistical_profile_likelihood: bool
    profiles: Dict[str, OptimizedProfile1D]
    total_forward_evaluations: int
    max_total_budget: int
    budget_exhausted: bool
    optimizer_method: str
    methodology_note: str

    def to_json_dict(self) -> Dict[str, Any]:
        return sanitize_for_json(
            {
                "analysis_kind": self.analysis_kind,
                "is_statistical_profile_likelihood": self.is_statistical_profile_likelihood,
                "methodology_note": self.methodology_note,
                "optimizer_method": self.optimizer_method,
                "total_forward_evaluations": int(self.total_forward_evaluations),
                "max_total_budget": int(self.max_total_budget),
                "budget_exhausted": bool(self.budget_exhausted),
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
                        "converged_flags": list(p.converged_flags),
                        "reference_normalized": p.reference_normalized,
                        "reference_physical": p.reference_physical,
                        "profile_min_cost": p.profile_min_cost,
                        "profile_argmin_physical": p.profile_argmin_physical,
                        "profile_argmin_normalized": p.profile_argmin_normalized,
                        "profile_minimum_at_grid_boundary": p.profile_minimum_at_grid_boundary,
                        "is_flat_within_tol": p.is_flat_within_tol,
                        "total_evaluations_for_parameter": p.total_evaluations_for_parameter,
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
    total_evals = 1

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
                total_evals += 1

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
        total_forward_evaluations=total_evals,
    )


# Alias retrocompatibile
compute_cost_slices = compute_one_at_a_time_cost_slices


def compute_optimized_objective_profiles(
    objective_callable: Callable[[np.ndarray], float],
    theta_ref: np.ndarray,
    transform: ParameterCoordinateTransform,
    num_grid_points: int = 9,
    max_evals_per_point: int = 25,
    max_total_evaluations: int = 900,
    optimizer_method: str = "L-BFGS-B",
    boundary_tol: float = 1.0e-3,
    flat_tol: float = 1.0e-12,
    is_statistical_profile_likelihood: bool = False,
) -> OptimizedObjectiveProfilesReport:
    """
    Calcola i profili ottimizzati del funzionale obiettivo:
        P_j(theta_{j,k}) = min_{theta_{-j} in Theta_{-j}} J_obj(theta_{j,k}, theta_{-j})
    utilizzando continuazione warm-start dal valore di riferimento verso gli estremi
    della griglia, rispettando i bounds [0, 1]^3 nello spazio normalizzato e un
    limite esplicito `max_total_evaluations` sul numero totale di valutazioni.
    """
    theta_phys_0 = np.asarray(theta_ref, dtype=np.float64)
    theta_tilde_0 = np.asarray(transform.to_normalized(theta_phys_0), dtype=np.float64)
    ref_cost = float(objective_callable(theta_phys_0))
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
        hit_bounds_mask: List[bool] = [False] * m_pts
        at_bounds_names: List[List[str]] = [[] for _ in range(m_pts)]
        evals_per_pt: List[int] = [0] * m_pts
        conv_flags: List[bool] = [False] * m_pts

        ref_idx = int(np.argmin(np.abs(grid_norm - t_ref_j)))

        # Ordine di visita: partiamo da ref_idx, saliamo verso 1.0 e poi scendiamo verso 0.0 (warm-start)
        up_branch = list(range(ref_idx, m_pts))
        down_branch = list(range(ref_idx - 1, -1, -1))

        def _solve_point(k_idx: int, warm_nuisance_norm: np.ndarray) -> np.ndarray:
            nonlocal total_evals, budget_exhausted
            val_norm = float(grid_norm[k_idx])

            # Valutazione OAT al punto (con i parametri di disturbo a theta_tilde_0)
            t_oat = theta_tilde_0.copy()
            t_oat[j] = val_norm
            p_oat = transform.from_normalized(t_oat)
            if abs(val_norm - t_ref_j) < 1.0e-15:
                c_oat = ref_cost
                pt_evals = 0
            else:
                if total_evals >= max_total_evaluations:
                    budget_exhausted = True
                    return warm_nuisance_norm
                c_oat = float(objective_callable(p_oat))
                total_evals += 1
                pt_evals = 1
            oat_costs[k_idx] = c_oat

            # Se siamo esattamente al punto di riferimento e il costo è già ~0, ottimo noto
            if abs(val_norm - t_ref_j) < 1.0e-15 and c_oat <= 1.0e-14:
                prof_costs[k_idx] = c_oat
                opt_norm_mat[k_idx] = t_oat
                opt_phys_mat[k_idx] = p_oat
                evals_per_pt[k_idx] = pt_evals
                conv_flags[k_idx] = True
                return theta_tilde_0[nuisance_indices].copy()

            remaining_budget = max_total_evaluations - total_evals
            point_cap = min(max_evals_per_point, remaining_budget)
            if point_cap <= 1:
                if remaining_budget <= 0:
                    budget_exhausted = True
                prof_costs[k_idx] = c_oat
                opt_norm_mat[k_idx] = t_oat
                opt_phys_mat[k_idx] = p_oat
                evals_per_pt[k_idx] = pt_evals
                conv_flags[k_idx] = False
                return warm_nuisance_norm

            best_u = warm_nuisance_norm.copy()
            best_c = c_oat
            if not np.allclose(warm_nuisance_norm, theta_tilde_0[nuisance_indices]):
                t_warm = np.zeros(4, dtype=np.float64)
                t_warm[j] = val_norm
                t_warm[nuisance_indices] = np.clip(warm_nuisance_norm, 0.0, 1.0)
                c_warm = float(objective_callable(transform.from_normalized(t_warm)))
                total_evals += 1
                pt_evals += 1
                if c_warm < best_c:
                    best_c = c_warm
                    best_u = np.clip(warm_nuisance_norm, 0.0, 1.0)
                else:
                    best_u = theta_tilde_0[nuisance_indices].copy()

            # Se l'obiettivo è già nullo a precisione di macchina (es. parametro non identificabile con q=0),
            # non serve avviare iterazioni L-BFGS-B
            if best_c <= 1.0e-15:
                t_final = np.zeros(4, dtype=np.float64)
                t_final[j] = val_norm
                t_final[nuisance_indices] = best_u
                prof_costs[k_idx] = best_c
                opt_norm_mat[k_idx] = t_final
                opt_phys_mat[k_idx] = transform.from_normalized(t_final)
                evals_per_pt[k_idx] = pt_evals
                conv_flags[k_idx] = True
                return best_u

            local_eval_count = 0
            max_opt_evals = max(1, min(max_evals_per_point - pt_evals, max_total_evaluations - total_evals))

            class _PointCapReached(Exception):
                pass

            def _sub_obj(u_3d: np.ndarray) -> float:
                nonlocal total_evals, pt_evals, local_eval_count, best_c, best_u, budget_exhausted
                if local_eval_count >= max_opt_evals or total_evals >= max_total_evaluations:
                    if total_evals >= max_total_evaluations:
                        budget_exhausted = True
                    raise _PointCapReached()
                u_clipped = np.clip(np.asarray(u_3d, dtype=np.float64), 0.0, 1.0)
                t_full = np.zeros(4, dtype=np.float64)
                t_full[j] = val_norm
                t_full[nuisance_indices] = u_clipped
                val_c = float(objective_callable(transform.from_normalized(t_full)))
                total_evals += 1
                pt_evals += 1
                local_eval_count += 1
                if val_c < best_c:
                    best_c = val_c
                    best_u = u_clipped.copy()
                return val_c

            converged = False
            try:
                res = minimize(
                    _sub_obj,
                    x0=best_u,
                    method=optimizer_method,
                    bounds=[(0.0, 1.0)] * 3,
                    options={"maxfun": max_opt_evals, "maxiter": max_opt_evals, "ftol": 1.0e-10, "eps": 1.0e-3}
                    if optimizer_method.upper() == "L-BFGS-B"
                    else {"maxfev": max_opt_evals, "ftol": 1.0e-9},
                )
                converged = bool(res.success)
            except _PointCapReached:
                converged = False

            t_opt = np.zeros(4, dtype=np.float64)
            t_opt[j] = val_norm
            t_opt[nuisance_indices] = best_u
            p_opt = transform.from_normalized(t_opt)

            prof_costs[k_idx] = best_c
            opt_norm_mat[k_idx] = t_opt
            opt_phys_mat[k_idx] = p_opt
            evals_per_pt[k_idx] = pt_evals
            conv_flags[k_idx] = converged

            bound_names_k: List[str] = []
            for loc_i, u_val in enumerate(best_u):
                if u_val <= boundary_tol or u_val >= (1.0 - boundary_tol):
                    bound_names_k.append(nuisance_names[loc_i])
            hit_bounds_mask[k_idx] = len(bound_names_k) > 0
            at_bounds_names[k_idx] = bound_names_k
            return best_u

        warm_u = theta_tilde_0[nuisance_indices].copy()
        for k_idx in up_branch:
            warm_u = _solve_point(k_idx, warm_u)

        warm_u = opt_norm_mat[ref_idx, nuisance_indices].copy()
        for k_idx in down_branch:
            warm_u = _solve_point(k_idx, warm_u)

        valid_mask = np.isfinite(prof_costs)
        if np.any(valid_mask):
            min_idx = int(np.nanargmin(prof_costs))
            min_cost = float(prof_costs[min_idx])
            argmin_phys = float(grid_phys[min_idx])
            argmin_norm = float(grid_norm[min_idx])
            is_flat = bool((np.nanmax(prof_costs) - np.nanmin(prof_costs)) <= flat_tol)
            min_at_bdy = bool(
                (not is_flat) and (min_idx == 0 or min_idx == (m_pts - 1))
            )
        else:
            min_cost = float("nan")
            argmin_phys = float("nan")
            argmin_norm = float("nan")
            is_flat = False
            min_at_bdy = False

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
            converged_flags=conv_flags,
            reference_normalized=t_ref_j,
            reference_physical=float(theta_phys_0[j]),
            profile_min_cost=min_cost,
            profile_argmin_physical=argmin_phys,
            profile_argmin_normalized=argmin_norm,
            profile_minimum_at_grid_boundary=min_at_bdy,
            is_flat_within_tol=is_flat,
            total_evaluations_for_parameter=int(sum(evals_per_pt)),
        )

    note = (
        "Optimized Objective Profiles P_j(theta_j) = min_{theta_{-j} in Theta_{-j}} J_obj(theta_j, theta_{-j}). "
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
        total_forward_evaluations=total_evals,
        max_total_budget=int(max_total_evaluations),
        budget_exhausted=budget_exhausted,
        optimizer_method=optimizer_method,
        methodology_note=note,
    )