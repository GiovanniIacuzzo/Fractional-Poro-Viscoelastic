"""
python/reteco/identifiability/__init__.py
Esporta gli strumenti di analisi di identificabilità locale (Jacobiano, audit FD, SVD)
e pratica (One-at-a-Time Cost Slices e Optimized Objective Profiles).
"""

from __future__ import annotations

from .profile_likelihood import (
    CostSlice1D,
    OneAtATimeCostSlicesReport,
    OptimizedObjectiveProfilesReport,
    OptimizedProfile1D,
    build_misfit_objective_from_observations,
    compute_cost_slices,
    compute_one_at_a_time_cost_slices,
    compute_optimized_objective_profiles,
)
from .sensitivity_jacobian import (
    PARAMETER_NAMES,
    FDStepStabilityReport,
    JacobianAuditResult,
    ObservationWeightingSpec,
    compute_normalized_jacobian,
    compute_normalized_jacobian_with_audit,
    verify_fd_step_stability,
)
from .svd_conditioning import (
    LocalIdentifiabilityReport,
    analyze_jacobian_svd,
    sanitize_for_json,
)

__all__ = [
    "PARAMETER_NAMES",
    "CostSlice1D",
    "FDStepStabilityReport",
    "JacobianAuditResult",
    "LocalIdentifiabilityReport",
    "ObservationWeightingSpec",
    "OneAtATimeCostSlicesReport",
    "OptimizedObjectiveProfilesReport",
    "OptimizedProfile1D",
    "analyze_jacobian_svd",
    "build_misfit_objective_from_observations",
    "compute_cost_slices",
    "compute_normalized_jacobian",
    "compute_normalized_jacobian_with_audit",
    "compute_one_at_a_time_cost_slices",
    "compute_optimized_objective_profiles",
    "sanitize_for_json",
    "verify_fd_step_stability",
]