#!/usr/bin/env python3
"""Anatomical tracer-only local identifiability pilot, with regional coefficients.

Run from the Fractional-Poro-Viscoelastic repository root, after copying this
file to experiments/. Does NOT alter the existing solver or manuscript artifacts.

For each material profile, a single set of 6-parameter finite-difference forward
solves is evaluated using TWO pre-registered observation designs (balanced vs
edema-only). Four-parameter sub-Jacobians and six-parameter nuisance analyses are
then compared without additional forward solves.

All sensors and perturbation scales are selected independent of model sensitivities.
No numerical inversion or profile likelihood is claimed by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from reteco import _core, load_master_config
from reteco.config_schema import FixedPhysicsModel
from reteco.anatomy.regional_materials import load_regional_materials
from reteco.forward.wrapper import RetecoForwardProblem2D
from reteco.forward.target_generator import build_anatomical_driving_Callbacks
from reteco.observation.q1_exact import ExactQ1TracerObservationOperator

NAMES = ("alpha", "beta", "D", "E", "K_h", "A_v")


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def baseline_parameters(master: Any) -> np.ndarray:
    theta = master.parameter_space_theta.theta_dagger_array()
    phys = master.fixed_physics
    av = float(phys.eta_v) * float(phys.tau_v) ** float(theta[1])
    return np.r_[theta, float(phys.K_h), av].astype(float)


def normalized_spans(master: Any) -> np.ndarray:
    """Delta physical/ln parameter per unit normalized step. Not uncertainty priors."""
    b = master.parameter_space_theta.bounds
    return np.array([
        b.alpha[1] - b.alpha[0],
        b.beta[1] - b.beta[0],
        math.log(b.D[1] / b.D[0]),
        math.log(b.E[1] / b.E[0]),
        math.log(4.0),   # K_h: nominal / 2 to nominal * 2
        math.log(4.0),   # A_v: nominal / 2 to nominal * 2
    ], dtype=float)


def perturb(base: np.ndarray, spans: np.ndarray, index: int, delta: float) -> np.ndarray:
    value = base.copy()
    if index in (0, 1):
        value[index] += delta * spans[index]
        if not (0.0 < value[index] < 1.0):
            raise ValueError(f"Fractional order {NAMES[index]} outside (0,1)")
    else:
        value[index] *= math.exp(delta * spans[index])
    return value


def get_edema_points(problem: RetecoForwardProblem2D, count: int) -> tuple[np.ndarray, dict]:
    """Deterministic maximin design among interior edema cells, not fitted to data."""
    labels = np.asarray(problem.cell_material_ids, dtype=int)
    quad = np.asarray(problem.cell_quad_connectivity, dtype=int)
    dofs = np.asarray(problem.cell_tracer_dof_connectivity, dtype=int)
    centers = np.mean(np.asarray(problem.tracer_pts)[dofs], axis=1)
    edema = np.flatnonzero(labels == 2)
    if edema.size < count:
        raise ValueError(f"Only {edema.size} edema cells for {count} sensors")

    # Reject cells sharing a polygon edge with another label or the exterior.
    neighbors: dict[tuple[int, int], list[int]] = {}
    for ci, q in enumerate(quad):
        for ia, ib in zip(q, np.roll(q, -1)):
            edge = tuple(sorted((int(ia), int(ib))))
            neighbors.setdefault(edge, []).append(ci)
    interior = np.ones(len(quad), dtype=bool)
    for members in neighbors.values():
        if len(members) != 2 or labels[members[0]] != labels[members[1]]:
            for ci in members:
                interior[ci] = False
    choices = np.flatnonzero((labels == 2) & interior)
    method = "edema interior-cell centers; deterministic maximin"
    if choices.size < count:
        choices = edema
        method = "edema cell centers (including interface cells); deterministic maximin"
    order = np.lexsort((centers[choices, 1], centers[choices, 0]))
    choices = choices[order]
    points = centers[choices]
    selection = [0]
    distance2 = np.sum((points - points[0]) ** 2, axis=1)
    while len(selection) < count:
        distance2[selection] = -1
        k = int(np.argmax(distance2))
        if distance2[k] < 0:
            raise RuntimeError("Cannot choose nonduplicate edema sensors")
        selection.append(k)
        distance2 = np.minimum(distance2, np.sum((points - points[k]) ** 2, axis=1))
    chosen = choices[np.asarray(selection, dtype=int)]
    return centers[chosen], {"method": method, "cell_indices": chosen.tolist(),
                             "num_total_edema_cells": int(len(edema)),
                             "num_interior_edema_cells": int(np.sum((labels == 2) & interior))}


def svd_info(jac: np.ndarray) -> dict:
    u, s, vt = np.linalg.svd(jac, full_matrices=False)
    tol = float(s[0] * 1e-8) if len(s) else 0.0
    rank = int(np.sum(s > tol))
    return {
        "singular_values": s.tolist(), "rank_at_relative_1e-8": rank,
        "condition_if_full_rank": float(s[0] / s[-1]) if rank == jac.shape[1] and s[-1] > 0 else None,
        "weakest_right_singular_vector": dict(zip(NAMES[:jac.shape[1]], vt[-1].tolist())),
        "column_norms": dict(zip(NAMES[:jac.shape[1]], np.linalg.norm(jac, axis=0).tolist())),
    }


def nuisance_info(jac6: np.ndarray) -> dict:
    """Project four-parameter sensitivities off the nuisance-column subspace."""
    j4, jn = jac6[:, :4], jac6[:, 4:]
    u, s, _ = np.linalg.svd(jn, full_matrices=False)
    effective_rank = int(np.sum(s > s[0] * 1e-8)) if s.size and s[0] else 0
    q = u[:, :effective_rank]
    projected = j4 - q @ (q.T @ j4)
    sv = np.linalg.svd(projected, compute_uv=False)
    nullvec = np.linalg.svd(projected, full_matrices=False)[2][-1]
    correlations = np.zeros((6, 6))
    norms = np.linalg.norm(jac6, axis=0)
    for i in range(6):
        for k in range(6):
            correlations[i, k] = float(np.dot(jac6[:, i], jac6[:, k]) / (norms[i]*norms[k])) if norms[i]*norms[k] else np.nan
    return {
        "nuisance_rank": effective_rank,
        "effective_singular_values_for_4_parameters": sv.tolist(),
        "effective_FIM_eigenvalues_for_4_parameters": (sv * sv).tolist(),
        "weakest_effective_direction": dict(zip(NAMES[:4], nullvec.tolist())),
        "absolute_column_cosine_matrix": np.abs(correlations).tolist(),
        "names": list(NAMES),
    }


def calculate(args: argparse.Namespace) -> None:
    p_cfg, p_anat, p_mesh, p_geom, p_mat, p_bal = map(resolve, (
        args.physics_config, args.anatomical_config, args.mesh, args.geometry_report,
        args.materials_config, args.sensor_spec,
    ))
    output = resolve(args.out_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    for p in (p_cfg, p_anat, p_mesh, p_geom, p_mat, p_bal):
        if not p.is_file():
            raise FileNotFoundError(p)

    geom = json.loads(p_geom.read_text(encoding="utf-8"))
    if not geom.get("all_mandatory_checks_passed", False):
        raise ValueError("Geometry report did not pass mandatory checks")
    declared_mesh = geom.get("exported_files", {}).get("gmsh_msh")
    if declared_mesh and resolve(declared_mesh) != p_mesh:
        raise ValueError("Mesh does not match geometry report")
    bal_spec = json.loads(p_bal.read_text(encoding="utf-8"))
    balanced = np.asarray(bal_spec["sensor_points"], dtype=float)
    if balanced.ndim != 2 or balanced.shape[1] != 2 or len(balanced) < 4 or not np.isfinite(balanced).all():
        raise ValueError("Balanced sensor coordinates must be finite Nx2, N >= 4")

    master = load_master_config(p_cfg)
    anatomy = yaml.safe_load(p_anat.read_text(encoding="utf-8"))
    fwd = anatomy["forward_simulation"]
    final_t = float(fwd["final_time_T"])
    n_steps = int(args.time_steps or fwd["time_steps_M"])
    grid = _core.TimeGrid.make_graded(final_t, n_steps, float(fwd["grading_r"]))
    obs_times = np.linspace(final_t/5, final_t, 5)
    base = baseline_parameters(master)
    spans = normalized_spans(master)

    output.mkdir(parents=True)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_parameters": dict(zip(NAMES, base.tolist())),
        "parameter_normalized_spans": dict(zip(NAMES, spans.tolist())),
        "nuisance_bounds": {"K_h": [float(base[4]/2), float(base[4]*2)],
                            "A_v": [float(base[5]/2), float(base[5]*2)]},
        "time_steps": n_steps, "T_final_adim": final_t,
        "observation_times": obs_times.tolist(), "noise_fraction": args.noise_fraction,
        "fd_step": args.fd_step, "extra_fd_step": args.extra_fd_step,
        "hashes": {str(p.name): sha256(p) for p in (p_cfg,p_anat,p_mesh,p_geom,p_mat,p_bal)},
        "note": "Geometric local sensitivity. Gaussian iid noise is an assumed diagnostic model, not empirical calibration. Regional contrast factors are fixed hypotheses.",
    }
    all_results, arrays, rows = {}, {}, []
    sensors_by_design: dict[str, np.ndarray] = {}

    for material_name in args.material_profiles:
        region_profile = load_regional_materials(p_mat, material_name, float(master.fixed_physics.phi_0))
        print(f"[materials: {material_name}]", flush=True)

        def evaluate(params: np.ndarray, get_full: bool = False):
            beta = float(params[1])
            tau_v = float(master.fixed_physics.tau_v)
            eta_v = float(params[5]) / tau_v ** beta
            phys_dict = master.fixed_physics.model_dump()
            phys_dict["K_h"] = float(params[4])
            phys_dict["eta_v"] = eta_v
            phys = FixedPhysicsModel.model_validate(phys_dict)
            problem = RetecoForwardProblem2D(
                physics_model=phys, scales=master.characteristic_scales,
                time_grid=grid, mesh_file_path=p_mesh,
                dirichlet_disp_boundary_ids=fwd.get("dirichlet_disp_boundary_ids", [1]),
                fe_deg_disp=int(fwd.get("fe_deg_disp", 2)),
                fe_deg_press=int(fwd.get("fe_deg_press", 1)),
                fe_deg_tracer=int(fwd.get("fe_deg_tracer", 1)),
                use_supg=bool(fwd.get("use_supg", False)),
                region_materials=region_profile,
            )
            c0, mech, fluid, _ = build_anatomical_driving_Callbacks(
                problem=problem, driving_cfg=fwd["driving_protocol"], geometry_report_path=p_geom,
            )
            if not sensors_by_design:
                edema_pts, edema_meta = get_edema_points(problem, len(balanced))
                sensors_by_design["balanced"] = balanced
                sensors_by_design["edema"] = edema_pts
                meta["edema_design"] = edema_meta
                meta["balanced_design"] = bal_spec.get("method", "previous matched pilot")
            traj = problem.solve_trajectory(
                np.asarray(params[:4], dtype=float), c0=c0,
                F_mech_func=mech, S_fluid_func=fluid,
                enforce_initial_compatibility=True,
                compute_diagnostics=False,
            )
            if not np.isfinite(traj.c_hist).all():
                raise RuntimeError("Nonfinite tracer solution")
            observations = {}
            for design, pts in sensors_by_design.items():
                op = ExactQ1TracerObservationOperator(problem, pts, obs_times)
                observations[design] = np.asarray(op.apply(problem, traj), dtype=float).ravel()
            return (observations, problem, traj) if get_full else observations

        ref_obs, ref_problem, ref_traj = evaluate(base, get_full=True)
        meta.setdefault("num_cells", int(ref_problem.num_active_cells))
        profile_result = {
            "min_reference_concentration": float(np.min(ref_traj.c_hist)),
            "max_reference_concentration": float(np.max(ref_traj.c_hist)),
            "mass_label": "not computed; prior reports cover mass conservation",
            "max_darcy_velocity_adim": float(ref_traj.max_darcy_velocity),
        }

        # If requested, compare the unit regional map with *no* regional map.
        if args.check_unit_materials and material_name == "homogeneous":
            profile_save = region_profile
            region_profile = None
            _, _, reference_no_map = evaluate(base, get_full=True)
            region_profile = profile_save
            profile_result["homogeneous_map_vs_no_map"] = {
                "relative_d": float(np.linalg.norm(ref_traj.d_hist-reference_no_map.d_hist) / max(np.linalg.norm(reference_no_map.d_hist),1e-15)),
                "relative_p": float(np.linalg.norm(ref_traj.p_hist-reference_no_map.p_hist) / max(np.linalg.norm(reference_no_map.p_hist),1e-15)),
                "relative_c": float(np.linalg.norm(ref_traj.c_hist-reference_no_map.c_hist) / max(np.linalg.norm(reference_no_map.c_hist),1e-15)),
            }
            print("  unit mapping check:", profile_result["homogeneous_map_vs_no_map"], flush=True)

        for design, y in ref_obs.items():
            arrays[f"{material_name}_{design}_sensors"] = sensors_by_design[design]
            arrays[f"{material_name}_{design}_y0"] = y

        steps = [float(args.fd_step)]
        if args.extra_fd_step is not None:
            steps.append(float(args.extra_fd_step))
        all_jacobians: dict[float, dict[str, np.ndarray]] = {}
        for step in steps:
            jacts = {d: np.zeros((len(ref_obs[d]), 6), dtype=float) for d in ref_obs}
            for j, param in enumerate(NAMES):
                plus = evaluate(perturb(base, spans, j, +step))
                minus = evaluate(perturb(base, spans, j, -step))
                for design in ref_obs:
                    jacts[design][:, j] = (plus[design]-minus[design]) / (2*step)
                print(f"  derivative {param:5s}: step={step:g} OK",flush=True)
            all_jacobians[step] = jacts

        for design, j6 in all_jacobians[steps[0]].items():
            y = ref_obs[design]
            sigma = float(args.noise_fraction * np.linalg.norm(y) / math.sqrt(len(y)))
            if sigma <= 0:
                raise ValueError("Nonpositive assumed noise scale")
            jw = j6 / sigma
            j4w = jw[:, :4]
            metrics = {
                "num_sensors": int(len(sensors_by_design[design])),
                "num_observations": int(len(y)),
                "y0_norm": float(np.linalg.norm(y)),
                "assumed_sigma_per_observation": sigma,
                "svd_4": svd_info(j4w),
                "svd_6": svd_info(jw),
                "nuisance_projection": nuisance_info(jw),
            }
            if len(steps) > 1:
                j_alt = all_jacobians[steps[1]][design]
                metrics["finite_difference_audit"] = {
                    "relative_frobenius_change": float(np.linalg.norm(j_alt-j6) / max(np.linalg.norm(j6),1e-15)),
                    "relative_column_changes": dict(zip(NAMES,[
                        float(np.linalg.norm(j_alt[:, k]-j6[:, k]) / max(np.linalg.norm(j6[:, k]),1e-15))
                        for k in range(6)
                    ])),
                }
            profile_result[design] = metrics
            arrays[f"{material_name}_{design}_J6"] = j6
            arrays[f"{material_name}_{design}_J6_noise_whitened"] = jw
            rows.append({
                "material_profile": material_name, "observation_design": design,
                "n_obs": len(y), "sigma_noise_assumed": sigma,
                "smin_4": metrics["svd_4"]["singular_values"][-1],
                "smin_6": metrics["svd_6"]["singular_values"][-1],
                "smin_4_after_nuisance": metrics["nuisance_projection"]["effective_singular_values_for_4_parameters"][-1],
                "cond_4": metrics["svd_4"]["condition_if_full_rank"],
                "cond_6": metrics["svd_6"]["condition_if_full_rank"],
            })
        all_results[material_name] = profile_result

    meta["sensor_points"] = {k: v.tolist() for k,v in sensors_by_design.items()}
    (output / "summary.json").write_text(json.dumps({"metadata":meta,"results":all_results},indent=2,ensure_ascii=False),encoding="utf-8")
    np.savez_compressed(output / "jacobians_and_observations.npz", **arrays)
    with (output / "comparison.csv").open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / "sensor_points.json").write_text(json.dumps(meta["sensor_points"],indent=2),encoding="utf-8")
    print(f"Completed. Results in: {output}",flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-config", default="configs/scales_and_physics.yaml")
    parser.add_argument("--anatomical-config", default="configs/brats_anatomical_forward.yaml")
    parser.add_argument("--mesh", default="artifacts/02_anatomical_forward_clinical/meshes/BraTS20_Training_001_slice067_h3.0mm.msh")
    parser.add_argument("--geometry-report", default="artifacts/02_anatomical_forward_clinical/geometry_quality_report.json")
    parser.add_argument("--materials-config", default="configs/brats_region_materials_phaseBC.yaml")
    parser.add_argument("--sensor-spec", default="artifacts/16_regional_identifiability_pilot/sensor_points.json")
    parser.add_argument("--material-profiles", nargs="+", choices=("homogeneous","exploratory_contrast"),default=["homogeneous","exploratory_contrast"])
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument("--noise-fraction", type=float, default=0.05)
    parser.add_argument("--fd-step", type=float, default=1e-3)
    parser.add_argument("--extra-fd-step", type=float, default=None, help="Optional 2nd step for FD stability; doubles perturbed forward solves")
    parser.add_argument("--check-unit-materials",action="store_true",help="Extra forward solve to compare homogeneous mapping with no mapping")
    parser.add_argument("--out-dir",default="artifacts/16_anatomical_svd_pilot")
    args=parser.parse_args()
    if args.noise_fraction <= 0 or not (0 < args.fd_step < .01) or (args.extra_fd_step is not None and not (0 < args.extra_fd_step < .01)):
        parser.error("noise-fraction must be positive and FD steps must be in (0,0.01)")
    calculate(args)


if __name__ == "__main__":
    main()
