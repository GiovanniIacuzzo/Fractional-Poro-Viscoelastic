#!/usr/bin/env python3
"""Anatomy-only, cross-discretization inverse twin experiment (BraTS 2D).

PREPARE: compare anatomical 3mm/48, 3mm/96, 2mm/48 and 2mm/96
          on *identical frozen sensor coordinates* and regional coefficients.
          Generate clean fine-grid (2mm/96) target observations.
INVERT:  recover (alpha,beta,D,E) on a selected *anatomical* inverse mesh:
          coarse=3mm/48 (legacy default), or fine=2mm/48 (spatial-bias check),
          against a selected frozen anatomical target (fine_96 by default).
          K_h and A_v remain fixed at their reference values.

No unit-square inverse simulations. No clinical dynamical observations: BraTS
images define geometry/regions; forcing and observations are synthetic.
Nothing in the existing experiment directories is modified.
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

import numpy as np
import yaml
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from reteco import _core, load_master_config
from reteco.anatomy.regional_materials import load_regional_materials
from reteco.forward.target_generator import build_anatomical_driving_Callbacks
from reteco.forward.wrapper import RetecoForwardProblem2D
from reteco.observation.q1_exact import ExactQ1TracerObservationOperator
from reteco.config_schema import ParameterCoordinateTransform

PARAMS = ("alpha", "beta", "D", "E")
DESIGNS = ("balanced", "edema")


def pathof(value: str | Path) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path: Path, obj: dict | list) -> None:
    path.write_text(json.dumps(clean_json(obj), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_source(args):
    paths = {key: pathof(getattr(args, key)) for key in (
        "physics_config", "anatomical_config", "mesh_coarse", "mesh_fine",
        "geometry_report", "materials_config")}
    refdir = pathof(args.phase16_reference)
    paths["phase16_summary"] = refdir / "summary.json"
    paths["phase16_npz"] = refdir / "jacobians_and_observations.npz"
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    source = json.loads(paths["phase16_summary"].read_text(encoding="utf-8"))
    meta = source["metadata"]
    if args.material_profile not in source["results"]:
        raise ValueError(f"Missing regional material profile {args.material_profile!r} in Phase 16")
    # Freeze the files defining the original Phase-16 anatomical reference.
    for name in ("physics_config", "anatomical_config", "mesh_coarse", "geometry_report", "materials_config"):
        p = paths[name]
        expected = meta["hashes"].get(p.name)
        if expected != digest(p):
            raise ValueError(f"Phase-16 provenance SHA256 mismatch for {name}: {p}")
    geom = json.loads(paths["geometry_report"].read_text(encoding="utf-8"))
    if not geom.get("all_mandatory_checks_passed", False):
        raise ValueError("Anatomical geometry quality report failed")
    declared = geom.get("exported_files", {}).get("gmsh_msh")
    if declared and pathof(declared) != paths["mesh_coarse"]:
        raise ValueError("Coarse anatomical mesh does not match the verified geometry report")
    if paths["mesh_coarse"] == paths["mesh_fine"] or digest(paths["mesh_coarse"]) == digest(paths["mesh_fine"]):
        raise ValueError("Target and inverse must use distinct anatomical meshes")

    master = load_master_config(paths["physics_config"])
    cfg = yaml.safe_load(paths["anatomical_config"].read_text(encoding="utf-8"))
    fwd = cfg["forward_simulation"]
    if not math.isclose(float(fwd["final_time_T"]), float(meta["T_final_adim"]), abs_tol=1e-12):
        raise ValueError("Phase-16 and anatomical final times disagree")
    if int(meta["time_steps"]) != int(args.coarse_steps):
        raise ValueError("Inverse coarse time grid must match frozen Phase 16 (48 steps)")
    if not math.isclose(float(master.fixed_physics.tau_v), 1.0, abs_tol=1e-12):
        raise ValueError("Four-parameter FEM reuse assumes tau_v=1, so A_v=eta_v is fixed as beta varies")
    theta_ref = master.parameter_space_theta.theta_dagger_array()
    theta_phase16 = np.array([float(meta["reference_parameters"][p]) for p in PARAMS], dtype=float)
    if not np.allclose(theta_ref, theta_phase16, atol=1e-12, rtol=0):
        raise ValueError("Frozen Phase-16 reference theta differs from master physics")
    if not math.isclose(float(meta["reference_parameters"]["K_h"]), float(master.fixed_physics.K_h), abs_tol=1e-12):
        raise ValueError("K_h disagrees with reference")
    if not math.isclose(float(meta["reference_parameters"]["A_v"]), float(master.fixed_physics.eta_v), abs_tol=1e-12):
        raise ValueError("A_v disagrees with reference")
    if args.fine_steps <= args.coarse_steps:
        raise ValueError("Fine temporal grid must have more steps than coarse grid")
    obs_times = np.asarray(meta["observation_times"], dtype=float)
    sensors, frozen_y0 = {}, {}
    with np.load(paths["phase16_npz"]) as npz:
        for design in DESIGNS:
            stem = f"{args.material_profile}_{design}"
            sensors[design] = np.asarray(npz[stem + "_sensors"], dtype=float)
            frozen_y0[design] = np.asarray(npz[stem + "_y0"], dtype=float).ravel()
            if sensors[design].ndim != 2 or sensors[design].shape[1] != 2:
                raise ValueError(f"Invalid frozen sensors for {design}")
            if len(frozen_y0[design]) != len(sensors[design]) * len(obs_times):
                raise ValueError(f"Frozen observation shape mismatch for {design}")
    materials = load_regional_materials(paths["materials_config"], args.material_profile,
                                        float(master.fixed_physics.phi_0))
    return paths, master, fwd, theta_ref, obs_times, sensors, frozen_y0, materials


class AnatomicalEvaluator:
    """Reuse a single assembled FEM problem for four-parameter evaluations."""
    def __init__(self, master, fwd, mesh: Path, geometry_report: Path, materials,
                 steps: int, obs_times, sensors):
        self.mesh = mesh
        self.grid = _core.TimeGrid.make_graded(
            float(fwd["final_time_T"]), int(steps), float(fwd["grading_r"]))
        self.problem = RetecoForwardProblem2D(
            physics_model=master.fixed_physics,
            scales=master.characteristic_scales,
            time_grid=self.grid,
            mesh_file_path=str(mesh),
            dirichlet_disp_boundary_ids=fwd.get("dirichlet_disp_boundary_ids", [1]),
            fe_deg_disp=int(fwd.get("fe_deg_disp", 2)),
            fe_deg_press=int(fwd.get("fe_deg_press", 1)),
            fe_deg_tracer=int(fwd.get("fe_deg_tracer", 1)),
            use_supg=bool(fwd.get("use_supg", False)),
            region_materials=materials,
        )
        self.c0, self.force, self.fluid, self.forcing_meta = build_anatomical_driving_Callbacks(
            problem=self.problem, driving_cfg=fwd["driving_protocol"],
            geometry_report_path=geometry_report)
        self.operators = {d: ExactQ1TracerObservationOperator(
            self.problem, pts, obs_times) for d, pts in sensors.items()}
        self.cache: dict[tuple[float, ...], dict[str, np.ndarray]] = {}
        self.nsolves = 0

    def evaluate(self, theta, with_diagnostics=False):
        theta = np.asarray(theta, dtype=float)
        key = tuple(np.round(theta, 14))
        if not with_diagnostics and key in self.cache:
            return self.cache[key], None
        traj = self.problem.solve_trajectory(
            theta_vec=theta, c0=self.c0,
            F_mech_func=self.force, S_fluid_func=self.fluid,
            enforce_initial_compatibility=True, compute_diagnostics=False)
        self.nsolves += 1
        if not (np.isfinite(traj.c_hist).all() and np.isfinite(traj.d_hist).all()
                and np.isfinite(traj.p_hist).all()):
            raise RuntimeError(f"Nonfinite trajectory on {self.mesh}")
        result = {d: np.asarray(op.apply(self.problem, traj), dtype=float).ravel()
                  for d, op in self.operators.items()}
        self.cache[key] = result
        if not with_diagnostics:
            return result, None
        mass_weights = np.asarray(self.problem.M_phi_c_csr @ np.ones(self.problem.num_tracer_dofs)).ravel()
        masses = np.asarray(traj.c_hist @ mass_weights, dtype=float)
        drift = float(np.max(np.abs(masses - masses[0])) / max(abs(masses[0]), 1e-14))
        info = {
            "num_cells": int(self.problem.num_active_cells),
            "material_region_ids": sorted(set(int(x) for x in self.problem.cell_material_ids)),
            "num_tracer_dofs": int(self.problem.num_tracer_dofs),
            "num_disp_dofs": int(self.problem.num_disp_dofs),
            "num_pressure_dofs": int(self.problem.num_press_dofs),
            "final_d_l2": float(self.problem.compute_disp_L2_norm(traj.d_hist[-1])),
            "final_p_l2": float(self.problem.compute_press_L2_norm(traj.p_hist[-1])),
            "final_c_l2": float(self.problem.compute_tracer_L2_norm(traj.c_hist[-1])),
            "min_c_node_all_times": float(np.min(traj.c_hist)),
            "negative_c_dof_count": int(np.count_nonzero(traj.c_hist < -1e-10)),
            "relative_tracer_mass_drift": drift,
            "max_darcy_velocity_adim": float(traj.max_darcy_velocity),
        }
        return result, info


def error_metrics(y, other, sigma):
    diff = np.asarray(y) - np.asarray(other)
    l2 = float(np.linalg.norm(diff))
    return {
        "absolute_l2": l2,
        "relative_l2": l2 / max(float(np.linalg.norm(other)), 1e-15),
        "noise_whitened_l2": l2 / sigma,
        "noise_whitened_rms": l2 / (sigma * math.sqrt(len(diff))),
        "half_chi2": 0.5 * (l2/sigma)**2,
    }


def manifest_base(args, paths, theta, obs_times, sensors):
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "18_anatomical_mesh_independent_inverse",
        "domain": "MRI-derived BraTS 2D parenchyma proxy for BOTH target and inverse",
        "data_kind": "anatomy-guided synthetic forcing and synthetic tracer observations",
        "material_profile": args.material_profile,
        "regional_multipliers_are_given": True,
        "theta_reference": dict(zip(PARAMS, map(float, theta))),
        "unknowns_in_inverse": list(PARAMS),
        "known_coefficients": ["K_h", "A_v", "regional_material_multipliers"],
        "mesh_target_h_mm_nominal": 2.0,
        "mesh_inverse_h_mm_nominal": 3.0,
        "inverse_steps": int(args.coarse_steps),
        "target_steps": int(args.fine_steps),
        "observation_times": np.asarray(obs_times).tolist(),
        "sensor_counts": {d: len(pts) for d, pts in sensors.items()},
        "physics_scales_and_config_hashes": {k: digest(p) for k, p in paths.items()},
        "reference_source": "Phase 16 registered sensor coordinates and coarse-model observations",
        "noise_assumption": "iid Gaussian with common absolute SD = fraction * balanced coarse reference RMS; no clinical noise calibration",
        "interpretation": "cross-discretization synthetic inverse on anatomical domain; not validation using clinical dynamic measurements",
    }


def prepare(args):
    paths, master, fwd, theta, times, sensors, frozen_y0, materials = load_source(args)
    out = pathof(args.out_dir)
    if out.exists():
        raise FileExistsError(f"Output exists: {out}")
    out.mkdir(parents=True)
    metadata = manifest_base(args, paths, theta, times, sensors)
    save_json(out / "manifest.json", metadata)
    sigma = float(args.noise_fraction * np.linalg.norm(frozen_y0["balanced"])
                  / math.sqrt(len(frozen_y0["balanced"])))
    if not (0 < sigma < math.inf):
        raise ValueError("Nonpositive or nonfinite assumed noise scale")
    specs = (
        ("coarse_48", "mesh_coarse", args.coarse_steps),
        ("coarse_96", "mesh_coarse", args.fine_steps),
        ("fine_48", "mesh_fine", args.coarse_steps),
        ("fine_96", "mesh_fine", args.fine_steps),
    )
    arrays = {"observation_times": times, "theta_ref": theta}
    for design, points in sensors.items():
        arrays[f"sensor_{design}"] = points
    data = {}
    diagnostics = {}
    for label, mesh_key, steps in specs:
        print(f"[anatomy] {label}: {paths[mesh_key].name}; steps={steps}", flush=True)
        solver = AnatomicalEvaluator(master, fwd, paths[mesh_key], paths["geometry_report"],
                                     materials, steps, times, sensors)
        if mesh_key == "mesh_fine" and solver.problem.num_active_cells <= diagnostics["coarse_48"]["num_cells"]:
            raise ValueError("Fine anatomical mesh does not contain more cells than coarse mesh")
        observations, info = solver.evaluate(theta, with_diagnostics=True)
        diagnostics[label] = info
        data[label] = observations
        for d, y in observations.items():
            arrays[f"{label}_{d}"] = y
        # The coarse reference must reproduce Phase 16 exactly (up to solver rounding).
        if label == "coarse_48":
            for design in DESIGNS:
                err = float(np.linalg.norm(observations[design] - frozen_y0[design]))
                tol = 1e-8 * max(1.0, float(np.linalg.norm(frozen_y0[design])))
                if err > tol:
                    raise RuntimeError(f"Phase-16 frozen anatomy mismatch ({design}): {err:.3g} > {tol:.3g}")
            print("[OK] Coarse anatomical observations reproduce Phase 16", flush=True)
        print(f"      min(c)={info['min_c_node_all_times']:.6g}; "
              f"mass drift={info['relative_tracer_mass_drift']:.3e}", flush=True)
        # Keep completed stages safely on disk even if a later mesh fails.
        np.savez_compressed(out / "observations.npz", **arrays)
        save_json(out / "diagnostics_partial.json", diagnostics)

    compares = {}
    for design in DESIGNS:
        yy = data["fine_96"][design]
        compares[design] = {
            "temporal_3mm_48_vs_96": error_metrics(data["coarse_48"][design], data["coarse_96"][design], sigma),
            "spatial_48steps_3mm_vs_2mm": error_metrics(data["coarse_48"][design], data["fine_48"][design], sigma),
            "coarse_vs_fine_target": error_metrics(data["coarse_48"][design], yy, sigma),
            "temporal_2mm_48_vs_96": error_metrics(data["fine_48"][design], yy, sigma),
            "spatial_96steps_3mm_vs_2mm": error_metrics(data["coarse_96"][design], yy, sigma),
        }
    result = {
        "sigma_common_diagnostic": sigma,
        "noise_fraction": args.noise_fraction,
        "diagnostics_by_discretization": diagnostics,
        "observation_discretization_discrepancies": compares,
        "target": "fine_96", "inverse": "coarse_48",
        "frozen_phase16_match": True,
    }
    save_json(out / "summary.json", result)
    if (out / "diagnostics_partial.json").exists():
        (out / "diagnostics_partial.json").unlink()
    for d in DESIGNS:
        m = compares[d]["coarse_vs_fine_target"]
        print(f"[target-inverse/{d}] relative={100*m['relative_l2']:.4f}% "
              f"whitened L2={m['noise_whitened_l2']:.5g}  "
              f"noise RMS units={m['noise_whitened_rms']:.5g}", flush=True)
    print(f"[DONE] Anatomical cross-discretization reference: {out}", flush=True)


def finite_difference_jacobian(residual, u, h):
    """Stable O(h^2) normalized-coordinate differences; 2-sided interior."""
    u = np.asarray(u, dtype=float)
    y0 = residual(u)
    jac = np.empty((len(y0), len(u)), dtype=float)
    for k in range(len(u)):
        if h <= u[k] <= 1.0-h:
            p, m = u.copy(), u.copy()
            p[k] += h
            m[k] -= h
            jac[:, k] = (residual(p) - residual(m)) / (2*h)
        elif u[k] < h:
            step = min(h, (1.0-u[k]) / 2)
            p1, p2 = u.copy(), u.copy()
            p1[k] += step; p2[k] += 2*step
            jac[:, k] = (-3*y0 + 4*residual(p1) - residual(p2)) / (2*step)
        else:
            step = min(h, u[k] / 2)
            m1, m2 = u.copy(), u.copy()
            m1[k] -= step; m2[k] -= 2*step
            jac[:, k] = (3*y0 - 4*residual(m1) + residual(m2)) / (2*step)
    return jac


def invert(args):
    out = pathof(args.out_dir)
    if out.exists():
        raise FileExistsError(f"Output exists: {out}")
    prep = pathof(args.prepared_reference)
    for required in ("manifest.json", "summary.json", "observations.npz"):
        if not (prep / required).is_file():
            raise FileNotFoundError(f"Require Phase-18 prepared reference {prep / required}")
    source_manifest = json.loads((prep / "manifest.json").read_text(encoding="utf-8"))
    paths, master, fwd, theta, times, sensors, frozen_y0, materials = load_source(args)
    # Only re-use a frozen target with the same physical/geometry configuration.
    for name, p in paths.items():
        if source_manifest["physics_scales_and_config_hashes"].get(name) != digest(p):
            raise ValueError(f"Prepared anatomical target hash mismatch for {name}")
    if source_manifest["material_profile"] != args.material_profile:
        raise ValueError("Regional profile differs from prepared target")
    if not np.allclose([source_manifest["theta_reference"][k] for k in PARAMS], theta,
                       rtol=0, atol=1e-13):
        raise ValueError("Prepared target reference parameters differ")

    with np.load(prep / "observations.npz") as data:
        target_key = f"{args.target_discretization}_{args.design}"
        if target_key not in data:
            raise ValueError(f"Prepared observation file does not include {target_key}")
        y_clean = np.asarray(data[target_key], dtype=float)
        inverse_ref_key = f"{args.inverse_mesh}_48_{args.design}"
        if inverse_ref_key not in data:
            raise ValueError(f"Prepared observation file does not contain {inverse_ref_key}")
        y_inverse_saved = np.asarray(data[inverse_ref_key], dtype=float)
        s_saved = np.asarray(data[f"sensor_{args.design}"], dtype=float)
    if not np.array_equal(s_saved, sensors[args.design]):
        raise ValueError("Frozen sensor coordinates changed")
    if y_clean.shape != y_inverse_saved.shape or not np.all(np.isfinite(y_clean)):
        raise ValueError("Invalid selected target observations")
    summ = json.loads((prep / "summary.json").read_text(encoding="utf-8"))
    sigma = float(summ["sigma_common_diagnostic"])
    if not (0.0 <= args.inject_noise <= 1.0):
        raise ValueError("Noise fraction must lie between 0 and 1")
    if not math.isclose(args.noise_fraction, float(summ["noise_fraction"]), abs_tol=1e-14):
        raise ValueError("Noise normalizer must match the prepared cross-discretization study")
    rng = np.random.default_rng(args.seed)
    actual_sd = float(args.inject_noise / args.noise_fraction * sigma) if args.inject_noise else 0.0
    y_data = y_clean + (actual_sd * rng.normal(size=y_clean.shape) if actual_sd else 0.0)
    inverse_mesh_key = "mesh_fine" if args.inverse_mesh == "fine" else "mesh_coarse"
    solver = AnatomicalEvaluator(master, fwd, paths[inverse_mesh_key], paths["geometry_report"],
                                 materials, args.coarse_steps, times,
                                 {args.design: sensors[args.design]})
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    z_truth = np.asarray(transform.to_normalized(theta), dtype=float)
    y_true_coarse, _ = solver.evaluate(theta)
    ref_error = float(np.linalg.norm(y_true_coarse[args.design] - y_inverse_saved))
    if ref_error > 1e-8 * max(1.0, float(np.linalg.norm(y_inverse_saved))):
        raise RuntimeError(f"Anatomical inverse-mesh reference mismatch ({args.inverse_mesh}): {ref_error:g}")

    def res(z):
        z = np.asarray(z, dtype=float)
        th = np.asarray(transform.from_normalized(z), dtype=float)
        yy, _ = solver.evaluate(th)
        return (yy[args.design] - y_data) / sigma

    out.mkdir(parents=True)
    config = {
        **manifest_base(args, paths, theta, times, {args.design: sensors[args.design]}),
        "stage": "inverse", "prepared_reference_dir": str(prep),
        "target_discretization": args.target_discretization,
        "inverse_mesh": args.inverse_mesh,
        "inverse_mesh_sha256": digest(paths[inverse_mesh_key]),
        "mesh_inverse_h_mm_nominal": 2.0 if args.inverse_mesh == "fine" else 3.0,
        "mesh_target_h_mm_nominal": 2.0 if args.target_discretization.startswith("fine_") else 3.0,
        "target_steps": args.fine_steps if args.target_discretization.endswith("_96") else args.coarse_steps,
        "target_is_cross_spatial_mesh": bool(
            args.target_discretization.split("_")[0] != args.inverse_mesh
        ),
        "target_is_same_solver_discretization": bool(
            args.target_discretization == f"{args.inverse_mesh}_48"
        ),
        "target_source": f"observations.npz:{args.target_discretization}_{args.design}",
        "prepared_manifest_sha256": digest(prep / "manifest.json"),
        "prepared_summary_sha256": digest(prep / "summary.json"),
        "design": args.design, "starts": args.starts,
        "start_points_are_preassigned_and_not_the_truth": True,
        "max_nfev": args.max_nfev,
        "fd_step_normalized": args.fd_step,
        "noise_added_fraction": args.inject_noise,
        "noise_rng_seed": args.seed,
        "sigma_used_in_objective": sigma,
        "actual_noise_sd": actual_sd,
        "optimizer": "scipy.optimize.least_squares TRF, bounded unit cube, explicit FD Jacobian",
    }
    save_json(out / "manifest.json", config)
    cost_at_truth = .5 * float(np.dot(res(z_truth), res(z_truth)))
    print(f"[diagnostic] inverse_mesh={args.inverse_mesh}, "
          f"target={args.target_discretization}, "
          f"cost at known truth={cost_at_truth:.8g}", flush=True)
    starts = [np.array(s, dtype=float) for s in (
        (.55, .55, .55, .55), (.20, .80, .35, .75), (.80, .20, .75, .35),
    )][:args.starts]
    if args.starts > 3:
        raise ValueError("At most three fixed starts are available")
    rows = []
    best_predict = None
    for i, z0 in enumerate(starts, 1):
        if np.allclose(z0, z_truth, atol=1e-5):
            raise RuntimeError("Initial point must differ from synthetic truth")
        print(f"[inverse] design={args.design}, start {i}/{len(starts)} "
              f"at z={np.round(z0, 3)}", flush=True)
        initial_cost = float(.5 * np.dot(res(z0), res(z0)))
        before = solver.nsolves
        opt = least_squares(
            res, z0, bounds=(np.zeros(4), np.ones(4)), method="trf",
            jac=lambda z: finite_difference_jacobian(res, z, args.fd_step),
            max_nfev=args.max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8,
        )
        recovered = np.asarray(transform.from_normalized(opt.x), dtype=float)
        residual = np.asarray(res(opt.x), dtype=float)
        y_fit, _ = solver.evaluate(recovered)
        best_objective = .5 * float(residual @ residual)
        row = {
            "start_id": i,
            "target_discretization": args.target_discretization,
            "cost_at_truth_half_chi2": cost_at_truth,
            "parameters_at_bound": [PARAMS[j] for j in range(4)
                                    if opt.x[j] < 1e-6 or opt.x[j] > 1.0-1e-6],
            "optimizer_success": bool(opt.success),
            "optimizer_status": int(opt.status), "optimizer_message": str(opt.message),
            "optimizer_nfev": int(opt.nfev), "optimizer_njev": int(opt.njev) if opt.njev is not None else None,
            "optimizer_optimality": float(opt.optimality), "unique_forward_solves": solver.nsolves-before,
            "initial_cost_half_chi2": initial_cost,
            "final_cost_half_chi2": best_objective,
            "final_residual_noise_whitened_l2": float(np.linalg.norm(residual)),
            "prediction_relative_error_vs_selected_target": float(np.linalg.norm(y_fit[args.design]-y_clean)
                                                                  / max(np.linalg.norm(y_clean),1e-15)),
            **{f"{k}_estimated":float(recovered[j]) for j,k in enumerate(PARAMS)},
            **{f"{k}_truth":float(theta[j]) for j,k in enumerate(PARAMS)},
            **{f"{k}_absolute_error":float(abs(recovered[j]-theta[j])) for j,k in enumerate(PARAMS)},
        }
        rows.append(row)
        with (out / "inverse_results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        save_json(out / "inverse_results.json", rows)
        if best_predict is None or best_objective < best_predict[0]:
            best_predict = best_objective, y_fit[args.design].copy(), recovered.copy()
            np.savez_compressed(out / "best_inverse_observations.npz",
                                y_target_clean=y_clean, y_data_used=y_data,
                                y_prediction=best_predict[1], theta_estimated=best_predict[2],
                                theta_truth=theta, sensors=sensors[args.design], observation_times=times)
        print(f"    1/2 chi2: {initial_cost:.6g} -> {best_objective:.6g} "
              f"(truth={cost_at_truth:.6g}), "
              f"success={opt.success}, beta={recovered[1]:.5g}, "
              f"forward solves={solver.nsolves-before}", flush=True)
    print(f"[DONE] Anatomical inverse outputs: {out}", flush=True)


def make_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("prepare", "inverse"), default="prepare")
    p.add_argument("--physics-config", default="configs/scales_and_physics.yaml")
    p.add_argument("--anatomical-config", default="configs/brats_anatomical_forward.yaml")
    p.add_argument("--mesh-coarse", default="artifacts/02_anatomical_forward_clinical/meshes/BraTS20_Training_001_slice067_h3.0mm.msh")
    p.add_argument("--mesh-fine", default="artifacts/02_anatomical_forward_clinical/meshes/BraTS20_Training_001_slice067_h2.0mm.msh")
    p.add_argument("--geometry-report", default="artifacts/02_anatomical_forward_clinical/geometry_quality_report.json")
    p.add_argument("--materials-config", default="configs/brats_region_materials_phaseBC.yaml")
    p.add_argument("--material-profile", default="exploratory_contrast")
    p.add_argument("--phase16-reference", default="artifacts/16_anatomical_fd_refinement")
    p.add_argument("--coarse-steps", type=int, default=48)
    p.add_argument("--fine-steps", type=int, default=96)
    p.add_argument("--noise-fraction", type=float, default=.05,
                   help="Common diagnostic noise SD, relative to balanced coarse RMS (not injected during prepare)")
    p.add_argument("--prepared-reference", default="artifacts/18_anatomical_cross_mesh_target")
    p.add_argument("--target-discretization", choices=("coarse_48", "coarse_96", "fine_48", "fine_96"),
                   default="fine_96",
                   help="Frozen anatomical observations used as inverse target. "
                        "coarse_48 is a solver consistency control; coarse_96 isolates temporal bias; "
                        "fine_96 is the original cross-mesh experiment.")
    p.add_argument("--inverse-mesh", choices=("coarse", "fine"), default="coarse",
                   help="Anatomical mesh used by the inverse FEM solver: coarse=3mm or fine=2mm; both use 48 steps.")
    p.add_argument("--design", choices=DESIGNS, default="balanced")
    p.add_argument("--inject-noise", type=float, default=0.0,
                   help="Optional fraction of signal RMS, as added iid Gaussian noise during inverse")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--starts", type=int, default=1)
    p.add_argument("--max-nfev", type=int, default=15)
    p.add_argument("--fd-step", type=float, default=5e-4)
    p.add_argument("--out-dir", default="artifacts/18_anatomical_cross_mesh_target")
    return p


def main():
    args = make_parser().parse_args()
    if args.coarse_steps < 2 or args.fine_steps <= args.coarse_steps:
        raise ValueError("Invalid coarse/fine time grids")
    if not (0 < args.noise_fraction < 1):
        raise ValueError("0 < noise-fraction < 1 is required")
    if args.mode == "prepare":
        prepare(args)
    else:
        if args.max_nfev < 3 or not (1 <= args.starts <= 3):
            raise ValueError("Require max-nfev>=3 and 1<=starts<=3")
        if not (0 < args.fd_step < .05):
            raise ValueError("Require 0 < fd-step < 0.05")
        invert(args)


if __name__ == "__main__":
    main()
