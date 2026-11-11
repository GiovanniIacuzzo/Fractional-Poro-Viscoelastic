#!/usr/bin/env python3
"""Nonlinear constrained beta profiles on the BraTS-region fractional Biot model.

Uses the *frozen* synthetic reference observations and sensor positions from
experiment 16. For every fixed beta value, minimizes the whitened concentration
misfit over (alpha,D,E), optionally also (K_h,A_v).

Profile values are local numerical upper bounds on the globally minimized
profile; they are NOT statistical confidence intervals or clinical inference.
No solver, old scripts, or existing outputs are modified.
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
from reteco.config_schema import FixedPhysicsModel
from reteco.anatomy.regional_materials import load_regional_materials
from reteco.forward.wrapper import RetecoForwardProblem2D
from reteco.forward.target_generator import build_anatomical_driving_Callbacks
from reteco.observation.q1_exact import ExactQ1TracerObservationOperator

NAMES = ("alpha", "beta", "D", "E", "K_h", "A_v")
FREE = {"four": (0, 2, 3), "six": (0, 2, 3, 4, 5)}


def pathof(p):
    p = Path(p)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def normalized(u, baseline, bnds):
    """Physical parameters -> unit-cube coordinates in Phase 16 convention."""
    z = np.empty(6, dtype=float)
    z[0] = (u[0]-bnds.alpha[0])/(bnds.alpha[1]-bnds.alpha[0])
    z[1] = (u[1]-bnds.beta[0])/(bnds.beta[1]-bnds.beta[0])
    for i, limits in [(2,bnds.D), (3,bnds.E)]:
        z[i] = math.log(u[i]/limits[0])/math.log(limits[1]/limits[0])
    for i in (4,5):
        z[i] = math.log(u[i]/(baseline[i]/2))/math.log(4)
    return z


def physical(z, baseline, bnds):
    """Unit-cube coordinates -> physical nondimensional parameters."""
    x = np.empty(6, dtype=float)
    x[0] = bnds.alpha[0] + z[0]*(bnds.alpha[1]-bnds.alpha[0])
    x[1] = bnds.beta[0] + z[1]*(bnds.beta[1]-bnds.beta[0])
    for i, limits in [(2,bnds.D), (3,bnds.E)]:
        x[i] = limits[0]*(limits[1]/limits[0])**z[i]
    for i in (4,5):
        x[i] = (baseline[i]/2)*4.**z[i]
    return x


def check_manifest(meta, paths):
    actual = {p.name: digest(p) for p in paths}
    for name, value in actual.items():
        if name not in meta["hashes"] or meta["hashes"][name] != value:
            raise ValueError(f"Reference/setup SHA256 mismatch for {name}; do not compare different experiments")
    if meta.get("time_steps") is None or meta.get("T_final_adim") is None:
        raise ValueError("Reference metadata lacks temporal discretization")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-dir", default="artifacts/16_anatomical_fd_refinement")
    ap.add_argument("--physics-config", default="configs/scales_and_physics.yaml")
    ap.add_argument("--anatomical-config", default="configs/brats_anatomical_forward.yaml")
    ap.add_argument("--mesh", default="artifacts/02_anatomical_forward_clinical/meshes/BraTS20_Training_001_slice067_h3.0mm.msh")
    ap.add_argument("--geometry-report", default="artifacts/02_anatomical_forward_clinical/geometry_quality_report.json")
    ap.add_argument("--materials-config", default="configs/brats_region_materials_phaseBC.yaml")
    ap.add_argument("--material-profile", default="exploratory_contrast")
    ap.add_argument("--designs", nargs="+", choices=("balanced","edema"), default=["balanced"])
    ap.add_argument("--models", nargs="+", choices=("four","six"), default=["six"])
    ap.add_argument("--beta-values", nargs="+", type=float, default=[.35,.45,.55])
    ap.add_argument("--max-nfev", type=int, default=12, help="Max optimizer base evaluations per local run; numerical Jacobian evaluations are additional")
    ap.add_argument("--noise-fraction", type=float, default=.05)
    ap.add_argument("--noise-policy", choices=("per_design","common"), default="common")
    ap.add_argument("--out-dir", default="artifacts/17_anatomical_beta_profiles_pilot")
    args = ap.parse_args()
    if args.max_nfev < 3 or not (0 < args.noise_fraction < 1):
        ap.error("max-nfev >= 3 and noise-fraction in (0,1) are required")
    if len(set(args.beta_values)) != len(args.beta_values):
        ap.error("beta-values must be distinct")

    pref = pathof(args.reference_dir)
    source_files = [pathof(x) for x in (
        args.physics_config, args.anatomical_config, args.mesh, args.geometry_report, args.materials_config)]
    if any(not f.is_file() for f in source_files):
        raise FileNotFoundError("A physics/config/mesh/geometry file does not exist")
    meta_file, npz_file = pref/"summary.json", pref/"jacobians_and_observations.npz"
    if not meta_file.is_file() or not npz_file.is_file():
        raise FileNotFoundError("Require experiment 16 summary.json and jacobians_and_observations.npz")
    source = json.loads(meta_file.read_text(encoding="utf-8"))
    meta = source["metadata"]
    if args.material_profile not in source["results"]:
        raise ValueError("Material profile absent from reference results")
    check_manifest(meta, source_files)
    for k, v in meta["reference_parameters"].items():
        if k not in NAMES or not np.isfinite(float(v)):
            raise ValueError("Malformed reference parameters")
    theta_ref = np.array([meta["reference_parameters"][k] for k in NAMES],dtype=float)
    if any(not (0 < v < 1) for v in (theta_ref[0],theta_ref[1])):
        raise ValueError("Reference fractional orders outside (0,1)")
    master = load_master_config(source_files[0])
    bnds = master.parameter_space_theta.bounds
    z_ref = normalized(theta_ref, theta_ref, bnds)
    if np.max(np.abs(physical(z_ref,theta_ref,bnds)-theta_ref)) > 1e-10:
        raise RuntimeError("Incorrect parameter transformation")
    if np.any(z_ref < 0) or np.any(z_ref > 1):
        raise ValueError("Reference parameters outside allowed box")
    if any(not (bnds.beta[0] < b < bnds.beta[1]) for b in args.beta_values):
        raise ValueError("Profile beta outside configured strict bounds")

    geom = json.loads(source_files[3].read_text(encoding="utf-8"))
    if not geom.get("all_mandatory_checks_passed",False):
        raise ValueError("Anatomical geometry report does not pass checks")
    if geom.get("exported_files",{}).get("gmsh_msh"):
        if pathof(geom["exported_files"]["gmsh_msh"]) != source_files[2]:
            raise ValueError("Mesh inconsistent with geometry report")
    fwd = yaml.safe_load(source_files[1].read_text(encoding="utf-8"))["forward_simulation"]
    grid = _core.TimeGrid.make_graded(
        float(meta["T_final_adim"]),int(meta["time_steps"]),float(fwd["grading_r"]))
    obs_times = np.asarray(meta["observation_times"], dtype=float)
    materials = load_regional_materials(source_files[4],args.material_profile,float(master.fixed_physics.phi_0))
    ref_data = np.load(npz_file)
    sensors = {}
    y_ref = {}
    j_ref = {}
    for design in args.designs:
        stem=f"{args.material_profile}_{design}"
        sensors[design]=np.asarray(ref_data[f"{stem}_sensors"],dtype=float)
        y_ref[design]=np.asarray(ref_data[f"{stem}_y0"],dtype=float)
        j_ref[design]=np.asarray(ref_data[f"{stem}_J6"],dtype=float)
        if j_ref[design].shape != (len(y_ref[design]),6):
            raise ValueError("Reference Jacobian shape mismatch")
        if not (len(y_ref[design])==len(sensors[design])*len(obs_times)):
            raise ValueError("Observation vector does not match sensor and time grid")
    ref_data.close()
    sigma_ref = float(args.noise_fraction*np.linalg.norm(y_ref["balanced"])/math.sqrt(len(y_ref["balanced"]))) if "balanced" in y_ref else None
    if args.noise_policy == "common" and sigma_ref is None:
        # If only edema is requested, load the frozen balanced reference for the common noise scale.
        with np.load(npz_file) as f:
            y_bal=np.asarray(f[f"{args.material_profile}_balanced_y0"],dtype=float)
        sigma_ref=float(args.noise_fraction*np.linalg.norm(y_bal)/math.sqrt(len(y_bal)))
    sigmas = {name: (sigma_ref if args.noise_policy == "common" else
             float(args.noise_fraction*np.linalg.norm(y_ref[name])/math.sqrt(len(y_ref[name]))))
              for name in args.designs}
    if min(sigmas.values())<=0: raise ValueError("Nonpositive assumed measurement noise")

    output = pathof(args.out_dir)
    if output.exists(): raise FileExistsError(f"Refusing overwrite of existing output {output}")
    output.mkdir(parents=True)
    manifest = {
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "purpose":"Constrained nonlinear locally optimized beta profiles; twin data, zero added noise",
        "reference_directory":str(pref),"reference_manifest_sha256":digest(meta_file),
        "profile":args.material_profile,"designs":args.designs,"models":args.models,
        "beta_values":args.beta_values,"max_nfev":args.max_nfev,
        "noise_policy":args.noise_policy,"noise_fraction":args.noise_fraction,
        "assumed_sigma":sigmas,"theta_reference":dict(zip(NAMES,theta_ref.tolist())),
        "method":"scipy.optimize.least_squares(TRF), bounded unit-cube coordinates, local numerical optimization",
        "interpretation":"Each reported local minimum is an upper bound for the global profile. Not a CI, not fitted to clinical observations.",
        "configuration_hashes":{f.name:digest(f) for f in source_files},
    }
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")

    cache = {}
    fev = 0
    def forward(z):
        nonlocal fev
        z=np.asarray(z,dtype=float)
        if not np.isfinite(z).all() or np.min(z)<-1e-11 or np.max(z)>1+1e-11:
            raise ValueError("Optimizer left parameter bounds")
        cache_key=tuple(np.round(z,13))
        if cache_key in cache:
            return cache[cache_key]
        theta=physical(z,theta_ref,bnds)
        phys=master.fixed_physics.model_dump()
        phys["K_h"]=float(theta[4])
        phys["eta_v"]=float(theta[5])/(float(phys["tau_v"])**float(theta[1]))
        phys=FixedPhysicsModel.model_validate(phys)
        prob=RetecoForwardProblem2D(
            physics_model=phys,scales=master.characteristic_scales,time_grid=grid,
            mesh_file_path=source_files[2],
            dirichlet_disp_boundary_ids=fwd.get("dirichlet_disp_boundary_ids",[1]),
            fe_deg_disp=int(fwd.get("fe_deg_disp",2)),
            fe_deg_press=int(fwd.get("fe_deg_press",1)),
            fe_deg_tracer=int(fwd.get("fe_deg_tracer",1)),
            use_supg=bool(fwd.get("use_supg",False)),region_materials=materials)
        c0, force, fluid, _ = build_anatomical_driving_Callbacks(
            prob, fwd["driving_protocol"], geometry_report_path=source_files[3])
        trajectory=prob.solve_trajectory(theta[:4],c0=c0,F_mech_func=force,
                                      S_fluid_func=fluid,enforce_initial_compatibility=True)
        if not np.isfinite(trajectory.c_hist).all():
            raise RuntimeError("Nonfinite forward trajectory")
        result={}
        for design, points in sensors.items():
            observer=ExactQ1TracerObservationOperator(prob,points,obs_times)
            result[design]=np.asarray(observer.apply(prob,trajectory),dtype=float).ravel()
        fev+=1
        cache[cache_key]=result
        return result

    observed_baseline=forward(z_ref)
    for design in args.designs:
        error=np.linalg.norm(observed_baseline[design]-y_ref[design])
        limit=1e-8*max(1.,np.linalg.norm(y_ref[design]))
        if error>limit:
            raise RuntimeError(f"Frozen reference mismatch for {design}: {error:g} > {limit:g}")
    print("[OK] Frozen forward reference and observations match Phase 16",flush=True)

    rows=[]
    columns=["model","design","beta","objective_half_chi2","normalized_residual_l2",
             "optimizer_success","optimizer_message","optimality","optimizer_nfev",
             "new_forward_solves","alpha","D","E","K_h","A_v", "is_reference_beta"]

    for design in args.designs:
      for model in args.models:
        free=np.asarray(FREE[model],dtype=int)
        ordered=sorted(args.beta_values,key=lambda b:abs(b-theta_ref[1]))
        for beta in ordered:
            z_fix=z_ref.copy()
            z_fix[1]=(beta-bnds.beta[0])/(bnds.beta[1]-bnds.beta[0])
            delta_beta=z_fix[1]-z_ref[1]
            # Local Jacobian-based compensation is only an initial guess;
            # actual objective is recomputed by the full nonlinear model.
            rhs=-j_ref[design][:,1]*delta_beta
            best_delta=np.linalg.lstsq(j_ref[design][:,free],rhs,rcond=None)[0]
            z_start=z_fix.copy()
            z_start[free]=np.clip(z_ref[free]+best_delta,1e-7,1-1e-7)

            count_before=fev
            if abs(beta-theta_ref[1])<1e-12:
                sol=z_ref.copy()
                success=True
                message="Exactly the frozen twin reference"
                nfev=0
                optimality=0.
            else:
                def resid(u):
                    zi=z_fix.copy();zi[free]=u
                    return (forward(zi)[design]-y_ref[design])/sigmas[design]
                # This is a local solver and can converge to distinct minima.
                result=least_squares(resid,z_start[free],bounds=(np.zeros(len(free)),np.ones(len(free))),
                                     method="trf",jac="2-point",max_nfev=args.max_nfev,
                                     ftol=1e-7,xtol=1e-7,gtol=1e-7)
                sol=z_fix.copy();sol[free]=result.x
                success=bool(result.success)
                message=str(result.message)
                nfev=int(result.nfev)
                optimality=float(result.optimality)
            y_fit=forward(sol)[design]
            r=(y_fit-y_ref[design])/sigmas[design]
            theta=physical(sol,theta_ref,bnds)
            row={"model":model,"design":design,"beta":float(beta),
                "objective_half_chi2":float(.5*np.dot(r,r)),
                "normalized_residual_l2":float(np.linalg.norm(r)),
                "optimizer_success":success,"optimizer_message":message,
                "optimality":optimality,"optimizer_nfev":nfev,
                "new_forward_solves":fev-count_before,
                "alpha":float(theta[0]),"D":float(theta[2]),"E":float(theta[3]),
                "K_h":float(theta[4]),"A_v":float(theta[5]),
                "is_reference_beta":bool(abs(beta-theta_ref[1])<1e-12)}
            rows.append(row)
            with (output/"profiles.csv").open("w",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=columns)
                writer.writeheader();writer.writerows(rows)
            (output/"profiles.json").write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding="utf-8")
            print(f"[{model:4s}/{design:8s}] beta={beta:.3f}  objective={row['objective_half_chi2']:.7g}  success={success}  new forwards={row['new_forward_solves']}",flush=True)
    print(f"Finished {len(rows)} profile points; unique forward simulations: {fev}; output: {output}",flush=True)


if __name__=="__main__":
    main()
