#!/usr/bin/env python3
"""
scripts/run_anatomical_forward_simulation.py
Esegue una simulazione diretta completa (Fase 2) sulla mesh quadrilatera anatomica
prodotta e verificata nella Fase 1, salvando i diagnostici numerici e le figure PDF.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT_DIR / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from reteco import _core, load_master_config
from reteco.forward.target_generator import build_anatomical_driving_Callbacks
from reteco.forward.wrapper import RetecoForwardProblem2D
from reteco.visualization.anatomical_forward_plots import (
    plot_anatomical_energy_and_mass_diagnostics,
)
from reteco.visualization.verification_plots import (
    plot_anatomical_quad_trajectory_snapshots,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="RETECO Phase 2 — Anatomical Quad-Mesh Forward Solver")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/brats_anatomical_forward.yaml",
        help="Percorso alla configurazione YAML anatomica.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    out_dir = Path(cfg["output_dir"])
    geom_report_path = out_dir / "geometry_quality_report.json"
    if not geom_report_path.exists():
        raise FileNotFoundError(
            f"Report geometrico di Fase 1 non trovato in {geom_report_path}. "
            "Esegui prima `scripts/run_brats_anatomical_pipeline.py`."
        )

    geom_report = json.loads(geom_report_path.read_text(encoding="utf-8"))
    if not geom_report.get("all_mandatory_checks_passed", False):
        raise RuntimeError("La mesh anatomica di Fase 1 non ha superato tutti i controlli obbligatori.")

    msh_path = Path(geom_report["exported_files"]["gmsh_msh"])
    master = load_master_config(cfg["physics_config_path"])
    fwd_cfg = cfg["forward_simulation"]

    grid = _core.TimeGrid.make_graded(
        float(fwd_cfg["final_time_T"]),
        int(fwd_cfg["time_steps_M"]),
        float(fwd_cfg["grading_r"]),
    )

    problem = RetecoForwardProblem2D(
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid,
        fe_deg_disp=int(fwd_cfg.get("fe_deg_disp", 2)),
        fe_deg_press=int(fwd_cfg.get("fe_deg_press", 1)),
        fe_deg_tracer=int(fwd_cfg.get("fe_deg_tracer", 1)),
        use_supg=bool(fwd_cfg.get("use_supg", False)),
        mesh_file_path=msh_path,
        dirichlet_disp_boundary_ids=fwd_cfg.get("dirichlet_disp_boundary_ids", [1]),
    )

    c0, F_mech_func, S_fluid_func, forcing_meta = build_anatomical_driving_Callbacks(
        problem=problem,
        driving_cfg=fwd_cfg["driving_protocol"],
        geometry_report_path=geom_report_path,
    )

    theta_vec = master.parameter_space_theta.theta_dagger_array()
    traj = problem.solve_trajectory(
        theta_vec=theta_vec,
        c0=c0,
        F_mech_func=F_mech_func,
        S_fluid_func=S_fluid_func,
        S_tracer_func=None,
        enforce_initial_compatibility=True,
        compute_diagnostics=True,
    )

    fig_snapshots = out_dir / "fig05_anatomical_forward_quad_snapshots.pdf"
    fig_diag = out_dir / "fig06_anatomical_energy_mass_diagnostics.pdf"
    plot_anatomical_quad_trajectory_snapshots(problem, traj, fig_snapshots)
    plot_anatomical_energy_and_mass_diagnostics(traj.diagnostics, fig_diag)

    masses = np.array([d.tracer_total_mass for d in traj.diagnostics], dtype=np.float64)
    rel_mass_drift = float(np.max(np.abs(masses - masses[0])) / masses[0])

    summary = {
        "subject_id": cfg["subject_id"],
        "domain_representation_type": geom_report["domain_representation_type"],
        "mesh_file_imported": str(msh_path),
        "dof_counts": {
            "num_active_quad_cells": problem.num_active_cells,
            "num_vertices": int(problem.fem.num_vertices()),
            "num_disp_dofs_Q2x2": problem.num_disp_dofs,
            "num_press_dofs_Q1": problem.num_press_dofs,
            "num_tracer_dofs_Q1": problem.num_tracer_dofs,
            "dof_and_mesh_consistency_verified": bool(problem.fem.verify_dof_and_mesh_consistency()),
        },
        "boundary_conditions": problem.boundary_conditions_metadata(),
        "anatomy_guided_synthetic_forcing": forcing_meta,
        "trajectory_metrics": {
            "theta_evaluated": {
                "alpha": float(theta_vec[0]),
                "beta": float(theta_vec[1]),
                "D": float(theta_vec[2]),
                "E": float(theta_vec[3]),
            },
            "initial_compatibility_residual": traj.initial_compatibility_residual,
            "max_darcy_velocity_adim": traj.max_darcy_velocity,
            "initial_tracer_mass_adim": float(masses[0]),
            "final_tracer_mass_adim": float(masses[-1]),
            "max_relative_tracer_mass_drift": rel_mass_drift,
            "final_total_energy": float(traj.diagnostics[-1].total_energy),
            "final_cumulative_visco_dissipation": float(traj.diagnostics[-1].cumulative_visco_dissip),
            "final_cumulative_darcy_dissipation": float(traj.diagnostics[-1].cumulative_darcy_dissip),
            "final_energy_balance_residual": float(traj.diagnostics[-1].energy_balance_residual),
            "max_step_discrete_identity_error": float(
                max(d.step_discrete_identity_error for d in traj.diagnostics[1:])
            ),
        },
        "exported_figures": {
            "quad_field_snapshots_pdf": str(fig_snapshots),
            "energy_mass_diagnostics_pdf": str(fig_diag),
        },
    }

    summary_path = out_dir / "anatomical_forward_simulation_report.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())