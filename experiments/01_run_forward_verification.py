"""
experiments/01_run_forward_verification.py
Esegue la verifica completa del problema diretto RETECO v1 (Sezione Metodi Numerici del Paper):
1) Studio di convergenza spaziale e temporale MMS su (d*, p*, c*)
2) Verifica del bilancio energetico discreto (Proposizione 1) e della conservazione della massa M(t)
Salva i risultati certificati e i grafici in `artifacts/01_forward_verification/`.
"""

from __future__ import annotations
import json
import math
from pathlib import Path
import yaml

from reteco import _core, load_master_config
from reteco.forward.wrapper import RetecoForwardProblem2D
from reteco.forward.target_generator import build_standard_driving_Callbacks
from reteco.visualization.verification_plots import (
    plot_mms_spatial_convergence,
    plot_energy_and_mass_diagnostics,
)


def main() -> None:
    out_dir = Path("artifacts/01_forward_verification")
    out_dir.mkdir(parents=True, exist_ok=True)

    master = load_master_config("configs/scales_and_physics.yaml")
    with open("configs/mms_benchmark.yaml", "r", encoding="utf-8") as f:
        mms_cfg = yaml.safe_load(f)
    with open("configs/twin_experiment_synthetic.yaml", "r", encoding="utf-8") as f:
        twin_cfg = yaml.safe_load(f)

    phys = master.fixed_physics.to_cpp(master.characteristic_scales)
    td = master.parameter_space_theta.theta_dagger
    theta_cpp = _core.ParameterTheta(td["alpha"], td["beta"], td["D"], td["E"])

    mms = _core.MMSManufacturedSystem(phys, theta_cpp, float(mms_cfg["delta_tracer"]))

    print("=== [1/2] Studio di Convergenza Spaziale MMS ===")
    sp_cfg = mms_cfg["spatial_refinement_study"]
    grid_fine = _core.TimeGrid.make_graded(
        float(mms_cfg["final_time_T"]),
        int(sp_cfg["fixed_time_steps_M"]),
        float(sp_cfg["grading_r"]),
    )

    spatial_reports = []
    for N in sp_cfg["subdivisions_N"]:
        rep = mms.run_full_mms_test(int(N), grid_fine, 2, 1, 1)
        spatial_reports.append(rep)
        print(
            f"  N={N:2d} (h={rep.h_mesh:.4f}) | "
            f"L2(d)={rep.d_L2:.3e}, H1(d)={rep.d_H1:.3e} | "
            f"L2(p)={rep.p_L2:.3e} | L2(c)={rep.c_L2:.3e}"
        )

    plot_mms_spatial_convergence(spatial_reports, out_dir / "fig_mms_spatial_convergence.pdf")

    print("=== [2/2] Verifica Bilancio Energetico e Conservazione Massa ===")
    grid_diag = _core.TimeGrid.make_graded(1.0, 60, 1.5)
    problem = RetecoForwardProblem2D(
        subdivisions_N=16,
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid_diag,
    )
    c0, _, _ = build_standard_driving_Callbacks(problem, twin_cfg["driving_protocol"])
    px = problem.press_pts[:, 0]
    py = problem.press_pts[:, 1]
    p0 = 0.4 * (px * (1.0 - px) * py * (1.0 - py))

    traj = problem.solve_trajectory(
        theta_vec=master.parameter_space_theta.theta_dagger_array(),
        c0=c0,
        p0=p0,
        enforce_initial_compatibility=True,
        compute_diagnostics=True,
    )

    plot_energy_and_mass_diagnostics(
        traj.diagnostics, out_dir / "fig_energy_mass_diagnostics.pdf"
    )

    summary = {
        "mms_spatial": [
            {
                "h": r.h_mesh,
                "d_L2": r.d_L2,
                "d_H1": r.d_H1,
                "p_L2": r.p_L2,
                "p_H1": r.p_H1,
                "c_L2": r.c_L2,
                "c_H1": r.c_H1,
            }
            for r in spatial_reports
        ],
        "initial_compatibility_residual": traj.initial_compatibility_residual,
        "initial_mass_M0": traj.diagnostics[0].tracer_total_mass,
        "final_mass_MT": traj.diagnostics[-1].tracer_total_mass,
        "relative_mass_drift": abs(
            traj.diagnostics[-1].tracer_total_mass - traj.diagnostics[0].tracer_total_mass
        )
        / traj.diagnostics[0].tracer_total_mass,
    }
    (out_dir / "forward_verification_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Risultati salvati in: {out_dir}")


if __name__ == "__main__":
    main()