#!/usr/bin/env python3
"""
experiments/02_run_anatomical_forward.py
Esperimento completo del forward model poro-viscoelastico frazionario RETECO v1
sulla geometria anatomica 2D ricavata da BraTS 2020.

Non altera il ruolo di `experiments/01_run_forward_verification.py` (che resta
il benchmark di verifica MMS sul quadrato unitario).
Salva tutti gli artefatti (JSON, NPZ, mesh e 7 figure PDF) nella directory configurata.
"""

from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from matplotlib.path import Path as MplPath
import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT_DIR / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from reteco import _core, load_master_config
from reteco.anatomy.brats_geometry import (
    AnatomicalCoordinateTransform2D,
    MeshGenerationConfig,
    build_anatomical_masks_2d,
    compute_geometry_quality_report,
    export_dealii_quad_mesh,
    generate_anatomical_quad_mesh_2d,
    load_brats_pipeline_config,
    load_brats_subject,
    load_l0_from_physics_config,
    select_anatomical_slice,
)
from reteco.forward.target_generator import build_anatomical_driving_Callbacks
from reteco.forward.wrapper import RetecoForwardProblem2D
from reteco.visualization.anatomical_forward_plots import (
    plot_anatomical_energy_and_mass_diagnostics,
    plot_fig01_multimodal_anatomy_and_source,
    plot_fig02_mask_and_quad_mesh_quality,
    plot_fig03_multiphysics_fields_on_quad_mesh,
    plot_fig04_temporal_evolution_comparable_scales,
    plot_fig05_mesh_resolution_sensitivity,
    plot_fig06_fractional_parameters_sensitivity,
)


def verify_source_inside_computational_domain(
    problem: RetecoForwardProblem2D,
    forcing_meta: Dict[str, Any],
    geom_report_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Requisito 1: Controlla che la posizione della sorgente sia interna al dominio
    di calcolo (celle quadrilatere della mesh) e registra la distanza dal centroide
    della lesione sia in millimetri sia in unità adimensionali L0.
    """
    les_mm = np.array(
        geom_report_dict["lesion_centroids"]["center_physical_mm"][:2], dtype=np.float64
    )
    les_adim = np.array(
        geom_report_dict["lesion_centroids"]["center_mesh_adim"], dtype=np.float64
    )

    src_mm = np.array(
        forcing_meta["fluid_injection_source"]["center_physical_mm"], dtype=np.float64
    )
    src_adim = np.array(
        forcing_meta["fluid_injection_source"]["center_mesh_adim"], dtype=np.float64
    )
    sigma_mm = float(forcing_meta["fluid_injection_source"]["sigma_physical_mm"])
    sigma_adim = float(forcing_meta["fluid_injection_source"]["sigma_mesh_adim"])

    dist_mm = float(np.linalg.norm(src_mm - les_mm))
    dist_adim = float(np.linalg.norm(src_adim - les_adim))

    verts = problem.mesh_vertices
    quads = problem.cell_quad_connectivity
    containing_cell_idx = -1
    for q_idx in range(quads.shape[0]):
        poly = verts[quads[q_idx]]
        if MplPath(poly).contains_point(src_adim, radius=1.0e-5):
            containing_cell_idx = int(q_idx)
            break

    if containing_cell_idx < 0:
        raise RuntimeError(
            f"Controllo Requisito 1 FALLITO: la posizione della sorgente {src_adim.tolist()} "
            "non cade all'interno del dominio di calcolo della mesh quadrilatera."
        )

    return {
        "source_inside_computational_domain": True,
        "containing_quad_cell_index": containing_cell_idx,
        "lesion_centroid_physical_mm": les_mm.tolist(),
        "lesion_centroid_mesh_adim": les_adim.tolist(),
        "source_center_physical_mm": src_mm.tolist(),
        "source_center_mesh_adim": src_adim.tolist(),
        "distance_source_to_lesion_centroid_mm": dist_mm,
        "distance_source_to_lesion_centroid_adim": dist_adim,
        "fluid_source_sigma_mm": sigma_mm,
        "fluid_source_sigma_adim": sigma_adim,
    }


def _extract_horizontal_profile_through_lesion(
    problem: RetecoForwardProblem2D,
    c_nodal: np.ndarray,
    lesion_center_adim: Tuple[float, float],
    band_half_width_adim: float = 0.22,
) -> Tuple[np.ndarray, np.ndarray]:
    pts = problem.tracer_pts
    mask_band = np.abs(pts[:, 1] - lesion_center_adim[1]) <= band_half_width_adim
    if np.sum(mask_band) < 4:
        mask_band = np.abs(pts[:, 1] - lesion_center_adim[1]) <= 0.45
    x_sel = pts[mask_band, 0]
    c_sel = c_nodal[mask_band]
    order = np.argsort(x_sel)
    return x_sel[order], c_sel[order]


def _compute_trajectory_series_metrics(
    problem: RetecoForwardProblem2D,
    traj: Any,
    lesion_center_adim: Tuple[float, float],
) -> Dict[str, Any]:
    """
    Calcola le serie temporali genuine ad ogni istante t_n:
    - ||d_h(t_n)||_{L^2(Omega)} = sqrt(d^T M_d d) tramite la matrice di massa vettoriale [Q_2]^2;
    - |d_h(t_n)|_{K_0} = sqrt(d^T K_0 d) registrata separatamente come seminorma di rigidezza;
    - ||p_h(t_n)||_{L^2(Omega)} = sqrt(p^T M_p p) tramite la matrice di massa Q_1;
    - ||c_h(t_n)||_{L^2(Omega)} = sqrt(c^T M_c c) tramite la matrice di massa Q_1.
    """
    n_t = len(traj.times)
    c_l2 = np.zeros(n_t, dtype=np.float64)
    p_l2 = np.zeros(n_t, dtype=np.float64)
    d_l2 = np.zeros(n_t, dtype=np.float64)
    d_stiff_seminorm = np.zeros(n_t, dtype=np.float64)

    for n in range(n_t):
        cn = traj.c_hist[n]
        pn = traj.p_hist[n]
        dn = traj.d_hist[n]
        c_l2[n] = problem.compute_tracer_L2_norm(cn)
        p_l2[n] = problem.compute_press_L2_norm(pn)
        d_l2[n] = problem.compute_disp_L2_norm(dn)
        d_stiff_seminorm[n] = problem.compute_disp_stiffness_seminorm(dn)

    dist2_c = np.sum((problem.tracer_pts - np.array(lesion_center_adim)) ** 2, axis=1)
    les_node_c = int(np.argmin(dist2_c))
    c_les_series = traj.c_hist[:, les_node_c]
    p_les_series = traj.p_hist[:, les_node_c]

    energy_series = np.array([d.total_energy for d in traj.diagnostics], dtype=np.float64)
    visco_series = np.array([d.cumulative_visco_dissip for d in traj.diagnostics], dtype=np.float64)
    darcy_series = np.array([d.cumulative_darcy_dissip for d in traj.diagnostics], dtype=np.float64)
    work_series = np.array([d.cumulative_external_work for d in traj.diagnostics], dtype=np.float64)
    res_series = np.array([d.energy_balance_residual for d in traj.diagnostics], dtype=np.float64)
    mass_series = np.array([d.tracer_total_mass for d in traj.diagnostics], dtype=np.float64)

    prof_x, prof_c = _extract_horizontal_profile_through_lesion(
        problem, traj.c_hist[-1], lesion_center_adim
    )

    return {
        "times": traj.times.tolist(),
        "c_l2_norm_series": c_l2.tolist(),
        "p_l2_norm_series": p_l2.tolist(),
        "press_l2_norm_series": p_l2.tolist(),
        "disp_l2_norm_series": d_l2.tolist(),
        "disp_stiffness_seminorm_series": d_stiff_seminorm.tolist(),
        "c_lesion_center_series": c_les_series.tolist(),
        "p_lesion_center_series": p_les_series.tolist(),
        "energy_series": energy_series.tolist(),
        "visco_dissipation_series": visco_series.tolist(),
        "darcy_dissipation_series": darcy_series.tolist(),
        "external_work_series": work_series.tolist(),
        "energy_balance_residual_series": res_series.tolist(),
        "tracer_mass_series": mass_series.tolist(),
        "profile_x_adim": prof_x.tolist(),
        "profile_c_final": prof_c.tolist(),
    }


def run_anatomical_forward_experiment(
    config_path: str | Path = "configs/brats_anatomical_forward.yaml",
) -> Dict[str, Any]:
    cfg_path = Path(config_path)
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw_cfg = yaml.safe_load(fh)

    pipe_cfg = load_brats_pipeline_config(cfg_path)
    out_dir = Path(raw_cfg.get("experiment_output_dir", "artifacts/02_anatomical_forward_clinical"))
    out_dir.mkdir(parents=True, exist_ok=True)
    meshes_dir = out_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    if not pipe_cfg.subject_dir.exists():
        raise FileNotFoundError(
            f"Directory soggetto BraTS non trovata: {pipe_cfg.subject_dir.resolve()}. "
            "Nessun dato sintetico può essere generato automaticamente."
        )

    l0_m, l0_mm = load_l0_from_physics_config(pipe_cfg.physics_config_path)
    master = load_master_config(pipe_cfg.physics_config_path)

    volumes = load_brats_subject(
        subject_dir=pipe_cfg.subject_dir,
        subject_id=pipe_cfg.subject_id,
        modality_filenames=pipe_cfg.modality_filenames,
        wm_mask_path=pipe_cfg.wm_mask_path,
        allow_grid_realignment=pipe_cfg.allow_grid_realignment,
        remap_label_3_to_4=pipe_cfg.remap_label_3_to_4,
        provenance_cfg=pipe_cfg.provenance,
    )
    slice_sel = select_anatomical_slice(
        volumes=volumes,
        mode=pipe_cfg.slice_selection_mode,
        manual_slice_index=pipe_cfg.manual_slice_index,
    )
    masks = build_anatomical_masks_2d(
        slice_sel=slice_sel,
        spacing_xy_mm=(volumes.spacing_mm[0], volumes.spacing_mm[1]),
        cfg=pipe_cfg.support_mask,
    )
    transform = AnatomicalCoordinateTransform2D.from_ras_affine_and_mask(
        ras_affine=volumes.ras_affine,
        slice_index=slice_sel.slice_index,
        support_mask=masks.brain_support_mask,
        L0_m=l0_m,
    )

    ref_mesh = generate_anatomical_quad_mesh_2d(
        masks=masks,
        slice_sel=slice_sel,
        transform=transform,
        mesh_cfg=pipe_cfg.mesh,
    )
    ref_basename = (
        f"{pipe_cfg.subject_id}_slice{slice_sel.slice_index:03d}"
        f"_h{pipe_cfg.mesh.target_element_size_mm:.1f}mm"
    )
    ref_exported = export_dealii_quad_mesh(ref_mesh, transform, meshes_dir, basename=ref_basename)
    geom_report = compute_geometry_quality_report(
        volumes=volumes,
        slice_sel=slice_sel,
        masks=masks,
        transform=transform,
        mesh=ref_mesh,
        exported_files=ref_exported,
        thresholds=pipe_cfg.quality_thresholds,
    )
    if not geom_report.all_mandatory_checks_passed:
        raise RuntimeError(
            f"Controlli geometrici obbligatori falliti sulla mesh di riferimento: {geom_report.failed_checks}"
        )

    geom_report_path = out_dir / "geometry_quality_report.json"
    geom_report_dict = asdict(geom_report)
    geom_report_path.write_text(json.dumps(geom_report_dict, indent=2, ensure_ascii=False), encoding="utf-8")

    fwd_cfg = raw_cfg["forward_simulation"]
    grid = _core.TimeGrid.make_graded(
        float(fwd_cfg["final_time_T"]),
        int(fwd_cfg["time_steps_M"]),
        float(fwd_cfg["grading_r"]),
    )
    problem_ref = RetecoForwardProblem2D(
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid,
        fe_deg_disp=int(fwd_cfg.get("fe_deg_disp", 2)),
        fe_deg_press=int(fwd_cfg.get("fe_deg_press", 1)),
        fe_deg_tracer=int(fwd_cfg.get("fe_deg_tracer", 1)),
        use_supg=bool(fwd_cfg.get("use_supg", False)),
        mesh_file_path=ref_exported["gmsh_msh"],
        dirichlet_disp_boundary_ids=fwd_cfg.get("dirichlet_disp_boundary_ids", [1]),
    )

    c0_ref, F_mech_ref, S_fluid_ref, forcing_meta = build_anatomical_driving_Callbacks(
        problem=problem_ref,
        driving_cfg=fwd_cfg["driving_protocol"],
        geometry_report_path=geom_report_path,
    )

    source_verification = verify_source_inside_computational_domain(
        problem=problem_ref,
        forcing_meta=forcing_meta,
        geom_report_dict=geom_report_dict,
    )

    theta_dagger = master.parameter_space_theta.theta_dagger_array()
    traj_ref = problem_ref.solve_trajectory(
        theta_vec=theta_dagger,
        c0=c0_ref,
        F_mech_func=F_mech_ref,
        S_fluid_func=S_fluid_ref,
        S_tracer_func=None,
        enforce_initial_compatibility=True,
        compute_diagnostics=True,
    )

    target_times = fwd_cfg.get("snapshot_target_times_adim", [0.05, 0.15, 0.35, float(fwd_cfg["final_time_T"])])
    snapshot_indices: List[int] = []
    for tt in target_times:
        idx = int(np.argmin(np.abs(traj_ref.times - float(tt))))
        if idx not in snapshot_indices:
            snapshot_indices.append(idx)

    darcy_flux_snapshots = {
        f"step_{idx:03d}": np.asarray(problem_ref.fem.compute_cell_darcy_fluxes(traj_ref.p_hist[idx]))
        for idx in snapshot_indices
    }
    np.savez_compressed(
        out_dir / "reference_trajectory_snapshots.npz",
        times_adim=traj_ref.times,
        times_phys_s=traj_ref.times * float(master.characteristic_scales.T0_s),
        snapshot_indices=np.array(snapshot_indices, dtype=np.int32),
        d_snapshots_adim=traj_ref.d_hist[snapshot_indices],
        p_snapshots_adim=traj_ref.p_hist[snapshot_indices],
        c_snapshots_adim=traj_ref.c_hist[snapshot_indices],
        mesh_vertices_adim=problem_ref.mesh_vertices,
        cell_quad_connectivity=problem_ref.cell_quad_connectivity,
        cell_tracer_dof_connectivity=problem_ref.cell_tracer_dof_connectivity,
        **darcy_flux_snapshots,
    )

    lesion_center_adim = tuple(source_verification["lesion_centroid_mesh_adim"])
    ref_series_metrics = _compute_trajectory_series_metrics(problem_ref, traj_ref, lesion_center_adim)

    # Requisito 4 & 5: Studio di SENSIBILITÀ ALLA MESH con controllo qualità obbligatorio
    # e identificazione della mesh più fine tramite min(target_element_size_mm)
    resolutions_mm = raw_cfg.get("mesh_sensitivity_study", {}).get("resolutions_mm", [4.0, 3.0, 2.0])
    if len(resolutions_mm) < 3:
        raise ValueError("mesh_sensitivity_study deve includere almeno 3 risoluzioni.")

    mesh_sensitivity_records: List[Dict[str, Any]] = []
    mesh_plot_payloads: List[Dict[str, Any]] = []

    ref_src_center_adim = np.asarray(
        forcing_meta["fluid_injection_source"]["center_mesh_adim"], dtype=np.float64
    )
    ref_src_sigma_adim = float(forcing_meta["fluid_injection_source"]["sigma_mesh_adim"])

    for h_val in resolutions_mm:
        h_mm = float(h_val)
        m_cfg = MeshGenerationConfig(
            target_element_size_mm=h_mm,
            boundary_projection_iterations=pipe_cfg.mesh.boundary_projection_iterations,
            laplacian_smoothing_iterations=pipe_cfg.mesh.laplacian_smoothing_iterations,
            laplacian_relaxation=pipe_cfg.mesh.laplacian_relaxation,
            contour_simplify_tolerance_mm=pipe_cfg.mesh.contour_simplify_tolerance_mm,
            min_hole_area_to_mesh_mm2=pipe_cfg.mesh.min_hole_area_to_mesh_mm2,
        )
        m_obj = generate_anatomical_quad_mesh_2d(masks, slice_sel, transform, m_cfg)
        b_name = f"{pipe_cfg.subject_id}_slice{slice_sel.slice_index:03d}_h{h_mm:.1f}mm"
        exp_m = export_dealii_quad_mesh(m_obj, transform, meshes_dir, basename=b_name)
        rep_m = compute_geometry_quality_report(
            volumes, slice_sel, masks, transform, m_obj, exp_m, pipe_cfg.quality_thresholds
        )
        if not rep_m.all_mandatory_checks_passed:
            raise RuntimeError(
                f"Controlli di qualità geometrica FALLITI nello studio di sensibilità alla mesh "
                f"per target_element_size_mm={h_mm}: {rep_m.failed_checks}"
            )

        prob_m = RetecoForwardProblem2D(
            physics_model=master.fixed_physics,
            scales=master.characteristic_scales,
            time_grid=grid,
            fe_deg_disp=int(fwd_cfg.get("fe_deg_disp", 2)),
            fe_deg_press=int(fwd_cfg.get("fe_deg_press", 1)),
            fe_deg_tracer=int(fwd_cfg.get("fe_deg_tracer", 1)),
            use_supg=bool(fwd_cfg.get("use_supg", False)),
            mesh_file_path=exp_m["gmsh_msh"],
            dirichlet_disp_boundary_ids=fwd_cfg.get("dirichlet_disp_boundary_ids", [1]),
        )
        c0_m, F_m, Sf_m, forcing_m = build_anatomical_driving_Callbacks(
            prob_m, fwd_cfg["driving_protocol"], geometry_report_path=geom_report_path
        )

        # Verifica invarianza di centro e larghezza della sorgente in coordinate adimensionali
        cur_center_adim = np.asarray(
            forcing_m["fluid_injection_source"]["center_mesh_adim"], dtype=np.float64
        )
        cur_sigma_adim = float(forcing_m["fluid_injection_source"]["sigma_mesh_adim"])
        if (
            np.linalg.norm(cur_center_adim - ref_src_center_adim) > 1.0e-12
            or abs(cur_sigma_adim - ref_src_sigma_adim) > 1.0e-12
        ):
            raise RuntimeError(
                f"Incoerenza dei parametri adimensionali della sorgente alla risoluzione h={h_mm} mm."
            )
        verify_source_inside_computational_domain(prob_m, forcing_m, geom_report_dict)

        traj_m = prob_m.solve_trajectory(
            theta_vec=theta_dagger,
            c0=c0_m,
            F_mech_func=F_m,
            S_fluid_func=Sf_m,
            enforce_initial_compatibility=True,
            compute_diagnostics=True,
        )
        series_m = _compute_trajectory_series_metrics(prob_m, traj_m, lesion_center_adim)
        masses_m = np.array(series_m["tracer_mass_series"], dtype=np.float64)
        mass_drift_m = float(np.max(np.abs(masses_m - masses_m[0])) / masses_m[0])

        mesh_sensitivity_records.append(
            {
                "target_element_size_mm": h_mm,
                "target_element_size_adim": h_mm / l0_mm,
                "num_quad_cells": prob_m.num_active_cells,
                "num_vertices": int(prob_m.fem.num_vertices()),
                "num_disp_dofs": prob_m.num_disp_dofs,
                "num_press_dofs": prob_m.num_press_dofs,
                "num_tracer_dofs": prob_m.num_tracer_dofs,
                "total_dofs": prob_m.num_disp_dofs + prob_m.num_press_dofs + prob_m.num_tracer_dofs,
                "mesh_area_mm2": rep_m.mesh_area_mm2,
                "relative_area_error_vs_mask": rep_m.relative_area_error,
                "rasterized_dice_similarity": rep_m.rasterized_dice_similarity,
                "source_center_adim": cur_center_adim.tolist(),
                "source_sigma_adim": cur_sigma_adim,
                "initial_tracer_mass_adim": float(masses_m[0]),
                "final_tracer_mass_adim": float(masses_m[-1]),
                "max_relative_mass_drift": mass_drift_m,
                "final_c_l2_norm": float(series_m["c_l2_norm_series"][-1]),
                "final_p_l2_norm": float(series_m["p_l2_norm_series"][-1]),
                "final_disp_l2_norm": float(series_m["disp_l2_norm_series"][-1]),
                "final_disp_stiffness_seminorm": float(series_m["disp_stiffness_seminorm_series"][-1]),
                "final_c_at_lesion_center": float(series_m["c_lesion_center_series"][-1]),
                "final_total_energy": float(series_m["energy_series"][-1]),
                "final_visco_dissipation": float(series_m["visco_dissipation_series"][-1]),
                "max_darcy_velocity_adim": traj_m.max_darcy_velocity,
            }
        )

        mesh_plot_payloads.append(
            {
                "target_h_mm": h_mm,
                "num_cells": prob_m.num_active_cells,
                "total_dofs": prob_m.num_disp_dofs + prob_m.num_press_dofs + prob_m.num_tracer_dofs,
                "mesh_vertices_adim": prob_m.mesh_vertices,
                "cell_quad_connectivity": prob_m.cell_quad_connectivity,
                "final_cell_tracer": np.mean(
                    traj_m.c_hist[-1][prob_m.cell_tracer_dof_connectivity], axis=1
                ),
                "times": series_m["times"],
                "c_l2_series": series_m["c_l2_norm_series"],
                "energy_series": series_m["energy_series"],
                "profile_x_adim": series_m["profile_x_adim"],
                "profile_c_final": series_m["profile_c_final"],
            }
        )

    # Identificazione della mesh più fine in base al minimo target_element_size_mm
    finest_rec = min(mesh_sensitivity_records, key=lambda r: r["target_element_size_mm"])
    for rec in mesh_sensitivity_records:
        rec["finest_reference_h_mm"] = finest_rec["target_element_size_mm"]
        rec["rel_diff_final_c_l2_vs_finest"] = abs(
            rec["final_c_l2_norm"] - finest_rec["final_c_l2_norm"]
        ) / max(finest_rec["final_c_l2_norm"], 1.0e-12)
        rec["rel_diff_final_disp_l2_vs_finest"] = abs(
            rec["final_disp_l2_norm"] - finest_rec["final_disp_l2_norm"]
        ) / max(finest_rec["final_disp_l2_norm"], 1.0e-12)
        rec["rel_diff_final_energy_vs_finest"] = abs(
            rec["final_total_energy"] - finest_rec["final_total_energy"]
        ) / max(finest_rec["final_total_energy"], 1.0e-12)

    # Studio di sensibilità ai parametri frazionari alpha e beta
    frac_cfg = raw_cfg.get("fractional_sensitivity_study", {})
    alpha_values = frac_cfg.get("alpha_values", [0.35, 0.55, 0.70, 0.90])
    beta_values = frac_cfg.get("beta_values", [0.20, 0.45, 0.65, 0.85])

    alpha_sweep_results: List[Dict[str, Any]] = []
    for a_val in alpha_values:
        theta_a = np.array([float(a_val), theta_dagger[1], theta_dagger[2], theta_dagger[3]])
        traj_a = problem_ref.solve_trajectory(
            theta_vec=theta_a,
            c0=c0_ref,
            F_mech_func=F_mech_ref,
            S_fluid_func=S_fluid_ref,
            enforce_initial_compatibility=True,
            compute_diagnostics=True,
        )
        ser_a = _compute_trajectory_series_metrics(problem_ref, traj_a, lesion_center_adim)
        ser_a["alpha"] = float(a_val)
        ser_a["beta"] = float(theta_dagger[1])
        alpha_sweep_results.append(ser_a)

    beta_sweep_results: List[Dict[str, Any]] = []
    for b_val in beta_values:
        theta_b = np.array([theta_dagger[0], float(b_val), theta_dagger[2], theta_dagger[3]])
        traj_b = problem_ref.solve_trajectory(
            theta_vec=theta_b,
            c0=c0_ref,
            F_mech_func=F_mech_ref,
            S_fluid_func=S_fluid_ref,
            enforce_initial_compatibility=True,
            compute_diagnostics=True,
        )
        ser_b = _compute_trajectory_series_metrics(problem_ref, traj_b, lesion_center_adim)
        ser_b["alpha"] = float(theta_dagger[0])
        ser_b["beta"] = float(b_val)
        ser_b["max_darcy_velocity_adim"] = traj_b.max_darcy_velocity
        beta_sweep_results.append(ser_b)

    fig_paths = {
        "fig01_multimodal_anatomy_and_source_pdf": str(
            out_dir / "fig01_multimodal_anatomy_and_annotations.pdf"
        ),
        "fig02_mask_and_quad_mesh_quality_pdf": str(
            out_dir / "fig02_anatomical_mask_and_quad_mesh_quality.pdf"
        ),
        "fig03_multiphysics_fields_on_quad_mesh_pdf": str(
            out_dir / "fig03_multiphysics_fields_on_quad_mesh.pdf"
        ),
        "fig04_temporal_evolution_comparable_scales_pdf": str(
            out_dir / "fig04_temporal_evolution_comparable_scales.pdf"
        ),
        "fig05_mesh_resolution_sensitivity_pdf": str(
            out_dir / "fig05_mesh_resolution_sensitivity.pdf"
        ),
        "fig06_fractional_parameters_sensitivity_pdf": str(
            out_dir / "fig06_fractional_parameters_sensitivity.pdf"
        ),
        "fig07_energy_and_mass_diagnostics_pdf": str(
            out_dir / "fig07_energy_and_mass_diagnostics.pdf"
        ),
    }

    plot_fig01_multimodal_anatomy_and_source(
        volumes,
        slice_sel,
        masks,
        transform,
        ref_mesh,
        source_verification,
        fig_paths["fig01_multimodal_anatomy_and_source_pdf"],
    )
    plot_fig02_mask_and_quad_mesh_quality(
        masks,
        ref_mesh,
        geom_report,
        fig_paths["fig02_mask_and_quad_mesh_quality_pdf"],
    )
    plot_fig03_multiphysics_fields_on_quad_mesh(
        problem_ref,
        traj_ref,
        master.characteristic_scales,
        masks.domain_representation_type,
        fig_paths["fig03_multiphysics_fields_on_quad_mesh_pdf"],
    )
    plot_fig04_temporal_evolution_comparable_scales(
        problem_ref,
        traj_ref,
        snapshot_indices,
        master.characteristic_scales,
        masks.domain_representation_type,
        fig_paths["fig04_temporal_evolution_comparable_scales_pdf"],
    )
    plot_fig05_mesh_resolution_sensitivity(
        mesh_plot_payloads,
        fig_paths["fig05_mesh_resolution_sensitivity_pdf"],
    )
    plot_fig06_fractional_parameters_sensitivity(
        alpha_sweep_results,
        beta_sweep_results,
        fig_paths["fig06_fractional_parameters_sensitivity_pdf"],
    )
    plot_anatomical_energy_and_mass_diagnostics(
        traj_ref.diagnostics,
        fig_paths["fig07_energy_and_mass_diagnostics_pdf"],
    )

    masses_ref = np.array(ref_series_metrics["tracer_mass_series"], dtype=np.float64)
    rel_mass_drift_ref = float(np.max(np.abs(masses_ref - masses_ref[0])) / masses_ref[0])
    max_step_id_err = float(max(d.step_discrete_identity_error for d in traj_ref.diagnostics[1:]))

    consistency_checks = {
        "source_inside_computational_domain": source_verification["source_inside_computational_domain"],
        "dof_and_mesh_consistency": bool(problem_ref.fem.verify_dof_and_mesh_consistency()),
        "initial_mechanical_compatibility_satisfied": bool(traj_ref.initial_compatibility_residual < 1.0e-11),
        "tracer_mass_conserved_under_no_flux": bool(rel_mass_drift_ref < 1.0e-11),
        "stepwise_discrete_energy_identity_satisfied": bool(max_step_id_err < 1.0e-11),
        "all_three_mesh_resolutions_completed": bool(len(mesh_sensitivity_records) >= 3),
    }

    experiment_summary = {
        "experiment_id": "02_anatomical_forward_brats2020",
        "source_data_kind": volumes.source_data_kind,
        "provenance_verified": volumes.provenance_verified,
        "input_provenance": geom_report.input_provenance,
        "scientific_disclaimers": {
            "domain_representation": masks.domain_representation_type,
            "warnings": masks.warnings,
            "clinical_validation_disclaimer": (
                "NO_CLINICAL_VALIDATION_CLAIMED: This experiment evaluates a mathematical "
                "forward model with anatomy-guided synthetic forcing on a 2D axial parenchyma "
                "proxy extracted from static BraTS 2020 MRI. Simulated fields do NOT represent "
                "observed clinical measurements."
            ),
            "mesh_comparison_terminology": (
                "MESH_SENSITIVITY_ONLY: Since no analytical exact solution exists on the "
                "anatomical geometry, multi-mesh comparisons quantify mesh resolution sensitivity "
                "and NOT theoretical asymptotic convergence orders."
            ),
            "energy_residual_interpretation": (
                "FORCED_ENERGY_IDENTITY_NOTE: Under active mechanical and fluid forcing, total "
                "energy E(t_n) is driven by external work W_ext(t_n) and does not decay monotonically. "
                "R_n = (E(t_n) + D_beta(t_n) + D_h(t_n)) - (E(0) + W_ext(t_n)) equals the cumulative "
                "implicit-Euler numerical dissipation (-sum D_{num,m}), while the algebraic stepwise "
                "identity |(R_n - R_{n-1}) + D_{num,n}| = 0 holds to machine precision."
            ),
        },
        "characteristic_scales": master.characteristic_scales.model_dump(),
        "fixed_physics": master.fixed_physics.model_dump(),
        "reference_parameters_theta": {
            "alpha": float(theta_dagger[0]),
            "beta": float(theta_dagger[1]),
            "D_adim": float(theta_dagger[2]),
            "E_adim": float(theta_dagger[3]),
            "D_phys_m2_s": float(theta_dagger[2])
            * (master.characteristic_scales.L0_m ** 2)
            / master.characteristic_scales.T0_s,
            "E_phys_Pa": float(theta_dagger[3]) * master.characteristic_scales.E0_Pa,
        },
        "nifti_and_slice_metadata": {
            "subject_id": volumes.subject_id,
            "source_data_kind": volumes.source_data_kind,
            "orientation_convention": volumes.orientation_convention,
            "volume_shape_3d": list(volumes.shape),
            "voxel_spacing_mm": list(volumes.spacing_mm),
            "selected_slice_index": slice_sel.slice_index,
            "selection_mode": slice_sel.selection_mode,
            "physical_z_mm": slice_sel.physical_z_mm,
            "segmentation_labels_report": geom_report.segmentation_labels_report,
        },
        "coordinate_transform": transform.to_metadata_dict(),
        "source_verification": source_verification,
        "boundary_conditions": problem_ref.boundary_conditions_metadata(),
        "forcing_configuration": forcing_meta,
        "reference_trajectory_diagnostics": {
            "mesh_version_file": ref_exported["gmsh_msh"],
            "num_quad_cells": problem_ref.num_active_cells,
            "num_disp_dofs": problem_ref.num_disp_dofs,
            "num_press_dofs": problem_ref.num_press_dofs,
            "num_tracer_dofs": problem_ref.num_tracer_dofs,
            "initial_compatibility_residual": traj_ref.initial_compatibility_residual,
            "initial_tracer_mass_adim": float(masses_ref[0]),
            "final_tracer_mass_adim": float(masses_ref[-1]),
            "max_relative_tracer_mass_drift": rel_mass_drift_ref,
            "final_disp_l2_norm_adim": float(ref_series_metrics["disp_l2_norm_series"][-1]),
            "final_disp_stiffness_seminorm_adim": float(
                ref_series_metrics["disp_stiffness_seminorm_series"][-1]
            ),
            "final_press_l2_norm_adim": float(ref_series_metrics["p_l2_norm_series"][-1]),
            "final_tracer_l2_norm_adim": float(ref_series_metrics["c_l2_norm_series"][-1]),
            "max_darcy_velocity_adim": traj_ref.max_darcy_velocity,
            "max_darcy_velocity_phys_mm_s": traj_ref.max_darcy_velocity
            * (l0_mm / float(master.characteristic_scales.T0_s)),
            "initial_total_energy_adim": float(traj_ref.diagnostics[0].total_energy),
            "final_total_energy_adim": float(traj_ref.diagnostics[-1].total_energy),
            "final_cumulative_visco_dissipation_adim": float(
                traj_ref.diagnostics[-1].cumulative_visco_dissip
            ),
            "final_cumulative_darcy_dissipation_adim": float(
                traj_ref.diagnostics[-1].cumulative_darcy_dissip
            ),
            "final_cumulative_external_work_adim": float(
                traj_ref.diagnostics[-1].cumulative_external_work
            ),
            "final_energy_balance_residual_adim": float(
                traj_ref.diagnostics[-1].energy_balance_residual
            ),
            "max_step_discrete_identity_error": max_step_id_err,
        },
        "mesh_sensitivity_study": {
            "terminology": "sensibilità alla mesh (no exact analytical solution)",
            "finest_reference_h_mm": finest_rec["target_element_size_mm"],
            "resolutions_evaluated": mesh_sensitivity_records,
        },
        "fractional_sensitivity_study": {
            "alpha_sweep_summary": [
                {
                    "alpha": item["alpha"],
                    "beta": item["beta"],
                    "final_c_lesion_center": item["c_lesion_center_series"][-1],
                    "final_c_l2_norm": item["c_l2_norm_series"][-1],
                }
                for item in alpha_sweep_results
            ],
            "beta_sweep_summary": [
                {
                    "alpha": item["alpha"],
                    "beta": item["beta"],
                    "final_disp_l2_norm": item["disp_l2_norm_series"][-1],
                    "final_disp_stiffness_seminorm": item["disp_stiffness_seminorm_series"][-1],
                    "final_press_l2_norm": item["p_l2_norm_series"][-1],
                    "final_visco_dissipation": item["visco_dissipation_series"][-1],
                    "max_darcy_velocity_adim": item["max_darcy_velocity_adim"],
                }
                for item in beta_sweep_results
            ],
        },
        "consistency_checks": consistency_checks,
        "all_mandatory_checks_passed": all(consistency_checks.values()),
        "generated_artifacts": {
            "geometry_quality_report_json": str(geom_report_path),
            "reference_snapshots_npz": str(out_dir / "reference_trajectory_snapshots.npz"),
            "experiment_summary_json": str(out_dir / "anatomical_forward_experiment_summary.json"),
            **fig_paths,
        },
    }

    summary_json_path = out_dir / "anatomical_forward_experiment_summary.json"
    summary_json_path.write_text(
        json.dumps(experiment_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return experiment_summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="RETECO v1 — Experiment 02: Anatomical Forward Model & Sensitivity Studies"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/brats_anatomical_forward.yaml",
        help="Percorso alla configurazione YAML anatomica.",
    )
    args = parser.parse_args(argv)

    summary = run_anatomical_forward_experiment(args.config)
    print(json.dumps(summary["consistency_checks"], indent=2))
    print(f"Artefatti completi salvati in: {summary['generated_artifacts']['experiment_summary_json']}")
    return 0 if summary["all_mandatory_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())