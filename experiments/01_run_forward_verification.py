"""
experiments/01_run_forward_verification.py
Protocollo completo di consolidamento e verifica del Problema Diretto RETECO v1:
1) Studio di convergenza MMS spaziale (con M coordinato a N^2 per isolare gli ordini spaziali)
   e temporale (su tutti e 3 i campi d, p, c) + generazione Tabella LaTeX e Figura 1
2) Verifica dell'identità energetica discreta (Proposizione 1) e conservazione della massa -> Figura 2
3) Studio parametrico del ruolo fisico di alpha (memoria di trasporto) e beta (memoria viscoelastica) -> Figura 3
4) Ispezione metrica dei volumi clinici BraTS (`data/01_raw/BraTS20_Training_001_*.nii`) con
   segmentazione dell'edema peritumorale e mappe multifisiche (||d||, p, q, c) -> Figura 4
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Dict
import numpy as np
import yaml

from reteco import _core, load_master_config
from reteco.forward.wrapper import RetecoForwardProblem2D
from reteco.forward.target_generator import build_standard_driving_Callbacks
from reteco.visualization.verification_plots import (
    export_mms_latex_table,
    plot_mms_complete_convergence,
    plot_energy_and_mass_diagnostics,
    plot_fractional_orders_sensitivity,
    plot_multiphysics_fields_and_brats_overlay,
)


def inspect_brats_nifti_slice(raw_dir: Path, L0_m: float) -> Dict | None:
    """
    Carica i volumi NIfTI `BraTS20_Training_001_t1.nii` e `BraTS20_Training_001_seg.nii`
    applicando la matrice affine reale dello scanner (voxel -> mm -> L0) e separa:
    - Edema peritumorale nella sostanza bianca (label == 2)
    - Nucleo tumorale necrotico/attivo (label == 1 o label == 4)
    """
    t1_path = raw_dir / "BraTS20_Training_001_t1.nii"
    seg_path = raw_dir / "BraTS20_Training_001_seg.nii"
    if not (t1_path.exists() and seg_path.exists()):
        return None

    import nibabel as nib

    t1_img = nib.load(str(t1_path))
    seg_img = nib.load(str(seg_path))
    t1_vol = t1_img.get_fdata(dtype=np.float64)
    seg_vol = np.rint(seg_img.get_fdata()).astype(np.int32)
    zooms_mm = tuple(float(z) for z in t1_img.header.get_zooms()[:3])

    # Selezioniamo la slice assiale con la massima estensione di lesione/edema peritumorale
    lesion_counts_per_slice = np.sum(seg_vol > 0, axis=(0, 1))
    best_slice = int(np.argmax(lesion_counts_per_slice))

    t1_slice = t1_vol[:, :, best_slice]
    seg_slice = seg_vol[:, :, best_slice]

    nx, ny = t1_slice.shape
    dx_mm, dy_mm = zooms_mm[0], zooms_mm[1]
    voxel_area_mm2 = dx_mm * dy_mm

    brain_mask = t1_slice > 0
    edema_mask = seg_slice == 2
    tumor_core_mask = (seg_slice == 1) | (seg_slice == 4)

    brain_area_mm2 = float(np.sum(brain_mask) * voxel_area_mm2)
    edema_area_mm2 = float(np.sum(edema_mask) * voxel_area_mm2)
    tumor_core_area_mm2 = float(np.sum(tumor_core_mask) * voxel_area_mm2)

    # Coordinate fisiche in mm centrate sulla scatola cranica della slice
    width_mm = nx * dx_mm
    height_mm = ny * dy_mm
    extent_mm = [0.0, width_mm, 0.0, height_mm]

    lesion_indices = np.argwhere(seg_slice > 0)
    centroid_ij = np.mean(lesion_indices, axis=0)
    centroid_mm = (float(centroid_ij[0] * dx_mm), float(centroid_ij[1] * dy_mm))

    # Posizione relativa normalizzata in [0, 1]^2 dentro il bounding box cerebrale
    brain_ij = np.argwhere(brain_mask)
    min_ij = np.min(brain_ij, axis=0)
    max_ij = np.max(brain_ij, axis=0)
    rel_pos = (centroid_ij - min_ij) / np.maximum(max_ij - min_ij, 1.0)

    return {
        "slice_index": best_slice,
        "voxel_spacing_mm": list(zooms_mm),
        "brain_slice_area_mm2": brain_area_mm2,
        "peritumoral_white_matter_edema_area_mm2": edema_area_mm2,
        "tumor_core_area_mm2": tumor_core_area_mm2,
        "edema_fraction_of_slice": edema_area_mm2 / max(brain_area_mm2, 1.0),
        "lesion_centroid_mm": list(centroid_mm),
        "lesion_normalized_coords": [float(rel_pos[0]), float(rel_pos[1])],
        "L0_mm": L0_m * 1000.0,
        "t1_slice": t1_slice,
        "seg_slice": seg_slice,
        "extent_mm": extent_mm,
    }


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
    T_mms = float(mms_cfg["final_time_T"])

    # -------------------------------------------------------------------------
    # 1. STUDIO DI CONVERGENZA SPAZIALE E TEMPORALE MMS
    # -------------------------------------------------------------------------
    print("=== [1/4] Studio di Convergenza Spaziale e Temporale MMS ===")
    sp_cfg = mms_cfg["spatial_refinement_study"]
    N_list = sp_cfg["subdivisions_N"]
    M_coord_list = sp_cfg["coordinated_time_steps_M"]

    spatial_reports = []
    for idx, (N, M_step) in enumerate(zip(N_list, M_coord_list)):
        grid_sp = _core.TimeGrid.make_graded(T_mms, int(M_step), float(sp_cfg["grading_r"]))
        rep = mms.run_full_mms_test(int(N), grid_sp, 2, 1, 1)
        spatial_reports.append(rep)
        if idx == 0:
            print(
                f"  [Spaziale] N={N:2d} (M={M_step:4d}, h={rep.h_mesh:.4f}) | "
                f"L2(d)={rep.d_L2:.3e}, H1(d)={rep.d_H1:.3e} | "
                f"L2(p)={rep.p_L2:.3e} | L2(c)={rep.c_L2:.3e}"
            )
        else:
            prev = spatial_reports[idx - 1]
            lh = math.log(prev.h_mesh / rep.h_mesh)
            rd = math.log(prev.d_L2 / rep.d_L2) / lh
            rp = math.log(prev.p_L2 / rep.p_L2) / lh
            rc = math.log(prev.c_L2 / rep.c_L2) / lh
            print(
                f"  [Spaziale] N={N:2d} (M={M_step:4d}, h={rep.h_mesh:.4f}) | "
                f"L2(d)={rep.d_L2:.3e} (EOC={rd:.2f}) | "
                f"L2(p)={rep.p_L2:.3e} (EOC={rp:.2f}) | "
                f"L2(c)={rep.c_L2:.3e} (EOC={rc:.2f})"
            )

    tm_cfg = mms_cfg["temporal_refinement_study"]
    N_fixed = int(tm_cfg["fixed_subdivisions_N"])
    temporal_reports = []
    for idx, M_t in enumerate(tm_cfg["time_steps_M"]):
        grid_tm = _core.TimeGrid.make_graded(T_mms, int(M_t), float(tm_cfg["grading_r"]))
        rep_t = mms.run_full_mms_test(N_fixed, grid_tm, 2, 2, 2)
        temporal_reports.append(rep_t)
        if idx > 0:
            prev = temporal_reports[idx - 1]
            ldt = math.log(prev.dt_max / rep_t.dt_max)
            rd_t = math.log(prev.d_L2 / rep_t.d_L2) / ldt
            rp_t = math.log(prev.p_L2 / rep_t.p_L2) / ldt
            rc_t = math.log(prev.c_L2 / rep_t.c_L2) / ldt
            print(
                f"  [Temporale] M={M_t:3d} (dt={rep_t.dt_max:.4f}) | "
                f"EOC_t(d)={rd_t:.2f}, EOC_t(p)={rp_t:.2f}, EOC_t(c)={rc_t:.2f}"
            )

    export_mms_latex_table(
        spatial_reports, temporal_reports, out_dir / "table_mms_convergence.tex"
    )
    plot_mms_complete_convergence(
        spatial_reports, temporal_reports, out_dir / "fig_01_mms_convergence_complete.pdf"
    )

    # -------------------------------------------------------------------------
    # 2. BILANCIO ENERGETICO DISCRETO E CONSERVAZIONE DELLA MASSA
    # -------------------------------------------------------------------------
    print("=== [2/4] Verifica Identità Energetica Discreta e Conservazione Massa ===")
    dt_ref_residuals = []
    traj_fine_diag = None
    for M_diag in [20, 40, 80, 160]:
        grid_d = _core.TimeGrid.make_graded(0.8, M_diag, 1.0)
        prob_d = RetecoForwardProblem2D(
            subdivisions_N=14,
            physics_model=master.fixed_physics,
            scales=master.characteristic_scales,
            time_grid=grid_d,
        )
        c0_d, _, _ = build_standard_driving_Callbacks(prob_d, twin_cfg["driving_protocol"])
        px, py = prob_d.press_pts[:, 0], prob_d.press_pts[:, 1]
        p0_d = 0.5 * np.cos(np.pi * px) * np.cos(np.pi * py)
        tr_d = prob_d.solve_trajectory(
            theta_vec=master.parameter_space_theta.theta_dagger_array(),
            c0=c0_d,
            p0=p0_d,
            enforce_initial_compatibility=True,
            compute_diagnostics=True,
        )
        dt_val = 0.8 / M_diag
        res_end = abs(tr_d.diagnostics[-1].energy_balance_residual)
        dt_ref_residuals.append((dt_val, res_end))
        if M_diag == 80:
            traj_fine_diag = tr_d

    assert traj_fine_diag is not None
    plot_energy_and_mass_diagnostics(
        traj_fine_diag.diagnostics,
        dt_ref_residuals,
        out_dir / "fig_02_energy_and_mass_conservation.pdf",
    )

    # -------------------------------------------------------------------------
    # 3. STUDIO DI SENSIBILITÀ FISICA AGLI ORDINI FRAZIONARI (alpha, beta)
    # -------------------------------------------------------------------------
    print("=== [3/4] Studio di Sensibilità Fisica agli Ordini Frazionari (alpha, beta) ===")
    grid_sens = _core.TimeGrid.make_graded(1.0, 60, 1.5)
    prob_sens = RetecoForwardProblem2D(
        subdivisions_N=14,
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid_sens,
    )
    c0_s, F_s, Sf_s = build_standard_driving_Callbacks(prob_sens, twin_cfg["driving_protocol"])
    base_theta = master.parameter_space_theta.theta_dagger_array()

    alpha_curves = {}
    for a_val in [0.30, 0.50, 0.70, 0.90]:
        th = base_theta.copy()
        th[0] = a_val
        tr = prob_sens.solve_trajectory(th, c0=c0_s, F_mech_func=F_s, S_fluid_func=Sf_s)
        c_peak = np.max(tr.c_hist, axis=1)
        alpha_curves[a_val] = (tr.times, c_peak)

    beta_curves = {}
    for b_val in [0.20, 0.45, 0.70, 0.90]:
        th = base_theta.copy()
        th[1] = b_val
        tr = prob_sens.solve_trajectory(
            th, c0=c0_s, F_mech_func=F_s, S_fluid_func=Sf_s, compute_diagnostics=True
        )
        p_l2_t = np.array(
            [math.sqrt(2.0 * d.fluid_storage_energy / master.fixed_physics.S_p) for d in tr.diagnostics]
        )
        beta_curves[b_val] = (tr.times, p_l2_t, tr.max_darcy_velocity)

    plot_fractional_orders_sensitivity(
        alpha_curves, beta_curves, out_dir / "fig_03_fractional_orders_physics_sensitivity.pdf"
    )

    # -------------------------------------------------------------------------
    # 4. ISPEZIONE ANATOMICA BraTS E CAMPI MULTIFISICI 2D (||d||, p, q, c)
    # -------------------------------------------------------------------------
    print("=== [4/4] Analisi Volumi NIfTI BraTS e Mappe Multifisiche 2D ===")
    brats_info = inspect_brats_nifti_slice(
        Path("data/01_raw"), master.characteristic_scales.L0_m
    )
    if brats_info is not None:
        print(
            f"  [BraTS20_Training_001] Slice assiale ottimale: z={brats_info['slice_index']} | "
            f"Area parenchima={brats_info['brain_slice_area_mm2']:.1f} mm^2 | "
            f"Area edema sostanza bianca (label 2)={brats_info['peritumoral_white_matter_edema_area_mm2']:.1f} mm^2 | "
            f"Area core tumorale (label 1,4)={brats_info['tumor_core_area_mm2']:.1f} mm^2"
        )
        inj_center = tuple(brats_info["lesion_normalized_coords"])
    else:
        inj_center = (0.5, 0.5)

    drv_custom = dict(twin_cfg["driving_protocol"])
    drv_custom["initial_tracer_gaussian"] = dict(drv_custom["initial_tracer_gaussian"])
    drv_custom["fluid_injection_source"] = dict(drv_custom["fluid_injection_source"])
    drv_custom["initial_tracer_gaussian"]["center"] = list(inj_center)
    drv_custom["fluid_injection_source"]["center"] = list(inj_center)

    grid_2d = _core.TimeGrid.make_graded(1.0, 60, 1.5)
    prob_2d = RetecoForwardProblem2D(
        subdivisions_N=20,
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid_2d,
    )
    c0_2d, F_2d, Sf_2d = build_standard_driving_Callbacks(prob_2d, drv_custom)
    tr_2d = prob_2d.solve_trajectory(
        base_theta, c0=c0_2d, F_mech_func=F_2d, S_fluid_func=Sf_2d, compute_diagnostics=True
    )

    plot_multiphysics_fields_and_brats_overlay(
        disp_pts=prob_2d.disp_pts,
        press_pts=prob_2d.press_pts,
        tracer_pts=prob_2d.tracer_pts,
        d_final=tr_2d.d_hist[-1],
        p_final=tr_2d.p_hist[-1],
        c_final=tr_2d.c_hist[-1],
        K_h=master.fixed_physics.K_h,
        output_path=out_dir / "fig_04_brats_multiphysics_fields.pdf",
        brats_data=brats_info,
    )

    # Esportazione JSON completa (senza gli array 2D pesanti dell'immagine)
    brats_meta = None
    if brats_info is not None:
        brats_meta = {
            k: v
            for k, v in brats_info.items()
            if k not in ("t1_slice", "seg_slice")
        }

    max_step_id_err = max(d.step_discrete_identity_error for d in traj_fine_diag.diagnostics[1:])
    summary = {
        "mms_spatial": [
            {
                "h": r.h_mesh,
                "dt_max": r.dt_max,
                "d_L2": r.d_L2,
                "d_H1": r.d_H1,
                "p_L2": r.p_L2,
                "p_H1": r.p_H1,
                "c_L2": r.c_L2,
                "c_H1": r.c_H1,
            }
            for r in spatial_reports
        ],
        "mms_temporal": [
            {
                "dt_max": r.dt_max,
                "d_L2": r.d_L2,
                "p_L2": r.p_L2,
                "c_L2": r.c_L2,
            }
            for r in temporal_reports
        ],
        "energy_and_mass_verification": {
            "initial_compatibility_residual": traj_fine_diag.initial_compatibility_residual,
            "max_stepwise_discrete_identity_error": max_step_id_err,
            "dt_refinement_residuals": [
                {"dt": dt_v, "abs_energy_residual_RN": res_v}
                for dt_v, res_v in dt_ref_residuals
            ],
            "initial_mass_M0": traj_fine_diag.diagnostics[0].tracer_total_mass,
            "final_mass_MT": traj_fine_diag.diagnostics[-1].tracer_total_mass,
            "relative_mass_drift": abs(
                traj_fine_diag.diagnostics[-1].tracer_total_mass
                - traj_fine_diag.diagnostics[0].tracer_total_mass
            )
            / traj_fine_diag.diagnostics[0].tracer_total_mass,
        },
        "brats_anatomical_pathology_metadata": brats_meta,
    }
    (out_dir / "forward_verification_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Tutte le 4 figure, la tabella LaTeX e il report JSON sono in: {out_dir}")


if __name__ == "__main__":
    main()