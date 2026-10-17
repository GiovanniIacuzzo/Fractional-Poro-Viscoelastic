"""
python/reteco/visualization/verification_plots.py
Suite grafica di qualità editoriale (Q1) per la validazione del Problema Diretto RETECO v1:
1) Convergenza spaziale (L2 & H1) e temporale (L2) del sistema accoppiato MMS + Tabella LaTeX
2) Bilancio energetico discreto, verifica dell'identità di passo e conservazione della massa M(t)
3) Sensibilità fisica delle traiettorie agli ordini frazionari alpha (trasporto) e beta (viscoelasticità)
4) Mappe multifisiche 2D (||d||, p + frecce di Darcy q, c) e integrazione con le maschere MRI BraTS
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Sequence, Any
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from .. import _core


plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
})


def export_mms_latex_table(
    spatial_reports: Sequence[_core.MMSErrorReport],
    temporal_reports: Sequence[_core.MMSErrorReport],
    output_tex_path: str | Path,
) -> str:
    """Esporta la tabella LaTeX completa con errori e ordini di convergenza (EOC) spaziali e temporali."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Coupled MMS verification on $\Omega=(0,1)^2$: spatial ($Q_2$-$Q_1$-$Q_1$) and temporal (L1-Backward Euler) convergence rates at $T=0.5$.}",
        r"\label{tab:mms_convergence}",
        r"\small",
        r"\begin{tabular}{ccccccccccccc}",
        r"\toprule",
        r"$h$ & $\|\mathbf{e}_{\mathbf{d}}\|_{L^2}$ & EOC & $\|\mathbf{e}_{\mathbf{d}}\|_{H^1}$ & EOC & "
        r"$\|e_p\|_{L^2}$ & EOC & $\|e_p\|_{H^1}$ & EOC & $\|e_c\|_{L^2}$ & EOC & $\|e_c\|_{H^1}$ & EOC \\",
        r"\midrule",
    ]

    for i, r in enumerate(spatial_reports):
        if i == 0:
            lines.append(
                f"{r.h_mesh:.4f} & {r.d_L2:.2e} & -- & {r.d_H1:.2e} & -- & "
                f"{r.p_L2:.2e} & -- & {r.p_H1:.2e} & -- & {r.c_L2:.2e} & -- & {r.c_H1:.2e} & -- \\\\"
            )
        else:
            prev = spatial_reports[i - 1]
            lh = math.log(prev.h_mesh / r.h_mesh)
            rd_l2 = math.log(prev.d_L2 / r.d_L2) / lh
            rd_h1 = math.log(prev.d_H1 / r.d_H1) / lh
            rp_l2 = math.log(prev.p_L2 / r.p_L2) / lh
            rp_h1 = math.log(prev.p_H1 / r.p_H1) / lh
            rc_l2 = math.log(prev.c_L2 / r.c_L2) / lh
            rc_h1 = math.log(prev.c_H1 / r.c_H1) / lh
            lines.append(
                f"{r.h_mesh:.4f} & {r.d_L2:.2e} & {rd_l2:.2f} & {r.d_H1:.2e} & {rd_h1:.2f} & "
                f"{r.p_L2:.2e} & {rp_l2:.2f} & {r.p_H1:.2e} & {rp_h1:.2f} & "
                f"{r.c_L2:.2e} & {rc_l2:.2f} & {r.c_H1:.2e} & {rc_h1:.2f} \\\\"
            )

    lines.extend([
        r"\midrule",
        r"$\Delta t$ & $\|\mathbf{e}_{\mathbf{d}}\|_{L^2}$ & EOC$_t$ & & & "
        r"$\|e_p\|_{L^2}$ & EOC$_t$ & & & $\|e_c\|_{L^2}$ & EOC$_t$ & & \\",
        r"\midrule",
    ])

    for i, r in enumerate(temporal_reports):
        if i == 0:
            lines.append(
                f"{r.dt_max:.4f} & {r.d_L2:.2e} & -- & & & "
                f"{r.p_L2:.2e} & -- & & & {r.c_L2:.2e} & -- & & \\\\"
            )
        else:
            prev = temporal_reports[i - 1]
            ldt = math.log(prev.dt_max / r.dt_max)
            rd_t = math.log(prev.d_L2 / r.d_L2) / ldt
            rp_t = math.log(prev.p_L2 / r.p_L2) / ldt
            rc_t = math.log(prev.c_L2 / r.c_L2) / ldt
            lines.append(
                f"{r.dt_max:.4f} & {r.d_L2:.2e} & {rd_t:.2f} & & & "
                f"{r.p_L2:.2e} & {rp_t:.2f} & & & {r.c_L2:.2e} & {rc_t:.2f} & & \\\\"
            )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    content = "\n".join(lines)
    out = Path(output_tex_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return content


def plot_mms_complete_convergence(
    spatial_reports: Sequence[_core.MMSErrorReport],
    temporal_reports: Sequence[_core.MMSErrorReport],
    output_path: str | Path,
) -> None:
    """
    Figura 1 (3 pannelli):
    (a) Convergenza spaziale in norma L^2(Omega)
    (b) Convergenza spaziale in norma H^1(Omega)
    (c) Convergenza temporale in norma L^2(Omega) al variare di Delta t
    """
    h = np.array([r.h_mesh for r in spatial_reports])
    d_l2 = np.array([r.d_L2 for r in spatial_reports])
    p_l2 = np.array([r.p_L2 for r in spatial_reports])
    c_l2 = np.array([r.c_L2 for r in spatial_reports])

    d_h1 = np.array([r.d_H1 for r in spatial_reports])
    p_h1 = np.array([r.p_H1 for r in spatial_reports])
    c_h1 = np.array([r.c_H1 for r in spatial_reports])

    dt = np.array([r.dt_max for r in temporal_reports])
    dt_d = np.array([r.d_L2 for r in temporal_reports])
    dt_p = np.array([r.p_L2 for r in temporal_reports])
    dt_c = np.array([r.c_L2 for r in temporal_reports])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 4.4), dpi=180)

    # Pannello (a): L2 spaziale
    ax1.loglog(h, d_l2, "o-", lw=2, color="#1f77b4", label=r"$\|\mathbf{d}_h - \mathbf{d}^*\|_{L^2}$ ($Q_2$)")
    ax1.loglog(h, p_l2, "s-", lw=2, color="#d62728", label=r"$\|p_h - p^*\|_{L^2}$ ($Q_1$)")
    ax1.loglog(h, c_l2, "^-", lw=2, color="#2ca02c", label=r"$\|c_h - c^*\|_{L^2}$ ($Q_1$)")
    ax1.loglog(h, d_l2[0] * (h / h[0]) ** 3, "k--", alpha=0.55, label=r"$\mathcal{O}(h^3)$ slope")
    ax1.loglog(h, p_l2[0] * (h / h[0]) ** 2, "k:", alpha=0.65, label=r"$\mathcal{O}(h^2)$ slope")
    ax1.set_xlabel(r"Element diameter $h$")
    ax1.set_ylabel(r"$L^2(\Omega)$ Error at $t=T$")
    ax1.set_title(r"(a) Spatial Convergence in $L^2(\Omega)$")
    ax1.grid(True, which="both", ls=":", alpha=0.6)
    ax1.legend(loc="best")

    # Pannello (b): H1 spaziale
    ax2.loglog(h, d_h1, "o-", lw=2, color="#1f77b4", label=r"$\|\mathbf{d}_h - \mathbf{d}^*\|_{H^1}$ ($Q_2$)")
    ax2.loglog(h, p_h1, "s-", lw=2, color="#d62728", label=r"$\|p_h - p^*\|_{H^1}$ ($Q_1$)")
    ax2.loglog(h, c_h1, "^-", lw=2, color="#2ca02c", label=r"$\|c_h - c^*\|_{H^1}$ ($Q_1$)")
    ax2.loglog(h, d_h1[0] * (h / h[0]) ** 2, "k--", alpha=0.55, label=r"$\mathcal{O}(h^2)$ slope")
    ax2.loglog(h, p_h1[0] * (h / h[0]) ** 1, "k:", alpha=0.65, label=r"$\mathcal{O}(h)$ slope")
    ax2.set_xlabel(r"Element diameter $h$")
    ax2.set_ylabel(r"$H^1(\Omega)$ Error at $t=T$")
    ax2.set_title(r"(b) Spatial Convergence in $H^1(\Omega)$")
    ax2.grid(True, which="both", ls=":", alpha=0.6)
    ax2.legend(loc="best")

    # Pannello (c): Temporale L2
    ax3.loglog(dt, dt_d, "o-", lw=2, color="#1f77b4", label=r"$\|\mathbf{d}_h - \mathbf{d}^*\|_{L^2}$")
    ax3.loglog(dt, dt_p, "s-", lw=2, color="#d62728", label=r"$\|p_h - p^*\|_{L^2}$")
    ax3.loglog(dt, dt_c, "^-", lw=2, color="#2ca02c", label=r"$\|c_h - c^*\|_{L^2}$")
    ax3.loglog(dt, dt_p[0] * (dt / dt[0]) ** 1, "k--", alpha=0.6, label=r"$\mathcal{O}(\Delta t)$ slope")
    ax3.set_xlabel(r"Time step size $\Delta t$")
    ax3.set_ylabel(r"$L^2(\Omega)$ Error at $t=T$")
    ax3.set_title("(c) Temporal Convergence (L1 / Backward Euler)")
    ax3.grid(True, which="both", ls=":", alpha=0.6)
    ax3.legend(loc="best")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

def plot_energy_and_mass_diagnostics(
    diagnostics: Sequence[_core.StepDiagnostics],
    dt_refinement_residuals: Sequence[tuple[float, float]],
    output_path: str | Path,
) -> None:
    """
    Figura di verifica per l'esperimento MMS/omogeneo 01 (3 pannelli):
    (a) Decadimento dell'energia E(t_n) e accumulo delle dissipazioni D_beta(t_n) e D_Darcy(t_n)
    (b) Identità energetica discreta esatta al passo e convergenza O(dt) del residuo |R_N|
    (c) Conservazione della massa del tracciante M(t_n) = M(0)
    """
    if len(dt_refinement_residuals) == 0:
        raise ValueError("dt_refinement_residuals non può essere vuoto in plot_energy_and_mass_diagnostics.")

    t = np.array([d.time for d in diagnostics], dtype=np.float64)
    e_el = np.array([d.elastic_energy for d in diagnostics], dtype=np.float64)
    e_fl = np.array([d.fluid_storage_energy for d in diagnostics], dtype=np.float64)
    e_tot = np.array([d.total_energy for d in diagnostics], dtype=np.float64)
    d_visco = np.array([d.cumulative_visco_dissip for d in diagnostics], dtype=np.float64)
    d_darcy = np.array([d.cumulative_darcy_dissip for d in diagnostics], dtype=np.float64)
    id_err = np.array([d.step_discrete_identity_error for d in diagnostics[1:]], dtype=np.float64)
    mass = np.array([d.tracer_total_mass for d in diagnostics], dtype=np.float64)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.8, 4.3), dpi=180)

    ax1.plot(t, e_tot, "b-", lw=2.2, label=r"Total Energy $\mathcal{E}(t_n)$")
    ax1.plot(t, e_el, "c--", lw=1.4, label=r"Elastic $\frac{1}{2}a_E(\mathbf{d}_h,\mathbf{d}_h)$")
    ax1.plot(t, e_fl, "m--", lw=1.4, label=r"Fluid Storage $\frac{S_p}{2}\|p_h\|_{L^2}^2$")
    ax1.plot(t, d_visco, "r-.", lw=1.8, label=r"Fractional Viscoelastic $\mathcal{D}_\beta(t_n)$")
    ax1.plot(t, d_darcy, "g:", lw=2.0, label=r"Darcy Hydraulic $\int_0^{t_n} a_h(p_h,p_h)\,ds$")
    ax1.set_xlabel(r"Dimensionless Time $t$")
    ax1.set_ylabel("Energy / Cumulative Dissipation")
    ax1.set_title("(a) Poro-Viscoelastic Energy Decay (Prop. 1)")
    ax1.grid(True, ls=":", alpha=0.6)
    ax1.legend(loc="best", fontsize=8)

    dts = np.array([item[0] for item in dt_refinement_residuals], dtype=np.float64)
    res_R = np.array([item[1] for item in dt_refinement_residuals], dtype=np.float64)
    ax2.loglog(dts, res_R, "ro-", lw=2, label=r"Cumulative Euler Dissipation $|R_N(\Delta t)|$")
    ax2.loglog(dts, res_R[0] * (dts / dts[0]), "k--", alpha=0.6, label=r"$\mathcal{O}(\Delta t)$ reference")
    max_id_err = float(np.max(id_err)) if id_err.size > 0 else 0.0
    ax2.text(
        0.05,
        0.12,
        f"Max stepwise identity error:\n"
        r"$\max_n |(R_n - R_{n-1}) + D_{\mathrm{num},n}|$"
        f"\n$= {max_id_err:.2e}$ (machine zero)",
        transform=ax2.transAxes,
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f8f9fa", "edgecolor": "#adb5bd"},
    )
    ax2.set_xlabel(r"Time step size $\Delta t$")
    ax2.set_ylabel(r"Final Energy Discrepancy $|R_N|$")
    ax2.set_title("(b) Discrete Energy Identity Convergence")
    ax2.grid(True, which="both", ls=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=8)

    rel_mass_err = np.abs(mass - mass[0]) / max(abs(mass[0]), 1.0e-15)
    ax3.semilogy(t, np.maximum(rel_mass_err, 1.0e-16), "o-", color="#6f42c1", markersize=3.5, lw=1.6)
    ax3.axhline(1.0e-12, color="k", ls="--", alpha=0.5, label=r"Tolerance threshold $10^{-12}$")
    ax3.set_ylim(1.0e-16, 1.0e-10)
    ax3.set_xlabel(r"Dimensionless Time $t$")
    ax3.set_ylabel(r"Relative Mass Drift $|M(t_n) - M(0)| / M(0)$")
    ax3.set_title(f"(c) Tracer Mass Conservation ($M(0)={mass[0]:.4f}$)")
    ax3.grid(True, which="both", ls=":", alpha=0.6)
    ax3.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    
def plot_fractional_orders_sensitivity(
    alpha_curves: Dict[float, tuple[np.ndarray, np.ndarray]],
    beta_curves: Dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: str | Path,
) -> None:
    """
    Figura 3 (2 pannelli):
    (a) Effetto dell'ordine frazionario di trasporto alpha sul decadimento del picco di concentrazione max_x c(x,t)
    (b) Effetto dell'ordine viscoelastico beta sulla dinamica della pressione porale e del flusso di Darcy
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4), dpi=180)

    for alpha_val, (t_arr, c_peak) in sorted(alpha_curves.items()):
        ax1.plot(t_arr, c_peak, lw=2, label=rf"$\alpha = {alpha_val:.2f}$")
    ax1.set_xlabel(r"Dimensionless Time $t$")
    ax1.set_ylabel(r"Peak Tracer Concentration $\max_{\mathbf{x}\in\Omega} c_h(\mathbf{x}, t)$")
    ax1.set_title(r"(a) Anomalous Trapping Memory ($\alpha \in (0,1)$)")
    ax1.grid(True, ls=":", alpha=0.6)
    ax1.legend()

    for beta_val, (t_arr, p_rms, _) in sorted(beta_curves.items()):
        ax2.plot(t_arr, p_rms, lw=2, label=rf"$\beta = {beta_val:.2f}$")
    ax2.set_xlabel(r"Dimensionless Time $t$")
    ax2.set_ylabel(r"Spatial RMS Interstitial Pressure $\|p_h(\cdot, t)\|_{L^2}$")
    ax2.set_title(r"(b) Fractional Kelvin–Voigt Matrix Memory ($\beta \in (0,1)$)")
    ax2.grid(True, ls=":", alpha=0.6)
    ax2.legend()

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_multiphysics_fields_and_brats_overlay(
    disp_pts: np.ndarray,
    press_pts: np.ndarray,
    tracer_pts: np.ndarray,
    d_final: np.ndarray,
    p_final: np.ndarray,
    c_final: np.ndarray,
    K_h: float,
    output_path: str | Path,
    brats_data: Dict | None = None,
) -> None:
    """
    Figura 4 (4 pannelli):
    (a) Sezione MRI BraTS T1/FLAIR in coordinate fisiche [mm] con segmentazione dell'edema peritumorale
        nella sostanza bianca (label 2) e del core tumorale (label 1, 4) da `BraTS20_Training_001_seg.nii`
    (b) Campo di spostamento della matrice solida ||d_h(x, T)||
    (c) Pressione interstiziale p_h(x, T) + campo vettoriale del flusso di Darcy q_h = -K_h grad(p_h)
    (d) Concentrazione del tracciante passivo c_h(x, T)
    """
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.1), dpi=180)

    # Pannello (a): Anatomia e patologia BraTS reale (se disponibile)
    ax0 = axes[0]
    if brats_data is not None:
        t1_slice = brats_data["t1_slice"]
        seg_slice = brats_data["seg_slice"]
        extent_mm = brats_data["extent_mm"]
        ax0.imshow(t1_slice.T, cmap="gray", origin="lower", extent=extent_mm)
        # Sovrapposizione dell'edema peritumorale (seg == 2) e del core tumorale (seg == 1 o 4)
        edema_mask = np.ma.masked_where(seg_slice.T != 2, seg_slice.T)
        tumor_mask = np.ma.masked_where((seg_slice.T != 1) & (seg_slice.T != 4), seg_slice.T)
        ax0.imshow(edema_mask, cmap="Blues", alpha=0.45, origin="lower", extent=extent_mm)
        ax0.imshow(tumor_mask, cmap="Reds", alpha=0.60, origin="lower", extent=extent_mm)
        c_mm = brats_data["lesion_centroid_mm"]
        ax0.plot(c_mm[0], c_mm[1], "y*", markersize=10, label="Lesion Centroid")
        ax0.set_xlabel(r"$x_{\mathrm{phys}}$ [mm]")
        ax0.set_ylabel(r"$y_{\mathrm{phys}}$ [mm]")
        ax0.set_title("(a) BraTS T1 + Edema/Tumor Mask")
        ax0.legend(loc="upper right", fontsize=7.5)
    else:
        ax0.text(0.5, 0.5, "BraTS NIfTI not found", ha="center", va="center")
        ax0.set_title("(a) Anatomical Domain")

    # Pannello (b): Magnitudine dello spostamento solido ||d_h||
    # In deal.II FESystem(FE_Q, 2), i DOFs pari e dispari sui nodi support_points:
    # raggruppiamo per coordinate uniche dei punti di supporto di d
    ax1 = axes[1]
    dx = d_final[0::2]
    dy = d_final[1::2]
    pts_d_unique = disp_pts[0::2]
    d_mag = np.hypot(dx, dy)
    tri_d = mtri.Triangulation(pts_d_unique[:, 0], pts_d_unique[:, 1])
    im1 = ax1.tricontourf(tri_d, d_mag, levels=20, cmap="viridis")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_xlabel(r"Dimensionless $x^*$")
    ax1.set_ylabel(r"Dimensionless $y^*$")
    ax1.set_title(r"(b) Matrix Displacement $\Vert{}\mathbf{d}_h(\mathbf{x}, T)\Vert{}$")
    ax1.set_aspect("equal")

    # Pannello (c): Pressione porale p_h + frecce del flusso di Darcy q_h = -K_h grad(p_h)
    ax2 = axes[2]
    tri_p = mtri.Triangulation(press_pts[:, 0], press_pts[:, 1])
    im2 = ax2.tricontourf(tri_p, p_final, levels=20, cmap="coolwarm")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Calcolo di q_h su griglia regolare per quiver plot
    interp_p = mtri.LinearTriInterpolator(tri_p, p_final)
    gx, gy = np.meshgrid(np.linspace(0.08, 0.92, 11), np.linspace(0.08, 0.92, 11))
    dp_dx, dp_dy = interp_p.gradient(gx, gy)
    qx = -K_h * dp_dx
    qy = -K_h * dp_dy
    ax2.quiver(gx, gy, qx, qy, color="k", alpha=0.75, scale=None)
    ax2.set_xlabel(r"Dimensionless $x^*$")
    ax2.set_ylabel(r"Dimensionless $y^*$")
    ax2.set_title(r"(c) Pore Pressure $p_h$ & Darcy Flux $\mathbf{q}_h$")
    ax2.set_aspect("equal")

    # Pannello (d): Concentrazione del tracciante c_h
    ax3 = axes[3]
    tri_c = mtri.Triangulation(tracer_pts[:, 0], tracer_pts[:, 1])
    im3 = ax3.tricontourf(tri_c, c_final, levels=20, cmap="magma")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_xlabel(r"Dimensionless $x^*$")
    ax3.set_ylabel(r"Dimensionless $y^*$")
    ax3.set_title(r"(d) Passive Tracer Concentration $c_h(\mathbf{x}, T)$")
    ax3.set_aspect("equal")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

def plot_anatomical_quad_trajectory_snapshots(
    problem: Any,
    traj: Any,
    output_path: str | Path,
    title_prefix: str = "Anatomical Parenchyma Proxy (Quad-Mesh Native Connectivity)",
) -> None:
    """
    Visualizza la soluzione FEM (concentrazione c_h, pressione p_h, modulo spostamento |d_h|)
    usando ESCLUSIVAMENTE la connettività quadrilatera originale della Triangulation<2>
    (nessuna ricostruzione Delaunay dai punti di supporto).
    """
    from matplotlib.collections import PolyCollection

    verts = problem.mesh_vertices                     # (N_v, 2) in unità L0
    quads_v = problem.cell_quad_connectivity          # (N_cells, 4) CCW
    quads_c = problem.cell_tracer_dof_connectivity    # (N_cells, 4) CCW
    polys = verts[quads_v]

    # Valori medi per cella calcolati sui 4 DoF Q_1 ai vertici della cella quadrilatera
    c_init_cell = np.mean(traj.c_hist[0][quads_c], axis=1)
    c_final_cell = np.mean(traj.c_hist[-1][quads_c], axis=1)
    p_mid_idx = max(1, len(traj.times) // 2)
    p_mid_cell = np.mean(traj.p_hist[p_mid_idx][quads_c], axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
    fig.suptitle(
        f"{title_prefix}\n"
        f"Cells={problem.num_active_cells} quads | DoFs: d={problem.num_disp_dofs}, "
        f"p={problem.num_press_dofs}, c={problem.num_tracer_dofs} | "
        f"BC: d=0 on outer ID=1 (Model Hypothesis), traction-free on inner ID=2",
        fontsize=9.5,
        fontweight="bold",
    )

    panels = [
        (c_init_cell, "inferno", f"Tracer $c_h(x^*, t=0)$"),
        (p_mid_cell, "coolwarm", f"Pore Pressure $p_h(x^*, t={traj.times[p_mid_idx]:.2f})$"),
        (c_final_cell, "inferno", f"Tracer $c_h(x^*, t={traj.times[-1]:.2f})$"),
    ]

    x_min, x_max = float(np.min(verts[:, 0])), float(np.max(verts[:, 0]))
    y_min, y_max = float(np.min(verts[:, 1])), float(np.max(verts[:, 1]))
    pad = 0.25

    for ax, (vals, cmap_name, subtitle) in zip(axes, panels):
        pc = PolyCollection(
            polys,
            array=vals,
            cmap=cmap_name,
            edgecolors=(0.15, 0.15, 0.15, 0.22),
            linewidths=0.30,
        )
        ax.add_collection(pc)
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_aspect("equal")
        ax.set_title(subtitle, fontsize=10, fontweight="bold")
        ax.set_xlabel(r"$x^* = (x - x_0)/L_0$ [-]")
        ax.set_ylabel(r"$y^* = (y - y_0)/L_0$ [-]")
        fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=300)
    plt.close(fig)