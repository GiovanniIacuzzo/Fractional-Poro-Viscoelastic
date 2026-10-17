"""
python/reteco/visualization/anatomical_forward_plots.py
Figure scientifiche per `experiments/02_run_anatomical_forward.py`.
Rispetta rigorosamente:
- uso esclusivo della connettività quadrilatera nativa (nessuna triangolazione Delaunay);
- distinzione esplicita tra `brain_parenchyma_proxy` e sostanza bianca segmentata;
- distinzione tra coordinate/unità adimensionali e fisiche dimensionali;
- scale cromatiche condivise sull'evoluzione temporale;
- serie temporali reali calcolate ad ogni passo temporale (nessuna curva ricostruita da massimi).
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
import numpy as np


def _domain_tag_label(domain_rep: str) -> str:
    if domain_rep == "white_matter_segmented":
        return "Independent White Matter Segmentation"
    return "Brain Parenchyma Proxy (2D Axial Slice — No Independent WM Mask)"


def _cell_quad_values_from_q1(nodal_q1: np.ndarray, dof_quads: np.ndarray) -> np.ndarray:
    return np.mean(nodal_q1[dof_quads], axis=1)


def _compute_vertex_disp_magnitude_from_dofs(
    d_vec: np.ndarray, disp_pts: np.ndarray, mesh_vertices: np.ndarray
) -> np.ndarray:
    """
    Estrae il modulo dello spostamento ||d_h|| sui vertici della mesh quadrilatera
    associando ciascun vertice al corrispondente punto di supporto di [Q_2]^2.
    """
    ux = d_vec[0::2]
    uy = d_vec[1::2]
    u_pts = disp_pts[0::2]
    out = np.zeros(mesh_vertices.shape[0], dtype=np.float64)
    for v_idx in range(mesh_vertices.shape[0]):
        d2 = np.sum((u_pts - mesh_vertices[v_idx]) ** 2, axis=1)
        k = int(np.argmin(d2))
        out[v_idx] = float(np.hypot(ux[k], uy[k]))
    return out


# =============================================================================
# Figura 1: Controllo multimodale dell'anatomia, annotazioni BraTS e sorgente
# =============================================================================

def plot_fig01_multimodal_anatomy_and_source(
    volumes: Any,
    slice_sel: Any,
    masks: Any,
    transform: Any,
    mesh: Any,
    source_verification: Dict[str, Any],
    output_path: str | Path,
) -> None:
    corners_ij = np.array(
        [[0.0, 0.0], [float(slice_sel.seg_2d.shape[0] - 1), float(slice_sel.seg_2d.shape[1] - 1)]]
    )
    corners_mm = transform.voxel_ij_to_physical_mm(corners_ij)
    extent_mm = [
        float(np.min(corners_mm[:, 0])),
        float(np.max(corners_mm[:, 0])),
        float(np.min(corners_mm[:, 1])),
        float(np.max(corners_mm[:, 1])),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), constrained_layout=True)
    dom_str = _domain_tag_label(masks.domain_representation_type)
    fig.suptitle(
        f"Controllo Multimodale BraTS 2020 e Verifica Posizione Sorgente Sintetica\n"
        f"Soggetto: {volumes.subject_id} | Orientamento: {volumes.orientation_convention} | "
        f"Sezione k={slice_sel.slice_index} (z={slice_sel.physical_z_mm:.2f} mm) | {dom_str}",
        fontsize=11,
        fontweight="bold",
    )

    mod_list = [("T1", axes[0, 0]), ("T1ce", axes[0, 1]), ("T2", axes[0, 2]), ("FLAIR", axes[1, 0])]
    for mod_name, ax in mod_list:
        im = ax.imshow(slice_sel.modalities_2d[mod_name].T, cmap="gray", origin="lower", extent=extent_mm)
        ax.set_title(f"MRI {mod_name} (Spazio Fisico RAS+)", fontsize=10, fontweight="bold")
        ax.set_xlabel("x RAS+ [mm]")
        ax.set_ylabel("y RAS+ [mm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    # Pannello (1,1): Annotazioni tumorali BraTS (Label 1, 2, 4)
    ax_seg = axes[1, 1]
    ax_seg.imshow(slice_sel.modalities_2d["FLAIR"].T, cmap="gray", origin="lower", extent=extent_mm)
    rgba = np.zeros((slice_sel.seg_2d.shape[1], slice_sel.seg_2d.shape[0], 4), dtype=np.float64)
    rgba[masks.edema_mask.T] = [0.95, 0.80, 0.15, 0.55]
    rgba[masks.necrotic_non_enhancing_mask.T] = [0.85, 0.15, 0.15, 0.72]
    rgba[masks.enhancing_tumor_mask.T] = [0.10, 0.75, 0.95, 0.78]
    ax_seg.imshow(rgba, origin="lower", extent=extent_mm)
    ax_seg.plot(mesh.outer_contour_mm[:, 0], mesh.outer_contour_mm[:, 1], "lime", lw=1.3)
    handles = [
        Patch(facecolor=(0.85, 0.15, 0.15, 0.75), label="Label 1: NCR/NET"),
        Patch(facecolor=(0.95, 0.80, 0.15, 0.65), label="Label 2: Peritumoral Edema"),
        Patch(facecolor=(0.10, 0.75, 0.95, 0.75), label="Label 4: Enhancing Tumor"),
    ]
    ax_seg.set_title("Annotazioni BraTS (Label {1, 2, 4})", fontsize=10, fontweight="bold")
    ax_seg.set_xlabel("x RAS+ [mm]")
    ax_seg.set_ylabel("y RAS+ [mm]")
    ax_seg.legend(handles=handles, loc="lower left", fontsize=7.5, framealpha=0.9)

    # Pannello (1,2): Verifica coerenza centroide lesione vs sorgente nel dominio
    ax_src = axes[1, 2]
    ax_src.imshow(masks.brain_support_mask.T, cmap="Blues", origin="lower", extent=extent_mm, alpha=0.55)
    ax_src.plot(
        mesh.outer_contour_mm[:, 0],
        mesh.outer_contour_mm[:, 1],
        color="crimson",
        lw=1.6,
        label="Contorno esterno (ID=1)",
    )
    for idx_h, hole_mm in enumerate(mesh.inner_contours_mm):
        lbl = "Contorno interno (ID=2)" if idx_h == 0 else None
        ax_src.plot(hole_mm[:, 0], hole_mm[:, 1], color="darkorange", lw=1.6, ls="--", label=lbl)

    les_mm = source_verification["lesion_centroid_physical_mm"]
    src_mm = source_verification["source_center_physical_mm"]
    sigma_mm = source_verification["fluid_source_sigma_mm"]
    dist_mm = source_verification["distance_source_to_lesion_centroid_mm"]

    ax_src.scatter(
        [les_mm[0]],
        [les_mm[1]],
        color="magenta",
        marker="x",
        s=90,
        linewidths=2.4,
        label=f"Centroide lesione ({les_mm[0]:.1f}, {les_mm[1]:.1f}) mm",
    )
    ax_src.scatter(
        [src_mm[0]],
        [src_mm[1]],
        facecolors="none",
        edgecolors="black",
        marker="o",
        s=110,
        linewidths=2.0,
        label=f"Centro sorgente (dist={dist_mm:.2e} mm)",
    )
    theta_ang = np.linspace(0.0, 2.0 * np.pi, 120)
    ax_src.plot(
        src_mm[0] + sigma_mm * np.cos(theta_ang),
        src_mm[1] + sigma_mm * np.sin(theta_ang),
        "k--",
        lw=1.2,
        label=f"Raggio $\\sigma = {sigma_mm:.1f}$ mm",
    )
    ax_src.set_title("Verifica Sorgente Interna al Dominio", fontsize=10, fontweight="bold")
    ax_src.set_xlabel("x RAS+ [mm]")
    ax_src.set_ylabel("y RAS+ [mm]")
    ax_src.legend(loc="lower left", fontsize=7.5, framealpha=0.9)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=300)
    plt.close(fig)


# =============================================================================
# Figura 2: Maschera e mesh anatomica quadrilatera con metriche di qualità
# =============================================================================

def plot_fig02_mask_and_quad_mesh_quality(
    masks: Any,
    mesh: Any,
    geom_report: Any,
    output_path: str | Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9), constrained_layout=True)
    dom_str = _domain_tag_label(masks.domain_representation_type)
    fig.suptitle(
        f"Geometria e Qualità della Mesh Quadrilatera 2D — {dom_str}\n"
        f"Nodi={geom_report.num_nodes} | Celle={geom_report.num_quad_cells} quad (0 triangoli) | "
        f"Dice={geom_report.rasterized_dice_similarity:.4f} | "
        f"Errore Relativo Area={100.0 * geom_report.relative_area_error:.2f}%",
        fontsize=10.5,
        fontweight="bold",
    )

    # Pannello 0: Mesh in coordinate adimensionali del solver con frontiere ID=1 e ID=2
    ax0 = axes[0]
    polys_adim = mesh.vertices_adim[mesh.quads]
    pc0 = PolyCollection(
        polys_adim,
        facecolors=(0.20, 0.60, 0.95, 0.22),
        edgecolors="steelblue",
        linewidths=0.45,
    )
    ax0.add_collection(pc0)
    for b_idx in range(mesh.boundary_edges.shape[0]):
        u, v = mesh.boundary_edges[b_idx]
        bid = int(mesh.boundary_ids[b_idx])
        col = "crimson" if bid == 1 else "darkorange"
        ax0.plot(
            [mesh.vertices_adim[u, 0], mesh.vertices_adim[v, 0]],
            [mesh.vertices_adim[u, 1], mesh.vertices_adim[v, 1]],
            color=col,
            lw=1.4,
        )
    ax0.set_aspect("equal")
    pad = 0.35
    ax0.set_xlim(float(np.min(mesh.vertices_adim[:, 0])) - pad, float(np.max(mesh.vertices_adim[:, 0])) + pad)
    ax0.set_ylim(float(np.min(mesh.vertices_adim[:, 1])) - pad, float(np.max(mesh.vertices_adim[:, 1])) + pad)
    ax0.set_title("Mesh Quadrilatera su Dominio Adimensionale", fontsize=10, fontweight="bold")
    ax0.set_xlabel(r"$x^* = (x - x_0)/L_0$ [-] ($L_0 = 10$ mm)")
    ax0.set_ylabel(r"$y^* = (y - y_0)/L_0$ [-]")
    ax0.legend(
        handles=[
            Patch(facecolor="none", edgecolor="crimson", label=r"$\Gamma_{\mathrm{ext}}$ (ID=1: $\mathbf{d}=\mathbf{0}$, Model Hyp.)"),
            Patch(facecolor="none", edgecolor="darkorange", label=r"$\Gamma_{\mathrm{int}}$ (ID=2: $\boldsymbol{\sigma}\mathbf{n}=\mathbf{0}$)"),
        ],
        loc="lower left",
        fontsize=7.5,
    )

    # Pannello 1: Mappa spaziale delle aree delle celle in mm^2
    ax1 = axes[1]
    v_mm = mesh.vertices_mm[mesh.quads]
    x_q, y_q = v_mm[:, :, 0], v_mm[:, :, 1]
    cell_areas_mm2 = 0.5 * (
        (x_q[:, 0] * y_q[:, 1] - x_q[:, 1] * y_q[:, 0])
        + (x_q[:, 1] * y_q[:, 2] - x_q[:, 2] * y_q[:, 1])
        + (x_q[:, 2] * y_q[:, 3] - x_q[:, 3] * y_q[:, 2])
        + (x_q[:, 3] * y_q[:, 0] - x_q[:, 0] * y_q[:, 3])
    )
    pc1 = PolyCollection(
        v_mm,
        array=cell_areas_mm2,
        cmap="viridis",
        edgecolors=(0.1, 0.1, 0.1, 0.25),
        linewidths=0.35,
    )
    ax1.add_collection(pc1)
    ax1.set_aspect("equal")
    ax1.set_xlim(float(np.min(mesh.vertices_mm[:, 0])) - 4.0, float(np.max(mesh.vertices_mm[:, 0])) + 4.0)
    ax1.set_ylim(float(np.min(mesh.vertices_mm[:, 1])) - 4.0, float(np.max(mesh.vertices_mm[:, 1])) + 4.0)
    ax1.set_title("Distribuzione Spaziale Area Celle [mm²]", fontsize=10, fontweight="bold")
    ax1.set_xlabel("x RAS+ [mm]")
    ax1.set_ylabel("y RAS+ [mm]")
    cbar1 = fig.colorbar(pc1, ax=ax1, fraction=0.046, pad=0.03)
    cbar1.set_label("Area cella [mm²]")

    # Pannello 2: Istogramma dei rapporti d'aspetto e delle aree
    ax2 = axes[2]
    e0 = np.linalg.norm(v_mm[:, 1] - v_mm[:, 0], axis=1)
    e1 = np.linalg.norm(v_mm[:, 2] - v_mm[:, 1], axis=1)
    e2 = np.linalg.norm(v_mm[:, 3] - v_mm[:, 2], axis=1)
    e3 = np.linalg.norm(v_mm[:, 0] - v_mm[:, 3], axis=1)
    aspect_ratios = np.maximum.reduce([e0, e1, e2, e3]) / np.maximum(
        np.minimum.reduce([e0, e1, e2, e3]), 1.0e-12
    )
    ax2.hist(aspect_ratios, bins=20, color="steelblue", edgecolor="black", alpha=0.85)
    ax2.axvline(float(np.mean(aspect_ratios)), color="crimson", lw=2.0, ls="--", label=f"Media = {np.mean(aspect_ratios):.3f}")
    ax2.set_title("Qualità Elementi: Rapporto d'Aspetto", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Aspect Ratio (max edge / min edge) [-]")
    ax2.set_ylabel("Numero di celle quadrilatere")
    ax2.grid(True, ls=":", alpha=0.6)
    ax2.legend(loc="upper right")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=300)
    plt.close(fig)


# =============================================================================
# Figura 3: Campi multifisici sulla connettività reale della mesh (stato finale)
# =============================================================================

def plot_fig03_multiphysics_fields_on_quad_mesh(
    problem: Any,
    traj: Any,
    scales: Any,
    domain_rep: str,
    output_path: str | Path,
) -> None:
    verts_adim = problem.mesh_vertices
    quads_v = problem.cell_quad_connectivity
    quads_c = problem.cell_tracer_dof_connectivity
    polys_adim = verts_adim[quads_v]

    L0_mm = float(scales.L0_m) * 1000.0
    T0_s = float(scales.T0_s)
    E0_Pa = float(scales.E0_Pa)
    c_ref = float(scales.c_ref_mol_m3)

    t_final_adim = float(traj.times[-1])
    t_final_s = t_final_adim * T0_s

    # 1. Modulo spostamento ||d_h|| su cella (adimensionale e in mm)
    v_disp_adim = _compute_vertex_disp_magnitude_from_dofs(traj.d_hist[-1], problem.disp_pts, verts_adim)
    cell_disp_adim = np.mean(v_disp_adim[quads_v], axis=1)
    cell_disp_mm = cell_disp_adim * L0_mm

    # 2. Pressione interstiziale p_h su cella (adimensionale e in Pa)
    cell_p_adim = _cell_quad_values_from_q1(traj.p_hist[-1], quads_c)
    cell_p_Pa = cell_p_adim * E0_Pa

    # 3. Concentrazione del tracciante c_h su cella (adimensionale e in mol/m^3)
    cell_c_adim = _cell_quad_values_from_q1(traj.c_hist[-1], quads_c)
    cell_c_phys = cell_c_adim * c_ref

    # 4. Flusso di Darcy q_h = -K_h grad(p_h) su cella
    centers_adim = np.asarray(problem.fem.get_cell_centers(), dtype=np.float64)
    q_vec_adim = np.asarray(problem.fem.compute_cell_darcy_fluxes(traj.p_hist[-1]), dtype=np.float64)
    q_mag_adim = np.linalg.norm(q_vec_adim, axis=1)
    q_mag_mm_s = q_mag_adim * (L0_mm / T0_s)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.6), constrained_layout=True)
    dom_str = _domain_tag_label(domain_rep)
    fig.suptitle(
        f"Campi Multifisici a $t = {t_final_adim:.2f}$ ($t_{{\\mathrm{{phys}}}} = {t_final_s:.1f}$ s) "
        f"sulla Connettività Quadrilatera Reale\n"
        f"{dom_str} — Simulazione Sintetica Guidata dall'Anatomia (NON misura clinica)",
        fontsize=10.5,
        fontweight="bold",
    )

    x_min, x_max = float(np.min(verts_adim[:, 0])) - 0.3, float(np.max(verts_adim[:, 0])) + 0.3
    y_min, y_max = float(np.min(verts_adim[:, 1])) - 0.3, float(np.max(verts_adim[:, 1])) + 0.3

    panels = [
        (
            axes[0, 0],
            cell_disp_adim,
            "viridis",
            f"(a) Spostamento $\\|\\mathbf{{d}}_h\\|$ (max={np.max(cell_disp_mm):.3e} mm)",
            r"$\|\mathbf{d}_h\|$ [-]  ($\times 10\text{ mm}$ per [mm])",
        ),
        (
            axes[0, 1],
            cell_p_adim,
            "coolwarm",
            f"(b) Pressione $p_h$ (max={np.max(cell_p_Pa):.2f} Pa)",
            r"$p_h$ [-]  ($\times 1000\text{ Pa}$ per [Pa])",
        ),
        (
            axes[1, 0],
            cell_c_adim,
            "inferno",
            f"(c) Tracciante $c_h$ (max={np.max(cell_c_phys):.3f} mol/m³)",
            r"$c_h$ [-]  ($\times c_{\mathrm{ref}}$ per [mol/m³])",
        ),
        (
            axes[1, 1],
            q_mag_adim,
            "plasma",
            f"(d) Flusso di Darcy $\\|\\mathbf{{q}}_h\\| = \\|-K_h\\nabla p_h\\|$ (max={np.max(q_mag_mm_s):.3e} mm/s)",
            r"$\|\mathbf{q}_h\|$ [-]  ($\times 0.1\text{ mm/s}$)",
        ),
    ]

    for ax, arr, cmap_name, title, cbar_lbl in panels:
        pc = PolyCollection(
            polys_adim,
            array=arr,
            cmap=cmap_name,
            edgecolors=(0.15, 0.15, 0.15, 0.20),
            linewidths=0.30,
        )
        ax.add_collection(pc)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel(r"$x^* = (x - x_0)/L_0$ [-]")
        ax.set_ylabel(r"$y^* = (y - y_0)/L_0$ [-]")
        cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(cbar_lbl, fontsize=8.5)

    # Sovrapposizione del campo direzionale di Darcy sul pannello (d)
    step_q = max(1, centers_adim.shape[0] // 180)
    sel_q = np.arange(0, centers_adim.shape[0], step_q)
    axes[1, 1].quiver(
        centers_adim[sel_q, 0],
        centers_adim[sel_q, 1],
        q_vec_adim[sel_q, 0],
        q_vec_adim[sel_q, 1],
        color="white",
        alpha=0.80,
        scale_units="xy",
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=300)
    plt.close(fig)


# =============================================================================
# Figura 4: Evoluzione temporale dei campi con scale cromatiche confrontabili
# =============================================================================

def plot_fig04_temporal_evolution_comparable_scales(
    problem: Any,
    traj: Any,
    snapshot_indices: Sequence[int],
    scales: Any,
    domain_rep: str,
    output_path: str | Path,
) -> None:
    verts_adim = problem.mesh_vertices
    quads_v = problem.cell_quad_connectivity
    quads_c = problem.cell_tracer_dof_connectivity
    polys_adim = verts_adim[quads_v]

    n_snaps = len(snapshot_indices)
    disp_snaps = []
    press_snaps = []
    tracer_snaps = []

    for idx in snapshot_indices:
        v_d = _compute_vertex_disp_magnitude_from_dofs(traj.d_hist[idx], problem.disp_pts, verts_adim)
        disp_snaps.append(np.mean(v_d[quads_v], axis=1))
        press_snaps.append(_cell_quad_values_from_q1(traj.p_hist[idx], quads_c))
        tracer_snaps.append(_cell_quad_values_from_q1(traj.c_hist[idx], quads_c))

    # Scale cromatiche fisse e condivise su tutte le colonne temporali per ciascun campo
    d_vmin, d_vmax = 0.0, max(float(np.max(disp_snaps)), 1.0e-12)
    p_vmin, p_vmax = float(np.min(press_snaps)), max(float(np.max(press_snaps)), 1.0e-12)
    c_vmin, c_vmax = float(np.min(tracer_snaps)), max(float(np.max(tracer_snaps)), 1.0e-12)

    fig, axes = plt.subplots(3, n_snaps, figsize=(3.9 * n_snaps, 10.2), constrained_layout=True)
    dom_str = _domain_tag_label(domain_rep)
    fig.suptitle(
        f"Evoluzione Spazio-Temporale con Scale Cromatiche Bloccate e Confrontabili\n{dom_str}",
        fontsize=11,
        fontweight="bold",
    )

    x_min, x_max = float(np.min(verts_adim[:, 0])) - 0.25, float(np.max(verts_adim[:, 0])) + 0.25
    y_min, y_max = float(np.min(verts_adim[:, 1])) - 0.25, float(np.max(verts_adim[:, 1])) + 0.25
    T0_s = float(scales.T0_s)

    rows_spec = [
        (0, disp_snaps, "viridis", d_vmin, d_vmax, r"$\|\mathbf{d}_h\|$ [-]"),
        (1, press_snaps, "coolwarm", p_vmin, p_vmax, r"$p_h$ [-]"),
        (2, tracer_snaps, "inferno", c_vmin, c_vmax, r"$c_h$ [-]"),
    ]

    for row_idx, snaps_list, cmap_name, vmin, vmax, row_label in rows_spec:
        last_pc = None
        for col_idx, snap_step in enumerate(snapshot_indices):
            ax = axes[row_idx, col_idx]
            pc = PolyCollection(
                polys_adim,
                array=snaps_list[col_idx],
                cmap=cmap_name,
                clim=(vmin, vmax),
                edgecolors="none",
            )
            ax.add_collection(pc)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal")
            t_adim = float(traj.times[snap_step])
            if row_idx == 0:
                ax.set_title(
                    f"Passo n={snap_step}\n$t={t_adim:.3f}$ ({t_adim * T0_s:.1f} s)",
                    fontsize=9.5,
                    fontweight="bold",
                )
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\n$y^*$ [-]", fontsize=9)
            else:
                ax.set_yticklabels([])
            if row_idx == 2:
                ax.set_xlabel(r"$x^*$ [-]", fontsize=9)
            else:
                ax.set_xticklabels([])
            last_pc = pc

        if last_pc is not None:
            cb = fig.colorbar(last_pc, ax=axes[row_idx, :].tolist(), fraction=0.025, pad=0.02)
            cb.set_label(row_label, fontsize=8.5)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=300)
    plt.close(fig)


# =============================================================================
# Figura 5: Sensibilità della soluzione alla risoluzione della mesh (3 mesh)
# =============================================================================

def plot_fig05_mesh_resolution_sensitivity(
    mesh_results: List[Dict[str, Any]],
    output_path: str | Path,
) -> None:
    """
    Confronta almeno 3 risoluzioni della mesh anatomica (es. h = 4.0, 3.0, 2.0 mm).
    In assenza di soluzione analitica esatta sul dominio anatomico, i risultati
    sono denominati esplicitamente 'Sensibilità alla Risoluzione della Mesh'.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), constrained_layout=True)
    fig.suptitle(
        "Analisi di Sensibilità alla Risoluzione della Mesh sul Dominio Anatomico\n"
        "(Nessuna Soluzione Esatta Assunta — Confronto Multi-Risoluzione delle Osservabili Fisiche)",
        fontsize=11,
        fontweight="bold",
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Riga 0: Visualizzazione delle 3 mesh e della distribuzione finale del tracciante
    for idx, res in enumerate(mesh_results[:3]):
        ax = axes[0, idx]
        verts = res["mesh_vertices_adim"]
        quads = res["cell_quad_connectivity"]
        c_cell = res["final_cell_tracer"]
        pc = PolyCollection(
            verts[quads],
            array=c_cell,
            cmap="inferno",
            edgecolors=(0.2, 0.2, 0.2, 0.30),
            linewidths=0.35,
        )
        ax.add_collection(pc)
        ax.set_aspect("equal")
        ax.set_xlim(float(np.min(verts[:, 0])) - 0.3, float(np.max(verts[:, 0])) + 0.3)
        ax.set_ylim(float(np.min(verts[:, 1])) - 0.3, float(np.max(verts[:, 1])) + 0.3)
        ax.set_title(
            f"Risoluzione Target $h = {res['target_h_mm']:.1f}$ mm\n"
            f"Celle={res['num_cells']} quad | DoF tot={res['total_dofs']}",
            fontsize=9.5,
            fontweight="bold",
        )
        ax.set_xlabel(r"$x^*$ [-]")
        ax.set_ylabel(r"$y^*$ [-]")
        fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)

    # Pannello (1,0): Serie temporale della norma L2(Omega) della concentrazione c_h(t)
    ax_c = axes[1, 0]
    for idx, res in enumerate(mesh_results):
        lbl = f"$h={res['target_h_mm']:.1f}$ mm ({res['num_cells']} quad)"
        ax_c.plot(res["times"], res["c_l2_series"], color=colors[idx % len(colors)], lw=2.0, label=lbl)
    ax_c.set_title(r"Sensibilità Serie Temporale $\|c_h(t)\|_{L^2(\Omega)}$", fontsize=10, fontweight="bold")
    ax_c.set_xlabel("Tempo Adimensionale $t$ [-]")
    ax_c.set_ylabel(r"$\|c_h(t)\|_{L^2(\Omega)}$ [-]")
    ax_c.grid(True, ls=":", alpha=0.6)
    ax_c.legend(fontsize=8)

    # Pannello (1,1): Serie temporale dell'energia meccanico-idraulica totale E(t)
    ax_e = axes[1, 1]
    for idx, res in enumerate(mesh_results):
        lbl = f"$h={res['target_h_mm']:.1f}$ mm"
        ax_e.plot(res["times"], res["energy_series"], color=colors[idx % len(colors)], lw=2.0, label=lbl)
    ax_e.set_title(r"Sensibilità Energia Poro-Viscoelastica $\mathcal{E}(t)$", fontsize=10, fontweight="bold")
    ax_e.set_xlabel("Tempo Adimensionale $t$ [-]")
    ax_e.set_ylabel(r"$\mathcal{E}(t)$ [-]")
    ax_e.grid(True, ls=":", alpha=0.6)
    ax_e.legend(fontsize=8)

    # Pannello (1,2): Profilo spaziale 1D lungo la sezione orizzontale passante per il centro lesione
    ax_prof = axes[1, 2]
    for idx, res in enumerate(mesh_results):
        lbl = f"$h={res['target_h_mm']:.1f}$ mm"
        ax_prof.plot(
            res["profile_x_adim"],
            res["profile_c_final"],
            marker="o",
            markersize=3.0,
            color=colors[idx % len(colors)],
            lw=1.6,
            label=lbl,
        )
    ax_prof.set_title(r"Profilo Orizzontale $c_h(x^*, y^*_{\mathrm{lesion}}, T)$", fontsize=10, fontweight="bold")
    ax_prof.set_xlabel(r"$x^*$ lungo il centroide della lesione [-]")
    ax_prof.set_ylabel(r"$c_h(x^*, y^*_{\mathrm{lesion}}, T)$ [-]")
    ax_prof.grid(True, ls=":", alpha=0.6)
    ax_prof.legend(fontsize=8)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=300)
    plt.close(fig)


# =============================================================================
# Figura 6: Sensibilità ai parametri frazionari alpha e beta (serie temporali vere)
# =============================================================================

def plot_fig06_fractional_parameters_sensitivity(
    alpha_sweep: List[Dict[str, Any]],
    beta_sweep: List[Dict[str, Any]],
    output_path: str | Path,
) -> None:
    """
    Visualizza le serie temporali REALMENTE calcolate passo per passo dal solutore FEM
    al variare di alpha (riga superiore) e di beta (riga inferiore), a parità di
    dominio anatomico, carichi e condizioni al contorno.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 8.8), constrained_layout=True)
    fig.suptitle(
        "Analisi di Sensibilità agli Ordini Frazionari $\\alpha$ (Trasporto) e $\\beta$ (Viscoelasticità)\n"
        "(Serie Temporali Complete Calcolate Passo-Passo sulla Mesh Quadrilatera Anatomica)",
        fontsize=11,
        fontweight="bold",
    )

    cmap_a = plt.cm.plasma(np.linspace(0.15, 0.88, len(alpha_sweep)))
    cmap_b = plt.cm.viridis(np.linspace(0.15, 0.88, len(beta_sweep)))

    # --- RIGA 0: Sensibilità a alpha (con beta, D, E fissi) ---
    for idx, item in enumerate(alpha_sweep):
        t = np.asarray(item["times"], dtype=np.float64)
        lbl = f"$\\alpha = {item['alpha']:.2f}$"
        axes[0, 0].plot(t, item["c_lesion_center_series"], color=cmap_a[idx], lw=2.0, label=lbl)
        axes[0, 1].plot(t, item["c_l2_norm_series"], color=cmap_a[idx], lw=2.0, label=lbl)
        axes[0, 2].plot(
            item["profile_x_adim"],
            item["profile_c_final"],
            color=cmap_a[idx],
            lw=1.8,
            label=lbl,
        )

    axes[0, 0].set_title(r"(a) Tracciante al Centro Lesione $c_h(\mathbf{x}_{\mathrm{les}}, t)$ vs $\alpha$", fontsize=9.5, fontweight="bold")
    axes[0, 0].set_xlabel("Tempo Adimensionale $t$ [-]")
    axes[0, 0].set_ylabel(r"$c_h(\mathbf{x}_{\mathrm{les}}, t)$ [-]")
    axes[0, 0].grid(True, ls=":", alpha=0.6)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_title(r"(b) Norma Spaziale $\|c_h(\cdot, t)\|_{L^2(\Omega)}$ vs $\alpha$", fontsize=9.5, fontweight="bold")
    axes[0, 1].set_xlabel("Tempo Adimensionale $t$ [-]")
    axes[0, 1].set_ylabel(r"$\|c_h(\cdot, t)\|_{L^2(\Omega)}$ [-]")
    axes[0, 1].grid(True, ls=":", alpha=0.6)
    axes[0, 1].legend(fontsize=8)

    axes[0, 2].set_title(r"(c) Profilo Finale $c_h(x^*, y^*_{\mathrm{les}}, T)$ vs $\alpha$", fontsize=9.5, fontweight="bold")
    axes[0, 2].set_xlabel(r"$x^*$ [-]")
    axes[0, 2].set_ylabel(r"$c_h(x^*, y^*_{\mathrm{les}}, T)$ [-]")
    axes[0, 2].grid(True, ls=":", alpha=0.6)
    axes[0, 2].legend(fontsize=8)

    # --- RIGA 1: Sensibilità a beta (con alpha, D, E fissi) ---
    for idx, item in enumerate(beta_sweep):
        t = np.asarray(item["times"], dtype=np.float64)
        lbl = f"$\\beta = {item['beta']:.2f}$"
        axes[1, 0].plot(t, item["disp_l2_norm_series"], color=cmap_b[idx], lw=2.0, label=lbl)
        axes[1, 1].plot(t, item.get("press_l2_norm_series", item["p_l2_norm_series"]), color=cmap_b[idx], lw=2.0, label=lbl)
        axes[1, 2].plot(t, item["visco_dissipation_series"], color=cmap_b[idx], lw=2.0, label=lbl)

    axes[1, 0].set_title(r"(d) Norma Spostamento $\|\mathbf{d}_h(\cdot, t)\|_{L^2(\Omega)}$ vs $\beta$", fontsize=9.5, fontweight="bold")
    axes[1, 0].set_xlabel("Tempo Adimensionale $t$ [-]")
    axes[1, 0].set_ylabel(r"$\|\mathbf{d}_h(\cdot, t)\|_{L^2(\Omega)}$ [-]")
    axes[1, 0].grid(True, ls=":", alpha=0.6)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_title(r"(e) Norma Pressione $\|p_h(\cdot, t)\|_{L^2(\Omega)}$ vs $\beta$", fontsize=9.5, fontweight="bold")
    axes[1, 1].set_xlabel("Tempo Adimensionale $t$ [-]")
    axes[1, 1].set_ylabel(r"$\|p_h(\cdot, t)\|_{L^2(\Omega)}$ [-]")
    axes[1, 1].grid(True, ls=":", alpha=0.6)
    axes[1, 1].legend(fontsize=8)

    axes[1, 2].set_title(r"(f) Dissipazione Viscoelastica $\mathcal{D}_\beta(t)$ vs $\beta$", fontsize=9.5, fontweight="bold")
    axes[1, 2].set_xlabel("Tempo Adimensionale $t$ [-]")
    axes[1, 2].set_ylabel(r"$\mathcal{D}_\beta(t)$ [-]")
    axes[1, 2].grid(True, ls=":", alpha=0.6)
    axes[1, 2].legend(fontsize=8)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=300)
    plt.close(fig)

# =============================================================================
# Figura 7: Diagnostica energetica e di massa su singola traiettoria anatomica
# =============================================================================

def plot_anatomical_energy_and_mass_diagnostics(
    diagnostics: Sequence[Any],
    output_path: str | Path,
) -> None:
    """
    Genera una figura diagnostica a 4 pannelli per una singola traiettoria anatomica
    forzata o non forzata, accettando esclusivamente (diagnostics, output_path):
    (a) Energia iniziale E(0), energia totale E(t_n) e contributi elastico/idraulico;
    (b) Dissipazioni cumulative (viscoelastica D_beta, Darcy D_h) e lavoro esterno W_ext(t_n);
    (c) Residuo del bilancio energetico R_n = (E + D_beta + D_h) - (E(0) + W_ext),
        dissipazione numerica cumulativa di Eulero implicito sum D_{num,m} e difetto
        algebrico dell'identità discreta di passo |(R_n - R_{n-1}) + D_{num,n}|;
    (d) Deriva relativa della massa del tracciante |M(t_n) - M(0)| / M(0).
    """
    if len(diagnostics) == 0:
        raise ValueError("La sequenza `diagnostics` non può essere vuota.")

    t = np.array([d.time for d in diagnostics], dtype=np.float64)
    e_el = np.array([d.elastic_energy for d in diagnostics], dtype=np.float64)
    e_fl = np.array([d.fluid_storage_energy for d in diagnostics], dtype=np.float64)
    e_tot = np.array([d.total_energy for d in diagnostics], dtype=np.float64)
    d_visco = np.array([d.cumulative_visco_dissip for d in diagnostics], dtype=np.float64)
    d_darcy = np.array([d.cumulative_darcy_dissip for d in diagnostics], dtype=np.float64)
    w_ext = np.array([d.cumulative_external_work for d in diagnostics], dtype=np.float64)
    res_R = np.array([d.energy_balance_residual for d in diagnostics], dtype=np.float64)
    step_num_d = np.array([d.step_numerical_dissipation for d in diagnostics], dtype=np.float64)
    cum_num_d = np.cumsum(step_num_d)
    step_id_err = np.array([d.step_discrete_identity_error for d in diagnostics[1:]], dtype=np.float64)
    mass = np.array([d.tracer_total_mass for d in diagnostics], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.8), constrained_layout=True)
    fig.suptitle(
        "Diagnostica Energetica, Lavoro Esterno e Conservazione di Massa sulla Traiettoria Anatomica\n"
        r"($R_n = [\mathcal{E}(t_n) + \mathcal{D}_\beta(t_n) + \mathcal{D}_h(t_n)] - [\mathcal{E}(0) + W_{\mathrm{ext}}(t_n)]$: "
        "in presenza di forzanti l'energia totale non decade monotonicamente)",
        fontsize=10.5,
        fontweight="bold",
    )

    # Pannello (a): Energia iniziale, totale e contributi
    ax_a = axes[0, 0]
    ax_a.axhline(e_tot[0], color="gray", lw=1.2, ls=":", label=rf"Initial Energy $\mathcal{{E}}(0) = {e_tot[0]:.3e}$")
    ax_a.plot(t, e_tot, "b-", lw=2.2, label=r"Total Energy $\mathcal{E}(t_n)$")
    ax_a.plot(t, e_el, "c--", lw=1.6, label=r"Elastic $\mathcal{E}_{\mathrm{el}}(t_n) = \frac{E}{2}\|\mathbf{d}_h\|_{K_0}^2$")
    ax_a.plot(t, e_fl, "m-.", lw=1.6, label=r"Fluid Storage $\mathcal{E}_{\mathrm{fl}}(t_n) = \frac{S_p}{2}\|p_h\|_{M_p}^2$")
    ax_a.set_title("(a) Energia Iniziale, Totale e Contributi", fontsize=10, fontweight="bold")
    ax_a.set_xlabel("Tempo Adimensionale $t$ [-]")
    ax_a.set_ylabel("Energia Adimensionale [-]")
    ax_a.grid(True, ls=":", alpha=0.6)
    ax_a.legend(loc="best", fontsize=8)

    # Pannello (b): Dissipazioni fisiche cumulative e lavoro esterno
    ax_b = axes[0, 1]
    ax_b.plot(t, d_visco, "r-", lw=2.0, label=r"Viscoelastic Dissipation $\mathcal{D}_\beta(t_n)$")
    ax_b.plot(t, d_darcy, "g--", lw=2.0, label=r"Darcy Hydraulic Dissipation $\mathcal{D}_h(t_n)$")
    ax_b.plot(t, w_ext, "k-.", lw=1.8, label=r"Cumulative External Work $W_{\mathrm{ext}}(t_n)$")
    ax_b.set_title("(b) Dissipazioni Fisiche e Lavoro Esterno", fontsize=10, fontweight="bold")
    ax_b.set_xlabel("Tempo Adimensionale $t$ [-]")
    ax_b.set_ylabel("Lavoro / Dissipazione Cumulativa [-]")
    ax_b.grid(True, ls=":", alpha=0.6)
    ax_b.legend(loc="best", fontsize=8)

    # Pannello (c): Residuo energetico continuo vs identità algebrica discreta di passo
    ax_c = axes[1, 0]
    t_steps = t[1:] if t.size > 1 else t
    ax_c.semilogy(
        t,
        np.maximum(np.abs(res_R), 1.0e-18),
        "b-o",
        markersize=3.2,
        lw=1.7,
        label=r"Energy Balance Discrepancy $|R_n|$",
    )
    ax_c.semilogy(
        t,
        np.maximum(cum_num_d, 1.0e-18),
        "c--",
        lw=1.5,
        label=r"Cumul. Implicit Euler Dissip. $\sum_{m=1}^n D_{\mathrm{num},m}$",
    )
    if step_id_err.size > 0:
        ax_c.semilogy(
            t_steps,
            np.maximum(step_id_err, 1.0e-18),
            "r-s",
            markersize=3.0,
            lw=1.4,
            label=r"Algebraic Step Identity Defect $|(R_n - R_{n-1}) + D_{\mathrm{num},n}|$",
        )
    max_id_err = float(np.max(step_id_err)) if step_id_err.size > 0 else 0.0
    ax_c.text(
        0.04,
        0.10,
        f"Max algebraic step defect: {max_id_err:.2e}\n"
        f"Final Euler dissipation |R_N|: {abs(res_R[-1]):.3e}",
        transform=ax_c.transAxes,
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8f9fa", "edgecolor": "#adb5bd"},
    )
    ax_c.set_title("(c) Residuo Energetico $|R_n|$ vs Identità Discreta di Passo", fontsize=10, fontweight="bold")
    ax_c.set_xlabel("Tempo Adimensionale $t$ [-]")
    ax_c.set_ylabel("Magnitudine (scala log) [-]")
    ax_c.grid(True, which="both", ls=":", alpha=0.6)
    ax_c.legend(loc="best", fontsize=7.8)

    # Pannello (d): Conservazione della massa del tracciante
    ax_d = axes[1, 1]
    rel_mass_err = np.abs(mass - mass[0]) / max(abs(mass[0]), 1.0e-15)
    ax_d.semilogy(t, np.maximum(rel_mass_err, 1.0e-16), "o-", color="#6f42c1", markersize=3.5, lw=1.6)
    ax_d.axhline(1.0e-12, color="k", ls="--", alpha=0.55, label=r"Soglia di tolleranza $10^{-12}$")
    ax_d.set_ylim(1.0e-16, 1.0e-10)
    ax_d.set_title(
        f"(d) Deriva Relativa Massa Tracciante ($M(0) = {mass[0]:.4f}$)",
        fontsize=10,
        fontweight="bold",
    )
    ax_d.set_xlabel("Tempo Adimensionale $t$ [-]")
    ax_d.set_ylabel(r"$|M(t_n) - M(0)| / M(0)$ [-]")
    ax_d.grid(True, which="both", ls=":", alpha=0.6)
    ax_d.legend(loc="upper right", fontsize=8)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", dpi=300)
    plt.close(fig)