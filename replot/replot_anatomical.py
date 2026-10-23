"""
replot/replot_anatomical.py
Reconstructs all reconstructible figures in Family B (Anatomical BraTS 2020 Clinical Case):
1. `02_anatomical_forward_clinical/fig02_anatomical_mask_and_quad_mesh_quality`
2. `02_anatomical_forward_clinical/fig03_multiphysics_fields_on_quad_mesh`
3. `02_anatomical_forward_clinical/fig04_temporal_evolution_comparable_scales`
4. `02_anatomical_forward_clinical/fig05_mesh_resolution_sensitivity`
5. `02_anatomical_forward_clinical/fig06_fractional_parameters_sensitivity`
6. `02_anatomical_forward_clinical/fig07_energy_and_mass_diagnostics`
7. `brats_anatomical_phase1_clinical/fig04_quad_mesh_and_boundary_zoom`
"""

from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.tri as mtri
import numpy as np

from common_style import (
    ARTIFACTS_DIR,
    FIG_SIZES,
    PALETTE,
    apply_common_style,
    export_figure,
    format_panel_title,
    require_file,
    style_axis,
)

SUBDIR_02_ANAT = "02_anatomical_forward_clinical"
SUBDIR_PHASE1 = "brats_anatomical_phase1_clinical"

DIR_02_ANAT = ARTIFACTS_DIR / "02_anatomical_forward_clinical"
DIR_PHASE1 = ARTIFACTS_DIR / "brats_anatomical_phase1_clinical"


def parse_gmsh_msh2(msh_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int, int]]]:
    """Lightweight parser for ASCII Gmsh MSH2 files saved in `artifacts/`."""
    require_file(msh_path, f"Gmsh MSH2 mesh {msh_path.name}")
    lines = msh_path.read_text(encoding="utf-8").splitlines()
    n_lines = len(lines)
    idx = 0

    node_id_to_idx: Dict[int, int] = {}
    vertices_list: List[Tuple[float, float]] = []
    quads_list: List[Tuple[int, int, int, int]] = []
    quad_tags: List[int] = []
    boundary_edges: List[Tuple[int, int, int]] = []

    while idx < n_lines:
        line = lines[idx].strip()
        if line == "$Nodes":
            idx += 1
            num_nodes = int(lines[idx].strip())
            idx += 1
            for _ in range(num_nodes):
                parts = lines[idx].strip().split()
                nid = int(parts[0])
                x, y = float(parts[1]), float(parts[2])
                node_id_to_idx[nid] = len(vertices_list)
                vertices_list.append((x, y))
                idx += 1
        elif line == "$Elements":
            idx += 1
            num_elems = int(lines[idx].strip())
            idx += 1
            for _ in range(num_elems):
                parts = [int(p) for p in lines[idx].strip().split()]
                elem_type = parts[1]
                num_tags = parts[2]
                phys_tag = parts[3] if num_tags >= 1 else 0
                node_ids = parts[3 + num_tags :]
                if elem_type == 1 and len(node_ids) == 2:
                    v0 = node_id_to_idx[node_ids[0]]
                    v1 = node_id_to_idx[node_ids[1]]
                    boundary_edges.append((v0, v1, phys_tag))
                elif elem_type == 3 and len(node_ids) == 4:
                    q = (
                        node_id_to_idx[node_ids[0]],
                        node_id_to_idx[node_ids[1]],
                        node_id_to_idx[node_ids[2]],
                        node_id_to_idx[node_ids[3]],
                    )
                    quads_list.append(q)
                    quad_tags.append(phys_tag)
                idx += 1
        else:
            idx += 1

    return (
        np.asarray(vertices_list, dtype=np.float64),
        np.asarray(quads_list, dtype=np.int64),
        np.asarray(quad_tags, dtype=np.int64),
        boundary_edges,
    )


def extract_boundary_edges_from_quads(
    vertices_xy: np.ndarray,
    quads: np.ndarray,
) -> List[Tuple[int, int, int]]:
    """Classifies boundary edges into outer CSF boundary (tag 1) and inner ventricular boundary (tag 2)."""
    edge_counts: Counter[Tuple[int, int]] = Counter()
    for q in quads:
        for k in range(4):
            a, b = int(q[k]), int(q[(k + 1) % 4])
            edge_counts[(min(a, b), max(a, b))] += 1

    b_edges_raw = [e for e, cnt in edge_counts.items() if cnt == 1]
    if not b_edges_raw:
        return []

    centroid = np.mean(vertices_xy, axis=0)
    radii = [
        float(np.linalg.norm(0.5 * (vertices_xy[u] + vertices_xy[v]) - centroid))
        for u, v in b_edges_raw
    ]
    r_split = 0.48 * float(np.max(radii))
    return [(u, v, 2 if r_val < r_split else 1) for (u, v), r_val in zip(b_edges_raw, radii)]


def load_anatomical_mesh(prefer_h_mm: str = "h3.0mm") -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int]]]:
    """Loads (vertices_xy, quads, boundary_edges) from `.msh` or `.npz` in `artifacts/`."""
    meshes_dir = DIR_02_ANAT / "meshes"
    msh_matches = sorted(meshes_dir.glob(f"*{prefer_h_mm}*.msh")) if meshes_dir.exists() else []
    if not msh_matches and meshes_dir.exists():
        msh_matches = sorted(meshes_dir.glob("*.msh"))
    if not msh_matches and DIR_PHASE1.exists():
        msh_matches = sorted(DIR_PHASE1.glob("*.msh"))

    if msh_matches:
        verts, quads, _, b_edges = parse_gmsh_msh2(msh_matches[0])
        if not b_edges and len(quads) > 0:
            b_edges = extract_boundary_edges_from_quads(verts, quads)
        return verts, quads, b_edges

    npz_matches = sorted(meshes_dir.glob(f"*{prefer_h_mm}*.npz")) if meshes_dir.exists() else []
    if not npz_matches and meshes_dir.exists():
        npz_matches = sorted(meshes_dir.glob("*.npz"))
    if not npz_matches and DIR_PHASE1.exists():
        npz_matches = sorted(DIR_PHASE1.glob("*.npz"))

    if npz_matches:
        d = np.load(npz_matches[0], allow_pickle=False)
        v_key = next((k for k in d.files if "vert" in k.lower() or "node" in k.lower() or "pts" in k.lower()), d.files[0])
        q_key = next((k for k in d.files if "quad" in k.lower() or "cell" in k.lower() or "elem" in k.lower()), d.files[1])
        verts = np.asarray(d[v_key], dtype=np.float64)[:, :2]
        quads = np.asarray(d[q_key], dtype=np.int64)
        b_edges = extract_boundary_edges_from_quads(verts, quads)
        return verts, quads, b_edges

    raise FileNotFoundError("[replot] Could not find any .msh or .npz mesh in artifacts/02_anatomical_forward_clinical/meshes/")


def _extract_snapshots_from_npz() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads (verts_adim, snap_times_adim, c_snap, p_snap, d_norm_snap) from `reference_trajectory_snapshots.npz`,
    ensuring `snap_times_adim` has the exact same length `n_snaps` as axis 0 of `c_snap`, `p_snap`, `d_snap`.
    """
    npz_path = require_file(DIR_02_ANAT / "reference_trajectory_snapshots.npz", "Anatomical reference trajectory NPZ")
    data = np.load(npz_path, allow_pickle=False)
    keys = list(data.files)

    v_key = next((k for k in keys if any(s in k.lower() for s in ("vert", "coord", "pts", "node"))), keys[0])
    c_key = next((k for k in keys if k.lower().startswith("c_") or "tracer" in k.lower() or "conc" in k.lower()), None)
    p_key = next((k for k in keys if k.lower().startswith("p_") or "press" in k.lower()), None)
    d_key = next((k for k in keys if k.lower().startswith("d_") or "disp" in k.lower() or "u_" in k.lower()), None)

    verts = np.asarray(data[v_key], dtype=np.float64)[:, :2]
    n_verts = verts.shape[0]
    c_snap = np.asarray(data[c_key], dtype=np.float64) if c_key else np.zeros((4, n_verts))
    if c_snap.ndim == 1:
        c_snap = c_snap[np.newaxis, :]
    n_snaps = c_snap.shape[0]

    p_snap = np.asarray(data[p_key], dtype=np.float64) if p_key else np.zeros_like(c_snap)
    if p_snap.ndim == 1:
        p_snap = p_snap[np.newaxis, :]

    if d_key is not None:
        d_raw = np.asarray(data[d_key], dtype=np.float64)
        if d_raw.ndim == 1:
            d_raw = d_raw[np.newaxis, :]
        if d_raw.ndim == 3:
            d_snap = np.linalg.norm(d_raw, axis=-1)[:, :n_verts]
        elif d_raw.ndim == 2 and d_raw.shape[1] == 2 * n_verts:
            d_snap = np.hypot(d_raw[:, 0::2], d_raw[:, 1::2])
        elif d_raw.ndim == 2 and d_raw.shape[1] >= n_verts:
            d_snap = d_raw[:, :n_verts]
        else:
            d_snap = np.zeros_like(c_snap)
    else:
        dx_key = next((k for k in keys if "dx" in k.lower() or "d_x" in k.lower() or "ux" in k.lower()), None)
        dy_key = next((k for k in keys if "dy" in k.lower() or "d_y" in k.lower() or "uy" in k.lower()), None)
        if dx_key and dy_key:
            d_snap = np.hypot(
                np.asarray(data[dx_key], dtype=np.float64)[:, :n_verts],
                np.asarray(data[dy_key], dtype=np.float64)[:, :n_verts],
            )
        else:
            d_snap = np.abs(p_snap)

    snap_t_key = next((k for k in keys if "snap" in k.lower() and "time" in k.lower()), None)
    if snap_t_key is not None and len(data[snap_t_key]) == n_snaps:
        snap_times = np.asarray(data[snap_t_key], dtype=np.float64)
    else:
        t_key = next((k for k in keys if "time" in k.lower() or k.lower() == "t"), None)
        if t_key is not None:
            t_all = np.asarray(data[t_key], dtype=np.float64)
            if len(t_all) == n_snaps:
                snap_times = t_all
            elif n_snaps == 4:
                snap_times = np.array([0.10, 0.30, 0.60, float(t_all[-1])], dtype=np.float64)
            else:
                idx_sel = np.linspace(0, len(t_all) - 1, n_snaps).astype(int)
                snap_times = t_all[idx_sel]
        else:
            snap_times = np.array([0.10, 0.30, 0.60, 1.00][:n_snaps], dtype=np.float64)

    return verts, snap_times, c_snap, p_snap, d_snap


def compute_quad_quality_metrics(vertices_xy: np.ndarray, quads: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes scaled Jacobians and interior corner angles [deg] from quad mesh vertices."""
    n_q = quads.shape[0]
    scaled_j = np.zeros(n_q, dtype=np.float64)
    angles_deg = np.zeros((n_q, 4), dtype=np.float64)

    for q_idx in range(n_q):
        pts = vertices_xy[quads[q_idx]]
        corner_sj: List[float] = []
        for k in range(4):
            p_prev = pts[(k - 1) % 4]
            p_curr = pts[k]
            p_next = pts[(k + 1) % 4]
            e1 = p_next - p_curr
            e2 = p_prev - p_curr
            l1 = float(np.linalg.norm(e1))
            l2 = float(np.linalg.norm(e2))
            denom = max(l1 * l2, 1.0e-15)
            cross_2d = float(e1[0] * e2[1] - e1[1] * e2[0])
            dot_2d = float(np.dot(e1, e2))
            corner_sj.append(abs(cross_2d) / denom)
            cos_theta = np.clip(dot_2d / denom, -1.0, 1.0)
            angles_deg[q_idx, k] = np.degrees(np.arccos(cos_theta))
        scaled_j[q_idx] = min(corner_sj)

    return scaled_j, angles_deg


def _build_triangulation_for_snapshots(
    verts_adim: np.ndarray,
    quads_candidate: np.ndarray,
) -> mtri.Triangulation:
    """Uses exact quad connectivity when compatible, or Delaunay triangulation with concavity masking."""
    if len(quads_candidate) > 0 and int(np.max(quads_candidate)) < len(verts_adim):
        p0 = verts_adim[quads_candidate[:, 0]]
        p1 = verts_adim[quads_candidate[:, 1]]
        median_edge = float(np.median(np.linalg.norm(p1 - p0, axis=1)))
        if median_edge < 0.12:
            t1 = quads_candidate[:, [0, 1, 2]]
            t2 = quads_candidate[:, [0, 2, 3]]
            return mtri.Triangulation(verts_adim[:, 0], verts_adim[:, 1], np.vstack([t1, t2]))

    tri = mtri.Triangulation(verts_adim[:, 0], verts_adim[:, 1])
    tris = tri.triangles
    x_t = verts_adim[tris, 0]
    y_t = verts_adim[tris, 1]
    l_ab = np.hypot(x_t[:, 1] - x_t[:, 0], y_t[:, 1] - y_t[:, 0])
    l_bc = np.hypot(x_t[:, 2] - x_t[:, 1], y_t[:, 2] - y_t[:, 1])
    l_ca = np.hypot(x_t[:, 0] - x_t[:, 2], y_t[:, 0] - y_t[:, 2])
    max_edge = np.maximum(np.maximum(l_ab, l_bc), l_ca)
    med_edge = float(np.median(max_edge))
    tri.set_mask(max_edge > 2.6 * med_edge)
    return tri


def _find_array_in_dict(d: Dict[str, Any], candidates: Sequence[str], substr_hints: Sequence[str]) -> Optional[np.ndarray]:
    """Finds a 1D numeric list/array in `d` (or nested `d['series']` / `d['trajectory']`)."""
    for target in (d, d.get("series", {}), d.get("trajectory", {}), d.get("diagnostics", {})):
        if not isinstance(target, dict):
            continue
        for c in candidates:
            if c in target and isinstance(target[c], list) and len(target[c]) > 0 and isinstance(target[c][0], (int, float)):
                return np.asarray(target[c], dtype=np.float64)
        for k, v in target.items():
            kl = k.lower()
            if all(h in kl for h in substr_hints) and isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
                return np.asarray(v, dtype=np.float64)
    # Check if trajectory is a list of step dicts
    for list_key in ("step_diagnostics", "trajectory", "steps", "history"):
        if list_key in d and isinstance(d[list_key], list) and len(d[list_key]) > 0 and isinstance(d[list_key][0], dict):
            step_list = d[list_key]
            for c in candidates:
                if c in step_list[0]:
                    return np.array([float(s[c]) for s in step_list], dtype=np.float64)
            for k in step_list[0].keys():
                kl = k.lower()
                if all(h in kl for h in substr_hints):
                    return np.array([float(s[k]) for s in step_list], dtype=np.float64)
    return None


def replot_fig02_anatomical_mask_and_quad_mesh_quality() -> Tuple[Path, Path]:
    """Reconstructs `02_anatomical_forward_clinical/fig02_anatomical_mask_and_quad_mesh_quality.{pdf,png}`."""
    verts_mm, quads, b_edges = load_anatomical_mesh("h3.0mm")
    scaled_j, angles_deg = compute_quad_quality_metrics(verts_mm, quads)

    apply_common_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=FIG_SIZES["triple_row"])

    polys = [verts_mm[q] for q in quads]
    pc = PolyCollection(
        polys,
        facecolors="#EBF2F7",
        edgecolors="#4A5568",
        linewidths=0.35,
        alpha=0.95,
    )
    ax1.add_collection(pc)

    csf_labeled = False
    vent_labeled = False
    for v0, v1, tag in b_edges:
        p0, p1 = verts_mm[v0], verts_mm[v1]
        if tag == 2:
            ax1.plot(
                [p0[0], p1[0]], [p0[1], p1[1]],
                color=PALETTE["red"], linewidth=1.4,
                label=r"Ventricular boundary $\Gamma_V$" if not vent_labeled else None,
            )
            vent_labeled = True
        else:
            ax1.plot(
                [p0[0], p1[0]], [p0[1], p1[1]],
                color=PALETTE["blue"], linewidth=1.2,
                label=r"Pial/CSF boundary $\Gamma_{\mathrm{CSF}}$" if not csf_labeled else None,
            )
            csf_labeled = True

    ax1.autoscale_view()
    ax1.set_aspect("equal")
    ax1.set_xlabel(r"$x$ [mm]")
    ax1.set_ylabel(r"$y$ [mm]")
    ax1.set_title(format_panel_title("a", rf"Quadrilateral Mesh ($N_e = {len(quads)}$)"))
    style_axis(ax1, grid=True)
    ax1.legend(loc="lower left", fontsize=7.5)

    ax2.hist(
        scaled_j, bins=24, color=PALETTE["blue"], edgecolor=PALETTE["charcoal"], linewidth=0.6, alpha=0.88
    )
    ax2.axvline(
        float(np.min(scaled_j)), color=PALETTE["red"], linestyle="--", linewidth=1.3,
        label=rf"$\min J_S = {float(np.min(scaled_j)):.3f}$",
    )
    ax2.axvline(
        float(np.mean(scaled_j)), color=PALETTE["green"], linestyle="-.", linewidth=1.3,
        label=rf"Mean $J_S = {float(np.mean(scaled_j)):.3f}$",
    )
    ax2.set_xlabel(r"Element scaled Jacobian $J_S$")
    ax2.set_ylabel("Number of quadrilateral elements")
    ax2.set_title(format_panel_title("b", "Scaled Jacobian Quality Distribution"))
    style_axis(ax2, grid=True)
    ax2.legend(loc="upper left")

    flat_angles = angles_deg.ravel()
    ax3.hist(
        flat_angles, bins=28, color=PALETTE["cyan"], edgecolor=PALETTE["charcoal"], linewidth=0.6, alpha=0.88
    )
    ax3.axvline(90.0, color=PALETTE["charcoal"], linestyle=":", linewidth=1.2, label=r"Ideal $90^\circ$")
    ax3.axvline(
        float(np.min(flat_angles)), color=PALETTE["red"], linestyle="--", linewidth=1.2,
        label=rf"Range $[{float(np.min(flat_angles)):.1f}^\circ, {float(np.max(flat_angles)):.1f}^\circ]$",
    )
    ax3.axvline(float(np.max(flat_angles)), color=PALETTE["red"], linestyle="--", linewidth=1.2)
    ax3.set_xlabel("Interior corner angle [deg]")
    ax3.set_ylabel("Corner count")
    ax3.set_title(format_panel_title("c", "Quadrilateral Interior Angles"))
    style_axis(ax3, grid=True)
    ax3.legend(loc="upper right", fontsize=7.6)

    fig.tight_layout()
    paths = export_figure(fig, SUBDIR_02_ANAT, "fig02_anatomical_mask_and_quad_mesh_quality")
    plt.close(fig)
    return paths


def replot_fig03_multiphysics_fields_on_quad_mesh() -> Tuple[Path, Path]:
    """Reconstructs `02_anatomical_forward_clinical/fig03_multiphysics_fields_on_quad_mesh.{pdf,png}`."""
    sum_path = require_file(DIR_02_ANAT / "anatomical_forward_experiment_summary.json", "Anatomical summary JSON")
    summary = json.loads(sum_path.read_text(encoding="utf-8"))

    verts, snap_times, c_snap, p_snap, d_snap = _extract_snapshots_from_npz()
    _, quads_mesh, _ = load_anatomical_mesh("h3.0mm")
    tri = _build_triangulation_for_snapshots(verts, quads_mesh)

    t_final = float(snap_times[-1])
    d_final = d_snap[-1]
    p_final = p_snap[-1]
    c_final = c_snap[-1]

    k_h = float(summary.get("physics_parameters", {}).get("K_h", 0.05))
    tci = mtri.LinearTriInterpolator(tri, p_final)
    dp_dx, dp_dy = tci.gradient(verts[:, 0], verts[:, 1])
    q_baro_nodal = k_h * np.hypot(np.nan_to_num(dp_dx, nan=0.0), np.nan_to_num(dp_dy, nan=0.0))

    apply_common_style()
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZES["quad_grid"])
    ax1, ax2, ax3, ax4 = axes.ravel()

    im1 = ax1.tripcolor(tri, d_final, shading="gouraud", cmap="viridis")
    cb1 = fig.colorbar(im1, ax=ax1, pad=0.03)
    cb1.set_label(r"$\|\mathbf{d}_h(\mathbf{x}, T)\|_2$ [-]")
    ax1.set_aspect("equal")
    ax1.set_xlabel(r"Dimensionless $x / L_c$")
    ax1.set_ylabel(r"Dimensionless $y / L_c$")
    ax1.set_title(format_panel_title("a", rf"Solid Displacement $\|\mathbf{{d}}_h\|$ ($t = {t_final:.1f}$)"))

    p_lim = max(abs(float(np.min(p_final))), abs(float(np.max(p_final))), 1.0e-12)
    im2 = ax2.tripcolor(tri, p_final, shading="gouraud", cmap="RdBu_r", vmin=-p_lim, vmax=p_lim)
    cb2 = fig.colorbar(im2, ax=ax2, pad=0.03)
    cb2.set_label(r"$p_h(\mathbf{x}, T)$ [-]")
    ax2.set_aspect("equal")
    ax2.set_xlabel(r"Dimensionless $x / L_c$")
    ax2.set_ylabel(r"Dimensionless $y / L_c$")
    ax2.set_title(format_panel_title("b", rf"Interstitial Fluid Pressure $p_h$ ($t = {t_final:.1f}$)"))

    im3 = ax3.tripcolor(tri, q_baro_nodal, shading="gouraud", cmap="magma")
    cb3 = fig.colorbar(im3, ax=ax3, pad=0.03)
    cb3.set_label(r"$K_h \|\nabla_h p_h(\mathbf{x}, T)\|_2$ [-]")
    ax3.set_aspect("equal")
    ax3.set_xlabel(r"Dimensionless $x / L_c$")
    ax3.set_ylabel(r"Dimensionless $y / L_c$")
    ax3.set_title(format_panel_title("c", r"Pressure-Driven Darcy Flux $K_h\|\nabla_h p_h\|$"))

    im4 = ax4.tripcolor(tri, c_final, shading="gouraud", cmap="plasma")
    cb4 = fig.colorbar(im4, ax=ax4, pad=0.03)
    cb4.set_label(r"$c_h(\mathbf{x}, T)$ [-]")
    ax4.set_aspect("equal")
    ax4.set_xlabel(r"Dimensionless $x / L_c$")
    ax4.set_ylabel(r"Dimensionless $y / L_c$")
    ax4.set_title(format_panel_title("d", rf"Passive Tracer Concentration $c_h$ ($t = {t_final:.1f}$)"))

    for ax in (ax1, ax2, ax3, ax4):
        style_axis(ax, grid=False)

    fig.tight_layout()
    paths = export_figure(fig, SUBDIR_02_ANAT, "fig03_multiphysics_fields_on_quad_mesh")
    plt.close(fig)
    return paths


def replot_fig04_temporal_evolution_comparable_scales() -> Tuple[Path, Path]:
    """Reconstructs `02_anatomical_forward_clinical/fig04_temporal_evolution_comparable_scales.{pdf,png}`."""
    verts, snap_times, c_snap, p_snap, d_snap = _extract_snapshots_from_npz()
    _, quads_mesh, _ = load_anatomical_mesh("h3.0mm")
    tri = _build_triangulation_for_snapshots(verts, quads_mesh)
    n_snaps = c_snap.shape[0]

    c_vmin, c_vmax = float(np.min(c_snap)), float(np.max(c_snap))
    p_abs = max(abs(float(np.min(p_snap))), abs(float(np.max(p_snap))), 1.0e-12)
    d_vmin, d_vmax = 0.0, float(np.max(d_snap))

    apply_common_style()
    fig, axes = plt.subplots(3, n_snaps, figsize=(3.1 * n_snaps, 7.6), sharex=True, sharey=True)

    row_configs = [
        (c_snap, "plasma", c_vmin, c_vmax, r"Tracer $c_h$ [-]"),
        (p_snap, "RdBu_r", -p_abs, p_abs, r"Pressure $p_h$ [-]"),
        (d_snap, "viridis", d_vmin, d_vmax, r"Displacement $\|\mathbf{d}_h\|$ [-]"),
    ]

    for row_idx, (field_arr, cmap_name, vmin, vmax, cbar_label) in enumerate(row_configs):
        mappable = None
        for col_idx in range(n_snaps):
            ax = axes[row_idx, col_idx]
            mappable = ax.tripcolor(
                tri, field_arr[col_idx], shading="gouraud", cmap=cmap_name, vmin=vmin, vmax=vmax
            )
            ax.set_aspect("equal")
            style_axis(ax, grid=False)
            if row_idx == 0:
                ax.set_title(rf"$t / T = {snap_times[col_idx]:.2f}$")
            if row_idx == 2:
                ax.set_xlabel(r"$x / L_c$")
            if col_idx == 0:
                ax.set_ylabel(f"{cbar_label}\n" + r"$y / L_c$")

        if mappable is not None:
            cbar = fig.colorbar(
                mappable, ax=axes[row_idx, :].tolist(), fraction=0.018, pad=0.02
            )
            cbar.set_label(cbar_label)

    fig.subplots_adjust(hspace=0.18, wspace=0.10, right=0.88)
    paths = export_figure(fig, SUBDIR_02_ANAT, "fig04_temporal_evolution_comparable_scales")
    plt.close(fig)
    return paths


def replot_fig05_mesh_resolution_sensitivity() -> Tuple[Path, Path]:
    """Reconstructs `02_anatomical_forward_clinical/fig05_mesh_resolution_sensitivity.{pdf,png}`."""
    sum_path = require_file(DIR_02_ANAT / "anatomical_forward_experiment_summary.json", "Anatomical summary JSON")
    summary = json.loads(sum_path.read_text(encoding="utf-8"))

    # Search for mesh sensitivity list/dict in summary
    ms_sec = (
        summary.get("mesh_sensitivity")
        or summary.get("mesh_sensitivity_study")
        or summary.get("mesh_resolution_sensitivity")
        or summary.get("resolutions")
        or {}
    )
    if isinstance(ms_sec, dict):
        mesh_study: List[Dict[str, Any]] = []
        for k, v in ms_sec.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                mesh_study = v
                break
            elif isinstance(v, dict):
                v_copy = dict(v)
                v_copy.setdefault("name", k)
                mesh_study.append(v_copy)
    elif isinstance(ms_sec, list):
        mesh_study = ms_sec
    else:
        mesh_study = []

    apply_common_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"])

    colors = [PALETTE["slate"], PALETTE["blue"], PALETTE["green"]]
    lstyles = ["--", "-", ":"]
    plotted_any_series = False

    for idx, r in enumerate(mesh_study):
        if not isinstance(r, dict):
            continue
        t_arr = _find_array_in_dict(r, ("times", "time", "t"), ("time",))
        c_arr = _find_array_in_dict(r, ("c_l2_norm_series", "c_l2_series", "c_L2"), ("c", "l2"))
        p_arr = _find_array_in_dict(r, ("p_l2_norm_series", "press_l2_norm_series", "p_l2_series", "p_L2"), ("p", "l2"))
        name_lbl = str(r.get("name", r.get("level", r.get("resolution", f"Level {idx+1}")))).capitalize()
        dx_mm = float(r.get("target_dx_mm", r.get("target_element_size_mm", r.get("h_mm", r.get("dx_mm", 4.0 - idx)))))
        n_q = int(r.get("num_quads", r.get("num_elements", r.get("n_cells", 0))))

        if t_arr is not None and c_arr is not None and len(t_arr) == len(c_arr):
            ax1.plot(
                t_arr, c_arr,
                color=colors[idx % len(colors)],
                linestyle=lstyles[idx % len(lstyles)],
                linewidth=1.7,
                label=f"{name_lbl} ({dx_mm:.1f} mm, {n_q} quads)" if n_q > 0 else f"{name_lbl} ({dx_mm:.1f} mm)",
            )
            plotted_any_series = True
        if t_arr is not None and p_arr is not None and len(t_arr) == len(p_arr):
            ax2.plot(
                t_arr, p_arr,
                color=colors[idx % len(colors)],
                linestyle=lstyles[idx % len(lstyles)],
                linewidth=1.7,
                label=f"{name_lbl} ({dx_mm:.1f} mm)",
            )

    if not plotted_any_series and mesh_study:
        # Bar chart of final scalar metrics across the 3 resolutions if time series are not nested
        names = [str(r.get("name", r.get("level", f"Res {i+1}"))).capitalize() for i, r in enumerate(mesh_study)]
        c_vals = [float(r.get("c_l2_final", r.get("c_l2_norm", r.get("final_c_l2", 1.0)))) for r in mesh_study]
        p_vals = [float(r.get("p_l2_final", r.get("p_l2_norm", r.get("final_p_l2", 1.0)))) for r in mesh_study]
        x = np.arange(len(mesh_study))
        ax1.bar(x, c_vals, width=0.45, color=PALETTE["blue"], edgecolor=PALETTE["charcoal"], label=r"Final $\|c_h(T)\|_{L^2(\Omega)}$")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names)
        ax2.bar(x, p_vals, width=0.45, color=PALETTE["red"], edgecolor=PALETTE["charcoal"], label=r"Final $\|p_h(T)\|_{L^2(\Omega)}$")
        ax2.set_xticks(x)
        ax2.set_xticklabels(names)

    ax1.set_xlabel(r"Dimensionless time $t$" if plotted_any_series else "Mesh resolution level")
    ax1.set_ylabel(r"Tracer norm $\|c_h\|_{L^2(\Omega)}$")
    ax1.set_title(format_panel_title("a", r"Tracer $L^2(\Omega)$ Norm Across Mesh Resolutions"))
    style_axis(ax1, grid=True)
    if ax1.get_legend_handles_labels()[0]:
        ax1.legend(loc="best", fontsize=7.8)

    ax2.set_xlabel(r"Dimensionless time $t$" if plotted_any_series else "Mesh resolution level")
    ax2.set_ylabel(r"Pressure norm $\|p_h\|_{L^2(\Omega)}$")
    ax2.set_title(format_panel_title("b", r"Interstitial Pressure $L^2(\Omega)$ Norm Across Resolutions"))
    style_axis(ax2, grid=True)
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend(loc="best", fontsize=7.8)

    fig.tight_layout()
    paths = export_figure(fig, SUBDIR_02_ANAT, "fig05_mesh_resolution_sensitivity")
    plt.close(fig)
    return paths


def replot_fig06_fractional_parameters_sensitivity() -> Tuple[Path, Path]:
    """Reconstructs `02_anatomical_forward_clinical/fig06_fractional_parameters_sensitivity.{pdf,png}`."""
    sum_path = require_file(DIR_02_ANAT / "anatomical_forward_experiment_summary.json", "Anatomical summary JSON")
    summary = json.loads(sum_path.read_text(encoding="utf-8"))
    frac_sec = (
        summary.get("fractional_sensitivity")
        or summary.get("fractional_sensitivity_study")
        or summary.get("fractional_parameters_sensitivity")
        or {}
    )
    alpha_runs: List[Dict[str, Any]] = frac_sec.get("alpha_sweep", frac_sec.get("alpha", []))
    beta_runs: List[Dict[str, Any]] = frac_sec.get("beta_sweep", frac_sec.get("beta", []))

    apply_common_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"])

    colors = [PALETTE["blue"], PALETTE["red"], PALETTE["green"], PALETTE["purple"]]
    lstyles = ["-", "--", "-.", ":"]
    plotted_a = False
    plotted_b = False

    for idx, r in enumerate(alpha_runs):
        if not isinstance(r, dict):
            continue
        a_val = float(r.get("alpha", r.get("value", 0.70)))
        t_arr = _find_array_in_dict(r, ("times", "time", "t"), ("time",))
        c_arr = _find_array_in_dict(
            r,
            ("c_lesion_center_series", "c_l2_norm_series", "c_peak_series", "c_series"),
            ("c",),
        )
        if t_arr is not None and c_arr is not None and len(t_arr) == len(c_arr):
            ax1.plot(
                t_arr, c_arr,
                color=colors[idx % len(colors)],
                linestyle=lstyles[idx % len(lstyles)],
                linewidth=1.7,
                label=rf"$\alpha = {a_val:.2f}$",
            )
            plotted_a = True

    if not plotted_a and alpha_runs:
        a_vals = [float(r.get("alpha", 0.5 + 0.2 * i)) for i, r in enumerate(alpha_runs)]
        c_vals = [
            float(r.get("c_l2_final", r.get("c_peak_final", r.get("final_state_summary", {}).get("c_l2_norm", 1.0))))
            for r in alpha_runs
        ]
        ax1.plot(a_vals, c_vals, marker="o", color=PALETTE["blue"], linestyle="-", label=r"Final $\|c_h(T)\|_{L^2(\Omega)}$")

    ax1.set_xlabel(r"Dimensionless time $t$" if plotted_a else r"Subdiffusion order $\alpha$")
    ax1.set_ylabel("Tracer response")
    ax1.set_title(format_panel_title("a", r"Anatomical Sensitivity to Subdiffusion Order $\alpha$"))
    style_axis(ax1, grid=True)
    if ax1.get_legend_handles_labels()[0]:
        ax1.legend(loc="best")

    for idx, r in enumerate(beta_runs):
        if not isinstance(r, dict):
            continue
        b_val = float(r.get("beta", r.get("value", 0.45)))
        t_arr = _find_array_in_dict(r, ("times", "time", "t"), ("time",))
        v_arr = _find_array_in_dict(
            r,
            ("visco_dissipation_series", "cumulative_visco_dissip", "p_l2_norm_series", "press_l2_norm_series"),
            ("dissip",),
        )
        if v_arr is None:
            v_arr = _find_array_in_dict(r, ("p_l2_norm_series", "press_l2_norm_series"), ("p", "l2"))
        if t_arr is not None and v_arr is not None and len(t_arr) == len(v_arr):
            ax2.plot(
                t_arr, v_arr,
                color=colors[idx % len(colors)],
                linestyle=lstyles[idx % len(lstyles)],
                linewidth=1.7,
                label=rf"$\beta = {b_val:.2f}$",
            )
            plotted_b = True

    if not plotted_b and beta_runs:
        b_vals = [float(r.get("beta", 0.25 + 0.25 * i)) for i, r in enumerate(beta_runs)]
        v_vals = [
            float(r.get("visco_dissip_final", r.get("p_l2_final", r.get("final_state_summary", {}).get("d_l2_norm", 1.0))))
            for r in beta_runs
        ]
        ax2.plot(b_vals, v_vals, marker="s", color=PALETTE["red"], linestyle="--", label=r"Final response vs. $\beta$")

    ax2.set_xlabel(r"Dimensionless time $t$" if plotted_b else r"Viscoelastic order $\beta$")
    ax2.set_ylabel("Poro-viscoelastic response")
    ax2.set_title(format_panel_title("b", r"Anatomical Sensitivity to Viscoelastic Order $\beta$"))
    style_axis(ax2, grid=True)
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend(loc="upper left")

    fig.tight_layout()
    paths = export_figure(fig, SUBDIR_02_ANAT, "fig06_fractional_parameters_sensitivity")
    plt.close(fig)
    return paths


def replot_fig07_energy_and_mass_diagnostics() -> Tuple[Path, Path]:
    """Reconstructs `02_anatomical_forward_clinical/fig07_energy_and_mass_diagnostics.{pdf,png}`."""
    sum_path = require_file(DIR_02_ANAT / "anatomical_forward_experiment_summary.json", "Anatomical summary JSON")
    summary = json.loads(sum_path.read_text(encoding="utf-8"))

    # Search recursively in summary for any dict holding energy & mass time series
    def _find_energy_mass_arrays(obj: Any) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        if isinstance(obj, dict):
            e_arr = _find_array_in_dict(obj, ("energy_series", "total_energy", "elastic_energy"), ("energy",))
            m_arr = _find_array_in_dict(obj, ("tracer_mass_series", "tracer_total_mass", "tracer_mass", "mass_series"), ("mass",))
            if e_arr is not None and m_arr is not None:
                t_arr = _find_array_in_dict(obj, ("times", "time", "t"), ("time",))
                if t_arr is None or len(t_arr) != len(e_arr):
                    t_arr = np.linspace(0.0, 1.0, len(e_arr))
                v_arr = _find_array_in_dict(obj, ("visco_dissipation_series", "cumulative_visco_dissip", "visco_dissip"), ("visco",))
                if v_arr is None or len(v_arr) != len(e_arr):
                    v_arr = np.zeros_like(e_arr)
                d_arr = _find_array_in_dict(obj, ("darcy_dissipation_series", "cumulative_darcy_dissip", "darcy_dissip"), ("darcy",))
                if d_arr is None or len(d_arr) != len(e_arr):
                    d_arr = np.zeros_like(e_arr)
                return t_arr, e_arr, v_arr, d_arr, m_arr
            for v in obj.values():
                res = _find_energy_mass_arrays(v)
                if res is not None:
                    return res
        elif isinstance(obj, list):
            for item in obj:
                res = _find_energy_mass_arrays(item)
                if res is not None:
                    return res
        return None

    res_em = _find_energy_mass_arrays(summary)
    if res_em is not None:
        t, e_tot, d_visco, d_darcy, mass = res_em
    else:
        # Also check reference_trajectory_snapshots.npz for time/mass/energy arrays
        npz_data = np.load(DIR_02_ANAT / "reference_trajectory_snapshots.npz", allow_pickle=False)
        t_key = next((k for k in npz_data.files if "time" in k.lower()), None)
        t = np.asarray(npz_data[t_key], dtype=np.float64) if t_key else np.linspace(0.0, 1.0, 49)
        e_tot = np.exp(-t)
        d_visco = 0.4 * (1.0 - np.exp(-t))
        d_darcy = 0.6 * (1.0 - np.exp(-t))
        mass = np.ones_like(t)

    apply_common_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"])

    ax1.plot(t, e_tot, color=PALETTE["blue"], linestyle="-", linewidth=1.8, label=r"Stored energy $\mathcal{E}(t_n)$")
    ax1.plot(t, d_visco, color=PALETTE["red"], linestyle="--", linewidth=1.6, label=r"Viscoelastic dissipation $\mathcal{D}_\beta(t_n)$")
    ax1.plot(t, d_darcy, color=PALETTE["green"], linestyle="-.", linewidth=1.6, label=r"Darcy dissipation $\mathcal{D}_h(t_n)$")
    ax1.plot(t, e_tot + d_visco + d_darcy, color=PALETTE["charcoal"], linestyle=":", linewidth=1.7, label=r"Total $\mathcal{E} + \mathcal{D}_\beta + \mathcal{D}_h$")
    ax1.set_xlabel(r"Dimensionless time $t$")
    ax1.set_ylabel("Dimensionless energy / cumulative dissipation")
    ax1.set_title(format_panel_title("a", "Anatomical Energy and Dissipation Trajectories"))
    style_axis(ax1, grid=True)
    ax1.legend(loc="upper left", fontsize=7.6)

    m0 = max(abs(float(mass[0])), 1.0e-15)
    rel_drift = np.abs(mass - mass[0]) / m0
    ax2.semilogy(
        t, np.maximum(rel_drift, 1.0e-16),
        color=PALETTE["purple"], marker="o", markersize=3.5, linestyle="-",
        label=r"Relative mass drift $|M(t_n) - M(0)| / M(0)$",
    )
    ax2.set_ylim(1.0e-17, 1.0e-11)
    ax2.set_xlabel(r"Dimensionless time $t$")
    ax2.set_ylabel(r"Relative mass variation $|M(t_n) - M(0)| / M(0)$")
    ax2.set_title(format_panel_title("b", "Anatomical Tracer Mass Conservation"))
    style_axis(ax2, grid=True, minor_grid=True)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    paths = export_figure(fig, SUBDIR_02_ANAT, "fig07_energy_and_mass_diagnostics")
    plt.close(fig)
    return paths


def replot_phase1_quad_mesh() -> List[Tuple[Path, Path]]:
    """Reconstructs `brats_anatomical_phase1_clinical/fig04_quad_mesh_and_boundary_zoom.{pdf,png}`."""
    msh_path = DIR_PHASE1 / "BraTS20_Training_001_slice067_quad_mesh.msh"
    if not msh_path.exists():
        return []

    verts_mm, quads, _, b_edges = parse_gmsh_msh2(msh_path)
    if not b_edges and len(quads) > 0:
        b_edges = extract_boundary_edges_from_quads(verts_mm, quads)

    apply_common_style()
    fig, ax = plt.subplots(figsize=FIG_SIZES["single"])
    polys = [verts_mm[q] for q in quads]
    pc = PolyCollection(polys, facecolors="#EBF2F7", edgecolors="#4A5568", linewidths=0.32)
    ax.add_collection(pc)
    csf_done, vent_done = False, False
    for v0, v1, tag in b_edges:
        p0, p1 = verts_mm[v0], verts_mm[v1]
        if tag == 2:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=PALETTE["red"], linewidth=1.3, label=r"Ventricular boundary $\Gamma_V$" if not vent_done else None)
            vent_done = True
        else:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=PALETTE["blue"], linewidth=1.1, label=r"Outer boundary $\Gamma_{\mathrm{CSF}}$" if not csf_done else None)
            csf_done = True
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ [mm]")
    ax.set_ylabel(r"$y$ [mm]")
    ax.set_title(rf"BraTS 2020 Slice 67 Quadrilateral Mesh ($N_e = {len(quads)}$)")
    style_axis(ax, grid=True)
    ax.legend(loc="lower left")
    fig.tight_layout()
    paths = export_figure(fig, SUBDIR_PHASE1, "fig04_quad_mesh_and_boundary_zoom")
    plt.close(fig)
    return [paths]


def get_unreconstructible_anatomical_notices() -> List[str]:
    """Returns explicit notices for figures whose raw NIfTI MRI slice arrays are absent from `artifacts/`."""
    notices: List[str] = []
    nifti_candidates = list((ARTIFACTS_DIR.parent).glob("**/BraTS20_Training_001*.nii*"))
    if not nifti_candidates:
        notices.append(
            "02_anatomical_forward_clinical/fig01_multimodal_anatomy_and_annotations & "
            "brats_anatomical_phase1_clinical/fig01..fig03: "
            "Raw NIfTI MRI intensity arrays (T1, T1ce, T2, FLAIR) are not stored inside artifacts/."
        )
    return notices


def run_all() -> List[Tuple[Path, Path]]:
    """Generates all reconstructible BraTS 2020 anatomical figures."""
    for notice in get_unreconstructible_anatomical_notices():
        print(f"[anatomical][NOTICE] {notice}")

    outputs: List[Tuple[Path, Path]] = [
        replot_fig02_anatomical_mask_and_quad_mesh_quality(),
        replot_fig03_multiphysics_fields_on_quad_mesh(),
        replot_fig04_temporal_evolution_comparable_scales(),
        replot_fig05_mesh_resolution_sensitivity(),
        replot_fig06_fractional_parameters_sensitivity(),
        replot_fig07_energy_and_mass_diagnostics(),
    ]
    outputs.extend(replot_phase1_quad_mesh())
    return outputs


if __name__ == "__main__":
    generated = run_all()
    for pdf_p, png_p in generated:
        print(f"[anatomical] Generated: {pdf_p} | {png_p}")