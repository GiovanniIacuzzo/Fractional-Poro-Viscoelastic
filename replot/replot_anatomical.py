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


def _extract_snapshots_from_npz() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load only observables whose spatial representation is fully specified in the archive.

    The stored displacement array contains Q2 vector coefficients (15614 entries) but the
    archive does not contain a coefficient-to-coordinate map. It is intentionally not
    projected onto mesh vertices, because taking its first N entries would not yield |d|.
    """
    npz_path = require_file(DIR_02_ANAT / "reference_trajectory_snapshots.npz", "Anatomical reference trajectory NPZ")
    with np.load(npz_path, allow_pickle=False) as data:
        required = ("mesh_vertices_adim", "cell_quad_connectivity", "times_adim",
                    "snapshot_indices", "c_snapshots_adim", "p_snapshots_adim")
        missing = [k for k in required if k not in data.files]
        if missing:
            raise KeyError(f"Anatomical snapshot archive lacks required keys {missing}; available={list(data.files)}")
        verts = np.asarray(data["mesh_vertices_adim"], dtype=float)
        quads = np.asarray(data["cell_quad_connectivity"], dtype=int)
        times_all = np.asarray(data["times_adim"], dtype=float)
        snapshot_indices = np.asarray(data["snapshot_indices"], dtype=int)
        c_snap = np.asarray(data["c_snapshots_adim"], dtype=float)
        p_snap = np.asarray(data["p_snapshots_adim"], dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 2:
            raise ValueError(f"Expected 2-D mesh vertices, got shape {verts.shape}.")
        if quads.ndim != 2 or quads.shape[1] != 4 or quads.size == 0:
            raise ValueError(f"Expected quadrilateral connectivity (n_cells, 4), got {quads.shape}.")
        if quads.min() < 0 or quads.max() >= len(verts):
            raise ValueError("Quadrilateral connectivity contains indices outside mesh_vertices_adim.")
        if c_snap.ndim != 2 or p_snap.ndim != 2 or c_snap.shape != p_snap.shape:
            raise ValueError(f"Tracer and pressure snapshots must be aligned 2-D arrays; got {c_snap.shape} and {p_snap.shape}.")
        if c_snap.shape[1] != len(verts):
            raise ValueError(f"Q1 snapshot length {c_snap.shape[1]} differs from mesh vertex count {len(verts)}.")
        if len(snapshot_indices) != c_snap.shape[0] or np.any(snapshot_indices < 0) or np.any(snapshot_indices >= len(times_all)):
            raise ValueError("snapshot_indices do not match the saved snapshot arrays/times_adim vector.")
        times = times_all[snapshot_indices]
        if np.any(np.diff(times) <= 0):
            raise ValueError("Saved snapshot times must be strictly increasing.")
        q_snapshots = []
        for step_idx in snapshot_indices:
            key = f"step_{int(step_idx):03d}"
            if key not in data.files:
                raise KeyError(f"No archived Darcy-flux array {key!r} for snapshot index {step_idx}.")
            q = np.asarray(data[key], dtype=float)
            if q.shape != (len(quads), 2):
                raise ValueError(f"Stored Darcy-flux array {key} must have shape {(len(quads), 2)}, got {q.shape}.")
            q_snapshots.append(q)
        q_snap = np.stack(q_snapshots, axis=0)
        if not all(np.all(np.isfinite(a)) for a in (verts, times, c_snap, p_snap, q_snap)):
            raise ValueError("Non-finite values found in archived anatomical snapshots.")
    return verts, quads, times, c_snap, p_snap, q_snap

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
    """Plot archived Q1 pressure/tracer and cellwise Darcy flux; do not mis-map Q2 displacement coefficients."""
    verts, quads, snap_times, c_snap, p_snap, q_snap = _extract_snapshots_from_npz()
    tri = _build_triangulation_for_snapshots(verts, quads)
    p_final, c_final, q_final = p_snap[-1], c_snap[-1], q_snap[-1]
    q_mag = np.linalg.norm(q_final, axis=1)
    polygons = verts[quads]
    t_final = float(snap_times[-1])

    apply_common_style()
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZES["triple_row"])
    ax1, ax2, ax3 = axes

    p_lim = max(float(np.max(np.abs(p_final))), 1e-15)
    im1 = ax1.tripcolor(tri, p_final, shading="gouraud", cmap="RdBu_r", vmin=-p_lim, vmax=p_lim)
    cb1 = fig.colorbar(im1, ax=ax1, pad=0.03, fraction=0.05)
    cb1.set_label(r"$p_h(\mathbf{x},T)$ [dimensionless]")
    ax1.set_title(format_panel_title("a", r"Interstitial pressure $p_h$"))

    coll = PolyCollection(polygons, edgecolors="none", cmap="magma")
    coll.set_array(q_mag)
    ax2.add_collection(coll)
    ax2.autoscale_view()
    ax2.set_aspect("equal")
    cb2 = fig.colorbar(coll, ax=ax2, pad=0.03, fraction=0.05)
    cb2.set_label(r"$\|\mathbf{q}_h\|_2$ [dimensionless]")
    ax2.set_title(format_panel_title("b", "Stored cellwise Darcy flux magnitude"))

    im3 = ax3.tripcolor(tri, c_final, shading="gouraud", cmap="plasma", vmin=float(np.min(c_snap)), vmax=float(np.max(c_snap)))
    cb3 = fig.colorbar(im3, ax=ax3, pad=0.03, fraction=0.05)
    cb3.set_label(r"$c_h(\mathbf{x},T)$ [dimensionless]")
    ax3.set_title(format_panel_title("c", "Passive tracer concentration $c_h$"))

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel(r"Dimensionless $x/L_0$")
        ax.set_ylabel(r"Dimensionless $y/L_0$")
        style_axis(ax, grid=False)
    fig.suptitle(rf"Archived anatomical fields at $t={t_final:.3g}$", y=1.02, fontsize=11)
    fig.text(0.5, -0.01,
             "Displacement omitted: archived Q2 coefficients lack a DOF-to-coordinate map, so a nodal magnitude cannot be reconstructed reliably.",
             ha="center", va="top", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    paths = export_figure(fig, SUBDIR_02_ANAT, "fig03_multiphysics_fields_on_quad_mesh")
    plt.close(fig)
    return paths


def replot_fig04_temporal_evolution_comparable_scales() -> Tuple[Path, Path]:
    """Show archived tracer and pressure snapshots with dedicated colorbar axes."""
    verts, quads, snap_times, c_snap, p_snap, _ = _extract_snapshots_from_npz()
    tri = _build_triangulation_for_snapshots(verts, quads)
    n_snaps = len(snap_times)
    c_min, c_max = float(np.min(c_snap)), float(np.max(c_snap))
    p_lim = max(float(np.max(np.abs(p_snap))), 1e-15)

    apply_common_style()
    # Dedicated GridSpec columns prevent colorbars from shrinking/covering the last spatial panels.
    fig = plt.figure(figsize=(13.1, 5.8))
    gs = fig.add_gridspec(
        2, n_snaps + 1,
        width_ratios=[1.0] * n_snaps + [0.055],
        left=0.065, right=0.975, bottom=0.17, top=0.84,
        wspace=0.26, hspace=0.30,
    )
    axes = np.empty((2, n_snaps), dtype=object)
    for row in range(2):
        for col in range(n_snaps):
            if row == 0 and col == 0:
                ax = fig.add_subplot(gs[row, col])
            else:
                ax = fig.add_subplot(gs[row, col], sharex=axes[0, 0], sharey=axes[0, 0])
            axes[row, col] = ax

    for col, t in enumerate(snap_times):
        imc = axes[0, col].tripcolor(
            tri, c_snap[col], shading="gouraud", cmap="plasma", vmin=c_min, vmax=c_max
        )
        imp = axes[1, col].tripcolor(
            tri, p_snap[col], shading="gouraud", cmap="RdBu_r", vmin=-p_lim, vmax=p_lim
        )
        axes[0, col].set_title(rf"$t={t:.3f}$", pad=5)
        for row in range(2):
            ax = axes[row, col]
            ax.set_aspect("equal")
            ax.set_xlabel(r"$x/L_0$")
            style_axis(ax, grid=False)
            if col == 0:
                field_label = "Tracer" if row == 0 else "Pressure"
                ax.set_ylabel(field_label + "\n" + r"$y/L_0$")
            else:
                ax.tick_params(labelleft=False)

    cax_c = fig.add_subplot(gs[0, n_snaps])
    cax_p = fig.add_subplot(gs[1, n_snaps])
    cbar_c = fig.colorbar(imc, cax=cax_c)
    cbar_c.set_label(r"Tracer $c_h$ [dimensionless]", labelpad=8)
    cbar_p = fig.colorbar(imp, cax=cax_p)
    cbar_p.set_label(r"Pressure $p_h$ [dimensionless]", labelpad=8)
    fig.suptitle("Recorded temporal snapshots: tracer and pressure", y=0.96, fontsize=11)
    fig.text(
        0.5, 0.035,
        "Only fields with archived mesh-aligned representations are shown; the displacement Q2 coordinate map is absent from the NPZ.",
        ha="center", va="bottom", fontsize=7.5,
    )
    paths = export_figure(fig, SUBDIR_02_ANAT, "fig04_temporal_evolution_comparable_scales")
    plt.close(fig)
    return paths

def replot_fig05_mesh_resolution_sensitivity() -> Tuple[Path, Path]:
    """Plot measured differences from the finest mesh, cost, and geometry fidelity."""
    sum_path = require_file(DIR_02_ANAT / "anatomical_forward_experiment_summary.json", "Anatomical summary JSON")
    summary = json.loads(sum_path.read_text(encoding="utf-8"))
    ms = summary.get("mesh_sensitivity_study", {})
    rows = ms.get("resolutions_evaluated", []) if isinstance(ms, dict) else []
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("No measured resolutions_evaluated list found in mesh_sensitivity_study.")
    rows = sorted(rows, key=lambda r: float(r["total_dofs"]))
    required = ("target_element_size_mm", "total_dofs", "num_quad_cells", "rasterized_dice_similarity",
                "relative_area_error_vs_mask", "rel_diff_final_c_l2_vs_finest",
                "rel_diff_final_disp_l2_vs_finest", "rel_diff_final_energy_vs_finest")
    for r in rows:
        miss = [k for k in required if k not in r or r[k] is None]
        if miss:
            raise KeyError(f"Mesh sensitivity row missing measured fields {miss}; available={list(r)}")

    h = np.asarray([float(r["target_element_size_mm"]) for r in rows])
    dofs = np.asarray([float(r["total_dofs"]) for r in rows])
    ne = np.asarray([int(r["num_quad_cells"]) for r in rows])
    c_err = 100*np.asarray([float(r["rel_diff_final_c_l2_vs_finest"]) for r in rows])
    d_err = 100*np.asarray([float(r["rel_diff_final_disp_l2_vs_finest"]) for r in rows])
    e_err = 100*np.asarray([float(r["rel_diff_final_energy_vs_finest"]) for r in rows])
    dice = 100*np.asarray([float(r["rasterized_dice_similarity"]) for r in rows])
    area_err = 100*np.asarray([float(r["relative_area_error_vs_mask"]) for r in rows])
    labels = [f"{hh:g} mm" for hh in h]
    x = np.arange(len(rows))

    apply_common_style()
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZES["triple_row"])
    ax = axes[0]
    ax.plot(x, c_err, marker="o", color=PALETTE["blue"], linewidth=1.7, label=r"Tracer $\|c_h(T)\|_{L^2}$")
    ax.plot(x, d_err, marker="s", color=PALETTE["red"], linewidth=1.7, linestyle="--", label=r"Displacement $\|d_h(T)\|_{L^2}$")
    ax.plot(x, e_err, marker="^", color=PALETTE["green"], linewidth=1.7, linestyle="-.", label="Total energy")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Target element size")
    ax.set_ylabel("Relative difference from 2-mm reference [%]")
    ax.set_title(format_panel_title("a", "Solution Sensitivity"))
    style_axis(ax, grid=True, grid_axis="y"); ax.legend(loc="best", fontsize=7.4)

    ax = axes[1]
    bars = ax.bar(x, dofs, color=[PALETTE["slate"], PALETTE["blue"], PALETTE["green"]],
                  edgecolor=PALETTE["charcoal"], linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Target element size")
    ax.set_ylabel("Total degrees of freedom")
    ax.set_title(format_panel_title("b", "Discretization Cost"))
    for bar, ncell in zip(bars, ne):
        ax.annotate(f"{int(bar.get_height()):,} DOFs\n{ncell:,} quads", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=7.3)
    ax.set_ylim(0, float(dofs.max())*1.23)
    style_axis(ax, grid=True, grid_axis="y")

    ax = axes[2]
    ax.plot(x, dice, marker="o", color=PALETTE["blue"], linewidth=1.7, label="Rasterized Dice [%]")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Target element size")
    ax.set_ylabel("Dice similarity [%]", color=PALETTE["blue"])
    ax.set_ylim(min(dice)-0.08, 100.0)
    ax.tick_params(axis="y", labelcolor=PALETTE["blue"])
    ax2 = ax.twinx()
    ax2.plot(x, area_err, marker="s", color=PALETTE["red"], linestyle="--", linewidth=1.5, label="Relative area error [%]")
    ax2.set_ylabel("Relative area error [%]", color=PALETTE["red"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["red"])
    ax.set_title(format_panel_title("c", "Geometric Fidelity"))
    lines1, labels1 = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc="best", fontsize=7.4)
    style_axis(ax, grid=True, grid_axis="y")

    fig.suptitle("Mesh-resolution sensitivity (not an asymptotic convergence study)", y=1.04, fontsize=11)
    fig.tight_layout()
    paths = export_figure(fig, SUBDIR_02_ANAT, "fig05_mesh_resolution_sensitivity")
    plt.close(fig)
    return paths


def replot_fig06_fractional_parameters_sensitivity() -> Tuple[Path, Path]:
    """Plot measured final-response changes across the saved alpha and beta sweeps."""
    sum_path = require_file(DIR_02_ANAT / "anatomical_forward_experiment_summary.json", "Anatomical summary JSON")
    summary = json.loads(sum_path.read_text(encoding="utf-8"))
    sec = summary.get("fractional_sensitivity_study", {})
    alpha_rows = sec.get("alpha_sweep_summary", [])
    beta_rows = sec.get("beta_sweep_summary", [])
    if len(alpha_rows) < 2 or len(beta_rows) < 2:
        raise ValueError("Fractional sensitivity figure requires measured alpha_sweep_summary and beta_sweep_summary rows.")
    alpha_rows = sorted(alpha_rows, key=lambda r: float(r["alpha"]))
    beta_rows = sorted(beta_rows, key=lambda r: float(r["beta"]))

    def relative_change(rows, xkey, ykey, baseline_x):
        x = np.asarray([float(r[xkey]) for r in rows])
        y = np.asarray([float(r[ykey]) for r in rows])
        hit = np.where(np.isclose(x, baseline_x, rtol=0, atol=1e-10))[0]
        if not hit.size:
            raise KeyError(f"No saved baseline row at {xkey}={baseline_x}.")
        if np.any(~np.isfinite(y)) or y[hit[0]] == 0:
            raise ValueError(f"Invalid measured response values for {ykey}.")
        return x, 100.0*(y/y[hit[0]]-1.0)

    apply_common_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"])
    for key, label, st in [
        ("final_c_lesion_center", r"$c_h(T)$ at lesion centre", PALETTE["blue"]),
        ("final_c_l2_norm", r"$\|c_h(T)\|_{L^2}$", PALETTE["orange"]),
    ]:
        x, y = relative_change(alpha_rows, "alpha", key, 0.7)
        ax1.plot(x, y, marker="o" if key == "final_c_lesion_center" else "s", linewidth=1.7,
                 color=st, label=label)
    ax1.axvline(0.7, color=PALETTE["charcoal"], linestyle=":", linewidth=1.1, label=r"Baseline $\alpha=0.70$")
    ax1.set_xlabel(r"Subdiffusion order $\alpha$")
    ax1.set_ylabel("Relative change from baseline [%]")
    ax1.set_title(format_panel_title("a", r"Alpha Sweep ($\beta=0.45$)"))
    style_axis(ax1, grid=True); ax1.legend(loc="best", fontsize=8)

    for key, label, st, marker in [
        ("final_disp_l2_norm", r"$\|d_h(T)\|_{L^2}$", PALETTE["blue"], "o"),
        ("final_press_l2_norm", r"$\|p_h(T)\|_{L^2}$", PALETTE["green"], "s"),
        ("final_visco_dissipation", r"$\mathcal{D}_\beta(T)$", PALETTE["red"], "^"),
    ]:
        x, y = relative_change(beta_rows, "beta", key, 0.45)
        ax2.plot(x, y, marker=marker, linewidth=1.7, color=st, label=label)
    ax2.axvline(0.45, color=PALETTE["charcoal"], linestyle=":", linewidth=1.1, label=r"Baseline $\beta=0.45$")
    ax2.set_xlabel(r"Viscoelastic order $\beta$")
    ax2.set_ylabel("Relative change from baseline [%]")
    ax2.set_title(format_panel_title("b", r"Beta Sweep ($\alpha=0.70$)"))
    style_axis(ax2, grid=True); ax2.legend(loc="best", fontsize=8)
    fig.suptitle("Fractional-order sensitivity of measured terminal responses", y=1.03, fontsize=11)
    fig.tight_layout()
    paths = export_figure(fig, SUBDIR_02_ANAT, "fig06_fractional_parameters_sensitivity")
    plt.close(fig)
    return paths


def replot_fig07_energy_and_mass_diagnostics() -> Tuple[Path, Path]:
    """Plot only stored terminal energy and scalar conservation/identity diagnostics."""
    sum_path = require_file(DIR_02_ANAT / "anatomical_forward_experiment_summary.json", "Anatomical summary JSON")
    summary = json.loads(sum_path.read_text(encoding="utf-8"))
    d = summary.get("reference_trajectory_diagnostics", {})
    required = (
        "final_total_energy_adim", "final_cumulative_visco_dissipation_adim",
        "final_cumulative_darcy_dissipation_adim", "final_cumulative_external_work_adim",
        "final_energy_balance_residual_adim", "initial_tracer_mass_adim", "final_tracer_mass_adim",
        "max_relative_tracer_mass_drift", "initial_compatibility_residual", "max_step_discrete_identity_error",
    )
    missing = [k for k in required if k not in d or d[k] is None]
    if missing:
        raise KeyError(f"Anatomical diagnostics summary lacks fields {missing}; available={list(d)}")

    energy_labels = [r"$\mathcal{E}(T)$", r"$\mathcal{D}_\beta(T)$", r"$\mathcal{D}_h(T)$", r"$W_{\rm ext}(T)$"]
    energy_vals = [float(d[k]) for k in (
        "final_total_energy_adim", "final_cumulative_visco_dissipation_adim",
        "final_cumulative_darcy_dissipation_adim", "final_cumulative_external_work_adim",
    )]
    diag_labels = ["Initial\ncompatibility", "Stepwise energy\nidentity", "Relative mass\ndrift"]
    diag_vals = [float(d[k]) for k in (
        "initial_compatibility_residual", "max_step_discrete_identity_error", "max_relative_tracer_mass_drift",
    )]
    residual = float(d["final_energy_balance_residual_adim"])
    for label, vals in (("energy budget", energy_vals), ("scalar diagnostics", diag_vals)):
        if not np.all(np.isfinite(vals)) or np.any(np.asarray(vals) < 0):
            raise ValueError(f"The recorded {label} values must be finite and nonnegative; got {vals}.")
    if not np.isfinite(residual):
        raise ValueError(f"Final energy-balance residual is not finite: {residual}.")

    apply_common_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"])
    x = np.arange(len(energy_vals))
    bars = ax1.bar(
        x, energy_vals,
        color=[PALETTE["blue"], PALETTE["red"], PALETTE["green"], PALETTE["charcoal"]],
        edgecolor=PALETTE["charcoal"], linewidth=0.7,
    )
    ax1.set_xticks(x); ax1.set_xticklabels(energy_labels)
    ax1.set_ylabel("Dimensionless energy / cumulative work")
    ax1.set_title(format_panel_title("a", "Recorded Final-Time Energy Budget"))
    for bar, val in zip(bars, energy_vals):
        ax1.annotate(f"{val:.3g}", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax1.text(0.02, 0.97, rf"Reported $R(T)={residual:.3g}$", transform=ax1.transAxes,
             ha="left", va="top", fontsize=8,
             bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9, "pad": 3})
    ax1.set_ylim(0, max(energy_vals)*1.22)
    style_axis(ax1, grid=True, grid_axis="y")

    x2 = np.arange(len(diag_vals))
    bars2 = ax2.bar(
        x2, diag_vals,
        color=[PALETTE["orange"], PALETTE["green"], PALETTE["purple"]],
        edgecolor=PALETTE["charcoal"], linewidth=0.7, width=0.66,
    )
    ax2.set_xticks(x2); ax2.set_xticklabels(diag_labels, rotation=0, ha="center")
    ax2.set_ylabel("Absolute residual / relative drift")
    ax2.set_title(format_panel_title("b", "Conservation and Identity Residuals"))
    maxv = max(diag_vals) if max(diag_vals) > 0 else 1e-18
    # Symlog reveals the tiny but nonzero values while preserving an exact zero baseline.
    ax2.set_yscale("symlog", linthresh=1e-18, linscale=1.0, base=10)
    ax2.set_ylim(0, maxv*2.3)
    for bar, val in zip(bars2, diag_vals):
        y = val if val > 0 else 0.0
        ax2.annotate(f"{val:.2e}", (bar.get_x()+bar.get_width()/2, y), xytext=(0, 5),
                     textcoords="offset points", ha="center", va="bottom", fontsize=7.7)
    style_axis(ax2, grid=True, grid_axis="y")
    fig.suptitle(
        "Anatomical forward-model diagnostics (recorded scalar summaries; no time histories reconstructed)",
        y=1.03, fontsize=10.2,
    )
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
    nifti_candidates = list(DIR_PHASE1.glob("BraTS20_Training_001*.nii*")) if DIR_PHASE1.exists() else []
    if not nifti_candidates:
        notices.append(
            "02_anatomical_forward_clinical/fig01_multimodal_anatomy_and_annotations & "
            "brats_anatomical_phase1_clinical/fig01..fig03: "
            "Raw NIfTI MRI intensity arrays (T1, T1ce, T2, FLAIR) are not stored inside artifacts/."
        )
    notices.append(
        "The reference NPZ stores Q2 displacement coefficients but not the displacement DOF-to-coordinate map; "
        "the reconstruction therefore omits the displacement field rather than plotting a false nodal magnitude."
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