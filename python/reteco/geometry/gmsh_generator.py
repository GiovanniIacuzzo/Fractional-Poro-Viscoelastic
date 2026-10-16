"""
python/reteco/geometry/gmsh_generator.py
Generatore di mesh 2D con Gmsh che assegna esplicitamente Physical Tags distinti
a Gamma_D (vincolo spostamenti per disuguaglianza di Korn) e Gamma_N (trazioni assegnate),
senza alcun fallback silenzioso (Contratto Matematico Sez. 4).
"""

from __future__ import annotations
from pathlib import Path
import numpy as np


PHYSICAL_TAG_GAMMA_D = 1
PHYSICAL_TAG_GAMMA_N = 2
PHYSICAL_TAG_DOMAIN = 10


def generate_tagged_2d_mesh_gmsh(
    boundary_points_adim: np.ndarray,
    output_msh_path: str | Path,
    h_min: float = 0.03,
    h_max: float = 0.10,
    gamma_d_fraction: float = 0.25,
    refinement_center: tuple[float, float] = (0.0, 0.0),
) -> Path:
    """
    Costruisce una mesh 2D a partire da una poligonale adimensionale chiusa,
    partizionando il bordo in Gamma_D (misura positiva) e Gamma_N.
    """
    import gmsh

    pts = np.asarray(boundary_points_adim, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 4:
        raise ValueError("boundary_points_adim deve avere forma (N, 2) con N >= 4.")
    if not (0.05 <= gamma_d_fraction <= 1.0):
        raise ValueError("gamma_d_fraction deve appartenere a [0.05, 1.0] per garantire Korn.")

    out_path = Path(output_msh_path).with_suffix(".msh")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("reteco_2d_domain")
        geo = gmsh.model.geo

        n_pts = pts.shape[0]
        pt_tags = [geo.addPoint(float(pts[i, 0]), float(pts[i, 1]), 0.0, h_max) for i in range(n_pts)]
        line_tags = [
            geo.addLine(pt_tags[i], pt_tags[(i + 1) % n_pts]) for i in range(n_pts)
        ]

        loop_tag = geo.addCurveLoop(line_tags)
        surf_tag = geo.addPlaneSurface([loop_tag])
        center_tag = geo.addPoint(float(refinement_center[0]), float(refinement_center[1]), 0.0, h_min)
        geo.synchronize()

        n_gamma_d = max(1, int(round(gamma_d_fraction * n_pts)))
        gamma_d_lines = line_tags[:n_gamma_d]
        gamma_n_lines = line_tags[n_gamma_d:]

        gmsh.model.addPhysicalGroup(1, gamma_d_lines, PHYSICAL_TAG_GAMMA_D)
        gmsh.model.setPhysicalName(1, PHYSICAL_TAG_GAMMA_D, "Gamma_D_Dirichlet")

        if gamma_n_lines:
            gmsh.model.addPhysicalGroup(1, gamma_n_lines, PHYSICAL_TAG_GAMMA_N)
            gmsh.model.setPhysicalName(1, PHYSICAL_TAG_GAMMA_N, "Gamma_N_Neumann")

        gmsh.model.addPhysicalGroup(2, [surf_tag], PHYSICAL_TAG_DOMAIN)
        gmsh.model.setPhysicalName(2, PHYSICAL_TAG_DOMAIN, "Omega_Domain")

        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "PointsList", [center_tag])
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", h_min)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", h_max)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.1)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 0.5)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        gmsh.model.mesh.generate(2)
        gmsh.write(str(out_path))
    finally:
        gmsh.finalize()

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Generazione mesh Gmsh fallita per: {out_path}")
    return out_path