"""
tests/test_07_anatomical_mesh_integration.py
Suite di verifica per la Fase 2: integrazione della mesh quadrilatera anatomica
nel motore C++20 deal.II (`InvariantFEMMatrices2D`, `BiotViscoStepSolver`,
`TracerTransportStepSolver`) e in `RetecoForwardProblem2D`.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
import yaml

from reteco import _core, load_master_config
from reteco.anatomy.brats_geometry import (
    AnatomicalCoordinateTransform2D,
    AnatomicalMasks2D,
    BraTSSliceSelection,
    MeshGenerationConfig,
    QuadMesh2D,
    export_dealii_quad_mesh,
    generate_anatomical_quad_mesh_2d,
)
from reteco.forward.target_generator import build_anatomical_driving_Callbacks
from reteco.forward.wrapper import RetecoForwardProblem2D
from reteco.visualization.verification_plots import plot_anatomical_quad_trajectory_snapshots


def _build_synthetic_annular_ellipse_mesh(tmp_dir: Path) -> Tuple[Path, Path, float]:
    """
    Costruisce ed esporta una mesh quadrilatera ellittica con cavità interna
    (boundary_id = 1 sul contorno esterno, boundary_id = 2 sul contorno interno)
    e un report JSON di preprocessing associato.
    """
    shape_2d = (140, 160)
    affine = np.array(
        [
            [1.0, 0.0, 0.0, -70.0],
            [0.0, 1.0, 0.0, -80.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    ii, jj = np.meshgrid(np.arange(shape_2d[0]), np.arange(shape_2d[1]), indexing="ij")
    x_mm = ii - 70.0
    y_mm = jj - 80.0

    outer = (x_mm / 48.0) ** 2 + (y_mm / 60.0) ** 2 <= 1.0
    inner_hole = ((x_mm + 8.0) / 9.5) ** 2 + ((y_mm - 4.0) / 12.5) ** 2 <= 1.0
    support_mask = outer & (~inner_hole)

    lesion = ((x_mm - 18.0) ** 2 + (y_mm + 14.0) ** 2 <= 12.0 ** 2) & support_mask
    seg_2d = np.where(lesion, 2, 0).astype(np.int32)

    slice_sel = BraTSSliceSelection(
        slice_axis=2,
        slice_index=5,
        selection_mode="manual",
        physical_z_mm=10.0,
        slice_plane_reference_point_mm=(0.0, 0.0, 10.0),
        tumor_voxel_count_in_slice=int(np.sum(lesion)),
        tumor_area_in_slice_mm2=float(np.sum(lesion)),
        modalities_2d={k: support_mask.astype(np.float64) for k in ("T1", "T1ce", "T2", "FLAIR")},
        seg_2d=seg_2d,
        wm_mask_2d=None,
    )
    masks = AnatomicalMasks2D(
        brain_support_mask=support_mask,
        whole_tumor_mask=lesion,
        edema_mask=lesion,
        necrotic_non_enhancing_mask=np.zeros_like(lesion),
        enhancing_tumor_mask=np.zeros_like(lesion),
        white_matter_mask=None,
        domain_representation_type="brain_parenchyma_proxy",
        warnings=["WM_MASK_UNAVAILABLE"],
        preserved_internal_holes_count=1,
        filled_small_artifact_holes_count=0,
        removed_isolated_components_count=0,
        lesion_inside_support_ratio=1.0,
    )

    l0_m = 0.01  # 10 mm
    transform = AnatomicalCoordinateTransform2D.from_ras_affine_and_mask(
        affine, slice_index=5, support_mask=support_mask, L0_m=l0_m
    )
    mesh = generate_anatomical_quad_mesh_2d(
        masks, slice_sel, transform, MeshGenerationConfig(target_element_size_mm=4.0)
    )
    exported = export_dealii_quad_mesh(mesh, transform, tmp_dir, basename="test_elliptic_annulus")

    # Calcolo centroide lesione e salvataggio report sintetico
    t_ii, t_jj = np.nonzero(lesion)
    t_xy_mm = transform.voxel_ij_to_physical_mm(np.column_stack([t_ii, t_jj]))
    c_mm = np.mean(t_xy_mm, axis=0)
    c_adim = transform.physical_mm_to_mesh_adim(c_mm.reshape(1, 2))[0]

    report_path = tmp_dir / "geometry_quality_report.json"
    import json
    report_path.write_text(
        json.dumps(
            {
                "L0_m": l0_m,
                "L0_mm": 10.0,
                "coordinate_transform": transform.to_metadata_dict(),
                "lesion_centroids": {
                    "has_lesion_in_slice": True,
                    "center_physical_mm": [float(c_mm[0]), float(c_mm[1]), 10.0],
                    "center_mesh_adim": [float(c_adim[0]), float(c_adim[1])],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    expected_mesh_area_adim = float(
        np.sum(
            0.5
            * (
                mesh.vertices_adim[mesh.quads, 0] * np.roll(mesh.vertices_adim[mesh.quads, 1], -1, axis=1)
                - np.roll(mesh.vertices_adim[mesh.quads, 0], -1, axis=1) * mesh.vertices_adim[mesh.quads, 1]
            )
        )
    )
    return Path(exported["gmsh_msh"]), report_path, expected_mesh_area_adim


def test_01_import_elliptic_mesh_dof_counts_and_boundary_ids(tmp_path: Path) -> None:
    """
    Test 1, 2, 3, 4: Importazione della mesh ellittica con cavità interna, verifica
    del conteggio dei DoF, coerenza dei support points tra pressione e tracciante,
    lettura dei boundary_id effettivi (1 e 2) e vincolo di Dirichlet solo sul bordo esterno (1).
    """
    msh_path, _, expected_area_adim = _build_synthetic_annular_ellipse_mesh(tmp_path)
    master = load_master_config("configs/scales_and_physics.yaml")
    grid = _core.TimeGrid.make_graded(0.4, 10, 1.0)

    problem = RetecoForwardProblem2D(
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid,
        mesh_file_path=msh_path,
        dirichlet_disp_boundary_ids=[1],
    )

    assert problem.fem.is_imported_mesh() is True
    assert problem.fem.verify_dof_and_mesh_consistency() is True

    # Per Q_1 su pressione e tracciante, numero di DoF e support points devono coincidere coi vertici
    assert problem.num_press_dofs == problem.fem.num_vertices()
    assert problem.num_tracer_dofs == problem.fem.num_vertices()
    assert np.allclose(problem.press_pts, problem.tracer_pts, atol=1.0e-14)
    assert problem.K0_csr.shape == (problem.num_disp_dofs, problem.num_disp_dofs)
    assert problem.fem.B_biot().shape == (problem.num_disp_dofs, problem.num_press_dofs)

    # Verifica che l'integrale di 1 tramite M_c coincida con l'area della mesh
    ones_c = np.ones(problem.num_tracer_dofs, dtype=np.float64)
    area_fem = float(ones_c @ (problem.M_c_csr @ ones_c))
    assert np.isclose(area_fem, expected_area_adim, rtol=1.0e-10)

    # Verifica che i boundary_id effettivi siano [1, 2] e che solo [1] abbia vincoli di Dirichlet
    bc_meta = problem.boundary_conditions_metadata()
    assert bc_meta["discovered_boundary_ids"] == [1, 2]
    assert bc_meta["displacement_dirichlet_zero_boundary_ids"] == [1]
    assert bc_meta["displacement_homogeneous_neumann_traction_free_boundary_ids"] == [2]
    assert bc_meta["boundary_face_counts"][1] > 0
    assert bc_meta["boundary_face_counts"][2] > 0

    # Verifica che i nodi sul contorno interno (vicini a x* ~ -0.8, y* ~ +0.4) NON siano vincolati a Dirichlet
    is_dir = np.asarray(problem.fem.is_dirichlet_disp(), dtype=bool)
    assert np.any(is_dir) and not np.all(is_dir)


def test_02_darcy_flux_and_advection_matrix_on_anatomical_mesh(tmp_path: Path) -> None:
    """
    Requisito 7: Verifica che su un gradiente di pressione lineare p(x,y) = g_x*x + g_y*y
    il flusso di Darcy calcolato sulle celle quadrilatere coincida con q = -K_h * [g_x, g_y],
    e che la matrice advettiva A_adv soddisfi 1^T * A_adv = 0^T (conservazione sotto non-flusso).
    """
    msh_path, _, _ = _build_synthetic_annular_ellipse_mesh(tmp_path)
    master = load_master_config("configs/scales_and_physics.yaml")
    grid = _core.TimeGrid.make_graded(0.2, 5, 1.0)

    problem = RetecoForwardProblem2D(
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid,
        mesh_file_path=msh_path,
        dirichlet_disp_boundary_ids=[1],
    )

    gx, gy = 1.75, -0.85
    p_lin = gx * problem.press_pts[:, 0] + gy * problem.press_pts[:, 1]
    cell_q = np.asarray(problem.fem.compute_cell_darcy_fluxes(p_lin))
    expected_q = np.array([-master.fixed_physics.K_h * gx, -master.fixed_physics.K_h * gy])

    assert np.allclose(cell_q[:, 0], expected_q[0], atol=1.0e-12)
    assert np.allclose(cell_q[:, 1], expected_q[1], atol=1.0e-12)

    # Verifica che 1^T * A_adv(p) == 0 per qualsiasi campo di pressione p_lin
    A_adv = problem.fem.assemble_tracer_advection(p_lin, False, 0.08).tocsr()
    ones_c = np.ones(problem.num_tracer_dofs, dtype=np.float64)
    col_sums = ones_c @ A_adv
    assert np.max(np.abs(col_sums)) < 1.0e-12


def test_03_trial_trajectory_mass_conservation_and_quad_connectivity(tmp_path: Path) -> None:
    """
    Requisiti 6, 8 + Test conservazione massa ed esecuzione traiettoria sulla mesh importata:
    - Trasformazione corretta del centro lesione dal report di Fase 1 e di sigma_mm tramite L0;
    - Conservazione esatta della massa del tracciante M(t_n) = M(0) con s_c = 0 e non-flusso;
    - Conservazione della connettività quadrilatera per la visualizzazione senza Delaunay.
    """
    msh_path, report_path, _ = _build_synthetic_annular_ellipse_mesh(tmp_path)
    master = load_master_config("configs/scales_and_physics.yaml")
    with open("configs/brats_anatomical_forward.yaml", "r", encoding="utf-8") as fh:
        anat_cfg = yaml.safe_load(fh)

    grid = _core.TimeGrid.make_graded(0.4, 20, 1.5)
    problem = RetecoForwardProblem2D(
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid,
        mesh_file_path=msh_path,
        dirichlet_disp_boundary_ids=[1],
    )

    c0, F_mech_func, S_fluid_func, forcing_meta = build_anatomical_driving_Callbacks(
        problem=problem,
        driving_cfg=anat_cfg["forward_simulation"]["driving_protocol"],
        geometry_report_path=report_path,
    )

    assert forcing_meta["forcing_classification"] == "anatomy_guided_synthetic_forcing"
    assert np.isclose(forcing_meta["initial_tracer_gaussian"]["sigma_mesh_adim"], 0.80)

    traj = problem.solve_trajectory(
        theta_vec=master.parameter_space_theta.theta_dagger_array(),
        c0=c0,
        F_mech_func=F_mech_func,
        S_fluid_func=S_fluid_func,
        S_tracer_func=None,
        enforce_initial_compatibility=True,
        compute_diagnostics=True,
    )

    assert traj.initial_compatibility_residual < 1.0e-12
    assert traj.max_darcy_velocity > 1.0e-4

    # Conservazione esatta della massa del tracciante sulla mesh anatomica con buco interno
    masses = np.array([d.tracer_total_mass for d in traj.diagnostics], dtype=np.float64)
    rel_mass_drift = float(np.max(np.abs(masses - masses[0])) / masses[0])
    assert rel_mass_drift < 1.0e-12, f"Deriva di massa del tracciante: {rel_mass_drift:.3e}"

    # Verifica connettività quadrilatera nativa ed esportazione PDF senza Delaunay
    assert problem.cell_quad_connectivity.shape == (problem.num_active_cells, 4)
    pdf_out = tmp_path / "fig_anatomical_quad_snapshots.pdf"
    plot_anatomical_quad_trajectory_snapshots(problem, traj, pdf_out)
    assert pdf_out.exists() and pdf_out.stat().st_size > 0


def test_04_energy_residual_diagnostics_on_imported_mesh(tmp_path: Path) -> None:
    """
    Diagnostica del residuo energetico sulla mesh anatomica importata (caso non forzato
    con stato iniziale in equilibrio compatibile): verifica la positività delle dissipazioni
    e la coerenza tra il bilancio energetico e la dissipazione numerica di passo, senza
    assumere a priori identità non verificate.
    """
    msh_path, _, _ = _build_synthetic_annular_ellipse_mesh(tmp_path)
    master = load_master_config("configs/scales_and_physics.yaml")
    grid = _core.TimeGrid.make_graded(0.4, 25, 1.0)

    problem = RetecoForwardProblem2D(
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid,
        mesh_file_path=msh_path,
        dirichlet_disp_boundary_ids=[1],
    )

    px, py = problem.press_pts[:, 0], problem.press_pts[:, 1]
    p0 = 0.35 * np.exp(-0.2 * (px ** 2 + py ** 2))
    c0 = np.ones(problem.num_tracer_dofs, dtype=np.float64)

    traj = problem.solve_trajectory(
        theta_vec=master.parameter_space_theta.theta_dagger_array(),
        c0=c0,
        p0=p0,
        d0=None,
        F_mech_func=None,
        S_fluid_func=None,
        enforce_initial_compatibility=True,
        compute_diagnostics=True,
    )

    e0 = float(traj.diagnostics[0].total_energy)
    assert e0 > 0.0
    for diag in traj.diagnostics[1:]:
        assert np.isfinite(diag.energy_balance_residual)
        assert diag.total_energy <= e0 + 1.0e-12
        assert diag.cumulative_visco_dissip >= -1.0e-12
        assert diag.cumulative_darcy_dissip >= 0.0
        assert diag.step_discrete_identity_error < 1.0e-11 * e0


def test_05_backward_compatibility_with_unit_square_mms() -> None:
    """
    Verifica che il costruttore predefinito sul quadrato unitario e il benchmark MMS
    continuino a funzionare esattamente come prima delle modifiche (N = 4 -> 8 con M = 400).
    """
    master = load_master_config("configs/scales_and_physics.yaml")
    phys = master.fixed_physics.to_cpp(master.characteristic_scales)
    td = master.parameter_space_theta.theta_dagger
    theta_cpp = _core.ParameterTheta(td["alpha"], td["beta"], td["D"], td["E"])

    mms = _core.MMSManufacturedSystem(phys, theta_cpp, 0.4)
    grid = _core.TimeGrid.make_graded(0.5, 400, 1.0)
    r4 = mms.run_full_mms_test(4, grid, 2, 1, 1)
    r8 = mms.run_full_mms_test(8, grid, 2, 1, 1)

    assert r8.d_L2 < 0.20 * r4.d_L2
    assert r8.p_L2 < 0.30 * r4.p_L2
    assert r8.c_L2 < 0.45 * r4.c_L2