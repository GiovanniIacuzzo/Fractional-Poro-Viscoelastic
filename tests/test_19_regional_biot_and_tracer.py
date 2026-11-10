"""FEM-level regression of BraTS material ID wiring and weighted tracer mass.

Run after make build. Synthetic mesh only; never a biological parameter validation.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
from reteco import _core, load_master_config
from reteco.forward.wrapper import RetecoForwardProblem2D
from reteco.anatomy.regional_materials import load_regional_materials, validate_regional_materials


def _four_region_msh(path: Path):
    nodes = [(i / 2., j / 2.) for j in range(3) for i in range(3)]
    quads = [(1,2,5,4), (2,3,6,5), (4,5,8,7), (5,6,9,8)]
    # Gmsh material IDs exactly {0,1,2,4}; deal.II maps physical tags to materials.
    txt = ['$MeshFormat','2.2 0 8','$EndMeshFormat', '$Nodes',str(len(nodes))]
    txt += [f'{k+1} {x} {y} 0' for k,(x,y) in enumerate(nodes)]
    txt += ['$EndNodes','$Elements','12']
    border = [(1,2),(2,3),(3,6),(6,9),(9,8),(8,7),(7,4),(4,1)]
    for i,(a,b) in enumerate(border):
        txt.append(f'{i+1} 1 2 1 1 {a} {b}')
    for i,(quad,region) in enumerate(zip(quads,(0,1,2,4))):
        txt.append(f'{i+9} 3 2 {region} 10 '+ ' '.join(str(v) for v in quad))
    txt += ['$EndElements']
    path.write_text('\n'.join(txt)+'\n')
    return path


def _make_problem(mesh: Path, materials=None):
    m = load_master_config('configs/scales_and_physics.yaml')
    grid = _core.TimeGrid.make_graded(0.2, 4, 1.5)
    return RetecoForwardProblem2D(
        physics_model=m.fixed_physics, scales=m.characteristic_scales,
        time_grid=grid, mesh_file_path=mesh, region_materials=materials)


def test_uniform_map_identical_to_legacy(tmp_path):
    mesh=_four_region_msh(tmp_path/'quad_regions.msh')
    homogeneous=load_regional_materials('configs/brats_region_materials_phaseBC.yaml','homogeneous',.2)
    legacy=_make_problem(mesh)
    explicit=_make_problem(mesh,homogeneous)
    assert set(legacy.cell_material_ids)=={0,1,2,4}
    for name in ('K0_mech','Kvisco_mech','K_p','M_c','K_c','M_phi_c','K_phiD_c'):
        a=getattr(legacy.fem,name)().toarray()
        b=getattr(explicit.fem,name)().toarray()
        assert np.allclose(a,b,atol=2e-12),name
    assert np.allclose(legacy.fem.M_phi_c().toarray(),legacy.fem.M_c().toarray())
    assert np.allclose(legacy.fem.K_phiD_c().toarray(),legacy.fem.K_c().toarray())


def test_material_contrasts_affect_correct_blocks_and_cell_fluxes(tmp_path):
    mesh=_four_region_msh(tmp_path/'quad_regions.msh')
    contrast=load_regional_materials('configs/brats_region_materials_phaseBC.yaml','exploratory_contrast',.2)
    baseline=_make_problem(mesh)
    hetero=_make_problem(mesh,contrast)
    for name in ('K0_mech','K_p','M_phi_c','K_phiD_c'):
        diff=(getattr(hetero.fem,name)()-getattr(baseline.fem,name)()).toarray()
        assert np.linalg.norm(diff)>1e-9,name
    # Viscous kernel unchanged by pure E(x) variation.
    assert np.allclose(hetero.fem.Kvisco_mech().toarray(),baseline.fem.Kvisco_mech().toarray())
    ones=np.ones(hetero.num_tracer_dofs)
    # For constant c, integral phi(x) dx matches cellwise expected area.
    expected=.2*.25*sum(contrast[k]['porosity'] for k in (0,1,2,4))
    assert hetero.monitor.compute_tracer_storage_mass(ones)==pytest.approx(expected,abs=1e-12)
    p=hetero.press_pts[:,0]  # dp/dx=1 on each quadrilateral
    flux=np.asarray(hetero.fem.compute_cell_darcy_fluxes(p))
    for lab,q in zip(hetero.cell_material_ids,flux):
        assert q[0]==pytest.approx(-.2*contrast[int(lab)]['hydraulic_mobility'],abs=2e-12)
        assert q[1]==pytest.approx(0.,abs=2e-12)


def test_heterogeneous_closed_tracer_mass_and_input_validation(tmp_path):
    mesh=_four_region_msh(tmp_path/'quad_regions.msh')
    contrast=load_regional_materials('configs/brats_region_materials_phaseBC.yaml','exploratory_contrast',.2)
    problem=_make_problem(mesh,contrast)
    c0=problem.build_gaussian_initial_tracer(center=(.38,.48),sigma=.21,baseline=.10)
    traj=problem.solve_trajectory(
        np.array([.7,.45,.08,2.5]),c0=c0,compute_diagnostics=False)
    mass=np.array([problem.monitor.compute_tracer_storage_mass(c) for c in traj.c_hist])
    assert np.max(abs(mass-mass[0]))/mass[0]<1e-9
    invalid=dict(contrast)
    invalid[2]=dict(invalid[2],porosity=10.0)
    with pytest.raises(ValueError):
        validate_regional_materials(invalid,phi0=.2)
