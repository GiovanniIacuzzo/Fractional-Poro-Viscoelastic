"""Conforming graded quadrilateral mesh on a synthetic lesion, no patient data."""
from __future__ import annotations
import numpy as np
import pytest
from reteco.anatomy.brats_geometry import MeshGenerationConfig, _roi_banded_axis
from tests.test_07_anatomical_mesh_integration import _build_synthetic_annular_ellipse_mesh
from reteco.anatomy.brats_geometry import (
    AnatomicalCoordinateTransform2D, AnatomicalMasks2D, BraTSSliceSelection,
    generate_anatomical_quad_mesh_2d,
)


def test_banded_axis_monotonic_and_finer_at_roi():
    ax = _roi_banded_axis(0.,100.,3.,1.5,34.,59.,5.)
    delta=np.diff(ax)
    assert np.all(delta>0)
    mid=.5*(ax[1:]+ax[:-1])
    assert np.median(delta[(mid>38)&(mid<55)]) < np.median(delta[mid<10])
    with pytest.raises(ValueError):
        _roi_banded_axis(0,100,3,4,30,60,5)


def test_mesh_fine_roi_conforming_synthetic():
    s=100
    ii,jj=np.indices((s,s))
    support=((ii-50)**2+(jj-50)**2)<38**2
    lesion=((ii-65)**2+(jj-42)**2)<11**2
    seg=np.where(lesion,2,0)
    affine=np.eye(4);affine[:2,3]=(-50,-50)
    sl=BraTSSliceSelection(slice_axis=2,slice_index=0,selection_mode='manual',
        physical_z_mm=0.,slice_plane_reference_point_mm=(0.,0.,0.),
        tumor_voxel_count_in_slice=int(lesion.sum()),tumor_area_in_slice_mm2=float(lesion.sum()),
        modalities_2d={k:support.astype(float) for k in ('T1','T1ce','T2','FLAIR')},
        seg_2d=seg,wm_mask_2d=None)
    masks=AnatomicalMasks2D(brain_support_mask=support,whole_tumor_mask=lesion,
        edema_mask=lesion,necrotic_non_enhancing_mask=np.zeros_like(lesion),
        enhancing_tumor_mask=np.zeros_like(lesion),white_matter_mask=None,
        domain_representation_type='brain_parenchyma_proxy',warnings=[],
        preserved_internal_holes_count=0,filled_small_artifact_holes_count=0,
        removed_isolated_components_count=0,lesion_inside_support_ratio=1.)
    transform=AnatomicalCoordinateTransform2D.from_ras_affine_and_mask(
        affine,slice_index=0,support_mask=support,L0_m=.01)
    coarse=generate_anatomical_quad_mesh_2d(masks,sl,transform,
        MeshGenerationConfig(target_element_size_mm=3.,boundary_projection_iterations=0,
                             laplacian_smoothing_iterations=0))
    fine=generate_anatomical_quad_mesh_2d(masks,sl,transform,
        MeshGenerationConfig(target_element_size_mm=3.,roi_fine_element_size_mm=1.5,
                             roi_refinement_margin_mm=4.,boundary_projection_iterations=0,
                             laplacian_smoothing_iterations=0))
    assert len(fine.quads)>len(coarse.quads)
    assert np.sum(fine.cell_tumor_label==2)>np.sum(coarse.cell_tumor_label==2)
    # Conforming quad grid: each edge belongs to exactly one or two elements.
    edges={}
    for cell in fine.quads:
        for a,b in zip(cell,np.roll(cell,-1)):
            key=tuple(sorted((int(a),int(b))))
            edges[key]=edges.get(key,0)+1
    assert set(edges.values()) <= {1,2}
    assert np.all(np.isin(fine.cell_tumor_label,[0,2]))
