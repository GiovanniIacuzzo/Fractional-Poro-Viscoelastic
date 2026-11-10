"""Synthetic, FEM-independent regression tests for the ROI mesh audit."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np

SCRIPT=Path(__file__).resolve().parents[1]/'scripts'/'audit_brats_roi_mesh.py'
spec=importlib.util.spec_from_file_location('audit_brats_roi_mesh',SCRIPT)
audit=importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(audit)


def test_region_metrics_exact_cell_alignment():
    # 4 cells on a 4x4 voxel grid, so every cell contains 2x2 voxel centers.
    verts=np.array([[x,y] for x in (-.5,1.5,3.5) for y in (-.5,1.5,3.5)],float)
    quads=np.array([[0,3,4,1],[1,4,5,2],[3,6,7,4],[4,7,8,5]])
    labels=np.array([0,2,1,4])
    seg=np.array([[0,0,2,2], [0,0,2,2], [1,1,4,4], [1,1,4,4]],dtype=int)
    affine=np.eye(4)
    result,cellmap,pred,_=audit.analyze_roi_mesh(seg,verts,quads,labels,affine,0)
    assert np.array_equal(pred,seg)
    assert result['overlapping_voxel_claims']==0
    assert result['region_metrics']['2']['dice']==1.0
    assert result['region_metrics']['whole_tumor']['dice']==1.0
    assert result['num_cells_crossing_edema_interface']==0


def test_missing_edema_captured():
    verts=np.array([[-.5,-.5],[3.5,-.5],[3.5,3.5],[-.5,3.5]],float)
    quads=np.array([[0,1,2,3]])
    seg=np.zeros((4,4),int);seg[:2,:]=2
    result,_,_,_=audit.analyze_roi_mesh(seg,verts,quads,np.array([0]),np.eye(4),0)
    assert result['region_metrics']['2']['dice']==0.0
    assert result['num_cells_crossing_edema_interface']==1
    assert result['num_background_labeled_cells_intersecting_true_lesion']==1


def test_oblique_slice_rejected():
    aff=np.eye(4);aff[2,0]=.1
    try: audit.vertices_to_slice_ij(np.array([[0.,0.]]),aff,0)
    except ValueError as e: assert 'Oblique' in str(e)
    else: raise AssertionError('Expected exception for oblique slice')
