#!/usr/bin/env python3
"""Quantitative audit of BraTS tumor/edema label representation on the exported quad mesh.

This is a *geometry/annotation fidelity audit*, not a biomechanical or clinical
validation. It uses the original RAS+ BraTS segmentation and the already exported
mesh bundle; it does not change the mesh, solver, tissue properties, or manuscript.

Usage (from repository root):
  python scripts/audit_brats_roi_mesh.py \
      --config configs/brats_anatomical_forward_review_v2.yaml \
      --geometry-report artifacts/brats_anatomical_phase1_model_review_v2/geometry_quality_report.json \
      --out-dir artifacts/brats_roi_mesh_audit_v2
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.path import Path as PolygonPath
import numpy as np
from scipy import ndimage as ndi
import yaml

ROOT = Path(__file__).resolve().parents[1]
LABELS = {1: 'NCR/NET', 2: 'ED (peritumoral edema)', 4: 'ET (enhancing)'}
LABEL_COLORS = {1: '#E56667', 2: '#F5C542', 4: '#3196D2'}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _absolute(path: str | Path, *, base: Path = ROOT) -> Path:
    p = Path(path)
    return (p if p.is_absolute() else base / p).resolve()


def vertices_to_slice_ij(vertices_mm: np.ndarray, affine: np.ndarray, k: int) -> np.ndarray:
    """Physical xy -> continuous voxel i,j on a fixed axial section.

    Only axis-aligned-through-plane 2D slices are supported. An oblique 3D
    section requires a true plane/volume intersection rather than z=k sampling.
    """
    a = np.asarray(affine, dtype=float)
    if a.shape != (4, 4) or np.linalg.matrix_rank(a[:2, :2]) != 2:
        raise ValueError('Invalid voxel-to-physical affine for 2D audit.')
    if np.max(np.abs(a[2, :2])) > 1e-5:
        raise ValueError('Oblique slice: physical z varies with in-plane voxel index; audit not defined.')
    xy_shift = a[:2, 3] + k * a[:2, 2]
    return np.linalg.solve(a[:2, :2], np.asarray(vertices_mm, float).T - xy_shift[:, None]).T


def rasterize_cell_labels(
    ij_vertices: np.ndarray, quads: np.ndarray, cell_labels: np.ndarray,
    shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Assign voxel centers to exported mesh cells; -1 means outside all cells.

    Uses exact polygon point tests in voxel-index coordinates; this measures
    fidelity of *cell-wise material labels* to a voxel segmentation. It does not
    assert that quadrature-level properties are implemented in the FEM engine.
    """
    sx, sy = map(int, shape)
    cell_map = np.full((sx, sy), -1, dtype=np.int32)
    overlap_count = 0
    for cid, ids in enumerate(np.asarray(quads, dtype=np.int64)):
        poly = ij_vertices[ids]
        lo = np.maximum(np.floor(np.min(poly, axis=0) - 1).astype(int), [0, 0])
        hi = np.minimum(np.ceil(np.max(poly, axis=0) + 1).astype(int), [sx-1, sy-1])
        if np.any(lo > hi):
            continue
        ii, jj = np.meshgrid(np.arange(lo[0], hi[0]+1),
                             np.arange(lo[1], hi[1]+1), indexing='ij')
        xy = np.column_stack([ii.ravel(), jj.ravel()])
        inside = PolygonPath(poly).contains_points(xy, radius=1e-8)
        pos = xy[inside]
        if len(pos) == 0:
            continue
        old = cell_map[pos[:, 0], pos[:, 1]]
        overlap_count += int(np.count_nonzero(old >= 0))
        # Do not silently resolve overlapping mesh cells by last-write-wins.
        fresh = old < 0
        cell_map[pos[fresh, 0], pos[fresh, 1]] = int(cid)
    pred = np.full((sx, sy), -1, dtype=np.int32)
    valid = cell_map >= 0
    pred[valid] = np.asarray(cell_labels, dtype=np.int32)[cell_map[valid]]
    return cell_map, pred, overlap_count


def _metrics(ref: np.ndarray, pred: np.ndarray, voxel_mm: tuple[float,float]) -> dict[str, Any]:
    a = np.asarray(ref, bool)
    b = np.asarray(pred, bool)
    tp = int(np.count_nonzero(a & b)); fp = int(np.count_nonzero(~a & b))
    fn = int(np.count_nonzero(a & ~b))
    denom = (2 * tp + fp + fn)
    dice = (2 * tp / denom) if denom else 1.0
    iou = (tp / (tp + fp + fn)) if (tp+fp+fn) else 1.0
    d = {'reference_pixels':int(a.sum()), 'mesh_label_pixels':int(b.sum()),
         'true_positive_pixels':tp,'false_positive_pixels':fp,'false_negative_pixels':fn,
         'dice':round(dice,6),'iou':round(iou,6),
         'recall':round(tp/(tp+fn),6) if tp+fn else None,
         'precision':round(tp/(tp+fp),6) if tp+fp else None,
         'hausdorff95_boundary_mm':None}
    if np.any(a) and np.any(b):
        # Pixelized boundaries (95th percentile bidirectional surface distance).
        ea = a & ~ndi.binary_erosion(a, border_value=0)
        eb = b & ~ndi.binary_erosion(b, border_value=0)
        if np.any(ea) and np.any(eb):
            da = ndi.distance_transform_edt(~ea, sampling=voxel_mm)
            db = ndi.distance_transform_edt(~eb, sampling=voxel_mm)
            d['hausdorff95_boundary_mm'] = round(
                float(np.percentile(np.r_[da[eb], db[ea]],95)),5)
    return d


def analyze_roi_mesh(seg: np.ndarray, vertices_mm: np.ndarray, quads: np.ndarray,
                     cell_labels: np.ndarray, affine: np.ndarray, slice_index: int
                    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    seg = np.asarray(seg, dtype=np.int32)
    labels = np.asarray(cell_labels, dtype=np.int32)
    if seg.ndim != 2 or np.any(~np.isin(np.unique(seg), [0,1,2,4])):
        raise ValueError('Expected a 2D BraTS segmentation containing only labels 0,1,2,4.')
    if labels.ndim != 1 or labels.shape[0] != len(quads):
        raise ValueError('Cell label count does not match the number of quadrilaterals.')
    if np.any(~np.isin(labels,[0,1,2,4])):
        raise ValueError('Unexpected mesh material labels; expected BraTS 0,1,2,4.')
    ijv = vertices_to_slice_ij(vertices_mm, affine, slice_index)
    cmap, pred, overlap = rasterize_cell_labels(ijv, quads, labels, seg.shape)
    spacing = tuple(float(np.linalg.norm(affine[:3,j])) for j in (0,1))
    cell_areas_mm2 = []
    cell_diam_mm = []
    for quad in quads:
        p = vertices_mm[quad]
        x,y = p[:,0],p[:,1]
        cell_areas_mm2.append(abs(np.sum(x*np.roll(y,-1)-y*np.roll(x,-1)))*0.5)
        cell_diam_mm.append(float(max(np.linalg.norm(p[0]-p[2]),np.linalg.norm(p[1]-p[3]))))
    cell_areas_mm2 = np.asarray(cell_areas_mm2)
    cell_diam_mm = np.asarray(cell_diam_mm)
    region = {}
    for label,name in LABELS.items():
        sel = labels == label
        region[str(label)] = {'region_name':name,
                              'mesh_cells_center_labeled':int(sel.sum()),
                              'mesh_cell_area_mm2':round(float(cell_areas_mm2[sel].sum()),4),
                              'mean_cell_diameter_mm':round(float(cell_diam_mm[sel].mean()),4) if sel.any() else None,
                              'p95_cell_diameter_mm':round(float(np.percentile(cell_diam_mm[sel],95)),4) if sel.any() else None,
                              **_metrics(seg==label,pred==label,spacing)}
    region['whole_tumor'] = _metrics(np.isin(seg,[1,2,4]), np.isin(pred,[1,2,4]),spacing)
    # Cells intersected by multiple true BraTS labels are especially important
    # around the necrotic core, enhancing rim, and edema interfaces.
    mixed = 0; edema_mixed = 0; unlabeled_lesion = 0
    for cid in range(len(quads)):
        seen = np.unique(seg[cmap == cid])
        if len(seen) > 1:
            mixed += 1
            if 2 in seen:
                edema_mixed += 1
        if np.any(np.isin(seen,[1,2,4])) and labels[cid] == 0:
            unlabeled_lesion += 1
    ref_tumor = np.isin(seg,[1,2,4]); covered = cmap >= 0
    out = {'scope':'2D ROI/mapping audit only; not clinical validation',
           'slice_index':int(slice_index),'shape_2d':[int(x) for x in seg.shape],
           'voxel_spacing_mm':list(spacing),'num_quad_cells':len(quads),
           'num_mesh_vertices':len(vertices_mm),'overlapping_voxel_claims':int(overlap),
           'num_voxels_inside_mesh':int(covered.sum()),
           'true_tumor_voxels_inside_mesh':int(np.count_nonzero(ref_tumor & covered)),
           'true_tumor_voxels_outside_mesh':int(np.count_nonzero(ref_tumor & ~covered)),
           'tumor_coverage_fraction':round(float(np.count_nonzero(ref_tumor & covered)/max(ref_tumor.sum(),1)),6),
           'num_cells_crossing_any_brats_interface':mixed,
           'num_cells_crossing_edema_interface':edema_mixed,
           'num_background_labeled_cells_intersecting_true_lesion':unlabeled_lesion,
           'region_metrics':region,
           'interpretation_note':'BraTS labels encode radiologic regions; ED may include infiltrating tumor. '
                'A cell-material ID does not imply region-dependent FEM coefficients.'}
    return out, cmap, pred, ijv


def plot_audit(flair: np.ndarray, seg: np.ndarray, pred: np.ndarray,
               ij_vertices: np.ndarray, quads: np.ndarray, outfile: Path) -> None:
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    raw = np.asarray(flair, float)
    vals = raw[np.isfinite(raw) & (raw!=0)]
    lo,hi = tuple(np.percentile(vals,[2,98])) if vals.size else (0,1)
    if hi <= lo: hi=lo+1
    fig,axes=plt.subplots(1,3,figsize=(12.0,4.3),layout='constrained')
    legend = [Patch(facecolor=LABEL_COLORS[l],label=f'{l}: {LABELS[l]}') for l in LABELS]
    true_tumor=np.isin(seg,[1,2,4]); ii,jj=np.where(true_tumor)
    if len(ii):
        x0=max(int(np.min(ii))-18,0); x1=min(int(np.max(ii))+19,seg.shape[0])
        y0=max(int(np.min(jj))-18,0); y1=min(int(np.max(jj))+19,seg.shape[1])
    else:
        x0,y0=0,0; x1,y1=seg.shape
    for ax in axes:
        ax.imshow(raw.T,origin='lower',cmap='gray',vmin=lo,vmax=hi,interpolation='none')
        ax.set(xlim=(x0,x1),ylim=(y0,y1),xlabel='Voxel i (1 mm reference grid)',ylabel='Voxel j')
        ax.set_aspect('equal')
    def overlay(ax, labels_arr):
        rgba=np.zeros((seg.shape[0],seg.shape[1],4),float)
        import matplotlib.colors as mcolors
        for lb, hexcol in LABEL_COLORS.items():
            mask=labels_arr==lb
            rgba[mask,:3]=mcolors.to_rgb(hexcol)
            rgba[mask,3]=0.52
        ax.imshow(np.transpose(rgba,(1,0,2)),origin='lower',interpolation='none')
    overlay(axes[1],seg)
    overlay(axes[2],pred)
    coll=PolyCollection(ij_vertices[quads],facecolors='none',edgecolors='#52ffbd',
                        linewidths=0.27,alpha=0.7)
    axes[2].add_collection(coll)
    axes[0].set_title('FLAIR: selected BraTS slice')
    axes[1].set_title('Reference BraTS annotation')
    axes[2].set_title('Cellwise mesh labels + quad edges')
    fig.legend(handles=legend,loc='outside lower center',ncol=3,frameon=False,fontsize=8)
    fig.savefig(outfile,format='pdf',dpi=200)
    plt.close(fig)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/brats_anatomical_forward_review_v2.yaml')
    p.add_argument('--geometry-report',required=True)
    p.add_argument('--mesh-npz',default=None,help='Optional: override mesh bundle path from quality report.')
    p.add_argument('--out-dir',default='artifacts/brats_roi_mesh_audit_v2')
    args=p.parse_args()
    cfg_path=_absolute(args.config); report_path=_absolute(args.geometry_report)
    cfg=yaml.safe_load(cfg_path.read_text(encoding='utf8'))
    rep=json.loads(report_path.read_text(encoding='utf8'))
    if rep.get('source_data_kind') != 'clinical_brats2020' or not rep.get('provenance_verified'):
        raise ValueError('Geometry report does not attest provenance-verified BraTS volumes.')
    mesh_path=_absolute(args.mesh_npz or rep['exported_files']['numpy_bundle_npz'])
    subject_dir=_absolute(cfg['subject_dir']); fnames=cfg['modality_filenames']
    seg_path=(subject_dir/fnames['seg']).resolve()
    flair_path=(subject_dir/fnames['FLAIR']).resolve()
    expected_sha=rep.get('input_provenance',{}).get('sha256_by_volume',{}).get('seg')
    if expected_sha and _sha256(seg_path) != expected_sha:
        raise ValueError('Segmentation SHA-256 differs from provenance report. Wrong MRI subject/version!')
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError('nibabel is required. Use the repository environment (`source ./activate_reteco.sh`).') from exc
    seg_img=nib.as_closest_canonical(nib.load(str(seg_path)))
    flair_img=nib.as_closest_canonical(nib.load(str(flair_path)))
    with np.load(mesh_path) as z:
        k=int(z['slice_index']); verts=z['vertices_mm'].copy(); quads=z['quads'].copy()
        labels=z['cell_tumor_label'].copy(); affine=z['ras_affine'].copy()
    if not np.allclose(affine,seg_img.affine,atol=1e-5):
        raise ValueError('Segmentation RAS affine differs from mesh bundle affine.')
    if not np.allclose(affine,flair_img.affine,atol=1e-5):
        raise ValueError('FLAIR RAS affine differs from mesh bundle affine.')
    if int(rep['slice_index']) != k:
        raise ValueError('Mesh slice disagrees with geometry report.')
    seg=np.rint(np.asarray(seg_img.dataobj[:,:,k],dtype=float)).astype(np.int32)
    flair=np.asarray(flair_img.dataobj[:,:,k],dtype=float)
    audit,_,pred,ijv=analyze_roi_mesh(seg,verts,quads,labels,affine,k)
    audit['subject_id']=rep['subject_id']
    audit['segmentation_sha256']=_sha256(seg_path)
    audit['mesh_npz_sha256']=_sha256(mesh_path)
    audit['geometry_report_sha256']=_sha256(report_path)
    audit['geometry_report_all_mandatory_checks_passed']=bool(rep.get('all_mandatory_checks_passed',False))
    out=_absolute(args.out_dir)
    if out.exists():
        raise FileExistsError(f'Output directory already exists, refusing to overwrite: {out}')
    out.mkdir(parents=True)
    (out/'brats_roi_mesh_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf8')
    plot_audit(flair,seg,pred,ijv,quads,out/'fig_brats_roi_mesh_audit.pdf')
    print(f'ROI audit saved: {out}')
    for lbl,item in audit['region_metrics'].items():
        print(f"  {lbl}: Dice={item['dice']:.4f}, HD95={item['hausdorff95_boundary_mm']} mm, "
              f"recall={item['recall']}, mapped_cells={item.get('mesh_cells_center_labeled','-')}")
    print('  Cells crossing edema interfaces:',audit['num_cells_crossing_edema_interface'])
    print('  Tumor pixels outside computational domain:',audit['true_tumor_voxels_outside_mesh'])


if __name__=='__main__':
    main()
