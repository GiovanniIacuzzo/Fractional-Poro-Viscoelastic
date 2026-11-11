"""Shared smooth BraTS material raster, evaluated at FEM quadrature points.

One immutable physical field is sampled from the original canonical BraTS labels.
Its values DO NOT depend on which FEM mesh is being evaluated. Gaussian
regularization is a documented modelling choice, fixed in millimetres.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter

LABELS=(0,1,2,4)
KEYS=('stiffness','hydraulic_mobility','tracer_diffusivity','porosity')
MAGIC='RETECO_QP_MATERIAL_V1'


def create_material_file(seg_path, geometry_report, regional_materials,
                         output_path, smoothing_mm=1.0):
    import nibabel as nib
    if not 0.0 <= smoothing_mm <= 5.0:
        raise ValueError('smoothing_mm must lie in [0,5] physical millimeters')
    seg_path=Path(seg_path).resolve()
    report=json.loads(Path(geometry_report).read_text())
    expected_slice=int(report['slice_index'])
    img=nib.as_closest_canonical(nib.load(str(seg_path)))
    aff=np.asarray(img.affine,dtype=float)
    # Only a rectilinear canonical RAS+ grid can be encoded in this simple format.
    if (abs(aff[0,1])+abs(aff[0,2])+abs(aff[1,0])+abs(aff[1,2])>1e-8 or
        aff[0,0]<=0 or aff[1,1]<=0 or abs(aff[2,0])+abs(aff[2,1])>1e-8):
        raise ValueError('Segmentation affine must be rectilinear RAS+; no implicit resampling allowed')
    if not (0<=expected_slice<img.shape[2]):raise ValueError('Slice out of range')
    expected_aff=np.asarray(report['coordinate_transform']['ras_affine_4x4'],float)
    if not np.allclose(aff,expected_aff,atol=1e-6,rtol=0):
        raise ValueError('Segmentation affine disagrees with frozen BraTS geometry report')
    labels=np.asarray(np.asanyarray(img.dataobj)[:,:,expected_slice],dtype=np.int32)
    found=set(np.unique(labels).tolist())
    if not found.issubset(LABELS):raise ValueError(f'Unexpected BraTS labels: {found}')
    reg={int(k):v for k,v in regional_materials.items()}
    if set(reg)!=set(LABELS):raise ValueError('Require regional multipliers for labels 0,1,2,4')
    if not all(k in reg[j] for k in KEYS for j in LABELS):raise ValueError('Missing regional coefficient')
    sx=float(aff[0,0]);sy=float(aff[1,1]);L0=float(report['L0_mm'])
    origin=np.asarray(report['coordinate_transform']['origin_xy_mm'],float)
    x0=(float(aff[0,3])-origin[0])/L0
    y0=(float(aff[1,3])-origin[1])/L0
    dx=sx/L0;dy=sy/L0
    if smoothing_mm==0:
        weights=np.stack([(labels==j).astype(float) for j in LABELS],axis=-1)
    else:
        weights=np.stack([gaussian_filter((labels==j).astype(float),
                sigma=(smoothing_mm/sx,smoothing_mm/sy),mode='nearest') for j in LABELS],axis=-1)
        weights=np.maximum(weights,0)
        weights/=np.sum(weights,axis=-1,keepdims=True)
    field=np.zeros((*labels.shape,len(KEYS)),float)
    for i,key in enumerate(KEYS):
        field[:,:,i]=sum(weights[:,:,li]*float(reg[label][key]) for li,label in enumerate(LABELS))
    if not (np.isfinite(field).all() and (field>0).all()):raise ValueError('Invalid material field values')
    path=Path(output_path).resolve()
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('The material field is immutable; select a new output path')
    with path.open('w') as fh:
        nx,ny=labels.shape
        fh.write(f'{MAGIC} {nx} {ny} {x0:.17g} {y0:.17g} {dx:.17g} {dy:.17g}\n')
        for j in range(ny):
            for i in range(nx):
                fh.write(' '.join(f'{v:.17g}' for v in field[i,j,:])+'\n')
    meta={
        'source_segmentation':str(seg_path),'seg_sha256':hashlib.sha256(seg_path.read_bytes()).hexdigest(),
        'geometry_report_sha256':hashlib.sha256(Path(geometry_report).read_bytes()).hexdigest(),
        'region_multipliers':reg,'slice_index':expected_slice,
        'coordinate_system':'canonical RAS+, physical x/y converted to mesh adimensional via L0',
        'smoothing_gaussian_mm':float(smoothing_mm),
        'shape_xy_voxels':list(labels.shape),'spacing_mm':[sx,sy],
        'grid_origin_adimensional':[x0,y0],'grid_spacing_adimensional':[dx,dy],
        'material_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'discretization':'one fixed physical bilinearly interpolated smooth raster at every FEM quadrature point',
        'note':'No dependence on FEM cell material IDs; field itself is unchanged across mesh levels.'}
    (path.with_suffix(path.suffix+'.json')).write_text(json.dumps(meta,indent=2))
    return meta


def load_material_grid(path):
    path=Path(path)
    with path.open() as fh:
        parts=fh.readline().split()
        if len(parts)!=7 or parts[0]!=MAGIC:raise ValueError('Bad material field header')
        nx,ny=int(parts[1]),int(parts[2]);x0,y0,dx,dy=map(float,parts[3:])
        values=np.loadtxt(fh,dtype=float)
    if values.shape!=(nx*ny,4):raise ValueError('Truncated material field')
    return values.reshape(ny,nx,4).transpose(1,0,2),(x0,y0,dx,dy)
