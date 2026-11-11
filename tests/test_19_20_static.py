"""Lightweight tests that do NOT require the compiled deal.II module."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import numpy as np

BASE=Path(__file__).resolve().parents[1]


def load(rel,name):
    path=BASE/rel
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def test_richardson_vector_fit_known_power():
    # Import script 20 without importing compiled solver at module scope.
    sys.path.insert(0,str(BASE/'python'))
    m=load('experiments/20_run_anatomical_fixed_material_crossmesh.py','phase20_static_test')
    yy=np.array([1.5,2.5,4.0,0.7])
    c=np.array([.01,-.02,.03,.05])
    hs=[3.,2.,1.5]
    fit=m.richardson_vector_fit([yy+c*(h**1.8) for h in hs],hs)
    assert abs(fit['empirical_p']-1.8)<1e-3,fit
    assert fit['fit_residual_relative_to_3mm_2mm_difference']<1e-6


def test_material_field_source_constant_independent_of_mesh(tmp_path,monkeypatch):
    sys.path.insert(0,str(BASE/'python'))
    m=load('python/reteco/anatomy/fixed_material_field.py','fixed_material_field_test')
    labels=np.zeros((240,240,70),dtype=int)
    labels[100:140,100:145,67]=2
    labels[110:120,110:130,67]=1
    affine=np.array([[1.,0,0,-239],[0,1,0,0],[0,0,1,0],[0,0,0,1.]])
    img=types.SimpleNamespace(dataobj=labels,shape=labels.shape,affine=affine)
    fake_nib=types.SimpleNamespace(load=lambda p:img,as_closest_canonical=lambda im:im)
    monkeypatch.setitem(sys.modules,'nibabel',fake_nib)
    seg=tmp_path/'seg.nii';seg.write_bytes(b'testing artificial labels')
    report={'slice_index':67,'L0_mm':10.,
           'coordinate_transform':{'ras_affine_4x4':affine.tolist(),
               'origin_xy_mm':[-118.81270138848204,106.35169019860565]}}
    geom=tmp_path/'report.json';geom.write_text(json.dumps(report))
    contrasts={i:dict(stiffness=1.,hydraulic_mobility=1.,tracer_diffusivity=1.,porosity=1.) for i in (0,1,2,4)}
    contrasts[2].update(stiffness=.7,hydraulic_mobility=1.6,tracer_diffusivity=1.4,porosity=1.2)
    filename=tmp_path/'field.txt'
    meta=m.create_material_file(seg,geom,contrasts,filename,smoothing_mm=1.)
    field,geo=m.load_material_grid(filename)
    assert field.shape==(240,240,4)
    assert field[120,140,0]<.95
    assert abs(field[0,0,0]-1)<1e-6
    assert meta['smoothing_gaussian_mm']==1.
    assert np.all(field>0)


def test_patch_is_bounded_and_opt_in():
    installer=load('scripts/install_qp_material_field_patch.py','patch_check')
    # Review invariants in the source installer. It only touches header/cpp on --apply.
    assert installer.CPP.name=='invariant_matrices.cpp'
    assert 'RETECO_QP_MATERIAL_FIELD' in installer.CPP_STRUCT
    assert 'quadrature_point' in installer.transform.__code__.co_consts.__str__()
