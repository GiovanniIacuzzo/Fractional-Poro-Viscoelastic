#!/usr/bin/env python3
"""Three-level BraTS cross-mesh experiment with ONE physical material field.

The BraTS label image defines a single smooth coefficient raster in physical RAS
coordinates. An opt-in native C++ patch evaluates its bilinear interpolant at
quadrature points for elasticity, Darcy, porosity and tracer diffusion and
advection. This prevents different material-ID maps being mistaken for pure
spatial FEM error. Boundary approximations still vary by mesh and are reported.

Use modes:
 field : generate immutable material field and (if absent) 1.5-mm mesh, no FEM
 run   : 3 FEM forward solves + one beta-contrast check, using native QP patch
 all   : both. Never overwrites an existing run.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import yaml
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'));sys.path.insert(0,str(ROOT))
from reteco.anatomy.fixed_material_field import create_material_file,load_material_grid


def import18():
    p=Path(__file__).with_name('18_run_anatomical_mesh_independent_inverse.py')
    if not p.is_file():raise FileNotFoundError('Expected experiment 18 alongside experiment 20')
    spec=importlib.util.spec_from_file_location('phase18_cross_mesh_support',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def filehash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def plot_figures(out, observations, sensors, levels):
    for design in sensors:
        y3,y2,y15=(observations[f'{h}_{design}'] for h in levels)
        n=len(sensors[design])
        if len(y3)%n:continue
        d32=(y3-y2).reshape(-1,n);d215=(y2-y15).reshape(-1,n)
        z32=np.linalg.norm(d32,axis=0);z215=np.linalg.norm(d215,axis=0)
        fig,ax=plt.subplots(figsize=(6.4,3.1))
        ax.plot(np.arange(1,n+1),z32,'o-',label='3 mm vs 2 mm')
        ax.plot(np.arange(1,n+1),z215,'s--',label='2 mm vs 1.5 mm')
        ax.set_xlabel('Sensore anatomico congelato');ax.set_ylabel('Norma temporale della discrepanza')
        ax.grid(True,alpha=.24);ax.legend(fontsize=8);fig.tight_layout()
        fig.savefig(out/f'fig_fixedfield_sensor_error_{design}.pdf');plt.close(fig)


def mesh_area(solver):
    # Cell area by shoelace on the same quad connectivity read by the FEM.
    vertices=np.asarray(solver.problem.mesh_vertices,dtype=float)
    quads=np.asarray(solver.problem.cell_quad_connectivity,dtype=int)
    v=vertices[quads]
    return float(.5*np.abs(np.sum(v[:,:,0]*np.roll(v[:,:,1],-1,axis=1)-v[:,:,1]*np.roll(v[:,:,0],-1,axis=1),axis=1)).sum())


def richardson_vector_fit(ys,hs):
    # This is an EMPIRICAL vector fit, not a proof of asymptotic convergence.
    Y=np.stack([np.asarray(y,float) for y in ys],axis=0)
    h=np.asarray(hs,float)
    def fit(p):
        M=np.column_stack((np.ones(3),h**p))
        coef=np.linalg.lstsq(M,Y,rcond=None)[0]
        residual=Y-M@coef
        return float(np.linalg.norm(residual)**2),coef
    sol=minimize_scalar(lambda p:fit(p)[0],bounds=(.25,4),method='bounded',options={'xatol':1e-5})
    ss,coef=fit(sol.x)
    difference=float(np.linalg.norm(Y[0]-Y[1]))
    return {'empirical_p':float(sol.x),'at_fit_boundary':bool(sol.x<.26 or sol.x>3.99),
            'vector_fit_residual_l2':math.sqrt(max(ss,0)),
            'fit_residual_relative_to_3mm_2mm_difference':math.sqrt(max(ss,0))/max(difference,1e-15),
            'extrapolated_observation_norm':float(np.linalg.norm(coef[0])),
            'extrapolated_vs_1p5mm_relative':float(np.linalg.norm(coef[0]-Y[-1])/max(np.linalg.norm(Y[-1]),1e-15)),
            'fit_interpretation':'3-level nonuniform h vector-fit, requires asymptotic validation; boundaries also differ'}


def generate_mesh_1p5(args,path,geometry_path):
    if path.exists():
        print(f'[20] Reusing existing mesh: {path}',flush=True);return
    from dataclasses import asdict
    path.parent.mkdir(parents=True,exist_ok=True)
    from reteco.anatomy.brats_geometry import (
        load_brats_pipeline_config,load_brats_subject,select_anatomical_slice,
        build_anatomical_masks_2d,AnatomicalCoordinateTransform2D,
        MeshGenerationConfig,generate_anatomical_quad_mesh_2d,
        export_dealii_quad_mesh,compute_geometry_quality_report)
    cfg=load_brats_pipeline_config(args.anatomical_config)
    volumes=load_brats_subject(subject_dir=cfg.subject_dir,subject_id=cfg.subject_id,
        modality_filenames=cfg.modality_filenames,wm_mask_path=cfg.wm_mask_path,
        allow_grid_realignment=cfg.allow_grid_realignment,remap_label_3_to_4=cfg.remap_label_3_to_4,
        provenance_cfg=cfg.provenance)
    sel=select_anatomical_slice(volumes=volumes,mode=cfg.slice_selection_mode,
                                manual_slice_index=cfg.manual_slice_index)
    geodata=json.loads(Path(geometry_path).read_text())
    if int(sel.slice_index)!=int(geodata['slice_index']):
        raise RuntimeError('Generated mesh uses a different slice from the frozen experiment')
    masks=build_anatomical_masks_2d(slice_sel=sel,
        spacing_xy_mm=(volumes.spacing_mm[0],volumes.spacing_mm[1]),cfg=cfg.support_mask)
    transform=AnatomicalCoordinateTransform2D.from_ras_affine_and_mask(
        ras_affine=volumes.ras_affine,slice_index=sel.slice_index,
        support_mask=masks.brain_support_mask,L0_m=float(geodata['L0_m']))
    m=cfg.mesh
    mcfg=MeshGenerationConfig(target_element_size_mm=1.5,
        boundary_projection_iterations=m.boundary_projection_iterations,
        laplacian_smoothing_iterations=m.laplacian_smoothing_iterations,
        laplacian_relaxation=m.laplacian_relaxation,
        contour_simplify_tolerance_mm=m.contour_simplify_tolerance_mm,
        min_hole_area_to_mesh_mm2=m.min_hole_area_to_mesh_mm2)
    mm=generate_anatomical_quad_mesh_2d(masks,sel,transform,mcfg)
    name=f'{cfg.subject_id}_slice{sel.slice_index:03d}_h1.5mm'
    files=export_dealii_quad_mesh(mm,transform,path.parent,basename=name)
    generated=Path(files['gmsh_msh']).resolve()
    if generated!=path.resolve():raise RuntimeError(f'Unexpected generated path {generated}, wanted {path}')
    rep=compute_geometry_quality_report(volumes,sel,masks,transform,mm,files,cfg.quality_thresholds)
    (path.with_suffix('.quality.json')).write_text(json.dumps(asdict(rep),indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
    if not rep.all_mandatory_checks_passed:
        raise RuntimeError(f'Generated 1.5mm anatomical mesh failed geometry checks: {rep.failed_checks}')
    print(f'[20] Generated 1.5mm anatomical mesh: {path}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--mode',choices=('field','run','all'),default='field')
    p.add_argument('--resume',action='store_true',help='Continue a partial --mode run without rerunning completed forward solves')
    p.add_argument('--phase16-reference',default='artifacts/16_anatomical_fd_refinement')
    p.add_argument('--physics-config',default='configs/scales_and_physics.yaml')
    p.add_argument('--anatomical-config',default='configs/brats_anatomical_forward.yaml')
    p.add_argument('--mesh-coarse',default='artifacts/02_anatomical_forward_clinical/meshes/BraTS20_Training_001_slice067_h3.0mm.msh')
    p.add_argument('--mesh-fine',default='artifacts/02_anatomical_forward_clinical/meshes/BraTS20_Training_001_slice067_h2.0mm.msh')
    p.add_argument('--mesh-extra',default='artifacts/20_anatomical_fixed_material_crossmesh/meshes/BraTS20_Training_001_slice067_h1.5mm.msh')
    p.add_argument('--geometry-report',default='artifacts/02_anatomical_forward_clinical/geometry_quality_report.json')
    p.add_argument('--materials-config',default='configs/brats_region_materials_phaseBC.yaml')
    p.add_argument('--segmentation',default='data/01_raw/BraTS20_Training_001_seg.nii')
    p.add_argument('--material-profile',default='exploratory_contrast')
    p.add_argument('--smoothing-mm',type=float,default=1.0)
    p.add_argument('--coarse-steps',type=int,default=48)
    p.add_argument('--fine-steps',type=int,default=96)
    p.add_argument('--run-steps',type=int,default=48)
    p.add_argument('--out-dir',default='artifacts/20_anatomical_fixed_material_crossmesh')
    p.add_argument('--field-file',default=None)
    args=p.parse_args()
    mod=import18()
    if os.environ.get('RETECO_QP_MATERIAL_FIELD'):
        raise RuntimeError('Unset RETECO_QP_MATERIAL_FIELD before running; script manages immutable field internally')
    paths,master,fwd,theta,times,sensors,y0,materials=mod.load_source(args)
    out=mod.pathof(args.out_dir)
    field=mod.pathof(args.field_file) if args.field_file else out/'shared_brats_material_field.txt'
    extra=mod.pathof(args.mesh_extra)
    if args.mode in ('field','all'):
        if out.exists() and field.exists():raise FileExistsError(f'Field already generated: {field}')
        out.mkdir(parents=True,exist_ok=True)
        seg=mod.pathof(args.segmentation)
        if not seg.is_file():raise FileNotFoundError(f'Requires original BraTS segmentation: {seg}')
        generate_mesh_1p5(args,extra,paths['geometry_report'])
        create_material_file(seg,paths['geometry_report'],materials,field,args.smoothing_mm)
        print(f'[20] Frozen material raster: {field}',flush=True)
    if args.mode=='field':return
    if not field.is_file() or not extra.is_file():raise FileNotFoundError('Run --mode field first (or provide existing field/third mesh)')
    metadata_path=field.with_suffix(field.suffix+'.json')
    if not metadata_path.is_file() or json.loads(metadata_path.read_text())['material_file_sha256']!=filehash(field):
        raise RuntimeError('Material field checksum mismatch. Refusing to run.')
    values,_=load_material_grid(field)
    if float(values.max()-values.min())<.05:raise RuntimeError('Common field lacks expected regional contrast')
    os.environ['RETECO_QP_MATERIAL_FIELD']=str(field.resolve())
    run_out=out/'run'
    if run_out.exists() and not args.resume:
        raise FileExistsError(f'Run results already exist: {run_out}; specify --resume for incomplete runs')
    run_out.mkdir(parents=True,exist_ok=True)
    configs={'phase16_summary_sha256':filehash(paths['phase16_summary']),
             'material_raster_sha256':filehash(field),
             'mesh_3mm_sha256':filehash(paths['mesh_coarse']),
             'mesh_2mm_sha256':filehash(paths['mesh_fine']),
             'mesh_1p5mm_sha256':filehash(extra),
             'smoothing_mm':float(args.smoothing_mm),
             'run_steps':int(args.run_steps),'created_utc':datetime.now(timezone.utc).isoformat(),
             'statement':'Same bilinear field sampled at FEM quadrature, not mesh-cell region labels',
             'caveat':'Outer anatomical polygon still varies by discretization; results cannot isolate pure h-error from geometry error.'}
    manifest_path=run_out/'manifest.json'
    if args.resume:
        if not manifest_path.is_file():raise FileNotFoundError('No completed manifest to resume')
        saved=json.loads(manifest_path.read_text())
        for k,v in configs.items():
            if k!='created_utc' and saved.get(k)!=v:
                raise RuntimeError(f'Resume is unsafe because {k} changed')
        progress_file=run_out/'progress.json'
        if progress_file.is_file():
            record=json.loads(progress_file.read_text())
        elif (run_out/'summary.json').is_file():
            raise RuntimeError('Experiment 20 already finished; no need to resume')
        else:
            record={'meshes':{},'observations':{},'designs':{}}
    else:
        mod.save_json(manifest_path,configs)
        record={'meshes':{},'observations':{},'designs':{}}
    meshmap=[('3mm',paths['mesh_coarse']),('2mm',paths['mesh_fine']),('1p5mm',extra)]
    arrays={'theta_ref':theta,'observation_times':times}
    for d,s in sensors.items(): arrays['sensors_'+d]=s
    observations={}
    saved_observations=run_out/'observations.npz'
    if args.resume and saved_observations.exists():
        with np.load(saved_observations) as existing:
            for key in existing.files:
                arrays[key]=np.asarray(existing[key])
                if any(key==f'{label}_{d}' for label,_ in meshmap for d in sensors):
                    observations[key]=np.asarray(existing[key],float)
    solvers={}
    for label,mesh in meshmap:
        if label in record['meshes'] and all(f'{label}_{d}' in observations for d in sensors):
            print(f'[20] Reusing completed {label} from saved observations',flush=True)
            continue
        print(f'[20] {label}, time steps={args.run_steps}, mesh={mesh}',flush=True)
        solver=mod.AnatomicalEvaluator(master,fwd,mesh,paths['geometry_report'],materials,
                                      args.run_steps,times,sensors)
        # Mandatory ABI/patch verification: physical-field stiffness must not
        # depend on region_materials dict if QP patch is active.
        if label=='3mm':
            unit={int(k):{v:1.0 for v in ('stiffness','hydraulic_mobility','tracer_diffusivity','porosity')} for k in (0,1,2,4)}
            check=mod.AnatomicalEvaluator(master,fwd,mesh,paths['geometry_report'],unit,args.run_steps,times,sensors)
            import scipy.sparse as sparse
            tested={}
            for name in ('K0_csr','K_phiD_c_csr','M_phi_c_csr'):
                a=getattr(solver.problem,name);b=getattr(check.problem,name)
                rel=float(sparse.linalg.norm(a-b)/max(sparse.linalg.norm(a),1e-15))
                tested[name]=rel
                if rel>1e-11:
                    raise RuntimeError(f'Native module IGNORING common material field ({name} differs by {rel:g}). Rebuild patched C++ core!')
            record['matrix_region_id_invariance']=tested
            del check
            print('[20] PASS: QP field patch active (FEM invariant to cell material ID maps)',flush=True)
        y,stats=solver.evaluate(theta,with_diagnostics=True)
        observations.update({f'{label}_{d}':np.asarray(v,float) for d,v in y.items()})
        for d,v in y.items():arrays[f'{label}_{d}']=v
        record['meshes'][label]={**stats,'num_cells':solver.problem.num_active_cells,
                                  'area_adim2':mesh_area(solver),'mesh_sha256':filehash(mesh)}
        mod.save_json(run_out/'progress.json',record)
        np.savez_compressed(run_out/'observations.npz',**arrays)
        solvers[label]=solver
    # Direct difference M2 (alpha=.70, beta=1) vs M4 (alpha=.70, beta=.45)
    # at a common material field (2mm) to define a reference beta signal.
    beta_one=np.asarray(theta,float).copy();beta_one[1]=1.0
    if '2mm' not in solvers:
        solvers['2mm']=mod.AnatomicalEvaluator(master,fwd,paths['mesh_fine'],paths['geometry_report'],materials,
                                               args.run_steps,times,sensors)
    contrast,_=solvers['2mm'].evaluate(beta_one)
    for d in sensors:
        y3=observations[f'3mm_{d}'];y2=observations[f'2mm_{d}'];y15=observations[f'1p5mm_{d}']
        beta_error=np.asarray(contrast[d],float)-y2
        denom=max(float(np.linalg.norm(y2)),1e-15)
        record['designs'][d]={
            'relative_mesh_3mm_vs_2mm':float(np.linalg.norm(y3-y2)/denom),
            'relative_mesh_2mm_vs_1p5mm':float(np.linalg.norm(y2-y15)/denom),
            'relative_M2_vs_M4_beta_signal_2mm':float(np.linalg.norm(beta_error)/denom),
            'ratio_spatial_3mm_2mm_to_beta_signal':float(np.linalg.norm(y3-y2)/max(np.linalg.norm(beta_error),1e-15)),
            'richardson':richardson_vector_fit([y3,y2,y15],[3.,2.,1.5]),
            'top_sensor_discrepancy_norms':sorted((float(v) for v in np.linalg.norm((y3-y2).reshape(-1,len(sensors[d])),axis=0)),reverse=True),
        }
        arrays[f'2mm_{d}_M2_beta1']=np.asarray(contrast[d],float)
        print(f"[20/{d}] 3 vs 2 = {100*record['designs'][d]['relative_mesh_3mm_vs_2mm']:.3g}% "
              f"2 vs 1.5 = {100*record['designs'][d]['relative_mesh_2mm_vs_1p5mm']:.3g}% "
              f"M2-M4 signal = {100*record['designs'][d]['relative_M2_vs_M4_beta_signal_2mm']:.3g}%",flush=True)
    np.savez_compressed(run_out/'observations.npz',**arrays)
    mod.save_json(run_out/'summary.json',record)
    plot_figures(run_out,observations,sensors,[q for q,_ in meshmap])
    (run_out/'progress.json').unlink(missing_ok=True)
    print(f'[20] DONE: {run_out}',flush=True)

if __name__=='__main__':main()
