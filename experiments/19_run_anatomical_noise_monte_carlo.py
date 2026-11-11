#!/usr/bin/env python3
"""Reproducible nonlinear FEM Monte Carlo on frozen heterogeneous BraTS geometry.

Four-parameter inverse (alpha,beta,D,E), reference/solver both on the frozen
3-mm, 48-step anatomical mesh. This isolates observational noise from the
independently studied mesh discrepancy. Uses *real* forward FEM evaluations
for each noisy inverse, not a linearized substitute.

- Gaussian iid additive noise; SD = fraction times COMMON balanced reference RMS.
- Frozen Phase-16 16-sensor designs, 5 observation times, exact Q1 sampling.
- Box-constrained TRF least squares, deterministic multi-start and no prior.
- Progressive on-disk CSV/JSON with resumable experiments.
- Laplace/Gauss-Newton local covariance computed separately from frozen J4.
- A non-converged run is reported as such, never counted as successful.

Time budget: the full 30x4x2 design can entail thousands of expensive FEM solves.
Run a small pilot first and use --max-new-replicates to chunk/resume.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT/'python'))
sys.path.insert(0,str(ROOT))
from reteco.config_schema import ParameterCoordinateTransform


def import18():
    path=Path(__file__).with_name('18_run_anatomical_mesh_independent_inverse.py')
    if not path.exists(): raise FileNotFoundError('Copy experiment 18 beside this script')
    spec=importlib.util.spec_from_file_location('anatomical_phase18_support',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(p,d):
    tmp=p.with_name(p.name+'.partial')
    tmp.write_text(json.dumps(d,indent=2,allow_nan=False,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else x))
    os.replace(tmp,p)


def save_csv(path,rows,fields):
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    os.replace(tmp,path)


def plot_aggregation(rows, out, names):
    fig,ax=plt.subplots(figsize=(6.3,3.5))
    noise=sorted(set(float(r['noise_fraction']) for r in rows))
    x=[];q25=[];q50=[];q75=[];lap=[]
    for f in noise:
        pool=[r for r in rows if float(r['noise_fraction'])==f and str(r['solver_success']).lower()=='true']
        if not pool:continue
        vals=np.array([abs(float(r['beta_estimate'])-.45) for r in pool])
        x.append(f*100);q25.append(np.percentile(vals,25));q50.append(np.median(vals));q75.append(np.percentile(vals,75))
        lap.append(float(pool[0]['laplace_beta_sd']))
    if x:
        ax.plot(x,q50,'o-',label='Mediana |β̂−β†|, inversione FEM')
        ax.fill_between(x,q25,q75,alpha=.20,label='Intervallo interquartile')
        ax.plot(x,lap,'s--',label='σβ locale (Laplace)')
        ax.set_xscale('log');ax.set_yscale('symlog',linthresh=1e-3)
        ax.set_xlabel('Rumore nominale, % del segnale RMS comune')
        ax.set_ylabel('Errore assoluto su β / σβ')
        ax.grid(True,alpha=.25);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(out/'fig_noise_beta_error.pdf')
    plt.close(fig)


def aggregate(rows,levels,theta):
    records=[]
    for f in levels:
        pool=[r for r in rows if float(r['noise_fraction'])==float(f)]
        ok=[r for r in pool if str(r['solver_success']).lower()=='true']
        rec={'noise_fraction':float(f),'n_attempted':len(pool),'n_success':len(ok),
             'n_incomplete_or_failed':len(pool)-len(ok)}
        if ok:
            rec['beta_bound_fraction']=float(np.mean([str(r['beta_bound']).lower()=='true' for r in ok]))
            rec['median_abs_beta_error']=float(np.median([abs(float(r['beta_error'])) for r in ok]))
            rec['laplace_beta_sd']=float(ok[0]['laplace_beta_sd'])
            for name in ('alpha','beta','D','E'):
                e=np.array([float(r[f'{name}_error']) for r in ok],float)
                rec[name+'_bias']=float(np.mean(e))
                rec[name+'_rmse']=float(np.sqrt(np.mean(e**2)))
                rec[name+'_empirical_std']=float(np.std(e,ddof=1)) if len(e)>1 else None
        records.append(rec)
    return {'per_noise_level':records,
            'warning':'Statistics are descriptive; full-sample inference requires all prescribed replicates and convergence checks.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--phase16-reference',default='artifacts/16_anatomical_fd_refinement')
    parser.add_argument('--physics-config',default='configs/scales_and_physics.yaml')
    parser.add_argument('--anatomical-config',default='configs/brats_anatomical_forward.yaml')
    parser.add_argument('--mesh-coarse',default='artifacts/02_anatomical_forward_clinical/meshes/BraTS20_Training_001_slice067_h3.0mm.msh')
    parser.add_argument('--mesh-fine',default='artifacts/02_anatomical_forward_clinical/meshes/BraTS20_Training_001_slice067_h2.0mm.msh')
    parser.add_argument('--geometry-report',default='artifacts/02_anatomical_forward_clinical/geometry_quality_report.json')
    parser.add_argument('--materials-config',default='configs/brats_region_materials_phaseBC.yaml')
    parser.add_argument('--material-profile',default='exploratory_contrast')
    parser.add_argument('--coarse-steps',type=int,default=48)
    parser.add_argument('--fine-steps',type=int,default=96)
    parser.add_argument('--design',choices=('balanced','edema'),default='balanced')
    parser.add_argument('--noise-fractions',type=float,nargs='+',default=[.001,.005,.01,.05])
    parser.add_argument('--replicates',type=int,default=30)
    parser.add_argument('--max-new-replicates',type=int,default=0,help='0 means do all remaining, otherwise chunk by this many replica-level pairs')
    parser.add_argument('--starts',type=int,default=2,choices=(1,2,3))
    parser.add_argument('--max-nfev',type=int,default=20)
    parser.add_argument('--fd-step',type=float,default=5e-4)
    parser.add_argument('--jac',choices=('central','forward'),default='forward')
    parser.add_argument('--seed',type=int,default=20261111)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--out-dir',default='artifacts/19_anatomical_noise_monte_carlo')
    args=parser.parse_args()
    if not (args.replicates>=1 and args.max_nfev>=3 and 0<args.fd_step<.02):raise ValueError('Invalid numerical settings')
    if args.max_new_replicates<0 or any(not (0<f<1) for f in args.noise_fractions):raise ValueError('Invalid noise levels/batch')
    if len(set(args.noise_fractions))!=len(args.noise_fractions):raise ValueError('Duplicate noise level')
    mod=import18()
    # Absolutely avoid a persisted env field from Experiment 20.
    if os.getenv('RETECO_QP_MATERIAL_FIELD'):
        raise RuntimeError('Unset RETECO_QP_MATERIAL_FIELD for Phase 19 baseline')
    paths,master,fwd,theta,times,sensors,y0,materials=mod.load_source(args)
    if not np.isclose(args.coarse_steps,48):raise ValueError('Must match frozen 48-step anatomical Phase 16')
    ref_dir=mod.pathof(args.phase16_reference)
    with np.load(ref_dir/'jacobians_and_observations.npz') as npz:
        j4=np.asarray(npz[f'{args.material_profile}_{args.design}_J6'],float)[:,:4]
    design=args.design
    target=np.asarray(y0[design],float)
    ybal=np.asarray(y0['balanced'],float)
    rms_bal=float(np.linalg.norm(ybal)/np.sqrt(len(ybal)))
    if j4.shape!=(len(target),4):raise RuntimeError('Unexpected Jacobian shape')
    solver=mod.AnatomicalEvaluator(master,fwd,paths['mesh_coarse'],paths['geometry_report'],materials,args.coarse_steps,times,{design:sensors[design]})
    ref_check,_=solver.evaluate(theta)
    if np.linalg.norm(ref_check[design]-target)>1e-8*max(1.,np.linalg.norm(target)):
        raise RuntimeError('FEM reference does not match Phase 16. Stop; no simulations launched.')
    transform=ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    bounds=np.array([np.zeros(4),np.ones(4)])
    cov_z_reference=np.linalg.pinv(j4.T@j4,rcond=1e-12)
    if np.linalg.matrix_rank(j4)!=4:raise RuntimeError('Rank deficient local covariance: cannot use Laplace SD')
    ztruth=np.asarray(transform.to_normalized(theta),float)
    bspan=float(master.parameter_space_theta.bounds.beta[1]-master.parameter_space_theta.bounds.beta[0])
    out=mod.pathof(args.out_dir)
    cfg={'experiment':'19_anatomical_noise_monte_carlo','scope':'nonlinear FEM 4-param on heterogeneous BraTS 3mm/48',
         'noise_fraction_levels':args.noise_fractions,'n_replicates_per_level':args.replicates,
         'design':design,'starts':args.starts,'max_nfev':args.max_nfev,'fd_step':args.fd_step,
         'jac_scheme':args.jac,'seed':args.seed,'theta_truth':theta.tolist(),
         'noise_definition':'iid normal with sigma = fraction * RMS(frozen balanced tracer), common across designs',
         'regularization_lambda':0,'rms_balanced':rms_bal,
         'source_hashes':{k:sha(v) for k,v in paths.items()},
         'jacobian_archive_hash':sha(ref_dir/'jacobians_and_observations.npz'),
         'phase18_support_hash':sha(Path(mod.__file__)),'python_script_hash':sha(Path(__file__)),
         'determinism':'separate SeedSequence(seed,level_index,replica_index) per noise draw',
         'optimizer':'scipy.optimize.least_squares(TRF), bounds[0,1] 4 variables, multi-start fixed',
         'statistics':'Laplace covariance from existing local J4; Gaussian likelihood, no prior; nuisance fixed'}
    fields=['noise_fraction','replicate','noise_sigma','true_clean_cost','best_cost','solver_success','status','nfev','optimality','starts','n_fem_solves','alpha_estimate','beta_estimate','D_estimate','E_estimate','alpha_error','beta_error','D_error','E_error','beta_bound','laplace_beta_sd','noise_l2','prediction_relative_error','best_start_id']
    manifest=out/'manifest.json'
    if args.resume:
        if not manifest.is_file():raise FileNotFoundError('Cannot resume; manifest is absent')
        old=json.loads(manifest.read_text())
        for k,v in cfg.items():
            if old.get(k)!=v:raise ValueError(f'Cannot resume: configuration changed: {k}')
        p=out/'results.csv'
        with p.open() as f: rows=list(csv.DictReader(f))
    else:
        if out.exists():raise FileExistsError('Use a fresh --out-dir or --resume')
        out.mkdir(parents=True)
        save_json(manifest,{**cfg,'created_utc':datetime.now(timezone.utc).isoformat()})
        rows=[]
    finished={(float(r['noise_fraction']),int(r['replicate'])) for r in rows}
    first_starts=[np.array(v,float) for v in ((.55,.55,.55,.55),(.20,.80,.35,.75),(.80,.20,.75,.35))][:args.starts]
    nnew=0
    for il,fraction in enumerate(args.noise_fractions):
        sigma=fraction*rms_bal
        for rep in range(args.replicates):
            if (fraction,rep) in finished:continue
            if args.max_new_replicates and nnew>=args.max_new_replicates:break
            noise=np.random.default_rng(np.random.SeedSequence([args.seed,il,rep])).standard_normal(len(target))*sigma
            noisy=target+noise
            def residual(z):
                th=np.asarray(transform.from_normalized(np.asarray(z,float)),float)
                yp,_=solver.evaluate(th)
                return (yp[design]-noisy)/sigma
            def jacobian(z):
                if args.jac=='central':return mod.finite_difference_jacobian(residual,z,args.fd_step)
                # Explicit forward one-sided difference with bound-aware stencil.
                z=np.asarray(z,float);base=residual(z);a=np.zeros((len(base),4),float)
                for j in range(4):
                    dz=args.fd_step if z[j]+args.fd_step<=1. else -args.fd_step
                    zz=z.copy();zz[j]+=dz;a[:,j]=(residual(zz)-base)/dz
                return a
            before=solver.nsolves
            best=None
            for i,initial in enumerate(first_starts):
                print(f'[19] noise={100*fraction:.2f}% replica={rep+1}/{args.replicates} start {i+1}/{len(first_starts)}',flush=True)
                opt=least_squares(residual,initial.copy(),jac=jacobian,bounds=(bounds[0],bounds[1]),method='trf',
                                  ftol=1e-8,xtol=1e-8,gtol=1e-8,max_nfev=args.max_nfev)
                score=.5*float(np.dot(opt.fun,opt.fun))
                if best is None or score<best[0]:best=(score,opt,i+1)
            bestcost,opt,beststart=best
            z=np.asarray(opt.x,float)
            estimate=np.asarray(transform.from_normalized(z),float)
            ypred,_=solver.evaluate(estimate)
            diff=estimate-theta
            row={'noise_fraction':float(fraction),'replicate':rep,'noise_sigma':sigma,
                 'true_clean_cost':.5*float(np.dot(noise/sigma,noise/sigma)),
                 'best_cost':bestcost,'solver_success':bool(opt.success),'status':int(opt.status),
                 'nfev':int(opt.nfev),'optimality':float(opt.optimality),'starts':args.starts,
                 'n_fem_solves':solver.nsolves-before,
                 **{f'{key}_estimate':float(estimate[j]) for j,key in enumerate(mod.PARAMS)},
                 **{f'{key}_error':float(diff[j]) for j,key in enumerate(mod.PARAMS)},
                 'beta_bound':bool(z[1]<1e-5 or z[1]>1-1e-5),
                 'laplace_beta_sd':float(sigma*np.sqrt(cov_z_reference[1,1])*bspan),
                 'noise_l2':float(np.linalg.norm(noise)),
                 'prediction_relative_error':float(np.linalg.norm(ypred[design]-target)/max(np.linalg.norm(target),1e-15)),
                 'best_start_id':beststart}
            rows.append(row)
            save_csv(out/'results.csv',rows,fields)
            save_json(out/'summary.json',aggregate(rows,args.noise_fractions,theta))
            save_json(out/'progress.json',{'completed':len(rows),'requested':len(args.noise_fractions)*args.replicates,
                     'last_record':{'noise_fraction':fraction,'replicate':rep,'solver_success':bool(opt.success)},
                     'total_forward_solves_since_start':solver.nsolves})
            print(f"    best cost={bestcost:.4g}; beta={estimate[1]:.5f}; success={opt.success}; forward solves={solver.nsolves-before}",flush=True)
            nnew+=1
        if args.max_new_replicates and nnew>=args.max_new_replicates:break
    plot_aggregation(rows,out,mod.PARAMS)
    all_done=len(finished)+nnew>=len(args.noise_fractions)*args.replicates
    print(f'[19] completed {len(rows)}/{len(args.noise_fractions)*args.replicates}, all_done={all_done}; output={out}',flush=True)

if __name__=='__main__':main()
