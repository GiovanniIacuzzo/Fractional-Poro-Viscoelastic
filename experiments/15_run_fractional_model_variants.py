#!/usr/bin/env python3
"""Controlled M1--M4 forward ablation of two independent fractional memories.

M1: alpha=1, beta=1; M2: alpha<1, beta=1;
M3: alpha=1, beta<1; M4: alpha<1, beta<1.

All runs share the same mesh, source vectors, time grid and D,E. The program
produces numerical evidence for *model differences*, not proof that M4 is
physically preferable. It never writes to the legacy manuscript artefacts.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from reteco import _core, load_master_config
from reteco.forward.wrapper import RetecoForwardProblem2D
from reteco.forward.target_generator import (
    build_standard_driving_Callbacks, build_anatomical_driving_Callbacks,
)
from reteco.observation.q1_exact import ExactQ1TracerObservationOperator
from reteco.anatomy.regional_materials import load_regional_materials

REGION_NAMES = {0: 'background', 1: 'NCR_NET', 2: 'ED', 4: 'ET'}

NAMES = {
    'M1': 'Integer-order (1,1)',
    'M2': 'Tracer memory (alpha,1)',
    'M3': 'Mechanical memory (1,beta)',
    'M4': 'Coupled memory (alpha,beta)',
}
COLORS = {'M1': '#344054', 'M2': '#007F86', 'M3': '#C0782A', 'M4': '#623F98'}


def variant_parameters(theta_reference):
    """Do not change D,E between hypotheses; alpha,beta from the master config."""
    a, b, D, E = map(float, theta_reference)
    if not (0.0 < a < 1.0 and 0.0 < b < 1.0):
        raise ValueError('The fractional reference must have alpha,beta strictly below one.')
    return {
        'M1': np.array([1., 1., D, E]),
        'M2': np.array([a, 1., D, E]),
        'M3': np.array([1., b, D, E]),
        'M4': np.array([a, b, D, E]),
    }


def l2_mass_norm(mass, field):
    x = np.asarray(field, dtype=np.float64)
    return float(np.sqrt(max(float(x @ (mass @ x)), 0.0)))


def _tracer_initial_dose(problem, c0, baseline):
    """Physical storage-weighted dose, including and excluding the baseline.

    M_phi_c is the dimensionless storage matrix normalized by phi_0;
    multiplying by phi_0 matches stored_tracer_mass in regional_time_series.csv.
    """
    phi0 = float(problem.physics_cpp.phi_0)
    ones = np.ones(problem.num_tracer_dofs, dtype=float)
    mass = problem.M_phi_c_csr
    volume = phi0 * float(ones @ (mass @ ones))
    total = phi0 * float(ones @ (mass @ np.asarray(c0, dtype=float)))
    excess = total - baseline * volume
    return {'storage_volume': volume, 'total_stored_mass': total,
            'excess_mass_above_baseline': excess}


def _reference_excess_dose(directory, *, baseline, mesh_sha, driving_sha, physics_sha):
    """Read reference M4 dose from a matching homogeneous anatomical pilot.

    The original reference need not have used dose normalization, and its time
    discretization may differ. Geometry and physical protocol must match.
    """
    directory = Path(directory).resolve()
    meta_path = directory / 'run_manifest.json'
    csv_path = directory / 'regional_time_series.csv'
    if not meta_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError('Dose reference requires run_manifest.json and regional_time_series.csv')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if meta.get('domain') != 'anatomical' or meta.get('material_profile') != 'homogeneous':
        raise ValueError('Dose reference must come from a homogeneous anatomical experiment.')
    for key, current in (('mesh_sha256',mesh_sha),('driving_sha256',driving_sha),
                         ('physics_sha256',physics_sha)):
        if meta.get(key) != current:
            raise ValueError(f'Dose reference mismatch for {key}; do not compare different protocols.')
    reference_baseline = meta.get('driving_metadata',{}).get('initial_tracer_gaussian',{}).get('baseline')
    if reference_baseline is None or not np.isclose(float(reference_baseline), baseline, atol=1e-12, rtol=0):
        raise ValueError('Dose reference and current tracer baseline differ.')
    with csv_path.open('r',encoding='utf-8',newline='') as stream:
        rows = [row for row in csv.DictReader(stream)
                if row['model'] == 'M4' and np.isclose(float(row['time']),0.0,atol=1e-13)]
    if not rows:
        raise ValueError('Dose reference has no M4 records at t=0.')
    mass = sum(float(row['stored_tracer_mass']) for row in rows)
    volume = sum(float(row['storage_volume']) for row in rows)
    excess = mass - baseline * volume
    if not np.isfinite(excess) or excess <= 0:
        raise ValueError('Reference initial excess dose must be finite and positive.')
    return float(excess)


def _series(problem, traj):
    return {
        'd_L2': np.array([problem.compute_disp_L2_norm(x) for x in traj.d_hist]),
        'p_L2': np.array([problem.compute_press_L2_norm(x) for x in traj.p_hist]),
        'c_L2': np.array([problem.compute_tracer_L2_norm(x) for x in traj.c_hist]),
        # Storage-weighted mass is conserved for spatially varying porosity.
        'c_mass': np.asarray(traj.c_hist @ (problem.M_phi_c_csr @ np.ones(problem.num_tracer_dofs))).ravel(),
    }


def _choose_anatomical_sensors(problem, count=16):
    """Balanced deterministic sensor locations over radiological ROIs.

    Only synthetic observations: these are *not* MRI-derived tracer sensors.
    """
    con = np.asarray(problem.cell_tracer_dof_connectivity, dtype=int)
    coords = np.asarray(problem.tracer_pts)
    centers = np.mean(coords[con], axis=1)
    labels = np.asarray(problem.cell_material_ids, dtype=int)
    if centers.size == 0:
        raise ValueError('No cells found for anatomical observation locations.')
    samples = []
    for lab in REGION_NAMES:
        ids = np.where(labels == lab)[0]
        if not len(ids):
            continue
        order = ids[np.lexsort((centers[ids, 1], centers[ids, 0]))]
        selected = order[np.unique(np.linspace(0, len(order)-1, min(count//4,len(order)), dtype=int))]
        samples.extend(selected.tolist())
    if not samples:
        raise ValueError('No registered BraTS-region cells for sensors.')
    return centers[np.asarray(samples, dtype=int)]


def _q1_quad_integral_weights(problem):
    """2x2 Gauss quadrature weights for exact Q1 cell integrals on isoparametric quads.

    Returns cell areas and shape-function integral weights for each cell.
    """
    vertices = np.asarray(problem.mesh_vertices, dtype=float)[problem.cell_quad_connectivity]
    weights = np.zeros((len(vertices), 4), dtype=float)
    g = 1. / np.sqrt(3.)
    for xi in (-g, g):
        for eta in (-g, g):
            N = np.array([(1-xi)*(1-eta), (1+xi)*(1-eta),
                          (1+xi)*(1+eta), (1-xi)*(1+eta)], dtype=float) / 4.
            Nx = np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)]) / 4.
            Ny = np.array([-(1-xi), -(1+xi), (1+xi), (1-xi)]) / 4.
            a = np.einsum('i,cij->cj', Nx, vertices)
            b = np.einsum('i,cij->cj', Ny, vertices)
            det = a[:,0]*b[:,1] - a[:,1]*b[:,0]
            if not np.all(np.isfinite(det)) or np.min(det) <= 0:
                raise ValueError('Nonpositive Q1 cell Jacobian in ROI integration.')
            weights += det[:,None] * N[None,:]
    return np.sum(weights, axis=1), weights


def _export_region_series(out, problem, times, trajectories):
    area, weights = _q1_quad_integral_weights(problem)
    labels = np.asarray(problem.cell_material_ids, dtype=int)
    conn = np.asarray(problem.cell_tracer_dof_connectivity, dtype=int)
    phi_ref = float(problem.physics_cpp.phi_0)
    profile = problem.region_materials
    columns = ['model','time','region_id','region','area','storage_volume',
               'stored_tracer_mass','storage_weighted_mean_concentration',
               'mean_abs_darcy_flux_midpoint']
    records=[]
    for name, trajectory in trajectories.items():
        for n, t in enumerate(times):
            cell_integral = np.einsum('ci,ci->c', weights, trajectory.c_hist[n,conn])
            flux = np.asarray(problem.fem.compute_cell_darcy_fluxes(trajectory.p_hist[n]))
            qmag = np.linalg.norm(flux, axis=1)
            for tag, label in REGION_NAMES.items():
                which = labels == tag
                if not np.any(which):
                    continue
                phi = phi_ref * profile.get(tag, {}).get('porosity', 1.0)
                a = float(np.sum(area[which]))
                storage_volume = phi * a
                stored = float(phi*np.sum(cell_integral[which]))
                records.append({'model':name,'time':float(t),'region_id':tag,
                    'region':label,'area':a,'storage_volume':storage_volume,
                    'stored_tracer_mass':stored,
                    'storage_weighted_mean_concentration':stored/storage_volume,
                    'mean_abs_darcy_flux_midpoint':float(np.dot(area[which],qmag[which])/a)})
    with (out/'regional_time_series.csv').open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=columns)
        writer.writeheader();writer.writerows(records)

    # Regional curves serve the biomedical interpretation better than global norms alone.
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.0), layout='constrained')
    for ax, tag in zip(axes, (1,2,4)):
        for name in NAMES:
            sub=[rec for rec in records if rec['model']==name and rec['region_id']==tag]
            if sub:
                ax.plot([rec['time'] for rec in sub],
                        [rec['storage_weighted_mean_concentration'] for rec in sub],
                        label=name,color=COLORS[name],lw=1.7)
        ax.set(title=REGION_NAMES[tag], xlabel='Dimensionless time')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Region-mean tracer concentration')
    axes[-1].legend(frameon=False,fontsize=7,ncol=2)
    fig.savefig(out/'fig_regional_tracer_mean.pdf',bbox_inches='tight')
    plt.close(fig)
    return records


def _write_figures(out, problem, times, variants, summary):
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.bbox': 'tight'})
    for quantity, ylab in [('d_L2', r'$\|d_h(t)\|_{L^2}$'),
                           ('p_L2', r'$\|p_h(t)\|_{L^2}$'),
                           ('c_L2', r'$\|c_h(t)\|_{L^2}$')]:
        fig, ax = plt.subplots(figsize=(5.8, 3.3))
        for key in NAMES:
            ax.plot(times, summary[key][quantity], linewidth=1.9, color=COLORS[key],
                    label=NAMES[key])
        ax.set(xlabel='Dimensionless time', ylabel=ylab)
        ax.grid(alpha=0.16, linewidth=0.6)
        ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.savefig(out / f'fig_{quantity}_model_comparison.pdf')
        plt.close(fig)

    # Native quadrilaterals, never an unrelated Delaunay triangulation.
    v = np.asarray(problem.mesh_vertices, dtype=float)
    cells = np.asarray(problem.cell_quad_connectivity, dtype=int)
    polygons = v[cells]
    dofs = np.asarray(problem.cell_tracer_dof_connectivity, dtype=int)
    c_ref = variants['M4'].c_hist[-1]
    diff_fields = [variants[k].c_hist[-1] - c_ref for k in ('M1','M2','M3')]
    diffs_cell = [np.mean(x[dofs], axis=1) for x in diff_fields]
    limit = max(max(float(np.max(abs(x))) for x in diffs_cell), 1.e-15)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), layout='constrained')
    coll = None
    for ax, key, diff in zip(axes, ('M1','M2','M3'), diffs_cell):
        coll = PolyCollection(polygons, array=diff, cmap='RdBu_r',
                              clim=(-limit,limit), edgecolors='none')
        ax.add_collection(coll)
        ax.autoscale_view()
        ax.set(aspect='equal', title=f'{key} minus M4', xlabel='x / L0')
        if ax is axes[0]:
            ax.set_ylabel('y / L0')
    fig.colorbar(coll, ax=list(axes), shrink=0.76, label='Cell-mean concentration difference')
    fig.savefig(out / 'fig_final_concentration_differences.pdf')
    plt.close(fig)




def _write_anatomical_regions_figure(out, problem, full_trajectory):
    quads = np.asarray(problem.cell_quad_connectivity, dtype=int)
    polygons = np.asarray(problem.mesh_vertices)[quads]
    cell_tags = np.asarray(problem.cell_material_ids, dtype=int)
    cell_dofs = np.asarray(problem.cell_tracer_dof_connectivity, dtype=int)
    c_final = np.asarray(full_trajectory.c_hist[-1])[cell_dofs].mean(axis=1)
    colors = {0:'#d2d8e0',1:'#c94c50',2:'#f4cf69',4:'#528dbe'}
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1,2,figsize=(8.3,3.8),layout='constrained')
    pcs=PolyCollection(polygons,facecolors=[colors.get(int(k),'black') for k in cell_tags],edgecolors='none')
    axes[0].add_collection(pcs)
    axes[0].legend(handles=[Patch(facecolor=colors[k],label=REGION_NAMES[k]) for k in REGION_NAMES],
                   frameon=False,fontsize=7,loc='lower left')
    pc=PolyCollection(polygons,array=c_final,cmap='viridis',edgecolors='none')
    axes[1].add_collection(pc)
    fig.colorbar(pc,ax=axes[1],shrink=.8,label='Cell-mean tracer concentration')
    for ax in axes:
        ax.autoscale_view();ax.set_aspect('equal');ax.set_xlabel('x / L0')
    axes[0].set(ylabel='y / L0',title='MRI-derived BraTS material labels')
    axes[1].set(title='M4 tracer at final time')
    fig.savefig(out/'fig_anatomical_material_labels_and_tracer.pdf',bbox_inches='tight')
    plt.close(fig)

def run(args):
    physics_path = (ROOT / args.physics_config).resolve()
    master = load_master_config(physics_path)
    if args.domain == 'square':
        if args.material_profile is not None:
            raise ValueError('Regional material profiles are only supported with an anatomical imported mesh.')
        drive_path = (ROOT / args.square_config).resolve()
        raw = yaml.safe_load(drive_path.read_text(encoding='utf-8'))
        fwd = raw['inverse_discretization']
        N = args.N or (6 if args.preset == 'smoke' else int(fwd['subdivisions_N']))
        M = args.M or (12 if args.preset == 'smoke' else int(fwd['time_steps_M']))
        T = float(fwd['final_time_T'])
        r = float(fwd['grading_r'])
        grid = _core.TimeGrid.make_graded(T, M, r)
        problem = RetecoForwardProblem2D(
            physics_model=master.fixed_physics, scales=master.characteristic_scales,
            time_grid=grid, subdivisions_N=N, use_supg=False,
            fe_deg_disp=int(fwd.get('fe_deg_disp',2)),
            fe_deg_press=int(fwd.get('fe_deg_press',1)),
            fe_deg_tracer=int(fwd.get('fe_deg_tracer',1)),
        )
        c0, F, Sf = build_standard_driving_Callbacks(problem, raw['driving_protocol'])
        cfg_file = drive_path
        sensors = np.array(np.meshgrid(np.linspace(.17,.83,5), np.linspace(.17,.83,5),
                                       indexing='xy')).reshape(2,-1).T
        driving_meta = {'source': str(drive_path.relative_to(ROOT)),
                        'description': 'Standard synthetic forcing; fluid includes background extraction.'}
    else:
        if not args.mesh or not Path(args.mesh).is_file():
            raise FileNotFoundError('--mesh requires an existing anatomical quadrilateral .msh file (run the geometry pipeline first).')
        if not args.geometry_report or not Path(args.geometry_report).is_file():
            raise FileNotFoundError('--geometry-report requires the matching geometry_quality_report.json; do not use a domain-centroid fallback for a paper run.')
        if args.N is not None:
            raise ValueError('--N is only meaningful on the square; for anatomy change the mesh resolution.')
        report_obj = json.loads(Path(args.geometry_report).read_text(encoding='utf-8'))
        exported = report_obj.get('exported_files', {}).get('gmsh_msh')
        if exported:
            declared = Path(exported)
            declared = declared.resolve() if declared.is_absolute() else (ROOT/declared).resolve()
            if declared != Path(args.mesh).resolve():
                raise ValueError('Selected .msh does not match geometry report exported mesh path.')
        if not report_obj.get('all_mandatory_checks_passed', False):
            raise ValueError('Geometry quality checks did not pass; do not execute FEM.')
        drive_path = (ROOT / args.anatomical_config).resolve()
        raw = yaml.safe_load(drive_path.read_text(encoding='utf-8'))
        fwd = raw['forward_simulation']
        M = args.M or (12 if args.preset == 'smoke' else int(fwd['time_steps_M']))
        T = float(fwd['final_time_T'])
        r = float(fwd['grading_r'])
        grid = _core.TimeGrid.make_graded(T, M, r)
        profile = (load_regional_materials(ROOT / args.materials_config, args.material_profile,
                                           float(master.fixed_physics.phi_0))
                   if args.material_profile else None)
        problem = RetecoForwardProblem2D(
            physics_model=master.fixed_physics, scales=master.characteristic_scales,
            time_grid=grid, mesh_file_path=str(Path(args.mesh).resolve()),
            dirichlet_disp_boundary_ids=fwd.get('dirichlet_disp_boundary_ids',[1]),
            fe_deg_disp=int(fwd.get('fe_deg_disp',2)),
            fe_deg_press=int(fwd.get('fe_deg_press',1)),
            fe_deg_tracer=int(fwd.get('fe_deg_tracer',1)),
            use_supg=False,
            region_materials=profile,
        )
        c0, F, Sf, driving_meta = build_anatomical_driving_Callbacks(
            problem, fwd['driving_protocol'],
            geometry_report_path=Path(args.geometry_report).resolve() if args.geometry_report else None)
        sensors = (_choose_anatomical_sensors(problem)
                   if args.sensor_spec is None else np.asarray(
                       json.loads(Path(args.sensor_spec).read_text(encoding='utf-8'))['sensor_points'],
                       dtype=float))
        if sensors.ndim != 2 or sensors.shape[1] != 2 or not np.isfinite(sensors).all():
            raise ValueError('Sensor specification must contain a finite Nx2 array.')
        cfg_file = drive_path
        N = None

    # Keep the legacy fixed-initial-concentration protocol as the default.
    # For clinical-motivated comparisons, match the *stored excess tracer dose*
    # rather than the concentration, since phi(x) changes between profiles.
    dose_baseline = float(driving_meta.get('initial_tracer_gaussian',{}).get('baseline',0.0))
    dose_before = _tracer_initial_dose(problem, c0, dose_baseline)
    dose_factor = 1.0
    dose_reference_path = None
    if args.dose_reference is not None:
        if args.domain != 'anatomical':
            raise ValueError('--dose-reference is only valid for anatomical experiments.')
        dose_reference_path = str(Path(args.dose_reference).resolve())
        dose_target = _reference_excess_dose(
            args.dose_reference, baseline=dose_baseline,
            mesh_sha=hashlib.sha256(Path(args.mesh).read_bytes()).hexdigest(),
            driving_sha=hashlib.sha256(cfg_file.read_bytes()).hexdigest(),
            physics_sha=hashlib.sha256(physics_path.read_bytes()).hexdigest())
        if dose_before['excess_mass_above_baseline'] <= 0:
            raise ValueError('Cannot normalize a zero or negative tracer bolus.')
        dose_factor = dose_target / dose_before['excess_mass_above_baseline']
        c0 = dose_baseline + dose_factor * (np.asarray(c0,dtype=float) - dose_baseline)
        # Metadata in the driving generator describe the original initialization;
        # update its applied amplitude and stored mass after dose normalization.
        for key in ('initial_tracer_gaussian','initial_tracer_distribution'):
            if key in driving_meta:
                driving_meta[key]['initial_mass'] = _tracer_initial_dose(problem,c0,dose_baseline)['total_stored_mass'] / float(problem.physics_cpp.phi_0)
        if 'initial_tracer_gaussian' in driving_meta:
            driving_meta['initial_tracer_gaussian']['amplitude'] *= dose_factor
        if 'initial_tracer_mass' in driving_meta:
            driving_meta['initial_tracer_mass'] = _tracer_initial_dose(problem,c0,dose_baseline)['total_stored_mass'] / float(problem.physics_cpp.phi_0)
    dose_after = _tracer_initial_dose(problem,c0,dose_baseline)
    dose_metadata = {'protocol': 'equal_excess_dose' if dose_reference_path else 'fixed_initial_concentration',
                     'baseline_concentration': dose_baseline,
                     'dose_reference_directory': dose_reference_path,
                     'amplitude_scale_factor': float(dose_factor),
                     'before': dose_before, 'applied': dose_after}
    print('Initial tracer: protocol=%s, baseline=%g, excess dose=%g, amplitude factor=%g' %
          (dose_metadata['protocol'], dose_baseline,
           dose_after['excess_mass_above_baseline'],dose_factor), flush=True)

    if args.forcing == 'mechanical':
        Sf = None
    elif args.forcing == 'fluid':
        F = None

    params = variant_parameters(master.parameter_space_theta.theta_dagger_array())
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = Path(args.out_dir) if args.out_dir else ROOT / 'artifacts' / 'model_variant_comparison' / f'{args.domain}_{args.forcing}_{args.preset}_{stamp}'
    if output.exists():
        raise FileExistsError(f'Output exists: {output}. Choose a new --out-dir (never overwrite existing experiments).')
    output.mkdir(parents=True)
    if args.domain == 'anatomical':
        (output/'sensor_points.json').write_text(
            json.dumps({'sensor_points':sensors.tolist(),
                        'method':'synthetic Q1 sampling at frozen 2D spatial points'},indent=2),
            encoding='utf-8')
    times = np.asarray(grid.times(), dtype=float)
    observation_times = np.linspace(T / 5., T, 5)
    obs_operator = ExactQ1TracerObservationOperator(problem, sensors, observation_times)
    trajectories, series, observations = {}, {}, {}

    for name, theta in params.items():
        print(f'[{name}] alpha={theta[0]:.3f}, beta={theta[1]:.3f}; domain={args.domain}; forcing={args.forcing}', flush=True)
        trajectory = problem.solve_trajectory(theta, c0=c0, F_mech_func=F, S_fluid_func=Sf,
                                              enforce_initial_compatibility=True,
                                              compute_diagnostics=False)
        trajectories[name] = trajectory
        series[name] = _series(problem, trajectory)
        observations[name] = np.asarray(obs_operator.apply(problem, trajectory), dtype=float)
        relative_mass_drift = np.max(np.abs(series[name]['c_mass']-series[name]['c_mass'][0])) / max(abs(series[name]['c_mass'][0]), 1.e-12)
        print(f'      max |q|={trajectory.max_darcy_velocity:.5g}, relative tracer mass drift={relative_mass_drift:.3e}', flush=True)
        if relative_mass_drift > args.mass_tolerance:
            raise RuntimeError(f'Tracer mass conservation failed for {name}: {relative_mass_drift:.3e} > {args.mass_tolerance:.3e}')

    csv_columns = ['model','alpha','beta','D','E','d_L2_final','p_L2_final','c_L2_final','c_mass_initial',
                   'c_mass_final','relative_mass_drift','max_darcy_velocity',
                   'rel_c_difference_vs_M4','rel_d_difference_vs_M4','rel_p_difference_vs_M4',
                   'rel_observation_difference_vs_M4']
    rows = []
    full = trajectories['M4']
    for name, traj in trajectories.items():
        theta = params[name]
        mass_drift = float(np.max(np.abs(series[name]['c_mass']-series[name]['c_mass'][0])) / max(abs(series[name]['c_mass'][0]),1.e-12))
        row = dict(zip(['model','alpha','beta','D','E'], [name,*map(float,theta)]))
        row.update({'d_L2_final':float(series[name]['d_L2'][-1]),
                    'p_L2_final':float(series[name]['p_L2'][-1]),
                    'c_L2_final':float(series[name]['c_L2'][-1]),
                    'c_mass_initial':float(series[name]['c_mass'][0]),
                    'c_mass_final':float(series[name]['c_mass'][-1]),
                    'relative_mass_drift':mass_drift,
                    'max_darcy_velocity':float(traj.max_darcy_velocity)})
        for label, m, arr, ref in [('c',problem.M_c_csr,traj.c_hist[-1],full.c_hist[-1]),
                                   ('d',problem.M_d_csr,traj.d_hist[-1],full.d_hist[-1]),
                                   ('p',problem.M_p_csr,traj.p_hist[-1],full.p_hist[-1])]:
            row[f'rel_{label}_difference_vs_M4'] = l2_mass_norm(m,arr-ref)/max(l2_mass_norm(m,ref),1e-12)
        row['rel_observation_difference_vs_M4'] = float(
            np.linalg.norm(observations[name]-observations['M4'])/
            max(np.linalg.norm(observations['M4']),1.e-12))
        rows.append(row)
    with (output/'model_variant_summary.csv').open('w',encoding='utf-8',newline='') as fh:
        writer = csv.DictWriter(fh,fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(rows)

    arrays = {'times':times,'mesh_vertices':problem.mesh_vertices,
              'cell_quad_connectivity':problem.cell_quad_connectivity,
              'cell_tracer_dof_connectivity':problem.cell_tracer_dof_connectivity,
              'tracer_support_points':problem.tracer_pts,
              'sensor_points':sensors,'observation_times':observation_times,
              'initial_tracer':c0}
    for key, traj in trajectories.items():
        arrays.update({f'{key}_d_hist':traj.d_hist,f'{key}_p_hist':traj.p_hist,
                       f'{key}_c_hist':traj.c_hist,f'{key}_observations':observations[key]})
        arrays.update({f'{key}_{k}':v for k,v in series[key].items()})
    np.savez_compressed(output/'trajectories_and_observations.npz',**arrays)
    _write_figures(output, problem, times, trajectories, series)
    if args.domain == 'anatomical':
        _export_region_series(output, problem, times, trajectories)
        _write_anatomical_regions_figure(output, problem, trajectories['M4'])
    manifest = {'purpose':'controlled M1--M4 forward comparison (not empirical validation)',
                'domain':args.domain,'forcing':args.forcing,'preset':args.preset,
                'physics_config':str(physics_path),'physics_sha256':hashlib.sha256(physics_path.read_bytes()).hexdigest(),
                'driving_config':str(cfg_file),'driving_sha256':hashlib.sha256(cfg_file.read_bytes()).hexdigest(),
                'time_steps_M':M,'subdivisions_N':N,'final_time_T':T,'grading_r':r,
                'mesh_file_path':str(Path(args.mesh).resolve()) if args.mesh else None,
                'mesh_sha256':hashlib.sha256(Path(args.mesh).read_bytes()).hexdigest() if args.mesh else None,
                'geometry_report_path':str(Path(args.geometry_report).resolve()) if args.geometry_report else None,
                'python_version':platform.python_version(),'variant_params':{k:v.tolist() for k,v in params.items()},
                'observations':'isoparametric Q1 with linear time interpolation',
                'driving_metadata':driving_meta,
                'initial_tracer_dose_protocol':dose_metadata,
                'material_profile':args.material_profile if args.domain=='anatomical' else None,
                'sensor_spec_path':str(Path(args.sensor_spec).resolve()) if args.sensor_spec else None,
                'regional_material_multipliers':problem.region_materials,
                'regional_materials_config_sha256':(
                    hashlib.sha256((ROOT/args.materials_config).read_bytes()).hexdigest()
                    if args.domain=='anatomical' and args.material_profile else None),
                'regional_mass_definition':'integral phi(x)*c dx; c_mass saved normalized by phi_0',
                'regional_sampling':'Q1 2x2 quadrature, Darcy flux at cell midpoint',
                'anatomy_note':'2D MRI geometry; synthetic mechanical/fluid sources; no clinical calibration',
                'source_code':str(Path(__file__).resolve())}
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Outputs saved in: {output.resolve()}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain',choices=('square','anatomical'),default='square')
    parser.add_argument('--preset',choices=('smoke','paper'),default='smoke')
    parser.add_argument('--forcing',choices=('combined','mechanical','fluid'),default='combined')
    parser.add_argument('--N',type=int,default=None,help='Override square mesh subdivisions')
    parser.add_argument('--M',type=int,default=None,help='Override number of time steps')
    parser.add_argument('--physics-config',default='configs/scales_and_physics.yaml')
    parser.add_argument('--square-config',default='configs/twin_experiment_synthetic.yaml')
    parser.add_argument('--anatomical-config',default='configs/brats_anatomical_forward.yaml')
    parser.add_argument('--mesh',help='Previously exported anatomical quadrilateral .msh file')
    parser.add_argument('--geometry-report',help='Corresponding geometry_quality_report.json')
    parser.add_argument('--dose-reference',default=None,
                        help='Homogeneous anatomical pilot directory used to match the stored initial excess tracer mass (optional).')
    parser.add_argument('--sensor-spec',default=None,
                        help='Freeze exact sensor coordinates across mesh resolutions (JSON output from prior run).')
    parser.add_argument('--materials-config',default='configs/brats_region_materials_phaseBC.yaml')
    parser.add_argument('--material-profile',choices=('homogeneous','exploratory_contrast'),
                        default=None,help='Optional registered BraTS-region material contrasts.')
    parser.add_argument('--mass-tolerance',type=float,default=1.e-7)
    parser.add_argument('--out-dir',default=None)
    args=parser.parse_args()
    if args.N is not None and args.N < 2: parser.error('--N must be >=2')
    if args.M is not None and args.M < 2: parser.error('--M must be >=2')
    run(args)


if __name__=='__main__':
    main()
