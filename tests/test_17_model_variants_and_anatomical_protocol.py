"""Regression tests for the post-review model-comparison protocol.

Requires a *freshly rebuilt* native _core on the execution host.
"""
from __future__ import annotations
import math
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.sparse import eye
from reteco import _core, load_master_config
from reteco.forward.wrapper import RetecoForwardProblem2D
from reteco.forward.target_generator import build_anatomical_driving_Callbacks
from reteco.observation.q1_exact import ExactQ1TracerObservationOperator


def test_gamma_one_is_exact_backward_euler_on_graded_grid():
    grid = _core.TimeGrid.make_graded(1.0, 13, 1.7)
    mem = _core.CaputoL1Memory(1.0, grid, 2)
    previous = np.array([1.2, -0.3])
    mem.reset(previous)
    for n in range(1, 14):
        t = grid.time_at(n)
        u_n = np.array([np.exp(-t), t*t])
        expected = (u_n - previous) / grid.dt(n)
        np.testing.assert_allclose(mem.evaluate_discrete_derivative(n,u_n),expected,atol=2e-12,rtol=2e-12)
        assert np.isclose(mem.current_implicit_coeff(n), 1.0 / grid.dt(n))
        weights = np.asarray(mem.step_weights(n))
        np.testing.assert_array_equal(weights[:-1],0)
        assert np.isclose(weights[-1],1.0 / grid.dt(n))
        mem.commit_step(n,u_n)
        previous = u_n


def test_no_change_to_inverse_bounds_and_fractional_reference():
    from reteco.config_schema import ParameterCoordinateTransform
    master=load_master_config('configs/scales_and_physics.yaml')
    b=master.parameter_space_theta.bounds
    assert b.alpha[1] < 1.0 and b.beta[1] < 1.0
    from importlib.util import spec_from_file_location, module_from_spec
    # The executable experiment can also be imported without running the four trajectories.
    spec=spec_from_file_location('model_variants','experiments/15_run_fractional_model_variants.py')
    module=module_from_spec(spec)
    spec.loader.exec_module(module)
    values=module.variant_parameters(master.parameter_space_theta.theta_dagger_array())
    assert list(values)==['M1','M2','M3','M4']
    np.testing.assert_allclose(values['M1'][:2],[1,1])
    np.testing.assert_allclose(values['M2'][:2],[.7,1])
    np.testing.assert_allclose(values['M3'][:2],[1,.45])
    np.testing.assert_allclose(values['M4'][:2],[.7,.45])
    assert all(np.allclose(x[2:],[.08,2.50]) for x in values.values())


class _FakeFEM:
    def is_dirichlet_disp(self):
        return np.zeros(8, dtype=bool)


class _FakeAnatomicalProblem:
    def __init__(self):
        points=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
        self.press_pts=points.copy()
        self.tracer_pts=points.copy()
        self.disp_pts=np.repeat(points,2,axis=0)
        self.num_disp_dofs=8
        self.num_press_dofs=4
        self.num_tracer_dofs=4
        self.M_p_csr=eye(4,format='csr')
        self.M_c_csr=eye(4,format='csr')
        self.M_d_csr=eye(8,format='csr')
        self.fem=_FakeFEM()
        self.scales=SimpleNamespace(L0_m=0.01)


def test_anatomical_forcing_matches_yaml_and_baseline():
    p=_FakeAnatomicalProblem()
    config={
        'initial_tracer_gaussian':{'sigma_mm':8,'amplitude':1.,'baseline':.10},
        'fluid_injection_source':{'sigma_mm':9,'amplitude':.60,
                                  'temporal_profile':'exponential_decay','decay_rate':1.5},
        'mechanical_body_force':{'sigma_mm':12,'amplitude_x':.80,'amplitude_y':.60,
                                 'temporal_profile':'sinusoidal','omega_t':math.pi}
    }
    c0,F,S,meta=build_anatomical_driving_Callbacks(p,config)
    center=np.array([.5,.5])
    expected_c=.10+np.exp(-.5*np.sum((p.tracer_pts-center)**2,axis=1)/.8**2)
    np.testing.assert_allclose(c0,expected_c)
    assert not np.allclose(F(0.5),0)
    np.testing.assert_allclose(F(0.),0,atol=1e-14)
    np.testing.assert_allclose(F(1.),0,atol=1e-14)
    np.testing.assert_allclose(S(.4),np.exp(-1.5*.4)*S(0),rtol=1e-13)
    assert meta['mechanical_body_force']['active']
    assert meta['fluid_injection_source']['temporal_profile']=='exponential_decay'
    assert meta['initial_tracer_gaussian']['baseline']==.10
    with pytest.raises(ValueError,match='Unsupported anatomical temporal_profile'):
        broken={**config,'mechanical_body_force':{**config['mechanical_body_force'],'temporal_profile':'typo'}}
        _,F_bad,_,_=build_anatomical_driving_Callbacks(_FakeAnatomicalProblem(),broken)
        F_bad(.3)


def test_q1_exact_interpolation_reproduces_affine_and_bilinear_on_square():
    master=load_master_config('configs/scales_and_physics.yaml')
    grid=_core.TimeGrid.make_graded(.4,4,1.5)
    p=RetecoForwardProblem2D(physics_model=master.fixed_physics,scales=master.characteristic_scales,
                             time_grid=grid,subdivisions_N=3)
    sensors=np.array([[.14,.29],[.46,.53],[.82,.73]])
    obs=ExactQ1TracerObservationOperator(p,sensors,[.1,.3])
    result=obs.verify_linear_and_bilinear_reproduction(p,is_cartesian=True)
    assert result['max_abs_error_linear_field'] < 1.e-11
    assert result['max_abs_error_bilinear_field_cartesian'] < 1.e-11


def test_four_variants_advance_small_forward_problem_without_mass_loss():
    master=load_master_config('configs/scales_and_physics.yaml')
    grid=_core.TimeGrid.make_graded(.25,5,1.4)
    p=RetecoForwardProblem2D(physics_model=master.fixed_physics,scales=master.characteristic_scales,
                             time_grid=grid,subdivisions_N=3)
    c0=p.build_gaussian_initial_tracer()
    for alpha,beta in [(1.,1.),(.7,1.),(1.,.45),(.7,.45)]:
        traj=p.solve_trajectory(np.array([alpha,beta,.08,2.5]),c0=c0,
                                 compute_diagnostics=False)
        assert np.isfinite(traj.c_hist).all()
        weights=p.M_c_csr @ np.ones(p.num_tracer_dofs)
        mass=traj.c_hist @ weights
        assert np.max(abs(mass-mass[0])) < 1.e-9
