"""
python/reteco/forward/wrapper.py
Interfaccia ad alto livello tipizzata verso il motore C++20 deal.II (`reteco._core`).
Pre-assembla le matrici spaziali una sola volta e risolve la traiettoria temporale
per qualsiasi vettore di parametri theta = (alpha, beta, D, E).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import numpy as np

from .. import _core
from ..config_schema import CharacteristicScales, FixedPhysicsModel


@dataclass
class ForwardTrajectoryResult:
    times: np.ndarray                      # (N_t + 1,)
    d_hist: np.ndarray                     # (N_t + 1, N_d)
    p_hist: np.ndarray                     # (N_t + 1, N_p)
    c_hist: np.ndarray                     # (N_t + 1, N_c)
    max_darcy_velocity: float              # max_{n} ||q_h^n||_{L^\infty}
    initial_compatibility_residual: float  # ||E*K0*d0 - b*B*p0 - F(0)||_2 su V
    diagnostics: List[_core.StepDiagnostics]


class RetecoForwardProblem2D:
    """
    Incapsula il problema diretto RETECO v1 su dominio 2D discretizzato con deal.II:
    - Spostamento d_h in [Q_2]^2 (Taylor-Hood)
    - Pressione p_h in Q_1       (Taylor-Hood)
    - Concentrazione c_h in Q_1
    """

    def __init__(
        self,
        subdivisions_N: int,
        physics_model: FixedPhysicsModel,
        scales: CharacteristicScales,
        time_grid: _core.TimeGrid,
        fe_deg_disp: int = 2,
        fe_deg_press: int = 1,
        fe_deg_tracer: int = 1,
        dirichlet_all_displacement_boundary: bool = True,
        use_supg: bool = False,
    ):
        self.subdivisions_N = int(subdivisions_N)
        self.physics_cpp = physics_model.to_cpp(scales)
        self.time_grid = time_grid
        self.use_supg = bool(use_supg)

        self.fem = _core.InvariantFEMMatrices2D(
            self.subdivisions_N,
            self.physics_cpp,
            fe_deg_disp,
            fe_deg_press,
            fe_deg_tracer,
            dirichlet_all_displacement_boundary,
        )
        self.monitor = _core.EnergyMassMonitor(self.fem)

        self.disp_pts = np.asarray(self.fem.get_disp_support_points(), dtype=np.float64)
        self.press_pts = np.asarray(self.fem.get_press_support_points(), dtype=np.float64)
        self.tracer_pts = np.asarray(self.fem.get_tracer_support_points(), dtype=np.float64)

        # Pre-calcolo delle matrici di massa per integrazione veloce delle forzanti nodali
        self.M_p_csr = self.fem.M_p().tocsr()
        self.M_c_csr = self.fem.M_c().tocsr()
        self.K0_csr = self.fem.K0_mech().tocsr()

    @property
    def num_disp_dofs(self) -> int:
        return int(self.fem.num_disp_dofs())

    @property
    def num_press_dofs(self) -> int:
        return int(self.fem.num_press_dofs())

    @property
    def num_tracer_dofs(self) -> int:
        return int(self.fem.num_tracer_dofs())

    def build_gaussian_initial_tracer(
        self,
        center: tuple[float, float] = (0.5, 0.5),
        sigma: float = 0.10,
        amplitude: float = 1.0,
        baseline: float = 0.1,
    ) -> np.ndarray:
        x = self.tracer_pts[:, 0]
        y = self.tracer_pts[:, 1]
        r2 = (x - center[0]) ** 2 + (y - center[1]) ** 2
        return baseline + amplitude * np.exp(-r2 / (2.0 * (sigma ** 2)))

    def solve_trajectory(
        self,
        theta_vec: np.ndarray,
        c0: np.ndarray,
        p0: Optional[np.ndarray] = None,
        d0: Optional[np.ndarray] = None,
        F_mech_func: Optional[Callable[[ float], np.ndarray]] = None,
        S_fluid_func: Optional[Callable[[float], np.ndarray]] = None,
        S_tracer_func: Optional[Callable[[float], np.ndarray]] = None,
        enforce_initial_compatibility: bool = True,
        compute_diagnostics: bool = False,
    ) -> ForwardTrajectoryResult:
        """
        Esegue l'integrazione temporale completa per theta = (alpha, beta, D, E).
        Se enforce_initial_compatibility=True e d0 è None, calcola automaticamente d0
        dalla condizione di equilibrio iniziale a_E(d0, v) - b*(p0, div v) = <F(0), v>.
        """
        arr = np.asarray(theta_vec, dtype=np.float64)
        if arr.shape != (4,):
            raise ValueError(f"theta_vec deve avere 4 componenti (alpha, beta, D, E), ricevuto {arr.shape}.")

        theta_cpp = _core.ParameterTheta(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))

        n_steps = int(self.time_grid.num_steps())
        times = np.asarray(self.time_grid.times(), dtype=np.float64)

        p0_vec = np.zeros(self.num_press_dofs, dtype=np.float64) if p0 is None else np.asarray(p0, dtype=np.float64)
        c0_vec = np.asarray(c0, dtype=np.float64)
        F0_vec = (
            np.zeros(self.num_disp_dofs, dtype=np.float64)
            if F_mech_func is None
            else np.asarray(F_mech_func(0.0), dtype=np.float64)
        )

        if d0 is None and enforce_initial_compatibility:
            d0_vec = np.asarray(
                _core.InitialCompatibility.compute_compatible_d0(
                    self.fem, theta_cpp.E, p0_vec, F0_vec
                ),
                dtype=np.float64,
            )
        elif d0 is None:
            d0_vec = np.zeros(self.num_disp_dofs, dtype=np.float64)
        else:
            d0_vec = np.asarray(d0, dtype=np.float64)

        compat_res = float(
            _core.InitialCompatibility.compatibility_residual_norm(
                self.fem, theta_cpp.E, d0_vec, p0_vec, F0_vec
            )
        )

        biot_solver = _core.BiotViscoStepSolver(self.fem, theta_cpp, self.time_grid)
        tracer_solver = _core.TracerTransportStepSolver(
            self.fem, theta_cpp, self.time_grid, self.use_supg
        )
        biot_solver.reset(d0_vec, p0_vec)
        tracer_solver.reset(c0_vec)

        d_hist = np.zeros((n_steps + 1, self.num_disp_dofs), dtype=np.float64)
        p_hist = np.zeros((n_steps + 1, self.num_press_dofs), dtype=np.float64)
        c_hist = np.zeros((n_steps + 1, self.num_tracer_dofs), dtype=np.float64)

        d_hist[0, :] = d0_vec
        p_hist[0, :] = p0_vec
        c_hist[0, :] = c0_vec

        F_list = [F0_vec]
        Sf_list = [np.zeros(self.num_press_dofs, dtype=np.float64)]
        zero_d = np.zeros(self.num_disp_dofs, dtype=np.float64)
        zero_p = np.zeros(self.num_press_dofs, dtype=np.float64)
        zero_c = np.zeros(self.num_tracer_dofs, dtype=np.float64)

        max_q = 0.0

        for n in range(1, n_steps + 1):
            t_n = float(times[n])
            F_n = zero_d if F_mech_func is None else np.asarray(F_mech_func(t_n), dtype=np.float64)
            Sf_n = zero_p if S_fluid_func is None else np.asarray(S_fluid_func(t_n), dtype=np.float64)
            Sc_n = zero_c if S_tracer_func is None else np.asarray(S_tracer_func(t_n), dtype=np.float64)

            d_n, p_n = biot_solver.advance_step(n, F_n, Sf_n)
            c_n = tracer_solver.advance_step(n, p_n, Sc_n)

            d_hist[n, :] = d_n
            p_hist[n, :] = p_n
            c_hist[n, :] = c_n

            q_step = float(self.fem.compute_max_darcy_velocity(p_n))
            if q_step > max_q:
                max_q = q_step

            if compute_diagnostics:
                F_list.append(F_n)
                Sf_list.append(Sf_n)

        if not (np.all(np.isfinite(d_hist)) and np.all(np.isfinite(p_hist)) and np.all(np.isfinite(c_hist))):
            raise RuntimeError("Il solutore diretto ha prodotto valori non finiti (NaN/Inf).")

        diag_list: List[_core.StepDiagnostics] = []
        if compute_diagnostics:
            diag_list = list(
                self.monitor.evaluate_trajectory_diagnostics(
                    theta_cpp,
                    self.time_grid,
                    [d_hist[k] for k in range(n_steps + 1)],
                    [p_hist[k] for k in range(n_steps + 1)],
                    [c_hist[k] for k in range(n_steps + 1)],
                    F_list,
                    Sf_list,
                )
            )

        return ForwardTrajectoryResult(
            times=times,
            d_hist=d_hist,
            p_hist=p_hist,
            c_hist=c_hist,
            max_darcy_velocity=max_q,
            initial_compatibility_residual=compat_res,
            diagnostics=diag_list,
        )