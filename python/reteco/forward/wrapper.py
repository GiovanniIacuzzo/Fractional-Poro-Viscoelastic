"""
python/reteco/forward/wrapper.py
Interfaccia ad alto livello tipizzata verso il motore C++20 deal.II (`reteco._core`).
Supporta sia il quadrato unitario MMS [0,1]^2 sia la mesh anatomica quadrilatera
verificata, mantenendo la connettività originale delle celle per le visualizzazioni.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
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

    Quando `mesh_file_path` è None, costruisce la griglia regolare sul quadrato unitario [0,1]^2.
    Quando `mesh_file_path` è fornito (.msh o .inp), importa la mesh quadrilatera anatomica
    e applica il caso di riferimento del Contratto Matematico:
    - d = 0 sui bordi in `dirichlet_disp_boundary_ids` (default: [1] = contorno esterno,
      documentato come ipotesi di modello meccanico e NON come misura clinica);
    - trazione nulla (sigma * n = 0) sui contorni interni non vincolati (es. boundary_id = 2);
    - flusso normale di Darcy nullo (q . n = -K_h grad(p) . n = 0) su tutta la frontiera;
    - flusso normale totale del tracciante nullo (F_alpha . n = 0) su tutta la frontiera.
    """

    def __init__(
        self,
        subdivisions_N: Optional[int] = 16,
        physics_model: Optional[FixedPhysicsModel] = None,
        scales: Optional[CharacteristicScales] = None,
        time_grid: Optional[_core.TimeGrid] = None,
        fe_deg_disp: int = 2,
        fe_deg_press: int = 1,
        fe_deg_tracer: int = 1,
        dirichlet_all_displacement_boundary: bool = True,
        use_supg: bool = False,
        mesh_file_path: Optional[str | Path] = None,
        dirichlet_disp_boundary_ids: Optional[Sequence[int]] = None,
    ):
        if physics_model is None or scales is None or time_grid is None:
            raise ValueError("physics_model, scales e time_grid sono obbligatori.")

        self.scales = scales
        self.physics_model = physics_model
        self.physics_cpp = physics_model.to_cpp(scales)
        self.time_grid = time_grid
        self.use_supg = bool(use_supg)
        self.mesh_file_path = str(mesh_file_path) if mesh_file_path is not None else None

        if self.mesh_file_path is None:
            if subdivisions_N is None:
                raise ValueError("Specificare subdivisions_N quando mesh_file_path è None.")
            self.subdivisions_N = int(subdivisions_N)
            self.fem = _core.InvariantFEMMatrices2D(
                self.subdivisions_N,
                self.physics_cpp,
                fe_deg_disp,
                fe_deg_press,
                fe_deg_tracer,
                dirichlet_all_displacement_boundary,
            )
        else:
            self.subdivisions_N = 0
            b_ids = [1] if dirichlet_disp_boundary_ids is None else [int(b) for b in dirichlet_disp_boundary_ids]
            self.fem = _core.InvariantFEMMatrices2D(
                self.mesh_file_path,
                self.physics_cpp,
                fe_deg_disp,
                fe_deg_press,
                fe_deg_tracer,
                b_ids,
            )

        self.monitor = _core.EnergyMassMonitor(self.fem)

        self.disp_pts = np.asarray(self.fem.get_disp_support_points(), dtype=np.float64)
        self.press_pts = np.asarray(self.fem.get_press_support_points(), dtype=np.float64)
        self.tracer_pts = np.asarray(self.fem.get_tracer_support_points(), dtype=np.float64)

        # Connettività quadrilatera nativa preservata per le visualizzazioni (Requisito 8)
        self.mesh_vertices = np.asarray(self.fem.get_mesh_vertices(), dtype=np.float64)
        self.cell_quad_connectivity = np.asarray(self.fem.get_cell_quad_connectivity(), dtype=np.int64)
        self.cell_tracer_dof_connectivity = np.asarray(
            self.fem.get_cell_tracer_dof_connectivity(), dtype=np.int64
        )
        self.cell_material_ids = np.asarray(self.fem.get_cell_material_ids(), dtype=np.int32)

        # Pre-calcolo delle matrici di massa in formato CSR per integrazione veloce delle forzanti
        self.M_d_csr = self.fem.M_d().tocsr()
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

    @property
    def num_active_cells(self) -> int:
        return int(self.fem.num_active_cells())

    @property
    def boundary_face_counts(self) -> Dict[int, int]:
        return {int(k): int(v) for k, v in self.fem.get_boundary_face_counts().items()}

    def boundary_conditions_metadata(self) -> Dict[str, Any]:
        """
        Restituisce la documentazione esplicita delle condizioni al contorno applicate,
        distinguendo rigorosamente le ipotesi di modello da eventuali misure cliniche.
        """
        discovered = [int(b) for b in self.fem.get_boundary_ids()]
        dirichlet_d = [int(b) for b in self.fem.get_dirichlet_disp_boundary_ids()]
        neumann_d = [b for b in discovered if b not in dirichlet_d]
        return {
            "mesh_source": self.fem.mesh_source_description(),
            "is_imported_anatomical_mesh": bool(self.fem.is_imported_mesh()),
            "discovered_boundary_ids": discovered,
            "boundary_face_counts": self.boundary_face_counts,
            "displacement_dirichlet_zero_boundary_ids": dirichlet_d,
            "displacement_homogeneous_neumann_traction_free_boundary_ids": neumann_d,
            "darcy_normal_flux_boundary_condition": "q . n = -K_h grad(p) . n = 0 on all boundaries",
            "tracer_total_normal_flux_boundary_condition": "F_alpha . n = (c*q - phi_0*D*grad(c)) . n = 0 on all boundaries",
            "epistemic_status_note": (
                "MODEL_HYPOTHESIS: Homogeneous Dirichlet displacement d = 0 on outer boundary "
                "and zero traction on internal boundaries are mathematical modeling hypotheses "
                "for the 2D slice proxy and do NOT represent clinical measurements."
            ),
        }

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
        F_mech_func: Optional[Callable[[float], np.ndarray]] = None,
        S_fluid_func: Optional[Callable[[float], np.ndarray]] = None,
        S_tracer_func: Optional[Callable[[float], np.ndarray]] = None,
        enforce_initial_compatibility: bool = True,
        compute_diagnostics: bool = False,
    ) -> ForwardTrajectoryResult:
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

    def compute_disp_L2_norm(self, d_vec: np.ndarray) -> float:
        """Calcola ||d_h||_{L^2(Omega)} = sqrt(d^T M_d d) tramite la matrice di massa vettoriale."""
        arr = np.asarray(d_vec, dtype=np.float64)
        return float(np.sqrt(max(0.0, float(arr @ (self.M_d_csr @ arr)))))

    def compute_disp_stiffness_seminorm(self, d_vec: np.ndarray) -> float:
        """Calcola la seminorma elastica di rigidezza |d_h|_{K_0} = sqrt(d^T K_0 d)."""
        arr = np.asarray(d_vec, dtype=np.float64)
        return float(np.sqrt(max(0.0, float(arr @ (self.K0_csr @ arr)))))

    def compute_press_L2_norm(self, p_vec: np.ndarray) -> float:
        """Calcola ||p_h||_{L^2(Omega)} = sqrt(p^T M_p p) tramite la matrice di massa della pressione."""
        arr = np.asarray(p_vec, dtype=np.float64)
        return float(np.sqrt(max(0.0, float(arr @ (self.M_p_csr @ arr)))))

    def compute_tracer_L2_norm(self, c_vec: np.ndarray) -> float:
        """Calcola ||c_h||_{L^2(Omega)} = sqrt(c^T M_c c) tramite la matrice di massa del tracciante."""
        arr = np.asarray(c_vec, dtype=np.float64)
        return float(np.sqrt(max(0.0, float(arr @ (self.M_c_csr @ arr)))))