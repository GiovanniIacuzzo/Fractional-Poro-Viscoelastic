"""
python/reteco/forward/target_generator.py
Costruzione del protocollo di forzante sintetica e generazione del target Ground Truth
su discretizzazione indipendente/raffinata h_tar (Contratto Matematico Sez. 4, 8 e 9)
e su domini anatomici 2D importati.

Versione protocollo forzanti:
    FORCING_PROTOCOL_VERSION = "RETECO-v1-twin-forcing-v2-weak-volume-mass"
"""

from __future__ import annotations
import math
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np
import scipy.sparse as sp

from .wrapper import RetecoForwardProblem2D

FORCING_PROTOCOL_VERSION: str = "RETECO-v1-twin-forcing-v2-weak-volume-mass"


def _lagrange_basis_1d(nodes_ref: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """
    Valuta i polinomi base di Lagrange 1D sui nodi di riferimento `nodes_ref` in [0, 1]
    nei punti di quadratura `xi` in [0, 1].
    Restituisce un array di forma (len(nodes_ref), len(xi)).
    """
    n_basis = nodes_ref.size
    vals = np.ones((n_basis, xi.size), dtype=np.float64)
    for i in range(n_basis):
        for j in range(n_basis):
            if i != j:
                vals[i, :] *= (xi - nodes_ref[j]) / (nodes_ref[i] - nodes_ref[j])
    return vals


def _lagrange_basis_deriv_1d(nodes_ref: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """
    Valuta le derivate prime dei polinomi base di Lagrange 1D sui nodi di riferimento
    `nodes_ref` in [0, 1] nei punti di quadratura `xi` in [0, 1].
    Restituisce un array di forma (len(nodes_ref), len(xi)).
    """
    n_basis = nodes_ref.size
    dvals = np.zeros((n_basis, xi.size), dtype=np.float64)
    for i in range(n_basis):
        for k in range(n_basis):
            if k == i:
                continue
            term = np.ones(xi.size, dtype=np.float64) / (nodes_ref[i] - nodes_ref[k])
            for j in range(n_basis):
                if j != i and j != k:
                    term *= (xi - nodes_ref[j]) / (nodes_ref[i] - nodes_ref[j])
            dvals[i, :] += term
    return dvals


def _assemble_1d_mass_matrix(subdivisions_N: int, fe_degree: int) -> np.ndarray:
    """
    Assembla esattamente (con quadratura di Gauss-Legendre a fe_degree + 2 punti)
    la matrice di massa 1D per elementi continui Q_{fe_degree} su [0, 1].
    """
    n_intervals = int(subdivisions_N)
    deg = int(fe_degree)
    if n_intervals <= 0 or deg <= 0:
        raise ValueError("subdivisions_N e fe_degree devono essere interi positivi.")

    n_nodes_1d = n_intervals * deg + 1
    h = 1.0 / float(n_intervals)
    nodes_ref = np.linspace(0.0, 1.0, deg + 1, dtype=np.float64)

    q_pts_std, q_wts_std = np.polynomial.legendre.leggauss(deg + 2)
    q_pts_ref = 0.5 * (q_pts_std + 1.0)
    q_wts_ref = 0.5 * q_wts_std

    phi_vals = _lagrange_basis_1d(nodes_ref, q_pts_ref)
    local_mass = h * (phi_vals * q_wts_ref[None, :]) @ phi_vals.T

    m_1d = np.zeros((n_nodes_1d, n_nodes_1d), dtype=np.float64)
    for e in range(n_intervals):
        idx = np.arange(e * deg, (e + 1) * deg + 1)
        m_1d[np.ix_(idx, idx)] += local_mass

    return m_1d


def _assemble_1d_stiffness_matrix(subdivisions_N: int, fe_degree: int) -> np.ndarray:
    """
    Assembla esattamente la matrice di rigidezza 1D int_0^1 phi_i'(x) phi_j'(x) dx
    per elementi continui Q_{fe_degree} su [0, 1].
    """
    n_intervals = int(subdivisions_N)
    deg = int(fe_degree)
    if n_intervals <= 0 or deg <= 0:
        raise ValueError("subdivisions_N e fe_degree devono essere interi positivi.")

    n_nodes_1d = n_intervals * deg + 1
    h = 1.0 / float(n_intervals)
    nodes_ref = np.linspace(0.0, 1.0, deg + 1, dtype=np.float64)

    q_pts_std, q_wts_std = np.polynomial.legendre.leggauss(deg + 2)
    q_pts_ref = 0.5 * (q_pts_std + 1.0)
    q_wts_ref = 0.5 * q_wts_std

    dphi_vals = _lagrange_basis_deriv_1d(nodes_ref, q_pts_ref)
    local_stiff = (1.0 / h) * (dphi_vals * q_wts_ref[None, :]) @ dphi_vals.T

    k_1d = np.zeros((n_nodes_1d, n_nodes_1d), dtype=np.float64)
    for e in range(n_intervals):
        idx = np.arange(e * deg, (e + 1) * deg + 1)
        k_1d[np.ix_(idx, idx)] += local_stiff

    return k_1d


def build_scalar_lagrange_mass_matrix_csr(
    support_pts: np.ndarray,
    subdivisions_N: int,
    fe_degree: int,
) -> sp.csr_matrix:
    """
    Costruisce la matrice di massa scalare 2D M_{ij} = int_Omega phi_i(x) phi_j(x) dx
    sulla griglia cartesiana (0, 1)^2 ordinata esattamente secondo `support_pts`.
    """
    pts = np.asarray(support_pts, dtype=np.float64)
    n_intervals = int(subdivisions_N)
    deg = int(fe_degree)
    n_1d = n_intervals * deg + 1
    expected_n = n_1d * n_1d
    if pts.shape != (expected_n, 2):
        raise ValueError(
            f"Dimensione support_pts {pts.shape} non compatibile con N={n_intervals}, "
            f"grado={deg} (atteso {(expected_n, 2)})."
        )

    m_1d_sparse = sp.csr_matrix(_assemble_1d_mass_matrix(n_intervals, deg))
    m_2d_lex = sp.kron(m_1d_sparse, m_1d_sparse, format="csr")

    ix = np.rint(pts[:, 0] * (n_intervals * deg)).astype(np.int64)
    iy = np.rint(pts[:, 1] * (n_intervals * deg)).astype(np.int64)
    grid_x = ix / float(n_intervals * deg)
    grid_y = iy / float(n_intervals * deg)
    if not (np.allclose(pts[:, 0], grid_x, atol=1e-10) and np.allclose(pts[:, 1], grid_y, atol=1e-10)):
        raise ValueError("I punti di supporto non giacciono sul reticolo tensoriale Q_p di [0, 1]^2.")

    lex_of_dof = ix * n_1d + iy
    if np.unique(lex_of_dof).size != expected_n:
        raise ValueError("Mappa dei punti di supporto non biiettiva.")

    return m_2d_lex[lex_of_dof, :][:, lex_of_dof].tocsr()


def build_scalar_lagrange_stiffness_matrix_csr(
    support_pts: np.ndarray,
    subdivisions_N: int,
    fe_degree: int,
) -> sp.csr_matrix:
    """
    Costruisce la matrice di rigidezza Laplaciana scalare 2D:
        K_{ij} = int_Omega grad(phi_i(x)) . grad(phi_j(x)) dx
    sulla griglia cartesiana (0, 1)^2 ordinata esattamente secondo `support_pts`.
    """
    pts = np.asarray(support_pts, dtype=np.float64)
    n_intervals = int(subdivisions_N)
    deg = int(fe_degree)
    n_1d = n_intervals * deg + 1
    expected_n = n_1d * n_1d
    if pts.shape != (expected_n, 2):
        raise ValueError(
            f"Dimensione support_pts {pts.shape} non compatibile con N={n_intervals}, "
            f"grado={deg} (atteso {(expected_n, 2)})."
        )

    m_1d_sparse = sp.csr_matrix(_assemble_1d_mass_matrix(n_intervals, deg))
    k_1d_sparse = sp.csr_matrix(_assemble_1d_stiffness_matrix(n_intervals, deg))
    k_2d_lex = (
        sp.kron(k_1d_sparse, m_1d_sparse, format="csr")
        + sp.kron(m_1d_sparse, k_1d_sparse, format="csr")
    )

    ix = np.rint(pts[:, 0] * (n_intervals * deg)).astype(np.int64)
    iy = np.rint(pts[:, 1] * (n_intervals * deg)).astype(np.int64)
    lex_of_dof = ix * n_1d + iy
    return k_2d_lex[lex_of_dof, :][:, lex_of_dof].tocsr()


def identify_displacement_component_masks(
    problem: RetecoForwardProblem2D,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identifica le maschere booleane (mask_x, mask_y) dei gradi di liberta'
    delle componenti x e y dello spostamento in `problem.disp_pts` (sia su mesh
    cartesiana sia su mesh anatomica non strutturata).
    """
    if hasattr(problem, "_cached_disp_masks"):
        return problem._cached_disp_masks

    pts = np.asarray(problem.disp_pts, dtype=np.float64)
    n_dofs = pts.shape[0]
    if n_dofs % 2 != 0:
        raise ValueError(f"Il numero di DOFs di spostamento ({n_dofs}) deve essere pari.")

    key_coords = np.rint(pts * 1.0e11).astype(np.int64)
    node_groups: Dict[Tuple[int, int], list[int]] = {}
    for dof_idx in range(n_dofs):
        key = (int(key_coords[dof_idx, 0]), int(key_coords[dof_idx, 1]))
        node_groups.setdefault(key, []).append(dof_idx)

    if len(node_groups) != n_dofs // 2:
        raise ValueError(
            f"Numero di nodi geometrici distinti ({len(node_groups)}) diverso da n_dofs/2 ({n_dofs // 2})."
        )

    mask_x = np.zeros(n_dofs, dtype=bool)
    mask_y = np.zeros(n_dofs, dtype=bool)
    for pair in node_groups.values():
        if len(pair) != 2:
            raise ValueError(f"Attesi esattamente 2 DOFs per nodo, trovati {len(pair)}.")
        d0, d1 = (pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0])
        mask_x[d0] = True
        mask_y[d1] = True

    idx_x = np.flatnonzero(mask_x)
    idx_y = np.flatnonzero(mask_y)
    if not np.allclose(pts[idx_x], pts[idx_y], atol=1e-12):
        raise RuntimeError("I punti di supporto delle componenti x e y non coincidono.")

    b_csr = None
    if hasattr(problem, "B_csr"):
        b_csr = problem.B_csr
    elif hasattr(problem, "fem") and hasattr(problem.fem, "B_biot"):
        b_mat = problem.fem.B_biot()
        b_csr = b_mat.tocsr() if hasattr(b_mat, "tocsr") else sp.csr_matrix(b_mat)

    if b_csr is not None and b_csr.shape == (n_dofs, problem.num_press_dofs):
        ones_p = np.ones(problem.num_press_dofs, dtype=np.float64)
        domain_area = float(ones_p @ (problem.M_p_csr @ ones_p))

        d_x_stretch = np.zeros(n_dofs, dtype=np.float64)
        d_x_stretch[mask_x] = pts[mask_x, 0]
        div_x_int = float(ones_p @ (b_csr.T @ d_x_stretch))

        d_y_stretch = np.zeros(n_dofs, dtype=np.float64)
        d_y_stretch[mask_y] = pts[mask_y, 1]
        div_y_int = float(ones_p @ (b_csr.T @ d_y_stretch))

        d_x_shear = np.zeros(n_dofs, dtype=np.float64)
        d_x_shear[mask_x] = pts[mask_x, 1]
        div_xshear_int = float(ones_p @ (b_csr.T @ d_x_shear))

        if not (
            np.isclose(div_x_int, domain_area, rtol=1e-9, atol=1e-10)
            and np.isclose(div_y_int, domain_area, rtol=1e-9, atol=1e-10)
            and np.isclose(div_xshear_int, 0.0, atol=1e-9)
        ):
            raise RuntimeError(
                f"Verifica algebrica delle componenti di spostamento fallita: "
                f"int(div(x,0))={div_x_int:.6e}, int(div(0,y))={div_y_int:.6e} "
                f"(area={domain_area:.6e}), int(div(y,0))={div_xshear_int:.6e}."
            )

    problem._cached_disp_masks = (mask_x, mask_y)
    return mask_x, mask_y


def build_displacement_mass_matrix_csr(
    problem: RetecoForwardProblem2D,
) -> sp.csr_matrix:
    """
    Restituisce la matrice di massa vettoriale dello spostamento M_d su [Q_p]^2:
        (M_d)_{ij} = int_Omega phi_i(x) . phi_j(x) dx.
    Se `problem.M_d_csr` o `problem.fem.M_d()` è esposta dal core C++, la utilizza direttamente.
    """
    if hasattr(problem, "_cached_M_d_csr"):
        return problem._cached_M_d_csr

    if hasattr(problem, "M_d_csr") and problem.M_d_csr is not None:
        problem._cached_M_d_csr = problem.M_d_csr.tocsr()
        return problem._cached_M_d_csr

    if hasattr(problem, "fem") and hasattr(problem.fem, "M_d"):
        m_d_raw = problem.fem.M_d()
        m_d_csr = m_d_raw.tocsr() if hasattr(m_d_raw, "tocsr") else sp.csr_matrix(m_d_raw)
        problem._cached_M_d_csr = m_d_csr
        return m_d_csr

    mask_x, mask_y = identify_displacement_component_masks(problem)
    idx_x = np.flatnonzero(mask_x)
    idx_y = np.flatnonzero(mask_y)

    pts_x = problem.disp_pts[idx_x]
    pts_y = problem.disp_pts[idx_y]
    if not np.allclose(pts_x, pts_y, atol=1e-12):
        raise RuntimeError("I punti di supporto delle componenti x e y non coincidono.")

    n_nodes = idx_x.size
    n_1d = int(round(math.sqrt(n_nodes)))
    fe_deg_disp = int(getattr(problem, "fe_deg_disp", 2))
    subdivisions_N = getattr(problem, "subdivisions_N", None)
    if subdivisions_N is None:
        subdivisions_N = (n_1d - 1) // fe_deg_disp

    m_scalar = build_scalar_lagrange_mass_matrix_csr(
        pts_x, subdivisions_N=int(subdivisions_N), fe_degree=fe_deg_disp
    ).tocoo()

    n_dofs = problem.num_disp_dofs
    rows = np.concatenate([idx_x[m_scalar.row], idx_y[m_scalar.row]])
    cols = np.concatenate([idx_x[m_scalar.col], idx_y[m_scalar.col]])
    data = np.concatenate([m_scalar.data, m_scalar.data])

    m_d_csr = sp.coo_matrix((data, (rows, cols)), shape=(n_dofs, n_dofs)).tocsr()
    problem._cached_M_d_csr = m_d_csr
    return m_d_csr


def integrate_vector_body_force_quadrature(
    problem: RetecoForwardProblem2D,
    fx_callable: Callable[[np.ndarray, np.ndarray], np.ndarray],
    fy_callable: Callable[[np.ndarray, np.ndarray], np.ndarray],
    apply_dirichlet_homogenization: bool = False,
) -> np.ndarray:
    """
    Calcola direttamente con quadratura di Gauss-Legendre 2D su ogni elemento K
    l'integrale debole di riferimento:
        F_i = int_Omega phi_i(x) . f_s(x) dx.
    """
    mask_x, mask_y = identify_displacement_component_masks(problem)
    idx_x = np.flatnonzero(mask_x)
    idx_y = np.flatnonzero(mask_y)
    pts_x = problem.disp_pts[idx_x]

    n_nodes = idx_x.size
    n_1d = int(round(math.sqrt(n_nodes)))
    deg = int(getattr(problem, "fe_deg_disp", 2))
    n_intervals = int(getattr(problem, "subdivisions_N", (n_1d - 1) // deg))
    h = 1.0 / float(n_intervals)

    nodes_ref = np.linspace(0.0, 1.0, deg + 1, dtype=np.float64)
    q_pts_std, q_wts_std = np.polynomial.legendre.leggauss(deg + 3)
    q_pts_ref = 0.5 * (q_pts_std + 1.0)
    q_wts_ref = 0.5 * q_wts_std
    phi_1d = _lagrange_basis_1d(nodes_ref, q_pts_ref)

    rhs_lex_x = np.zeros((n_1d, n_1d), dtype=np.float64)
    rhs_lex_y = np.zeros((n_1d, n_1d), dtype=np.float64)

    for ex in range(n_intervals):
        x_phys = ex * h + h * q_pts_ref
        ix_loc = np.arange(ex * deg, (ex + 1) * deg + 1)
        for ey in range(n_intervals):
            y_phys = ey * h + h * q_pts_ref
            iy_loc = np.arange(ey * deg, (ey + 1) * deg + 1)

            xx, yy = np.meshgrid(x_phys, y_phys, indexing="ij")
            w_2d = (h * q_wts_ref)[:, None] * (h * q_wts_ref)[None, :]

            fx_vals = np.asarray(fx_callable(xx, yy), dtype=np.float64) * w_2d
            fy_vals = np.asarray(fy_callable(xx, yy), dtype=np.float64) * w_2d

            loc_x = phi_1d @ fx_vals @ phi_1d.T
            loc_y = phi_1d @ fy_vals @ phi_1d.T

            rhs_lex_x[np.ix_(ix_loc, iy_loc)] += loc_x
            rhs_lex_y[np.ix_(ix_loc, iy_loc)] += loc_y

    ix_dof = np.rint(pts_x[:, 0] * (n_intervals * deg)).astype(np.int64)
    iy_dof = np.rint(pts_x[:, 1] * (n_intervals * deg)).astype(np.int64)

    F_out = np.zeros(problem.num_disp_dofs, dtype=np.float64)
    F_out[idx_x] = rhs_lex_x[ix_dof, iy_dof]
    F_out[idx_y] = rhs_lex_y[ix_dof, iy_dof]

    if apply_dirichlet_homogenization:
        is_dir = np.asarray(problem.fem.is_dirichlet_disp(), dtype=bool)
        F_out[is_dir] = 0.0

    return F_out


def compute_spatial_forcing_vectors_and_diagnostics(
    problem: RetecoForwardProblem2D,
    driving_cfg: Dict,
) -> Dict[str, object]:
    """
    Costruisce i vettori spaziali base delle forzanti (prima della modulazione temporale
    sin(omega * t)) e calcola le grandezze integrali diagnostiche su Omega = (0, 1)^2.
    """
    sf_cfg = driving_cfg["fluid_injection_source"]
    sf_center = (float(sf_cfg["center"][0]), float(sf_cfg["center"][1]))
    sf_sigma = float(sf_cfg["sigma"])
    sf_amp = float(sf_cfg["amplitude"])
    sf_offset = float(sf_cfg.get("baseline_offset", 0.5))
    enforce_zero_mean = bool(sf_cfg.get("enforce_zero_mean", False))

    px = problem.press_pts[:, 0]
    py = problem.press_pts[:, 1]
    r2_p = (px - sf_center[0]) ** 2 + (py - sf_center[1]) ** 2
    raw_sf = sf_amp * (np.exp(-r2_p / (2.0 * (sf_sigma ** 2))) - sf_offset)

    ones_p = np.ones(problem.num_press_dofs, dtype=np.float64)
    domain_measure_p = float(ones_p @ (problem.M_p_csr @ ones_p))
    if enforce_zero_mean:
        mean_sf_fem = float(ones_p @ (problem.M_p_csr @ raw_sf)) / domain_measure_p
        raw_sf = raw_sf - mean_sf_fem

    Sf_spatial_rhs = np.asarray(problem.M_p_csr @ raw_sf, dtype=np.float64)
    fluid_source_spatial_integral = float(ones_p @ Sf_spatial_rhs)

    if enforce_zero_mean:
        fluid_source_continuous_integral_ref = 0.0
    else:
        rt2_sig = math.sqrt(2.0) * sf_sigma
        ix_gauss = math.sqrt(0.5 * math.pi) * sf_sigma * (
            math.erf((1.0 - sf_center[0]) / rt2_sig) + math.erf(sf_center[0] / rt2_sig)
        )
        iy_gauss = math.sqrt(0.5 * math.pi) * sf_sigma * (
            math.erf((1.0 - sf_center[1]) / rt2_sig) + math.erf(sf_center[1] / rt2_sig)
        )
        fluid_source_continuous_integral_ref = sf_amp * (ix_gauss * iy_gauss - sf_offset)

    mech_cfg = driving_cfg["mechanical_body_force"]
    amp_x = float(mech_cfg["amplitude_x"])
    amp_y = float(mech_cfg["amplitude_y"])
    omega = float(mech_cfg["omega_t"])

    mask_x, mask_y = identify_displacement_component_masks(problem)
    M_d_csr = build_displacement_mass_matrix_csr(problem)

    dx_pts = problem.disp_pts[:, 0]
    dy_pts = problem.disp_pts[:, 1]
    spatial_profile = np.sin(np.pi * dx_pts) * np.sin(np.pi * dy_pts)

    f_nodal_spatial = np.zeros(problem.num_disp_dofs, dtype=np.float64)
    f_nodal_spatial[mask_x] = amp_x * spatial_profile[mask_x]
    f_nodal_spatial[mask_y] = amp_y * spatial_profile[mask_y]

    F_spatial_unconstrained = np.asarray(M_d_csr @ f_nodal_spatial, dtype=np.float64)

    e_x = mask_x.astype(np.float64)
    e_y = mask_y.astype(np.float64)
    mech_resultant_x_spatial = float(e_x @ F_spatial_unconstrained)
    mech_resultant_y_spatial = float(e_y @ F_spatial_unconstrained)

    is_dir = np.asarray(problem.fem.is_dirichlet_disp(), dtype=bool)
    F_spatial_base = F_spatial_unconstrained.copy()
    F_spatial_base[is_dir] = 0.0

    mech_interior_resultant_x = float(e_x @ F_spatial_base)
    mech_interior_resultant_y = float(e_y @ F_spatial_base)

    analytical_factor = (2.0 / math.pi) ** 2
    mech_resultant_x_continuous_ref = amp_x * analytical_factor
    mech_resultant_y_continuous_ref = amp_y * analytical_factor

    return {
        "forcing_protocol_version": FORCING_PROTOCOL_VERSION,
        "omega_t": omega,
        "amplitude_x": amp_x,
        "amplitude_y": amp_y,
        "fluid_amplitude": sf_amp,
        "fluid_sigma": sf_sigma,
        "fluid_center": list(sf_center),
        "fluid_baseline_offset": sf_offset,
        "fluid_enforce_zero_mean": enforce_zero_mean,
        "Sf_spatial_rhs": Sf_spatial_rhs,
        "F_spatial_unconstrained": F_spatial_unconstrained,
        "F_spatial_base": F_spatial_base,
        "fluid_source_spatial_integral": fluid_source_spatial_integral,
        "fluid_source_continuous_integral_ref": fluid_source_continuous_integral_ref,
        "mech_resultant_x_spatial": mech_resultant_x_spatial,
        "mech_resultant_y_spatial": mech_resultant_y_spatial,
        "mech_interior_resultant_x": mech_interior_resultant_x,
        "mech_interior_resultant_y": mech_interior_resultant_y,
        "mech_resultant_x_continuous_ref": mech_resultant_x_continuous_ref,
        "mech_resultant_y_continuous_ref": mech_resultant_y_continuous_ref,
    }


def build_standard_driving_Callbacks(
    problem: RetecoForwardProblem2D,
    driving_cfg: Dict,
) -> Tuple[
    np.ndarray,
    Callable[[float], np.ndarray],
    Callable[[float], np.ndarray],
]:
    """
    Costruisce per il twin experiment sul quadrato unitario:
    1) Condizione iniziale del tracciante c_0(x) Gaussiana;
    2) Forzante meccanica di volume F_mech(t) assemblata con la matrice di massa vettoriale M_d
       rispettando le ampiezze direzionali amplitude_x e amplitude_y, nulla per t <= 0;
    3) Sorgente fluida S_fluid(t) assemblata con la matrice di massa della pressione M_p,
       nulla per t <= 0.
    """
    ic_cfg = driving_cfg["initial_tracer_gaussian"]
    c0 = problem.build_gaussian_initial_tracer(
        center=tuple(ic_cfg["center"]),
        sigma=float(ic_cfg["sigma"]),
        amplitude=float(ic_cfg["amplitude"]),
        baseline=float(ic_cfg["baseline"]),
    )

    info = compute_spatial_forcing_vectors_and_diagnostics(problem, driving_cfg)
    omega = float(info["omega_t"])
    F_spatial_base = np.asarray(info["F_spatial_base"], dtype=np.float64)
    Sf_spatial_rhs = np.asarray(info["Sf_spatial_rhs"], dtype=np.float64)

    def F_mech_func(t: float) -> np.ndarray:
        if t <= 0.0:
            return np.zeros(problem.num_disp_dofs, dtype=np.float64)
        return math.sin(omega * float(t)) * F_spatial_base

    def S_fluid_func(t: float) -> np.ndarray:
        if t <= 0.0:
            return np.zeros(problem.num_press_dofs, dtype=np.float64)
        return math.sin(omega * float(t)) * Sf_spatial_rhs

    return c0, F_mech_func, S_fluid_func


def _extract_center_from_geometry_report(report_obj: Any) -> Optional[Tuple[float, float]]:
    """Cerca ricorsivamente le coordinate adimensionali del centro lesione/dominio nel report JSON."""
    candidate_keys = (
        "lesion_centroid_adim",
        "lesion_center_adim",
        "tumor_centroid_adim",
        "core_centroid_adim",
        "wt_centroid_adim",
        "centroid_adim",
        "source_center_adim",
        "center_adim",
    )
    if isinstance(report_obj, dict):
        for k in candidate_keys:
            val = report_obj.get(k)
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                try:
                    c0, c1 = float(val[0]), float(val[1])
                    if np.isfinite(c0) and np.isfinite(c1):
                        return (c0, c1)
                except (TypeError, ValueError):
                    pass
        for v in report_obj.values():
            if isinstance(v, dict):
                found = _extract_center_from_geometry_report(v)
                if found is not None:
                    return found
    return None

def build_anatomical_driving_Callbacks(
    problem: Any,
    driving_cfg: Dict[str, Any],
    *,
    geometry_report_path: Optional[Any] = None,
    scales: Optional[Any] = None,
) -> tuple[np.ndarray, Callable[[float], np.ndarray], Callable[[float], np.ndarray], Dict[str, Any]]:
    """
    Costruisce le condizioni iniziali e le forzanti per il problema diretto su mesh anatomica,
    restituendo la quadrupla (c0, F_mech_func, S_fluid_func, forcing_meta) compatibile con
    tests/test_07_anatomical_mesh_integration.py ed experiments/02_run_anatomical_forward.py.
    """
    import json
    from pathlib import Path

    L0_mm = 10.0
    if scales is not None and hasattr(scales, "L0_mm"):
        L0_mm = float(scales.L0_mm)
    elif scales is not None and hasattr(scales, "L0_m"):
        L0_mm = float(scales.L0_m) * 1000.0
    elif hasattr(problem, "scales"):
        if hasattr(problem.scales, "L0_mm"):
            L0_mm = float(problem.scales.L0_mm)
        elif hasattr(problem.scales, "L0_m"):
            L0_mm = float(problem.scales.L0_m) * 1000.0

    geom_data: Dict[str, Any] = {}
    if geometry_report_path is not None:
        p = Path(geometry_report_path)
        if p.exists():
            try:
                geom_data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                geom_data = {}

    if "L0_mm" in geom_data:
        L0_mm = float(geom_data["L0_mm"])
    elif "L0_m" in geom_data:
        L0_mm = float(geom_data["L0_m"]) * 1000.0

    # Estrazione coordinate DoF tramite gli attributi press_pts e tracer_pts di RetecoForwardProblem2D
    coords_p = np.asarray(problem.press_pts, dtype=np.float64)
    coords_c = np.asarray(problem.tracer_pts, dtype=np.float64)
    bbox_min_adim = np.min(coords_p, axis=0)
    bbox_max_adim = np.max(coords_p, axis=0)
    domain_centroid_adim = np.mean(coords_p, axis=0)

    center_adim: Optional[np.ndarray] = None
    center_mm: Optional[np.ndarray] = None

    # 1. Lettura centroide lesione dal geometry_quality_report.json (struttura Fase 1)
    lesion_info = geom_data.get("lesion_centroids", {})
    if isinstance(lesion_info, dict):
        if "center_mesh_adim" in lesion_info and lesion_info["center_mesh_adim"] is not None:
            center_adim = np.asarray(lesion_info["center_mesh_adim"], dtype=np.float64)[:2]
        if "center_physical_mm" in lesion_info and lesion_info["center_physical_mm"] is not None:
            center_mm = np.asarray(lesion_info["center_physical_mm"], dtype=np.float64)[:2]

    # Fallback su altre chiavi piatte del report geometrico
    if center_adim is None:
        for key_cand in (
            "lesion_core_centroid_adim",
            "tumor_centroid_adim",
            "edema_centroid_adim",
            "centroid_adim",
        ):
            if key_cand in geom_data and geom_data[key_cand] is not None:
                center_adim = np.asarray(geom_data[key_cand], dtype=np.float64)[:2]
                break

    if center_adim is None and "lesion_core_centroid_mm" in geom_data:
        center_mm = np.asarray(geom_data["lesion_core_centroid_mm"], dtype=np.float64)[:2]
        center_adim = center_mm / L0_mm

    # 2. Fallback sulla configurazione YAML driving_cfg
    fluid_cfg = driving_cfg.get(
        "fluid_injection_source", driving_cfg.get("fluid_source", driving_cfg)
    )
    tracer_cfg = driving_cfg.get(
        "initial_tracer_gaussian", driving_cfg.get("initial_tracer_distribution", driving_cfg)
    )

    if center_adim is None:
        if isinstance(fluid_cfg, dict) and "center_mesh_adim" in fluid_cfg:
            center_adim = np.asarray(fluid_cfg["center_mesh_adim"], dtype=np.float64)[:2]
        elif isinstance(fluid_cfg, dict) and "center_adim" in fluid_cfg:
            center_adim = np.asarray(fluid_cfg["center_adim"], dtype=np.float64)[:2]
        elif isinstance(fluid_cfg, dict) and "center_physical_mm" in fluid_cfg:
            center_mm = np.asarray(fluid_cfg["center_physical_mm"], dtype=np.float64)[:2]
            center_adim = center_mm / L0_mm
        else:
            center_adim = domain_centroid_adim.copy()

    # Garantisce che il centro cada all'interno del bounding box computazionale
    margin = 0.02 * (bbox_max_adim - bbox_min_adim)
    center_adim = np.clip(center_adim, bbox_min_adim + margin, bbox_max_adim - margin)
    if center_mm is None:
        if "coordinate_transform" in geom_data and isinstance(geom_data["coordinate_transform"], dict):
            ct = geom_data["coordinate_transform"]
            origin_mm = np.asarray(ct.get("origin_physical_mm", [0.0, 0.0]), dtype=np.float64)[:2]
            center_mm = origin_mm + center_adim * L0_mm
        else:
            center_mm = center_adim * L0_mm

    # Raggio/deviazione standard sorgente fluida (in mm e adimensionale)
    if isinstance(fluid_cfg, dict) and "sigma_mm" in fluid_cfg:
        sigma_fluid_mm = float(fluid_cfg["sigma_mm"])
        sigma_fluid_adim = sigma_fluid_mm / L0_mm
    elif isinstance(fluid_cfg, dict) and "sigma_physical_mm" in fluid_cfg:
        sigma_fluid_mm = float(fluid_cfg["sigma_physical_mm"])
        sigma_fluid_adim = sigma_fluid_mm / L0_mm
    else:
        sigma_fluid_adim = float(
            (fluid_cfg.get("sigma_mesh_adim") if isinstance(fluid_cfg, dict) else None)
            or (fluid_cfg.get("sigma_adim") if isinstance(fluid_cfg, dict) else None)
            or driving_cfg.get("fluid_gaussian_sigma", 0.60)
        )
        sigma_fluid_mm = sigma_fluid_adim * L0_mm

    # Raggio/deviazione standard gaussiana iniziale tracciante (es. 8.0 mm / 10.0 mm = 0.80)
    if isinstance(tracer_cfg, dict) and "sigma_mm" in tracer_cfg:
        sigma_tracer_mm = float(tracer_cfg["sigma_mm"])
        sigma_tracer_adim = sigma_tracer_mm / L0_mm
    elif isinstance(tracer_cfg, dict) and "sigma_physical_mm" in tracer_cfg:
        sigma_tracer_mm = float(tracer_cfg["sigma_physical_mm"])
        sigma_tracer_adim = sigma_tracer_mm / L0_mm
    elif "tracer_initial_sigma_mm" in driving_cfg:
        sigma_tracer_mm = float(driving_cfg["tracer_initial_sigma_mm"])
        sigma_tracer_adim = sigma_tracer_mm / L0_mm
    else:
        sigma_tracer_adim = float(
            (tracer_cfg.get("sigma_mesh_adim") if isinstance(tracer_cfg, dict) else None)
            or (tracer_cfg.get("sigma_adim") if isinstance(tracer_cfg, dict) else None)
            or driving_cfg.get("tracer_initial_sigma", 0.80)
        )
        sigma_tracer_mm = sigma_tracer_adim * L0_mm

    tracer_amplitude = float(
        (tracer_cfg.get("amplitude", 1.0) if isinstance(tracer_cfg, dict) else 1.0)
    )
    fluid_amplitude = float(
        (fluid_cfg.get("amplitude", driving_cfg.get("fluid_source_amplitude", 0.35))
         if isinstance(fluid_cfg, dict)
         else driving_cfg.get("fluid_source_amplitude", 0.35))
    )

    # Costruzione vettore iniziale c0 sulla mesh anatomica
    dx_c = coords_c[:, 0] - float(center_adim[0])
    dy_c = coords_c[:, 1] - float(center_adim[1])
    c0 = tracer_amplitude * np.exp(
        -0.5 * (dx_c ** 2 + dy_c ** 2) / (sigma_tracer_adim ** 2)
    )

    # Costruzione sorgente fluida S_fluid_func sulla mesh anatomica
    dx_p = coords_p[:, 0] - float(center_adim[0])
    dy_p = coords_p[:, 1] - float(center_adim[1])
    spatial_fluid_profile = fluid_amplitude * np.exp(
        -0.5 * (dx_p ** 2 + dy_p ** 2) / (sigma_fluid_adim ** 2)
    )
    s_fluid_load_vec = problem.M_p_csr @ spatial_fluid_profile

    ramp_tau = float(driving_cfg.get("ramp_time_constant", 0.08))

    def S_fluid_func(t: float) -> np.ndarray:
        ramp = 1.0 - np.exp(-float(t) / max(ramp_tau, 1.0e-12)) if ramp_tau > 0.0 else 1.0
        return ramp * s_fluid_load_vec

    # Costruzione forzante meccanica F_mech_func (nulla se disattivata in anatomica, o via builder)
    f_mech_zero = np.zeros(problem.num_disp_dofs, dtype=np.float64)

    def F_mech_func(t: float) -> np.ndarray:
        return f_mech_zero

    ones_c = np.ones(problem.num_tracer_dofs, dtype=np.float64)
    initial_tracer_mass = float(ones_c @ (problem.M_c_csr @ c0))
    ones_p = np.ones(problem.num_press_dofs, dtype=np.float64)
    fluid_spatial_integral = float(ones_p @ s_fluid_load_vec)

    forcing_meta: Dict[str, Any] = {
        "forcing_classification": "anatomy_guided_synthetic_forcing",
        "L0_mm": float(L0_mm),
        "L0_m": float(L0_mm) / 1000.0,
        "geometry_report_path": str(geometry_report_path) if geometry_report_path is not None else None,
        "initial_tracer_mass": initial_tracer_mass,
        "initial_tracer_gaussian": {
            "center_mesh_adim": [float(center_adim[0]), float(center_adim[1])],
            "center_adim": [float(center_adim[0]), float(center_adim[1])],
            "center_physical_mm": [float(center_mm[0]), float(center_mm[1])],
            "sigma_mesh_adim": float(sigma_tracer_adim),
            "sigma_adim": float(sigma_tracer_adim),
            "sigma_mm": float(sigma_tracer_mm),
            "sigma_physical_mm": float(sigma_tracer_mm),
            "amplitude": float(tracer_amplitude),
            "initial_mass": initial_tracer_mass,
        },
        "initial_tracer_distribution": {
            "center_mesh_adim": [float(center_adim[0]), float(center_adim[1])],
            "center_adim": [float(center_adim[0]), float(center_adim[1])],
            "center_physical_mm": [float(center_mm[0]), float(center_mm[1])],
            "sigma_mesh_adim": float(sigma_tracer_adim),
            "sigma_adim": float(sigma_tracer_adim),
            "sigma_mm": float(sigma_tracer_mm),
            "sigma_physical_mm": float(sigma_tracer_mm),
            "initial_mass": initial_tracer_mass,
        },
        "fluid_injection_source": {
            "center_mesh_adim": [float(center_adim[0]), float(center_adim[1])],
            "center_adim": [float(center_adim[0]), float(center_adim[1])],
            "center_physical_mm": [float(center_mm[0]), float(center_mm[1])],
            "sigma_mesh_adim": float(sigma_fluid_adim),
            "sigma_adim": float(sigma_fluid_adim),
            "sigma_mm": float(sigma_fluid_mm),
            "sigma_physical_mm": float(sigma_fluid_mm),
            "radius_adim": float(sigma_fluid_adim),
            "radius_physical_mm": float(sigma_fluid_mm),
            "amplitude": float(fluid_amplitude),
            "spatial_integral": fluid_spatial_integral,
        },
        "mechanical_body_force": {
            "active": False,
            "Fx_integral": 0.0,
            "Fy_integral": 0.0,
        },
    }
    return c0, F_mech_func, S_fluid_func, forcing_meta