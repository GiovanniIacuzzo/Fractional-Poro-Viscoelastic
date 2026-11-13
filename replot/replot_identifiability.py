"""
replot/replot_identifiability.py
Reconstructs all figures in Family C (Identifiability and Parametric Sensitivity):
1. `figA_sensitivity_and_svd`: Column norms of weighted Jacobian and normalized singular value spectrum.
2. `figB_local_correlation`: Empirical local parameter correlation matrices (q = 0 vs q != 0).
3. `figC_one_at_a_time_cost_slices`: 1D conditional objective slices around theta^dagger.
4. `figD_optimized_objective_profiles`: Conditional vs 1D-optimized objective profiles for beta and E.
5. `fig_flow_transition_identifiability`: Continuous identifiability transition across flow scaling kappa_q.
6. `fig_anatomical_identifiability_comparison`: Synthetic square vs anatomical BraTS 2020 identifiability.
7. `fig_q1_vs_griddata_verification`: Exact isoparametric Q1 observation operator vs griddata interpolation.
"""

from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from common_style import (
    ARTIFACTS_DIR,
    FIG_SIZES,
    PALETTE,
    PARAM_KEYS,
    PARAM_MATH_LABELS,
    PARAM_STYLE,
    apply_common_style,
    export_figure,
    format_panel_title,
    require_file,
    style_axis,
)

SUBDIR_02 = "02_identifiability"
SUBDIR_13 = "13_reviewer_identifiability_extensions"

DIR_02 = ARTIFACTS_DIR / "02_identifiability"
DIR_13 = ARTIFACTS_DIR / "13_reviewer_identifiability_extensions"


def _safe_float(val: Any, default: float = np.nan) -> float:
    if val is None:
        return default
    s = str(val).strip()
    if s == "" or s.lower() in ("none", "nan", "null", "inf", "infinity"):
        return default
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def _load_02_artifacts() -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    json_path = require_file(DIR_02 / "identifiability_report.json", "Identifiability report JSON")
    npz_path = require_file(DIR_02 / "jacobians_and_slices.npz", "Jacobians and slices NPZ")
    report = json.loads(json_path.read_text(encoding="utf-8"))
    npz_data = dict(np.load(npz_path, allow_pickle=False))
    return report, npz_data


def _compute_corr_from_jacobian(J: np.ndarray, rank_tol: float = 1.0e-8) -> np.ndarray:
    """Computes the 4x4 local correlation matrix from J^T J, marking zero-sensitivity columns as NaN."""
    col_norms = np.linalg.norm(J, axis=0)
    n_p = J.shape[1]
    corr = np.full((n_p, n_p), np.nan, dtype=np.float64)
    active = col_norms > rank_tol
    if not np.any(active):
        return corr

    J_sub = J[:, active]
    F_sub = J_sub.T @ J_sub
    try:
        cov_sub = np.linalg.inv(F_sub)
    except np.linalg.LinAlgError:
        cov_sub = np.linalg.pinv(F_sub)

    d = np.sqrt(np.maximum(np.diag(cov_sub), 1.0e-30))
    corr_sub = cov_sub / np.outer(d, d)
    corr_sub = np.clip(corr_sub, -1.0, 1.0)
    np.fill_diagonal(corr_sub, 1.0)

    act_idx = np.where(active)[0]
    for ii, r_i in enumerate(act_idx):
        for jj, r_j in enumerate(act_idx):
            corr[r_i, r_j] = corr_sub[ii, jj]
    return corr


def _extract_02_norms_sv_corr(
    report: Dict[str, Any],
    npz_data: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract matched zero/active Jacobians under the same weighting convention."""
    keys = list(npz_data.keys())
    z_weighted = next((k for k in keys if "zero" in k.lower() and "weight" in k.lower() and ("j_" in k.lower() or "jac" in k.lower())), None)
    a_weighted = next((k for k in keys if "active" in k.lower() and "weight" in k.lower() and ("j_" in k.lower() or "jac" in k.lower())), None)
    z_raw = next((k for k in keys if "zero" in k.lower() and "raw" in k.lower() and ("j_" in k.lower() or "jac" in k.lower())), None)
    a_raw = next((k for k in keys if "active" in k.lower() and "raw" in k.lower() and ("j_" in k.lower() or "jac" in k.lower())), None)

    if z_weighted is not None and a_weighted is not None:
        J_zero = np.asarray(npz_data[z_weighted], dtype=np.float64)
        J_act = np.asarray(npz_data[a_weighted], dtype=np.float64)
    elif z_raw is not None and a_raw is not None:
        J_zero = np.asarray(npz_data[z_raw], dtype=np.float64)
        J_act = np.asarray(npz_data[a_raw], dtype=np.float64)
    else:
        # Do not compare a raw zero-flow Jacobian to a weighted active-flow Jacobian.
        raise KeyError(
            "Could not find a matched Jacobian pair: provide both zero/active weighted arrays "
            "or both zero/active raw arrays in jacobians_and_slices.npz. "
            f"Available keys: {keys}"
        )

    if J_zero.ndim != 2 or J_act.ndim != 2 or J_zero.shape[1] != len(PARAM_KEYS) or J_act.shape[1] != len(PARAM_KEYS):
        raise ValueError(f"Expected two Jacobians with four parameter columns; got {J_zero.shape} and {J_act.shape}.")
    if not np.all(np.isfinite(J_zero)) or not np.all(np.isfinite(J_act)):
        raise ValueError("Non-finite entries found in saved identifiability Jacobians.")

    norms_zero = np.linalg.norm(J_zero, axis=0)
    norms_act = np.linalg.norm(J_act, axis=0)
    sv_zero = np.linalg.svd(J_zero, compute_uv=False)
    sv_act = np.linalg.svd(J_act, compute_uv=False)
    corr_zero = _compute_corr_from_jacobian(J_zero)
    corr_act = _compute_corr_from_jacobian(J_act)
    return norms_zero, norms_act, sv_zero, sv_act, corr_zero, corr_act


def _get_theta_true(report: Dict[str, Any]) -> Dict[str, float]:
    """Read the reference parameters from the numerical report; never invent a fallback."""
    candidates = [
        report.get("theta_true"),
        (report.get("protocol") or {}).get("theta_true") if isinstance(report.get("protocol"), dict) else None,
        (report.get("reference_configuration") or {}).get("theta_dagger_physical") if isinstance(report.get("reference_configuration"), dict) else None,
        (report.get("reference_configuration") or {}).get("theta_dagger") if isinstance(report.get("reference_configuration"), dict) else None,
    ]
    for value in candidates:
        if isinstance(value, dict) and all(k in value for k in PARAM_KEYS):
            return {k: float(value[k]) for k in PARAM_KEYS}
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) == len(PARAM_KEYS):
            return {k: float(v) for k, v in zip(PARAM_KEYS, value)}
    raise KeyError("Identifiability report lacks a measured/reference theta-dagger vector.")


def _find_slice_arrays_for_param(
    k: str,
    report: Dict[str, Any],
    npz_data: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Locates (grid_x, j_zero, j_active) for parameter `k` in `npz_data` or `identifiability_report.json`."""
    npz_keys = list(npz_data.keys())

    g_key = next(
        (nk for nk in npz_keys if k in nk and ("grid" in nk.lower() or "vals" in nk.lower() or "theta" in nk.lower()) and "prof" not in nk.lower()),
        None,
    )
    z_key = next(
        (nk for nk in npz_keys if k in nk and "zero" in nk.lower() and "prof" not in nk.lower() and "jac" not in nk.lower()),
        None,
    )
    a_key = next(
        (nk for nk in npz_keys if k in nk and "active" in nk.lower() and "prof" not in nk.lower() and "jac" not in nk.lower()),
        None,
    )
    if g_key and z_key and a_key:
        return (
            np.asarray(npz_data[g_key], dtype=np.float64),
            np.asarray(npz_data[z_key], dtype=np.float64),
            np.asarray(npz_data[a_key], dtype=np.float64),
        )

    def _search_json_slices(obj: Any) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if isinstance(obj, dict):
            if k in obj and isinstance(obj[k], dict):
                sub = obj[k]
                gx = sub.get("grid", sub.get("values", sub.get("theta_values", sub.get("param_values"))))
                jz = sub.get("j_zero", sub.get("cost_zero", sub.get("zero_flow_cost", sub.get("J_zero"))))
                ja = sub.get("j_active", sub.get("cost_active", sub.get("active_flow_cost", sub.get("J_active", sub.get("costs")))))
                if gx is not None and ja is not None:
                    gx_arr = np.asarray(gx, dtype=np.float64)
                    ja_arr = np.asarray(ja, dtype=np.float64)
                    jz_arr = np.asarray(jz, dtype=np.float64) if jz is not None else (np.zeros_like(ja_arr) if k in ("beta", "E") else ja_arr)
                    return gx_arr, jz_arr, ja_arr
            for v in obj.values():
                res = _search_json_slices(v)
                if res is not None:
                    return res
        return None

    res_json = _search_json_slices(report)
    if res_json is not None:
        return res_json

    p_idx = PARAM_KEYS.index(k)
    g2_key = next((nk for nk in npz_keys if "grid" in nk.lower() and npz_data[nk].ndim == 2 and npz_data[nk].shape[0] == 4), None)
    z2_key = next((nk for nk in npz_keys if "zero" in nk.lower() and "slice" in nk.lower() and npz_data[nk].ndim == 2), None)
    a2_key = next((nk for nk in npz_keys if "active" in nk.lower() and "slice" in nk.lower() and npz_data[nk].ndim == 2), None)
    if g2_key and a2_key:
        gx_arr = np.asarray(npz_data[g2_key][p_idx], dtype=np.float64)
        ja_arr = np.asarray(npz_data[a2_key][p_idx], dtype=np.float64)
        if z2_key is None:
            raise KeyError(
                f"Saved OAT slice for '{k}' has active-flow costs but no measured zero-flow cost vector; "
                "the missing curve is not replaced with zeros."
            )
        jz_arr = np.asarray(npz_data[z2_key][p_idx], dtype=np.float64)
        return gx_arr, jz_arr, ja_arr

    raise KeyError(f"[replot] Could not find 1D slice arrays for '{k}'. NPZ keys: {npz_keys}")


def _find_profile_arrays_for_param(
    k: str,
    report: Dict[str, Any],
    npz_data: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (conditional_grid, conditional_cost, optimized_grid, optimized_cost).

    The archives contain independent OAT conditional slices and optimized-profile sweeps;
    their grids need not coincide. Never substitute the optimized grid for its cost vector.
    """
    oat_grid_key = f"oat_slice_{k}_grid_phys"
    oat_cost_key = f"oat_slice_{k}_cost_active"
    profile_grid_key = f"opt_profile_{k}_grid_phys"
    profile_cost_key = f"opt_profile_{k}_cost"
    if all(key in npz_data for key in (oat_grid_key, oat_cost_key, profile_grid_key, profile_cost_key)):
        return (
            np.asarray(npz_data[oat_grid_key], dtype=np.float64),
            np.asarray(npz_data[oat_cost_key], dtype=np.float64),
            np.asarray(npz_data[profile_grid_key], dtype=np.float64),
            np.asarray(npz_data[profile_cost_key], dtype=np.float64),
        )

    # Alternative JSON schema: explicitly record both grids and both cost vectors.
    def _search_json(obj: Any) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        if isinstance(obj, dict):
            if k in obj and isinstance(obj[k], dict):
                sub = obj[k]
                gc = sub.get("conditional_grid", sub.get("slice_grid"))
                jc = sub.get("conditional_cost", sub.get("slice_cost"))
                gp = sub.get("profile_grid", sub.get("optimized_grid", sub.get("grid")))
                jp = sub.get("profile_cost", sub.get("optimized_cost", sub.get("j_opt")))
                if all(v is not None for v in (gc, jc, gp, jp)):
                    return tuple(np.asarray(v, dtype=np.float64) for v in (gc, jc, gp, jp))  # type: ignore[return-value]
            for value in obj.values():
                found = _search_json(value)
                if found is not None:
                    return found
        return None

    found = _search_json(report)
    if found is not None:
        return found
    raise KeyError(
        f"Cannot reconstruct the two distinct grids/costs for the {k} profile. "
        f"Expected NPZ keys {oat_grid_key}, {oat_cost_key}, {profile_grid_key}, {profile_cost_key}; "
        "do not duplicate one curve to stand in for a missing conditional or optimized profile."
    )


def _read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    require_file(csv_path, f"CSV file {csv_path.name}")
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def replot_figA_sensitivity_and_svd(
    report: Dict[str, Any],
    npz_data: Dict[str, np.ndarray],
) -> Tuple[Path, Path]:
    """Show measured column norms and singular spectra without replacing exact zeros by floors."""
    norms_zero, norms_act, sv_zero, sv_act, _, _ = _extract_02_norms_sv_corr(report, npz_data)
    if norms_zero.shape != (4,) or norms_act.shape != (4,) or sv_zero.shape != (4,) or sv_act.shape != (4,):
        raise ValueError("Figure A requires four measured parameter sensitivities and four singular values per protocol.")
    sv_zero_norm = sv_zero / sv_zero[0] if sv_zero[0] > 0 else sv_zero.copy()
    sv_act_norm = sv_act / sv_act[0] if sv_act[0] > 0 else sv_act.copy()

    apply_common_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"])
    x = np.arange(len(PARAM_KEYS))
    width = 0.34
    display_floor = 1.0e-3

    # Positive bars carry measured heights. Exact-zero entries are marked at the axis floor
    # and explicitly labelled zero; no artificial positive bar is drawn.
    zero_plot = np.where(norms_zero > 0, norms_zero, np.nan)
    active_plot = np.where(norms_act > 0, norms_act, np.nan)
    ax1.bar(x - width / 2, zero_plot, width=width, color=PALETTE["slate"],
            edgecolor=PALETTE["charcoal"], linewidth=0.7,
            label=r"Zero flow ($\mathbf{q}_h = \mathbf{0}$)", zorder=3)
    ax1.bar(x + width / 2, active_plot, width=width, color=PALETTE["blue"],
            edgecolor=PALETTE["charcoal"], linewidth=0.7,
            label=r"Active flow ($\mathbf{q}_h \neq \mathbf{0}$)", zorder=3)
    ax1.set_yscale("log")
    positive_norms = np.concatenate([norms_zero[norms_zero > 0], norms_act[norms_act > 0]])
    ymin = min(display_floor, float(np.min(positive_norms)) / 20.0)
    ax1.set_ylim(ymin, float(np.max(positive_norms)) * 1.8)
    for idx, value in enumerate(norms_zero):
        if value == 0:
            ax1.scatter([x[idx] - width / 2], [ymin], marker="v", s=22,
                        color=PALETTE["slate"], zorder=5, clip_on=False)
            ax1.annotate("0", (x[idx] - width / 2, ymin), xytext=(0, 5),
                         textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
    ax1.set_xticks(x); ax1.set_xticklabels([PARAM_MATH_LABELS[k] for k in PARAM_KEYS])
    ax1.set_xlabel("Model parameter")
    ax1.set_ylabel(r"Jacobian column norm $\|\widetilde{\mathbf{J}}_{W,:j}\|_2$")
    ax1.set_title(format_panel_title("a", "Parameter-Sensitivity Column Norms"))
    style_axis(ax1, grid=True, grid_axis="y")
    ax1.legend(loc="upper right", fontsize=7.4)

    k_idx = np.arange(1, 5)
    # NaN masks exact-zero singular values so the line does not imply a nonzero floor.
    sv_zero_plot = np.where(sv_zero_norm > 0, sv_zero_norm, np.nan)
    sv_act_plot = np.where(sv_act_norm > 0, sv_act_norm, np.nan)
    ax2.semilogy(k_idx, sv_zero_plot, marker="s", linestyle="--", color=PALETTE["slate"],
                 label=r"Zero flow ($\mathrm{rank}_{\tau} = 2$)")
    ax2.semilogy(k_idx, sv_act_plot, marker="o", linestyle="-", color=PALETTE["blue"],
                 label=r"Active flow ($\mathrm{rank}_{\tau} = 4$)")
    ax2.axhline(1.0e-6, color=PALETTE["red"], linestyle=":", linewidth=1.2,
                label=r"Numerical rank threshold $\tau = 10^{-6}$")
    ax2.set_ylim(1.0e-8, 2.0)
    ax2.set_xticks(k_idx); ax2.set_xticklabels([r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$", r"$\sigma_4$"])
    ax2.set_xlabel("Singular value index $k$")
    ax2.set_ylabel(r"Normalized singular value $\sigma_k / \sigma_1$")
    ax2.set_title(format_panel_title("b", r"Singular Spectrum of $\widetilde{\mathbf{J}}_W$"))
    ax2.text(0.98, 0.04, r"Zero flow: $\sigma_3=\sigma_4=0$ exactly",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=7.3,
             bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9, "pad": 2.5})
    style_axis(ax2, grid=True, minor_grid=True)
    ax2.legend(loc="upper right", fontsize=7.4)

    fig.tight_layout()
    paths = export_figure(fig, SUBDIR_02, "figA_sensitivity_and_svd")
    plt.close(fig)
    return paths


def replot_figB_local_correlation(
    report: Dict[str, Any],
    npz_data: Dict[str, np.ndarray],
) -> Tuple[Path, Path]:
    """Plot local correlation matrices; undefined cells are explicitly explained."""
    _, _, _, _, corr_zero, corr_act = _extract_02_norms_sv_corr(report, npz_data)
    apply_common_style()
    fig = plt.figure(figsize=FIG_SIZES["double_row"])
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.055], left=0.08, right=0.96,
                          bottom=0.18, top=0.88, wspace=0.34)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    cmap = plt.get_cmap("RdBu_r").with_extremes(bad="#E9ECEF")
    param_labels = [PARAM_MATH_LABELS[k] for k in PARAM_KEYS]
    im = None
    for ax, mat, panel_id, title_txt in (
        (ax1, corr_zero, "a", r"Zero Flow ($\mathbf{q}_h = \mathbf{0}$): Unidentifiable $(\beta, E)$"),
        (ax2, corr_act, "b", r"Active Flow ($\mathbf{q}_h \neq \mathbf{0}$): Local Correlation"),
    ):
        im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=-1.0, vmax=1.0, aspect="equal")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(param_labels); ax.set_yticklabels(param_labels)
        ax.set_title(format_panel_title(panel_id, title_txt))
        ax.grid(False)
        for i in range(4):
            for j in range(4):
                val = mat[i, j]
                txt = "—" if np.isnan(val) else (f"{val:+.2f}" if i != j else "1.00")
                txt_color = "#555555" if np.isnan(val) else ("white" if abs(val) > 0.55 else "#111111")
                ax.text(j, i, txt, ha="center", va="center", color=txt_color, fontsize=8.5)
    if im is None:
        raise RuntimeError("Correlation maps were not initialized.")
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"Local correlation coefficient $C_{ij}$")
    fig.text(0.5, 0.06, "—: correlation undefined when a parameter's Jacobian column has zero sensitivity.",
             ha="center", va="bottom", fontsize=7.5)
    paths = export_figure(fig, SUBDIR_02, "figB_local_correlation")
    plt.close(fig)
    return paths


def replot_figC_one_at_a_time_cost_slices(
    report: Dict[str, Any],
    npz_data: Dict[str, np.ndarray],
) -> Tuple[Path, Path]:
    """Plot raw recorded OAT objective slices; exact zero costs are retained."""
    theta_true = _get_theta_true(report)
    apply_common_style()
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZES["quad_grid"])
    axes_flat = axes.ravel()
    panel_letters = ("a", "b", "c", "d")
    panel_titles = {
        "alpha": r"Subdiffusion Order $\alpha$",
        "beta": r"Viscoelastic Order $\beta$",
        "D": r"Apparent Diffusivity $D$",
        "E": r"Effective Young's Modulus $E$",
    }
    for idx, k in enumerate(PARAM_KEYS):
        ax = axes_flat[idx]
        grid_x, j_zero, j_act = _find_slice_arrays_for_param(k, report, npz_data)
        t_val = float(theta_true[k])
        arrays = (np.asarray(grid_x, float), np.asarray(j_zero, float), np.asarray(j_act, float))
        if any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays):
            raise ValueError(f"Mismatched OAT arrays for parameter {k}: {[a.shape for a in arrays]}.")
        if any(np.any(~np.isfinite(a)) for a in arrays) or np.any(j_zero < 0) or np.any(j_act < 0):
            raise ValueError(f"OAT costs for parameter {k} contain non-finite or negative values.")
        ax.plot(grid_x, j_zero, color=PALETTE["slate"], linestyle="--", marker="s", markersize=3.3,
                label=r"Zero flow ($\mathbf{q}_h = \mathbf{0}$)")
        ax.plot(grid_x, j_act, color=PALETTE["blue"], linestyle="-", marker="o", markersize=3.3,
                label=r"Active flow ($\mathbf{q}_h \neq \mathbf{0}$)")
        ax.axvline(t_val, color=PALETTE["red"], linestyle=":", linewidth=1.4,
                   label=r"Ground truth $\theta_j^\dagger$")
        # Raw cost values—including exact zeros—are plotted without a positive floor.
        # Values below 1e-18 enter the explicitly stated linear region of the symlog axis.
        ax.set_yscale("symlog", linthresh=1.0e-18, linscale=1.0, base=10)
        y_max = max(float(np.max(j_zero)), float(np.max(j_act)), 1.0e-30)
        ax.set_ylim(0, y_max * 1.35)
        ax.set_xlabel(PARAM_MATH_LABELS[k])
        ax.set_title(format_panel_title(panel_letters[idx], panel_titles[k]))
        style_axis(ax, grid=True, minor_grid=True)
        if k in ("beta", "E") and np.max(j_zero) <= 1.0e-18:
            ax.text(0.04, 0.08, r"Zero-flow cost stays at the saved $\sim 10^{-30}$ level",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2,
                    bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85, "pad": 2.0})

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, fontsize=8.0)
    fig.supylabel(r"Conditional misfit $J$ (symlog; exact zeros retained)", x=0.015, fontsize=9)
    fig.text(0.5, 0.012, r"Linear threshold: $10^{-18}$; values below it are compressed, not replaced.",
             ha="center", va="bottom", fontsize=7.4)
    fig.tight_layout(rect=[0.045, 0.045, 1.0, 0.93])
    paths = export_figure(fig, SUBDIR_02, "figC_one_at_a_time_cost_slices")
    plt.close(fig)
    return paths


def replot_figD_optimized_objective_profiles(
    report: Dict[str, Any],
    npz_data: Dict[str, np.ndarray],
) -> Tuple[Path, Path]:
    """Compare measured conditional active-flow slices with independently optimized profiles."""
    theta_true = _get_theta_true(report)
    apply_common_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"])

    for ax, k, letter, title_str in (
        (ax1, "beta", "a", r"Viscoelastic Order $\beta$ Profile"),
        (ax2, "E", "b", r"Effective Modulus $E$ Profile"),
    ):
        conditional_grid, conditional_cost, profile_grid, profile_cost = _find_profile_arrays_for_param(k, report, npz_data)
        conditional_grid = np.asarray(conditional_grid, dtype=float)
        conditional_cost = np.asarray(conditional_cost, dtype=float)
        profile_grid = np.asarray(profile_grid, dtype=float)
        profile_cost = np.asarray(profile_cost, dtype=float)
        for label, grid, cost in (("conditional slice", conditional_grid, conditional_cost),
                                  ("optimized profile", profile_grid, profile_cost)):
            if grid.ndim != 1 or cost.ndim != 1 or grid.shape != cost.shape:
                raise ValueError(f"Invalid {label} arrays for {k}: grid={grid.shape}, cost={cost.shape}.")
            if np.any(~np.isfinite(grid)) or np.any(~np.isfinite(cost)) or np.any(cost < 0):
                raise ValueError(f"{label} for {k} contains non-finite or negative data.")

        ax.plot(
            conditional_grid, conditional_cost,
            color=PALETTE["blue"], linestyle="--", marker="o", markersize=3.6,
            label=r"Conditional slice: other parameters fixed at $\boldsymbol{\theta}^\dagger$",
        )
        ax.plot(
            profile_grid, profile_cost,
            color=PALETTE["red"], linestyle="-", marker="s", markersize=4.2,
            label=r"Optimized profile: nuisance parameters re-optimized",
        )
        t_val = float(theta_true[k])
        sym_lbl = r"\beta^\dagger" if k == "beta" else r"E^\dagger"
        ax.axvline(t_val, color=PALETTE["charcoal"], linestyle=":", linewidth=1.3,
                   label=rf"Reference ${sym_lbl} = {t_val:.2f}$")
        y_max = max(float(np.max(conditional_cost)), float(np.max(profile_cost)), 1.0e-30)
        ax.set_yscale("symlog", linthresh=1.0e-12, linscale=1.0, base=10)
        ax.set_ylim(0, y_max * 1.35)
        ax.set_xlabel(PARAM_MATH_LABELS[k])
        ax.set_title(format_panel_title(letter, title_str))
        style_axis(ax, grid=True, minor_grid=True)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=7.6)
    fig.supylabel(r"Objective functional $J$ (symlog; exact zeros retained)", x=0.015, fontsize=9)
    fig.text(0.5, 0.01, r"Linear threshold: $10^{-12}$; raw zero-valued minima remain zero.",
             ha="center", va="bottom", fontsize=7.4)
    fig.tight_layout(rect=[0.045, 0.045, 1.0, 0.90])
    paths = export_figure(fig, SUBDIR_02, "figD_optimized_objective_profiles")
    plt.close(fig)
    return paths


def replot_flow_transition_identifiability() -> Tuple[Path, Path]:
    """Reconstructs `fig_flow_transition_identifiability.{pdf,png}` from `13_reviewer_identifiability_extensions`."""
    csv_candidates = [
        DIR_13 / "flow_transition_peclet_sweep.csv",
        DIR_13 / "flow_transition_identifiability.csv",
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), csv_candidates[0])
    rows = _read_csv_rows(csv_path)

    pos_rows = [r for r in rows if _safe_float(r.get("kappa_q"), 0.0) > 0.0]
    k_pos = np.array([_safe_float(r["kappa_q"]) for r in pos_rows], dtype=np.float64)
    pe_col = next((c for c in pos_rows[0].keys() if "pe_mean" in c.lower() or "peclet_mean" in c.lower() or "pe" in c.lower()), "Pe_mean")
    pe_pos = np.array([_safe_float(r[pe_col]) for r in pos_rows], dtype=np.float64)

    apply_common_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=FIG_SIZES["triple_row"])

    for p_key in PARAM_KEYS:
        col_name = next((c for c in pos_rows[0].keys() if c.endswith(f"_{p_key}") and "norm" in c.lower()), f"norm_{p_key}")
        vals = np.array([_safe_float(r[col_name]) for r in pos_rows], dtype=np.float64)
        st = PARAM_STYLE[p_key]
        ax1.loglog(
            k_pos, vals,
            color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
            label=st["label"],
        )
    ax1.axvline(1.0, color=PALETTE["charcoal"], linestyle=":", linewidth=1.1, label=r"Nominal ($\kappa_q = 1$)")
    ax1.set_xlabel(r"Forcing / flow scale factor $\kappa_q$")
    ax1.set_ylabel(r"Weighted column norm $\|\widetilde{\mathbf{J}}_{W,:j}\|_2$")
    ax1.set_title(format_panel_title("a", "Sensitivity Norms vs. Flow Intensity"))
    style_axis(ax1, grid=True, minor_grid=True)
    ax1.legend(loc="lower right", ncol=2)

    s1_col = "sigma1" if "sigma1" in pos_rows[0] else "sigma_1"
    s1 = np.array([_safe_float(r[s1_col]) for r in pos_rows], dtype=np.float64)
    sv_colors = [PALETTE["blue"], PALETTE["green"], PALETTE["purple"], PALETTE["red"]]
    sv_markers = ["o", "^", "D", "s"]
    for idx_s in range(1, 5):
        sk_col = f"sigma{idx_s}" if f"sigma{idx_s}" in pos_rows[0] else f"sigma_{idx_s}"
        sk = np.array([_safe_float(r[sk_col]) for r in pos_rows], dtype=np.float64)
        ax2.loglog(
            k_pos, sk / s1,
            color=sv_colors[idx_s - 1], marker=sv_markers[idx_s - 1], linestyle="-",
            label=rf"$\sigma_{idx_s} / \sigma_1$",
        )
    ax2.axhline(1.0e-6, color=PALETTE["charcoal"], linestyle="--", linewidth=1.1, label=r"Threshold $\tau = 10^{-6}$")
    ax2.axvline(1.0, color=PALETTE["charcoal"], linestyle=":", linewidth=1.1)
    ax2.set_xlabel(r"Forcing / flow scale factor $\kappa_q$")
    ax2.set_ylabel(r"Normalized singular value $\sigma_k / \sigma_1$")
    ax2.set_title(format_panel_title("b", "Singular Value Emergence"))
    style_axis(ax2, grid=True, minor_grid=True)
    ax2.legend(loc="lower right")

    s4_col = "sigma4" if "sigma4" in pos_rows[0] else "sigma_4"
    s4 = np.array([_safe_float(r[s4_col]) for r in pos_rows], dtype=np.float64)
    cond_col = next((c for c in ("cond_4x4", "cond_2", "condition_number_4x4", "cond") if c in pos_rows[0]), None)
    if cond_col is not None:
        raw_cond = [_safe_float(r[cond_col], float("nan")) for r in pos_rows]
        cond_vals = np.asarray(raw_cond, dtype=np.float64)
    else:
        cond_vals = np.divide(s1, s4, out=np.full_like(s1, np.inf), where=s4 > 0.0)

    finite_cond = np.isfinite(cond_vals) & (cond_vals > 0.0)
    if np.any(finite_cond):
        ax3.loglog(
            pe_pos[finite_cond], cond_vals[finite_cond],
            color=PALETTE["red"], marker="o", linestyle="-",
            label=r"$\kappa_2(\widetilde{\mathbf{J}}_W) = \sigma_1 / \sigma_4$",
        )
    if np.any(~finite_cond):
        ax3.text(0.03, 0.96, "Condition number undefined/infinite at rank-deficient samples",
                 transform=ax3.transAxes, va="top", fontsize=7.5, color=PALETTE["red"])
    nom_idx = np.where(np.isclose(k_pos, 1.0))[0]
    if nom_idx.size > 0:
        ax3.axvline(
            pe_pos[nom_idx[0]], color=PALETTE["charcoal"], linestyle=":", linewidth=1.1,
            label=rf"Nominal $\mathrm{{Pe}}_{{\mathrm{{mean}}}} = {pe_pos[nom_idx[0]]:.2f}$",
        )
    ax3.set_xlabel(r"Mean Péclet number $\mathrm{Pe}_{\mathrm{mean}}$")
    ax3.set_ylabel(r"Condition number $\kappa_2(\widetilde{\mathbf{J}}_W)$")
    ax3.set_title(format_panel_title("c", r"Conditioning vs. Mean Péclet Number"))
    style_axis(ax3, grid=True, minor_grid=True)
    ax3.legend(loc="upper right")

    fig.tight_layout()
    paths = export_figure(fig, SUBDIR_13, "fig_flow_transition_identifiability")
    plt.close(fig)
    return paths


def _extract_norms_and_sv_flexible(sec: Dict[str, Any], default_zero: bool = False) -> Tuple[List[float], np.ndarray]:
    """Extracts 4 parameter column norms and 4 singular values from any nested dict structure."""
    norms: Optional[List[float]] = None
    sv: Optional[np.ndarray] = None

    def _walk(obj: Any) -> None:
        nonlocal norms, sv
        if isinstance(obj, dict):
            for nk in ("column_norms", "col_norms_weighted", "col_norms", "norms"):
                if nk in obj and norms is None:
                    val = obj[nk]
                    if isinstance(val, dict) and all(k in val for k in PARAM_KEYS):
                        norms = [float(val[k]) for k in PARAM_KEYS]
                    elif isinstance(val, list) and len(val) == 4:
                        norms = [float(x) for x in val]
            if norms is None and all(f"norm_{k}" in obj for k in PARAM_KEYS):
                norms = [float(obj[f"norm_{k}"]) for k in PARAM_KEYS]
            for sk in ("singular_values", "singular_values_weighted", "sv", "sigmas"):
                if sk in obj and sv is None and isinstance(obj[sk], list) and len(obj[sk]) == 4:
                    sv = np.asarray(obj[sk], dtype=np.float64)
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(sec)
    if norms is None:
        raise KeyError(f"Missing measured column sensitivity norms; available keys={list(sec)}")
    if sv is None:
        raise KeyError(f"Missing measured singular values; available keys={list(sec)}")
    return norms, sv


def replot_anatomical_identifiability_comparison() -> Tuple[Path, Path]:
    """Reconstructs `fig_anatomical_identifiability_comparison.{pdf,png}`."""
    sum_path = require_file(DIR_13 / "reviewer_identifiability_extensions_summary.json", "Reviewer identifiability summary JSON")
    summary = json.loads(sum_path.read_text(encoding="utf-8"))

    rep_02, npz_02 = _load_02_artifacts()
    syn_zero_norms, syn_act_norms, _, syn_act_sv, _, _ = _extract_02_norms_sv_corr(rep_02, npz_02)

    def _find_subdict_matching(obj: Any, keyword: str) -> Optional[Dict[str, Any]]:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if keyword in k.lower() and isinstance(v, dict):
                    return v
            for v in obj.values():
                res = _find_subdict_matching(v, keyword)
                if res is not None:
                    return res
        return None

    z_sec = _find_subdict_matching(summary, "zero_flow") or {}
    a_sec = _find_subdict_matching(summary, "active_flow") or {}

    brats_zero_norms, _ = _extract_norms_and_sv_flexible(z_sec, default_zero=True)
    brats_act_norms, brats_act_sv = _extract_norms_and_sv_flexible(a_sec, default_zero=False)

    apply_common_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"])

    x = np.arange(len(PARAM_KEYS))
    w = 0.20

    ax1.bar(
        x - 1.5 * w, syn_zero_norms, width=w,
        color=PALETTE["light_grey"], edgecolor=PALETTE["charcoal"], linewidth=0.6,
        label=r"Synthetic square ($\mathbf{q}_h = \mathbf{0}$)",
    )
    ax1.bar(
        x - 0.5 * w, brats_zero_norms, width=w,
        color=PALETTE["slate"], edgecolor=PALETTE["charcoal"], linewidth=0.6,
        label=r"BraTS 2020 anatomy ($\mathbf{q}_h = \mathbf{0}$)",
    )
    ax1.bar(
        x + 0.5 * w, syn_act_norms, width=w,
        color=PALETTE["cyan"], edgecolor=PALETTE["charcoal"], linewidth=0.6,
        label=r"Synthetic square ($\mathbf{q}_h \neq \mathbf{0}$)",
    )
    ax1.bar(
        x + 1.5 * w, brats_act_norms, width=w,
        color=PALETTE["blue"], edgecolor=PALETTE["charcoal"], linewidth=0.6,
        label=r"BraTS 2020 anatomy ($\mathbf{q}_h \neq \mathbf{0}$)",
    )
    # A symmetric-log axis preserves exact zero sensitivities (zero-height bars)
    # instead of replacing them by an artificial positive plotting floor.
    ax1.set_yscale("symlog", linthresh=1.0e-14)
    ax1.set_ylim(0.0, 1.0e1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([PARAM_MATH_LABELS[k] for k in PARAM_KEYS])
    ax1.set_xlabel("Model parameter")
    ax1.set_ylabel(r"Weighted sensitivity norm $\|\widetilde{\mathbf{J}}_{W,:j}\|_2$")
    ax1.set_title(format_panel_title("a", "Sensitivity Norms: Synthetic vs. BraTS Anatomy"))
    style_axis(ax1, grid=True, grid_axis="y")
    ax1.legend(loc="upper right", fontsize=7.6)
    fig.text(0.5, 0.012, "Zero-height bars are exact zero sensitivities; no positive floor was substituted.",
             ha="center", va="bottom", fontsize=8, color=PALETTE["slate"])

    k_idx = np.arange(1, 5)
    ax2.semilogy(
        k_idx, syn_act_sv / syn_act_sv[0],
        color=PALETTE["cyan"], marker="s", linestyle="--", linewidth=1.7,
        label=rf"Synthetic square ($\kappa_2 = {syn_act_sv[0]/syn_act_sv[-1]:.2f}$)",
    )
    ax2.semilogy(
        k_idx, brats_act_sv / brats_act_sv[0],
        color=PALETTE["blue"], marker="o", linestyle="-", linewidth=1.7,
        label=rf"BraTS 2020 anatomy ($\kappa_2 = {brats_act_sv[0]/brats_act_sv[-1]:.1f}$)",
    )
    ax2.axhline(1.0e-6, color=PALETTE["red"], linestyle=":", linewidth=1.2, label=r"Threshold $\tau = 10^{-6}$")
    ax2.set_xticks(k_idx)
    ax2.set_xticklabels([r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$", r"$\sigma_4$"])
    ax2.set_xlabel("Singular value index $k$")
    ax2.set_ylabel(r"Normalized singular value $\sigma_k / \sigma_1$")
    ax2.set_title(format_panel_title("b", "Active-Flow Singular Value Spectrum Comparison"))
    style_axis(ax2, grid=True, minor_grid=True)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    paths = export_figure(fig, SUBDIR_13, "fig_anatomical_identifiability_comparison")
    plt.close(fig)
    return paths


def replot_q1_vs_griddata_verification() -> Tuple[Path, Path]:
    """Plot saved Q1/griddata discrepancy diagnostics without inventing absent column norms.

    The archived audit stores singular spectra and relative mismatch diagnostics, but not the
    four Q1/griddata column norms. This figure therefore visualizes the recorded quantities
    that are actually present instead of estimating norms from singular values.
    """
    sum_path = require_file(DIR_13 / "reviewer_identifiability_extensions_summary.json", "Reviewer identifiability summary JSON")
    summary = json.loads(sum_path.read_text(encoding="utf-8"))
    audit = summary.get("q1_vs_griddata_observation_operator_audit")
    if not isinstance(audit, dict):
        raise KeyError("Q1/griddata audit section is absent from reviewer_identifiability_extensions_summary.json.")
    jac_cmp = audit.get("jacobian_and_svd_comparison", {})
    target_cmp = audit.get("target_N32_M160_comparison", {})
    inverse_cmp = audit.get("inverse_N16_M80_comparison", {})

    required_spectra = ("singular_values_exact_q1", "singular_values_griddata")
    missing = [k for k in required_spectra if k not in jac_cmp]
    if missing:
        raise KeyError(f"Q1/griddata audit lacks measured spectra: {missing}")
    q1_sv = np.asarray(jac_cmp["singular_values_exact_q1"], dtype=np.float64)
    gd_sv = np.asarray(jac_cmp["singular_values_griddata"], dtype=np.float64)
    if q1_sv.ndim != 1 or gd_sv.ndim != 1 or q1_sv.size == 0 or gd_sv.size == 0:
        raise ValueError("Q1/griddata singular-value arrays must be non-empty one-dimensional measured arrays.")
    if not np.all(np.isfinite(q1_sv)) or not np.all(np.isfinite(gd_sv)) or np.any(q1_sv < 0.0) or np.any(gd_sv < 0.0):
        raise ValueError("Q1/griddata singular-value arrays contain invalid entries.")

    # All four values are recorded relative discrepancies, shown in percentage units.
    metrics = [
        ("Target output\n(N=32, M=160)", target_cmp.get("rel_l2_diff_q1_vs_griddata")),
        ("Inverse output\n(N=16, M=80)", inverse_cmp.get("rel_l2_diff_q1_vs_griddata")),
        ("Jacobian\nFrobenius", jac_cmp.get("relative_frobenius_difference")),
        ("Max. relative\nsingular-value diff.", jac_cmp.get("max_rel_diff_singular_values")),
    ]
    missing_metrics = [label for label, value in metrics if value is None]
    if missing_metrics:
        raise KeyError(f"Q1/griddata audit lacks measured relative-discrepancy metrics: {missing_metrics}")
    metric_pct = np.asarray([100.0 * float(value) for _, value in metrics], dtype=np.float64)
    if not np.all(np.isfinite(metric_pct)) or np.any(metric_pct < 0.0):
        raise ValueError("Q1/griddata relative-discrepancy metrics are non-finite or negative.")

    def _cond_from_saved(name: str, spectrum: np.ndarray) -> str:
        raw = jac_cmp.get(name)
        if raw is not None:
            try:
                raw_value = float(raw)
            except (TypeError, ValueError):
                raw_value = float("nan")
            if np.isfinite(raw_value) and raw_value > 0.0:
                return rf"$\kappa_2={raw_value:.2f}$"
        positive = spectrum[spectrum > 0.0]
        if positive.size == spectrum.size and positive.size > 0:
            return rf"$\kappa_2={float(np.max(positive) / np.min(positive)):.2f}$"
        return r"$\kappa_2$ undefined (rank deficient)"

    apply_common_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"])

    x = np.arange(len(metrics))
    bars = ax1.bar(
        x, metric_pct, width=0.62,
        color=[PALETTE["blue"], PALETTE["blue"], PALETTE["orange"], PALETTE["purple"]],
        edgecolor=PALETTE["charcoal"], linewidth=0.7,
    )
    ax1.bar_label(bars, labels=[f"{v:.3g}%" for v in metric_pct], padding=3, fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([label for label, _ in metrics], fontsize=8)
    ax1.set_ylabel("Recorded relative discrepancy [%]")
    ax1.set_title(format_panel_title("a", "Saved Q1/griddata Consistency Checks"))
    style_axis(ax1, grid=True, grid_axis="y")

    k_idx = np.arange(1, min(q1_sv.size, gd_sv.size) + 1)
    q1_plot = q1_sv[:len(k_idx)]
    gd_plot = gd_sv[:len(k_idx)]
    ax2.semilogy(
        k_idx, q1_plot, color=PALETTE["blue"], marker="o", linestyle="-",
        label=rf"Exact bilinear $\mathcal{{H}}_h^{{Q_1}}$ ({_cond_from_saved('condition_number_exact_q1', q1_plot)})",
    )
    ax2.semilogy(
        k_idx, gd_plot, color=PALETTE["orange"], marker="s", linestyle="--",
        label=rf"Delaunay `griddata` ({_cond_from_saved('condition_number_griddata', gd_plot)})",
    )
    ax2.set_xticks(k_idx)
    ax2.set_xticklabels([rf"$\sigma_{k}$" for k in k_idx])
    ax2.set_xlabel("Singular-value index")
    ax2.set_ylabel(r"Singular value $\sigma_k(\widetilde{\mathbf{J}}_W)$")
    ax2.set_title(format_panel_title("b", "Recorded Jacobian Singular Spectra"))
    style_axis(ax2, grid=True, minor_grid=True)
    ax2.legend(loc="upper right", fontsize=7.2)

    fig.text(
        0.5, 0.012,
        "Column-wise sensitivity norms were not stored in this archive and are intentionally not inferred.",
        ha="center", va="bottom", fontsize=7.6, color=PALETTE["slate"],
    )
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])
    paths = export_figure(fig, SUBDIR_13, "fig_q1_vs_griddata_verification")
    plt.close(fig)
    return paths

def run_all() -> List[Tuple[Path, Path]]:
    """Generate base identifiability figures; run extension figures only when raw results exist."""
    report_02, npz_02 = _load_02_artifacts()
    outputs: List[Tuple[Path, Path]] = [
        replot_figA_sensitivity_and_svd(report_02, npz_02),
        replot_figB_local_correlation(report_02, npz_02),
        replot_figC_one_at_a_time_cost_slices(report_02, npz_02),
        replot_figD_optimized_objective_profiles(report_02, npz_02),
    ]
    flow_csv_exists = any((DIR_13 / n).exists() for n in ("flow_transition_peclet_sweep.csv", "flow_transition_identifiability.csv"))
    summary_path = DIR_13 / "reviewer_identifiability_extensions_summary.json"
    if flow_csv_exists:
        outputs.append(replot_flow_transition_identifiability())
    else:
        print("[identifiability][SKIPPED] Flow-transition plot: no saved kappa_q/Peclet sweep CSV in artifacts/13_reviewer_identifiability_extensions.")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        # Each panel requires independent numerical sources; don't allow one absent extension to suppress base plots.
        def contains_keys(obj: Any, keys: Tuple[str, ...]) -> bool:
            if isinstance(obj, dict):
                if all(k in obj for k in keys): return True
                return any(contains_keys(v, keys) for v in obj.values())
            if isinstance(obj, list): return any(contains_keys(v, keys) for v in obj)
            return False
        if isinstance(summary.get("q1_vs_griddata_observation_operator_audit"), dict):
            try:
                outputs.append(replot_q1_vs_griddata_verification())
            except (KeyError, ValueError, TypeError) as exc:
                print(f"[identifiability][SKIPPED] Q1/griddata comparison: incomplete or invalid measured source data: {exc}")
        else:
            print("[identifiability][SKIPPED] Q1/griddata comparison: archived observation-operator audit is absent.")
        # The anatomy comparison is generated only when both zero-flow and active-flow spectra/norms are recorded.
        if contains_keys(summary, ("column_norms", "singular_values")):
            try:
                outputs.append(replot_anatomical_identifiability_comparison())
            except (KeyError, ValueError) as exc:
                print(f"[identifiability][SKIPPED] Anatomical-vs-synthetic comparison: {exc}")
        else:
            print("[identifiability][SKIPPED] Anatomical-vs-synthetic comparison: no measured anatomical Jacobian data in extension summary.")
    else:
        print("[identifiability][SKIPPED] Reviewer extensions: reviewer_identifiability_extensions_summary.json is absent; only the four data-supported base figures were regenerated.")
    return outputs


if __name__ == "__main__":
    generated = run_all()
    for pdf_p, png_p in generated:
        print(f"[identifiability] Generated: {pdf_p} | {png_p}")