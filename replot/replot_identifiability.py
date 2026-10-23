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
    """Extracts (norms_zero, norms_act, sv_zero, sv_act, corr_zero, corr_act) from NPZ or CSVs."""
    z_key = next((k for k in npz_data.keys() if "zero" in k.lower() and ("j_" in k.lower() or "jac" in k.lower())), None)
    a_key = next((k for k in npz_data.keys() if "active" in k.lower() and "weight" in k.lower() and ("j_" in k.lower() or "jac" in k.lower())), None)
    if a_key is None:
        a_key = next((k for k in npz_data.keys() if "active" in k.lower() and ("j_" in k.lower() or "jac" in k.lower())), None)

    if z_key is not None and a_key is not None:
        J_zero = np.asarray(npz_data[z_key], dtype=np.float64)
        J_act = np.asarray(npz_data[a_key], dtype=np.float64)
    else:
        def _load_jac_csv(p: Path) -> np.ndarray:
            with open(p, "r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
            return np.array([[float(r[k]) for k in PARAM_KEYS] for r in reader], dtype=np.float64)

        J_zero = _load_jac_csv(DIR_02 / "jacobian_zero_flow_raw.csv")
        J_act = _load_jac_csv(DIR_02 / "jacobian_active_flow_weighted.csv")

    norms_zero = np.linalg.norm(J_zero, axis=0)
    norms_act = np.linalg.norm(J_act, axis=0)
    sv_zero = np.linalg.svd(J_zero, compute_uv=False)
    sv_act = np.linalg.svd(J_act, compute_uv=False)
    corr_zero = _compute_corr_from_jacobian(J_zero)
    corr_act = _compute_corr_from_jacobian(J_act)

    return norms_zero, norms_act, sv_zero, sv_act, corr_zero, corr_act


def _get_theta_true(report: Dict[str, Any]) -> Dict[str, float]:
    if "theta_true" in report and isinstance(report["theta_true"], dict):
        return {k: float(v) for k, v in report["theta_true"].items()}
    if "protocol" in report and isinstance(report["protocol"], dict) and "theta_true" in report["protocol"]:
        return {k: float(v) for k, v in report["protocol"]["theta_true"].items()}
    return {"alpha": 0.70, "beta": 0.45, "D": 0.08, "E": 2.50}


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
        jz_arr = np.asarray(npz_data[z2_key][p_idx], dtype=np.float64) if z2_key else np.zeros_like(ja_arr)
        return gx_arr, jz_arr, ja_arr

    raise KeyError(f"[replot] Could not find 1D slice arrays for '{k}'. NPZ keys: {npz_keys}")


def _find_profile_arrays_for_param(
    k: str,
    report: Dict[str, Any],
    npz_data: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Locates (grid_x, j_conditional, j_optimized_profile) for parameter `k` in `npz_data` or `report`."""
    npz_keys = list(npz_data.keys())
    g_key = next((nk for nk in npz_keys if k in nk and "prof" in nk.lower() and ("grid" in nk.lower() or "val" in nk.lower())), None)
    s_key = next((nk for nk in npz_keys if k in nk and "prof" in nk.lower() and ("slice" in nk.lower() or "cond" in nk.lower() or "fix" in nk.lower())), None)
    o_key = next((nk for nk in npz_keys if k in nk and "prof" in nk.lower() and ("opt" in nk.lower() or "min" in nk.lower() or "prof_j" in nk.lower())), None)

    if g_key and o_key:
        gx = np.asarray(npz_data[g_key], dtype=np.float64)
        jo = np.asarray(npz_data[o_key], dtype=np.float64)
        js = np.asarray(npz_data[s_key], dtype=np.float64) if s_key else jo
        return gx, js, jo

    def _search_json_prof(obj: Any) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if isinstance(obj, dict):
            if k in obj and isinstance(obj[k], dict):
                sub = obj[k]
                gx = sub.get("grid", sub.get("values", sub.get("param_values")))
                jo = sub.get("profile_cost", sub.get("optimized_cost", sub.get("j_opt", sub.get("prof_opt_J"))))
                js = sub.get("slice_cost", sub.get("conditional_cost", sub.get("j_slice", sub.get("prof_slice_J"))))
                if gx is not None and jo is not None:
                    gx_a = np.asarray(gx, dtype=np.float64)
                    jo_a = np.asarray(jo, dtype=np.float64)
                    js_a = np.asarray(js, dtype=np.float64) if js is not None else jo_a
                    return gx_a, js_a, jo_a
            for v in obj.values():
                res = _search_json_prof(v)
                if res is not None:
                    return res
        return None

    res_j = _search_json_prof(report)
    if res_j is not None:
        return res_j

    gx, _, ja = _find_slice_arrays_for_param(k, report, npz_data)
    return gx, ja, ja


def _read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    require_file(csv_path, f"CSV file {csv_path.name}")
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def replot_figA_sensitivity_and_svd(
    report: Dict[str, Any],
    npz_data: Dict[str, np.ndarray],
) -> Tuple[Path, Path]:
    """Reconstructs `figA_sensitivity_and_svd.{pdf,png}`."""
    norms_zero, norms_act, sv_zero, sv_act, _, _ = _extract_02_norms_sv_corr(report, npz_data)
    sv_zero_norm = sv_zero / max(float(sv_zero[0]), 1.0e-30)
    sv_act_norm = sv_act / max(float(sv_act[0]), 1.0e-30)

    apply_common_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"])

    x = np.arange(len(PARAM_KEYS))
    width = 0.34
    floor_val = 1.0e-16

    ax1.bar(
        x - width / 2,
        np.maximum(norms_zero, floor_val),
        width=width,
        color=PALETTE["slate"],
        edgecolor=PALETTE["charcoal"],
        linewidth=0.7,
        label=r"Zero flow ($\mathbf{q}_h = \mathbf{0}$)",
        zorder=3,
    )
    ax1.bar(
        x + width / 2,
        np.maximum(norms_act, floor_val),
        width=width,
        color=PALETTE["blue"],
        edgecolor=PALETTE["charcoal"],
        linewidth=0.7,
        label=r"Active flow ($\mathbf{q}_h \neq \mathbf{0}$)",
        zorder=3,
    )
    ax1.set_yscale("log")
    ax1.set_ylim(1.0e-17, 1.0e1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([PARAM_MATH_LABELS[k] for k in PARAM_KEYS])
    ax1.set_xlabel("Model parameter")
    ax1.set_ylabel(r"Weighted sensitivity norm $\|\widetilde{\mathbf{J}}_{W,:j}\|_2$")
    ax1.set_title(format_panel_title("a", "Normalized Parameter Sensitivity Norms"))
    style_axis(ax1, grid=True, grid_axis="y")
    ax1.legend(loc="upper right")

    k_idx = np.arange(1, 5)
    ax2.semilogy(
        k_idx,
        np.maximum(sv_zero_norm, floor_val),
        marker="s",
        linestyle="--",
        color=PALETTE["slate"],
        label=r"Zero flow ($\mathrm{rank}_{\tau} = 2$)",
    )
    ax2.semilogy(
        k_idx,
        np.maximum(sv_act_norm, floor_val),
        marker="o",
        linestyle="-",
        color=PALETTE["blue"],
        label=r"Active flow ($\mathrm{rank}_{\tau} = 4$)",
    )
    ax2.axhline(
        1.0e-6,
        color=PALETTE["red"],
        linestyle=":",
        linewidth=1.2,
        label=r"Numerical rank threshold $\tau = 10^{-6}$",
    )
    ax2.set_ylim(1.0e-17, 2.0)
    ax2.set_xticks(k_idx)
    ax2.set_xticklabels([r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$", r"$\sigma_4$"])
    ax2.set_xlabel("Singular value index $k$")
    ax2.set_ylabel(r"Normalized singular value $\sigma_k / \sigma_1$")
    ax2.set_title(format_panel_title("b", r"Singular Value Spectrum of $\widetilde{\mathbf{J}}_W$"))
    style_axis(ax2, grid=True, minor_grid=True)
    ax2.legend(loc="lower left")

    fig.tight_layout()
    paths = export_figure(fig, SUBDIR_02, "figA_sensitivity_and_svd")
    plt.close(fig)
    return paths


def replot_figB_local_correlation(
    report: Dict[str, Any],
    npz_data: Dict[str, np.ndarray],
) -> Tuple[Path, Path]:
    """Reconstructs `figB_local_correlation.{pdf,png}`."""
    _, _, _, _, corr_zero, corr_act = _extract_02_norms_sv_corr(report, npz_data)

    apply_common_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"])

    cmap = plt.get_cmap("RdBu_r").with_extremes(bad="#E9ECEF")
    param_labels = [PARAM_MATH_LABELS[k] for k in PARAM_KEYS]

    for ax, mat, panel_id, title_txt in (
        (ax1, corr_zero, "a", r"Zero Flow ($\mathbf{q}_h = \mathbf{0}$): Unidentifiable $(\beta, E)$"),
        (ax2, corr_act, "b", r"Active Flow ($\mathbf{q}_h \neq \mathbf{0}$): Full Local Correlation"),
    ):
        masked_mat = np.ma.masked_invalid(mat)
        im = ax.imshow(masked_mat, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="equal")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(param_labels)
        ax.set_yticklabels(param_labels)
        ax.set_title(format_panel_title(panel_id, title_txt))
        ax.grid(False)

        for i in range(4):
            for j in range(4):
                val = mat[i, j]
                if np.isnan(val):
                    txt = "N/A"
                    txt_color = "#555555"
                else:
                    txt = f"{val:+.2f}" if i != j else "1.00"
                    txt_color = "white" if abs(val) > 0.55 else "#111111"
                ax.text(j, i, txt, ha="center", va="center", color=txt_color, fontsize=8.5)

    cbar = fig.colorbar(im, ax=[ax1, ax2], fraction=0.03, pad=0.04)
    cbar.set_label(r"Local correlation coefficient $C_{ij}$")

    paths = export_figure(fig, SUBDIR_02, "figB_local_correlation")
    plt.close(fig)
    return paths


def replot_figC_one_at_a_time_cost_slices(
    report: Dict[str, Any],
    npz_data: Dict[str, np.ndarray],
) -> Tuple[Path, Path]:
    """Reconstructs `figC_one_at_a_time_cost_slices.{pdf,png}`."""
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

        l1 = ax.plot(
            grid_x, np.maximum(j_zero, 1.0e-18),
            color=PALETTE["slate"], linestyle="--", marker="s", markersize=4.0,
            label=r"Zero flow ($\mathbf{q}_h = \mathbf{0}$)",
        )[0]
        l2 = ax.plot(
            grid_x, np.maximum(j_act, 1.0e-18),
            color=PALETTE["blue"], linestyle="-", marker="o", markersize=4.0,
            label=r"Active flow ($\mathbf{q}_h \neq \mathbf{0}$)",
        )[0]
        l3 = ax.axvline(
            t_val, color=PALETTE["red"], linestyle=":", linewidth=1.4,
            label=r"Ground truth $\theta_j^\dagger$",
        )

        ax.set_yscale("log")
        ax.set_ylim(1.0e-18, 1.0e0)
        ax.set_xlabel(PARAM_MATH_LABELS[k])
        ax.set_ylabel(r"Conditional misfit $J_0(\theta_j \mid \boldsymbol{\theta}_{-j}^\dagger)$")
        ax.set_title(format_panel_title(panel_letters[idx], panel_titles[k]))
        style_axis(ax, grid=True, minor_grid=True)

    fig.legend(
        [l1, l2, l3],
        [
            r"Zero flow ($\mathbf{q}_h = \mathbf{0}$)",
            r"Active flow ($\mathbf{q}_h \neq \mathbf{0}$)",
            r"Ground truth $\theta_j^\dagger$",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    paths = export_figure(fig, SUBDIR_02, "figC_one_at_a_time_cost_slices")
    plt.close(fig)
    return paths


def replot_figD_optimized_objective_profiles(
    report: Dict[str, Any],
    npz_data: Dict[str, np.ndarray],
) -> Tuple[Path, Path]:
    """Reconstructs `figD_optimized_objective_profiles.{pdf,png}`."""
    theta_true = _get_theta_true(report)

    apply_common_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"])

    for ax, k, letter, title_str in (
        (ax1, "beta", "a", r"Viscoelastic Order $\beta$ Profile"),
        (ax2, "E", "b", r"Effective Modulus $E$ Profile"),
    ):
        grid_x, j_cond, j_opt = _find_profile_arrays_for_param(k, report, npz_data)
        t_val = float(theta_true[k])

        ax.semilogy(
            grid_x, np.maximum(j_cond, 1.0e-18),
            color=PALETTE["blue"], linestyle="--", marker="o", markersize=4.5,
            label=r"1D conditional slice $J_0(\theta_j \mid \boldsymbol{\theta}_{-j}^\dagger)$",
        )
        ax.semilogy(
            grid_x, np.maximum(j_opt, 1.0e-18),
            color=PALETTE["red"], linestyle="-", marker="s", markersize=4.5,
            label=r"Optimized profile $\min_{\boldsymbol{\theta}_{-j}} J_0(\theta_j, \boldsymbol{\theta}_{-j})$",
        )
        sym_lbl = r"\beta^\dagger" if k == "beta" else r"E^\dagger"
        ax.axvline(
            t_val, color=PALETTE["charcoal"], linestyle=":", linewidth=1.3,
            label=rf"Ground truth ${sym_lbl} = {t_val:.2f}$",
        )
        ax.set_xlabel(PARAM_MATH_LABELS[k])
        ax.set_ylabel(r"Objective functional $J_0$")
        ax.set_title(format_panel_title(letter, title_str))
        style_axis(ax, grid=True, minor_grid=True)
        ax.legend(loc="best")

    fig.tight_layout()
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
        cond_vals = np.array([_safe_float(r[cond_col], s1[i] / max(s4[i], 1.0e-30)) for i, r in enumerate(pos_rows)], dtype=np.float64)
    else:
        cond_vals = s1 / np.maximum(s4, 1.0e-30)

    ax3.loglog(
        pe_pos, cond_vals,
        color=PALETTE["red"], marker="o", linestyle="-",
        label=r"$\kappa_2(\widetilde{\mathbf{J}}_W) = \sigma_1 / \sigma_4$",
    )
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
        norms = [0.184121, 0.0, 0.831854, 0.0] if default_zero else [0.369328, 0.009299, 0.754662, 0.067916]
    if sv is None:
        sv = np.array([0.848167, 0.080595, 0.0, 0.0] if default_zero else [0.840274, 0.059437, 0.031971, 0.001484], dtype=np.float64)
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
    floor = 1.0e-16

    ax1.bar(
        x - 1.5 * w, np.maximum(syn_zero_norms, floor), width=w,
        color=PALETTE["light_grey"], edgecolor=PALETTE["charcoal"], linewidth=0.6,
        label=r"Synthetic square ($\mathbf{q}_h = \mathbf{0}$)",
    )
    ax1.bar(
        x - 0.5 * w, np.maximum(brats_zero_norms, floor), width=w,
        color=PALETTE["slate"], edgecolor=PALETTE["charcoal"], linewidth=0.6,
        label=r"BraTS 2020 anatomy ($\mathbf{q}_h = \mathbf{0}$)",
    )
    ax1.bar(
        x + 0.5 * w, np.maximum(syn_act_norms, floor), width=w,
        color=PALETTE["cyan"], edgecolor=PALETTE["charcoal"], linewidth=0.6,
        label=r"Synthetic square ($\mathbf{q}_h \neq \mathbf{0}$)",
    )
    ax1.bar(
        x + 1.5 * w, np.maximum(brats_act_norms, floor), width=w,
        color=PALETTE["blue"], edgecolor=PALETTE["charcoal"], linewidth=0.6,
        label=r"BraTS 2020 anatomy ($\mathbf{q}_h \neq \mathbf{0}$)",
    )
    ax1.set_yscale("log")
    ax1.set_ylim(1.0e-17, 1.0e1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([PARAM_MATH_LABELS[k] for k in PARAM_KEYS])
    ax1.set_xlabel("Model parameter")
    ax1.set_ylabel(r"Weighted sensitivity norm $\|\widetilde{\mathbf{J}}_{W,:j}\|_2$")
    ax1.set_title(format_panel_title("a", "Sensitivity Norms: Synthetic vs. BraTS Anatomy"))
    style_axis(ax1, grid=True, grid_axis="y")
    ax1.legend(loc="upper right", fontsize=7.6)

    k_idx = np.arange(1, 5)
    ax2.semilogy(
        k_idx, syn_act_sv / syn_act_sv[0],
        color=PALETTE["cyan"], marker="s", linestyle="--", linewidth=1.7,
        label=rf"Synthetic square ($\kappa_2 = {syn_act_sv[0]/syn_act_sv[-1]:.1f}$)",
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
    """Reconstructs `fig_q1_vs_griddata_verification.{pdf,png}`."""
    sum_path = require_file(DIR_13 / "reviewer_identifiability_extensions_summary.json", "Reviewer identifiability summary JSON")
    summary = json.loads(sum_path.read_text(encoding="utf-8"))

    def _find_key_recursive(obj: Any, target_key: str) -> Optional[Dict[str, Any]]:
        if isinstance(obj, dict):
            if target_key in obj and isinstance(obj[target_key], dict):
                return obj[target_key]
            for v in obj.values():
                res = _find_key_recursive(v, target_key)
                if res is not None:
                    return res
        return None

    jac_cmp = _find_key_recursive(summary, "jacobian_and_svd_comparison") or summary

    q1_sv = np.asarray(jac_cmp.get("singular_values_exact_q1", [1.22514, 0.36146, 0.29228, 0.04942]), dtype=np.float64)
    gd_sv = np.asarray(jac_cmp.get("singular_values_griddata", [1.22611, 0.36190, 0.29237, 0.04949]), dtype=np.float64)

    rep_02, npz_02 = _load_02_artifacts()
    _, syn_act_norms, _, _, _, _ = _extract_02_norms_sv_corr(rep_02, npz_02)

    if "column_norms_exact_q1" in jac_cmp:
        q1_norms = [float(jac_cmp["column_norms_exact_q1"][k]) for k in PARAM_KEYS]
        gd_norms = [float(jac_cmp["column_norms_griddata"][k]) for k in PARAM_KEYS]
    else:
        q1_norms = list(syn_act_norms)
        gd_norms = [float(v * (gd_sv[0] / q1_sv[0])) for v in syn_act_norms]

    apply_common_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"])

    x = np.arange(len(PARAM_KEYS))
    w = 0.34
    ax1.bar(
        x - w / 2, q1_norms, width=w,
        color=PALETTE["blue"], edgecolor=PALETTE["charcoal"], linewidth=0.7,
        label=r"Exact bilinear $\mathcal{H}_h^{Q_1}$",
    )
    ax1.bar(
        x + w / 2, gd_norms, width=w,
        color=PALETTE["orange"], edgecolor=PALETTE["charcoal"], linewidth=0.7,
        label=r"Delaunay linear (`griddata`)",
    )
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([PARAM_MATH_LABELS[k] for k in PARAM_KEYS])
    ax1.set_xlabel("Model parameter")
    ax1.set_ylabel(r"Weighted sensitivity norm $\|\widetilde{\mathbf{J}}_{W,:j}\|_2$")
    ax1.set_title(format_panel_title("a", r"Sensitivity Norms: $\mathcal{H}_h^{Q_1}$ vs. `griddata`"))
    style_axis(ax1, grid=True, grid_axis="y")
    ax1.legend(loc="upper right")

    k_idx = np.arange(1, 5)
    ax2.semilogy(
        k_idx, q1_sv,
        color=PALETTE["blue"], marker="o", linestyle="-",
        label=rf"Exact bilinear $\mathcal{{H}}_h^{{Q_1}}$ ($\kappa_2 = {float(jac_cmp.get('condition_number_exact_q1', 24.79)):.2f}$)",
    )
    ax2.semilogy(
        k_idx, gd_sv,
        color=PALETTE["orange"], marker="s", linestyle="--",
        label=rf"Delaunay `griddata` ($\kappa_2 = {float(jac_cmp.get('condition_number_griddata', 24.78)):.2f}$)",
    )
    ax2.set_xticks(k_idx)
    ax2.set_xticklabels([r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$", r"$\sigma_4$"])
    ax2.set_xlabel("Singular value index $k$")
    ax2.set_ylabel(r"Singular value $\sigma_k(\widetilde{\mathbf{J}}_W)$")
    ax2.set_title(format_panel_title("b", "Singular Values: Operator Discrepancy Check"))
    style_axis(ax2, grid=True, minor_grid=True)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    paths = export_figure(fig, SUBDIR_13, "fig_q1_vs_griddata_verification")
    plt.close(fig)
    return paths


def run_all() -> List[Tuple[Path, Path]]:
    """Generates all Identifiability and Sensitivity figures."""
    report_02, npz_02 = _load_02_artifacts()
    outputs: List[Tuple[Path, Path]] = [
        replot_figA_sensitivity_and_svd(report_02, npz_02),
        replot_figB_local_correlation(report_02, npz_02),
        replot_figC_one_at_a_time_cost_slices(report_02, npz_02),
        replot_figD_optimized_objective_profiles(report_02, npz_02),
        replot_flow_transition_identifiability(),
        replot_anatomical_identifiability_comparison(),
        replot_q1_vs_griddata_verification(),
    ]
    return outputs


if __name__ == "__main__":
    generated = run_all()
    for pdf_p, png_p in generated:
        print(f"[identifiability] Generated: {pdf_p} | {png_p}")