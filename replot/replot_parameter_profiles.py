"""
replot/replot_parameter_profiles.py
Reconstructs all figures in Family D (Parameter Landscapes, 1D Optimized Profiles,
and Mesh-Resolution Sensitivity of the beta profile):
1. `08_beta_landscape/fig_beta_conditional_landscape`: 1D conditional scans along beta around truth and CPSO optima.
2. `09_beta_profile/fig_beta_optimized_profiles`: 1D nuisance-optimized profile hat{J}(beta) and compensation trajectories.
3. `10_beta_mesh_sensitivity/fig_beta_mesh_sensitivity`: Comparison of hat{J}_data(beta) across N=16 and N=24 meshes.
"""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np

from common_style import (
    ARTIFACTS_DIR,
    FIG_SIZES,
    PALETTE,
    RUN_COLORS,
    RUN_LINESTYLES,
    apply_common_style,
    export_figure,
    format_panel_title,
    require_file,
    style_axis,
)

DIR_08 = ARTIFACTS_DIR / "08_beta_landscape"
DIR_09 = ARTIFACTS_DIR / "09_beta_profile"
DIR_10 = ARTIFACTS_DIR / "10_beta_mesh_sensitivity"

THETA_TRUE = {"alpha": 0.70, "beta": 0.45, "D": 0.08, "E": 2.50}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    require_file(path, f"Profile/landscape CSV {path.name}")
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _is_float_str(val: str) -> bool:
    try:
        float(val)
        return val.strip().lower() not in ("true", "false", "")
    except (ValueError, TypeError):
        return False


def _pick_numeric_column(row: Dict[str, str], exact_names: Sequence[str], substring_hint: str) -> str:
    """Picks a strictly numeric float column matching `exact_names` first, then `substring_hint`."""
    for name in exact_names:
        if name in row and _is_float_str(row[name]):
            return name
    for col, val in row.items():
        if substring_hint in col.lower() and _is_float_str(val):
            return col
    raise KeyError(f"[replot] Could not find numeric column for '{substring_hint}' in {list(row.keys())}")


def replot_beta_conditional_landscape() -> Tuple[Path, Path]:
    """
    Reconstructs `08_beta_landscape/fig_beta_conditional_landscape.{pdf,png}`
    directly from `beta_conditional_scans.csv`.
    """
    csv_path = require_file(DIR_08 / "beta_conditional_scans.csv", "08_beta_landscape conditional scans CSV")
    rows = _read_csv(csv_path)

    # Pick string grouping column (non-numeric identifier of the scan/section)
    group_col = next(
        (
            c for c in rows[0].keys()
            if any(w in c.lower() for w in ("section", "scan", "slice", "case", "seed", "ref", "group", "type", "label", "name"))
            and not _is_float_str(rows[0][c])
            and rows[0][c].strip().lower() not in ("true", "false")
        ),
        next(c for c in rows[0].keys() if not _is_float_str(rows[0][c])),
    )

    b_col = _pick_numeric_column(rows[0], ("beta", "beta_val", "beta_grid", "beta_fixed"), "beta")
    jt_col = _pick_numeric_column(rows[0], ("j_total", "J_total", "cost_total"), "total")
    jd_col = _pick_numeric_column(rows[0], ("j_data", "J_data", "cost_data"), "data")
    jr_col = _pick_numeric_column(rows[0], ("j_reg", "J_reg", "cost_reg"), "reg")

    groups: List[str] = []
    for r in rows:
        g = r[group_col]
        if g not in groups:
            groups.append(g)

    truth_groups = [g for g in groups if any(w in g.lower() for w in ("truth", "true", "dagger", "exact"))]
    seed_groups = [g for g in groups if g not in truth_groups]
    if not truth_groups and groups:
        truth_groups = [groups[0]]
        seed_groups = groups[1:]

    apply_common_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"])

    # Panel (a): 1D conditional scan around ground truth
    t_rows = [r for r in rows if r[group_col] == truth_groups[0]]
    b_true = np.array([float(r[b_col]) for r in t_rows], dtype=np.float64)
    jt_true = np.array([float(r[jt_col]) for r in t_rows], dtype=np.float64)
    jd_true = np.array([float(r[jd_col]) for r in t_rows], dtype=np.float64)
    jr_true = np.array([float(r[jr_col]) for r in t_rows], dtype=np.float64)

    ax1.plot(
        b_true, jt_true,
        color=PALETTE["charcoal"], linestyle="-", marker="o", markersize=4.0,
        label=r"$J_{\mathrm{total}}(\beta)$",
    )
    ax1.plot(
        b_true, jd_true,
        color=PALETTE["blue"], linestyle="--", marker="s", markersize=3.6,
        label=r"$J_{\mathrm{data}}(\beta)$",
    )
    ax1.plot(
        b_true, jr_true,
        color=PALETTE["orange"], linestyle=":",
        label=r"$J_{\mathrm{reg}}(\beta)$ ($\lambda = 10^{-4}$)",
    )
    objective_values = np.concatenate((jt_true, jd_true, jr_true))
    if np.all(objective_values > 0.0):
        ax1.set_yscale("log")
    else:
        ax1.set_yscale("symlog", linthresh=1.0e-12)
    min_idx = int(np.argmin(jt_true))
    ax1.plot(
        b_true[min_idx], jt_true[min_idx],
        marker="*", markersize=9.0, color=PALETTE["green"], linestyle="none",
        label=rf"Grid minimum ($\beta = {b_true[min_idx]:.2f}$)",
    )
    ax1.axvline(
        THETA_TRUE["beta"], color=PALETTE["red"], linestyle=":", linewidth=1.3,
        label=rf"Ground truth $\beta^\dagger = {THETA_TRUE['beta']:.2f}$",
    )
    ax1.set_xlabel(r"Viscoelastic order $\beta$")
    ax1.set_ylabel(r"Objective functional $J(\alpha^\dagger, \beta, D^\dagger, E^\dagger)$")
    ax1.set_title(format_panel_title("a", r"1D Slice Along $\beta$ at $(\alpha^\dagger, D^\dagger, E^\dagger)$"))
    style_axis(ax1, grid=True, minor_grid=True)
    ax1.legend(loc="best", fontsize=7.8)

    # Panel (b): 1D conditional scans around CPSO multi-seed optima
    for idx, sg in enumerate(seed_groups):
        s_rows = [r for r in rows if r[group_col] == sg]
        b_grid = np.array([float(r[b_col]) for r in s_rows], dtype=np.float64)
        j_tot = np.array([float(r[jt_col]) for r in s_rows], dtype=np.float64)
        color = RUN_COLORS[idx % len(RUN_COLORS)]
        ls = RUN_LINESTYLES[idx % len(RUN_LINESTYLES)]
        ax2.semilogy(
            b_grid, j_tot,
            color=color, linestyle=ls, linewidth=1.6,
            label=f"CPSO Seed {idx + 1} conditional slice",
        )
        m_i = int(np.argmin(j_tot))
        ax2.plot(b_grid[m_i], j_tot[m_i], marker="o", markersize=5.0, color=color)

    ax2.axvline(
        THETA_TRUE["beta"], color=PALETTE["red"], linestyle=":", linewidth=1.3,
        label=rf"Ground truth $\beta^\dagger = {THETA_TRUE['beta']:.2f}$",
    )
    ax2.set_xlabel(r"Viscoelastic order $\beta$")
    ax2.set_ylabel(r"Conditional objective $J_{\mathrm{total}}(\widehat{\alpha}_s, \beta, \widehat{D}_s, \widehat{E}_s)$")
    ax2.set_title(format_panel_title("b", r"1D Slices Along $\beta$ at CPSO Multi-Seed Optima"))
    style_axis(ax2, grid=True, minor_grid=True)
    ax2.legend(loc="best", fontsize=7.8)

    fig.tight_layout()
    paths = export_figure(fig, "08_beta_landscape", "fig_beta_conditional_landscape")
    plt.close(fig)
    return paths


def replot_beta_optimized_profiles() -> Tuple[Path, Path]:
    """Reconstructs `09_beta_profile/fig_beta_optimized_profiles.{pdf,png}`."""
    data_rows = _read_csv(DIR_09 / "profile_data_results.csv")
    reg_rows = _read_csv(DIR_09 / "profile_regularized_results.csv")
    comp_rows = _read_csv(DIR_09 / "beta_profiles_comparison.csv")

    b_col_d = _pick_numeric_column(data_rows[0], ("beta_fixed", "beta"), "beta")
    j_col_d = _pick_numeric_column(data_rows[0], ("profile_j_data", "j_data_opt", "j_data"), "data")
    beta_d = np.array([float(r[b_col_d]) for r in data_rows], dtype=np.float64)
    j_prof_data = np.array([float(r[j_col_d]) for r in data_rows], dtype=np.float64)
    a_hat_d = np.array([float(r["alpha_hat"]) for r in data_rows], dtype=np.float64)
    D_hat_d = np.array([float(r["D_hat"]) for r in data_rows], dtype=np.float64)
    E_hat_d = np.array([float(r["E_hat"]) for r in data_rows], dtype=np.float64)

    b_col_r = _pick_numeric_column(reg_rows[0], ("beta_fixed", "beta"), "beta")
    j_col_r = _pick_numeric_column(reg_rows[0], ("profile_j_total", "j_total_opt", "j_total"), "total")
    beta_r = np.array([float(r[b_col_r]) for r in reg_rows], dtype=np.float64)
    j_prof_tot = np.array([float(r[j_col_r]) for r in reg_rows], dtype=np.float64)
    a_hat_r = np.array([float(r["alpha_hat"]) for r in reg_rows], dtype=np.float64)
    D_hat_r = np.array([float(r["D_hat"]) for r in reg_rows], dtype=np.float64)
    E_hat_r = np.array([float(r["E_hat"]) for r in reg_rows], dtype=np.float64)

    # Sort each profile by beta before drawing connected curves.
    order_d = np.argsort(beta_d, kind="stable")
    beta_d, j_prof_data, a_hat_d, D_hat_d, E_hat_d = (x[order_d] for x in (beta_d, j_prof_data, a_hat_d, D_hat_d, E_hat_d))
    order_r = np.argsort(beta_r, kind="stable")
    beta_r, j_prof_tot, a_hat_r, D_hat_r, E_hat_r = (x[order_r] for x in (beta_r, j_prof_tot, a_hat_r, D_hat_r, E_hat_r))

    b_col_c = _pick_numeric_column(comp_rows[0], ("beta_fixed", "beta"), "beta")
    slice_col = next((c for c in comp_rows[0].keys() if ("slice" in c.lower() or "truth" in c.lower()) and _is_float_str(comp_rows[0][c])), None)
    beta_c = np.array([float(r[b_col_c]) for r in comp_rows], dtype=np.float64)
    j_slice_fixed = (
        np.array([float(r[slice_col]) for r in comp_rows], dtype=np.float64)
        if slice_col is not None else None
    )
    order_c = np.argsort(beta_c, kind="stable")
    beta_c = beta_c[order_c]
    if j_slice_fixed is not None:
        j_slice_fixed = j_slice_fixed[order_c]
    order_c = np.argsort(beta_c, kind="stable")
    beta_c = beta_c[order_c]
    if j_slice_fixed is not None:
        j_slice_fixed = j_slice_fixed[order_c]

    apply_common_style()
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZES["quad_grid"])
    ax1, ax2, ax3, ax4 = axes.ravel()

    if j_slice_fixed is not None:
        ax1.semilogy(
            beta_c, j_slice_fixed,
            color=PALETTE["slate"], linestyle=":", marker="^", markersize=4.0,
            label=r"Fixed slice $J_{\mathrm{data}}(\alpha^\dagger, \beta, D^\dagger, E^\dagger)$",
        )
    ax1.semilogy(
        beta_d, j_prof_data,
        color=PALETTE["blue"], linestyle="-", marker="o", markersize=4.5,
        label=r"Profile $\widehat{J}_{\mathrm{data}}(\beta) = \min_{\alpha,D,E} J_{\mathrm{data}}$",
    )
    ax1.semilogy(
        beta_r, j_prof_tot,
        color=PALETTE["red"], linestyle="--", marker="s", markersize=4.5,
        label=r"Profile $\widehat{J}_{\mathrm{total}}(\beta) = \min_{\alpha,D,E} J_{\mathrm{total}}$",
    )
    ax1.axvline(THETA_TRUE["beta"], color=PALETTE["charcoal"], linestyle=":", linewidth=1.2, label=r"$\beta^\dagger = 0.45$")
    ax1.set_xlabel(r"Viscoelastic order $\beta$")
    ax1.set_ylabel("Objective functional value")
    ax1.set_title(format_panel_title("a", r"1D Optimized Profiles Along $\beta$"))
    style_axis(ax1, grid=True, minor_grid=True)
    ax1.legend(loc="best", fontsize=7.6)

    ax2.plot(beta_d, a_hat_d, color=PALETTE["blue"], marker="o", linestyle="-", label=r"Data-only $\widehat{\alpha}(\beta)$")
    ax2.plot(beta_r, a_hat_r, color=PALETTE["red"], marker="s", linestyle="--", label=r"Regularized $\widehat{\alpha}_{\mathrm{reg}}(\beta)$")
    ax2.axhline(THETA_TRUE["alpha"], color=PALETTE["charcoal"], linestyle=":", label=rf"$\alpha^\dagger = {THETA_TRUE['alpha']:.2f}$")
    ax2.axvline(THETA_TRUE["beta"], color=PALETTE["charcoal"], linestyle=":")
    ax2.set_xlabel(r"Viscoelastic order $\beta$")
    ax2.set_ylabel(r"Compensating subdiffusion order $\widehat{\alpha}(\beta)$")
    ax2.set_title(format_panel_title("b", r"Nuisance Trajectory $\widehat{\alpha}(\beta)$"))
    style_axis(ax2, grid=True)
    ax2.legend(loc="best", fontsize=7.8)

    ax3.plot(beta_d, D_hat_d, color=PALETTE["blue"], marker="o", linestyle="-", label=r"Data-only $\widehat{D}(\beta)$")
    ax3.plot(beta_r, D_hat_r, color=PALETTE["red"], marker="s", linestyle="--", label=r"Regularized $\widehat{D}_{\mathrm{reg}}(\beta)$")
    ax3.axhline(THETA_TRUE["D"], color=PALETTE["charcoal"], linestyle=":", label=rf"$D^\dagger = {THETA_TRUE['D']:.2f}$")
    ax3.axvline(THETA_TRUE["beta"], color=PALETTE["charcoal"], linestyle=":")
    ax3.set_xlabel(r"Viscoelastic order $\beta$")
    ax3.set_ylabel(r"Compensating diffusivity $\widehat{D}(\beta)$")
    ax3.set_title(format_panel_title("c", r"Nuisance Trajectory $\widehat{D}(\beta)$"))
    style_axis(ax3, grid=True)
    ax3.legend(loc="best", fontsize=7.8)

    ax4.plot(beta_d, E_hat_d, color=PALETTE["blue"], marker="o", linestyle="-", label=r"Data-only $\widehat{E}(\beta)$")
    ax4.plot(beta_r, E_hat_r, color=PALETTE["red"], marker="s", linestyle="--", label=r"Regularized $\widehat{E}_{\mathrm{reg}}(\beta)$")
    ax4.axhline(THETA_TRUE["E"], color=PALETTE["charcoal"], linestyle=":", label=rf"$E^\dagger = {THETA_TRUE['E']:.2f}$")
    ax4.axvline(THETA_TRUE["beta"], color=PALETTE["charcoal"], linestyle=":")
    ax4.set_xlabel(r"Viscoelastic order $\beta$")
    ax4.set_ylabel(r"Compensating modulus $\widehat{E}(\beta)$")
    ax4.set_title(format_panel_title("d", r"Nuisance Trajectory $\widehat{E}(\beta)$"))
    style_axis(ax4, grid=True)
    ax4.legend(loc="best", fontsize=7.8)

    fig.tight_layout()
    paths = export_figure(fig, "09_beta_profile", "fig_beta_optimized_profiles")
    plt.close(fig)
    return paths


def replot_beta_mesh_sensitivity() -> Tuple[Path, Path]:
    """Reconstructs `10_beta_mesh_sensitivity/fig_beta_mesh_sensitivity.{pdf,png}`."""
    rows_16 = _read_csv(DIR_10 / "profile_data_N16_M80.csv")
    rows_24 = _read_csv(DIR_10 / "profile_data_N24_M120.csv")
    supp_path = DIR_10 / "supplementary_point_verifications.csv"
    supp_rows = _read_csv(supp_path) if supp_path.exists() else []

    b_col16 = _pick_numeric_column(rows_16[0], ("beta_fixed", "beta"), "beta")
    j_col16 = _pick_numeric_column(rows_16[0], ("profile_j_data", "j_data_opt", "j_data"), "data")
    b16 = np.array([float(r[b_col16]) for r in rows_16], dtype=np.float64)
    j16 = np.array([float(r[j_col16]) for r in rows_16], dtype=np.float64)
    a16 = np.array([float(r["alpha_hat"]) for r in rows_16], dtype=np.float64)
    d16 = np.array([float(r["D_hat"]) for r in rows_16], dtype=np.float64)
    e16 = np.array([float(r["E_hat"]) for r in rows_16], dtype=np.float64)

    b_col24 = _pick_numeric_column(rows_24[0], ("beta_fixed", "beta"), "beta")
    j_col24 = _pick_numeric_column(rows_24[0], ("profile_j_data", "j_data_opt", "j_data"), "data")
    b24 = np.array([float(r[b_col24]) for r in rows_24], dtype=np.float64)
    j24 = np.array([float(r[j_col24]) for r in rows_24], dtype=np.float64)
    a24 = np.array([float(r["alpha_hat"]) for r in rows_24], dtype=np.float64)
    d24 = np.array([float(r["D_hat"]) for r in rows_24], dtype=np.float64)
    e24 = np.array([float(r["E_hat"]) for r in rows_24], dtype=np.float64)

    apply_common_style()
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZES["quad_grid"])
    ax1, ax2, ax3, ax4 = axes.ravel()

    ax1.semilogy(
        b16, j16, color=PALETTE["blue"], marker="o", linestyle="-",
        label=r"Inversion grid ($N=16, M=80$)",
    )
    ax1.semilogy(
        b24, j24, color=PALETTE["green"], marker="s", linestyle="--",
        label=r"Refined grid ($N=24, M=120$)",
    )
    if supp_rows:
        b_col_s = _pick_numeric_column(supp_rows[0], ("beta", "beta_fixed"), "beta")
        j_col_s = _pick_numeric_column(supp_rows[0], ("j_data", "profile_j_data"), "data")
        b_supp = np.array([float(r[b_col_s]) for r in supp_rows], dtype=np.float64)
        j_supp = np.array([float(r[j_col_s]) for r in supp_rows], dtype=np.float64)
        ax1.plot(
            b_supp, j_supp,
            color=PALETTE["red"], marker="*", markersize=7.5, linestyle="none",
            label=r"Supplementary point checks",
        )
        objective_profile_values = np.concatenate((j16, j24, j_supp))
    else:
        objective_profile_values = np.concatenate((j16, j24))
    if np.all(objective_profile_values > 0.0):
        ax1.set_yscale("log")
    else:
        ax1.set_yscale("symlog", linthresh=1.0e-12)
    ax1.axvline(THETA_TRUE["beta"], color=PALETTE["charcoal"], linestyle=":", label=r"$\beta^\dagger = 0.45$")
    ax1.set_xlabel(r"Viscoelastic order $\beta$")
    ax1.set_ylabel(r"Optimized data misfit $\widehat{J}_{\mathrm{data}}(\beta)$")
    ax1.set_title(format_panel_title("a", r"Mesh Sensitivity of Profile $\widehat{J}_{\mathrm{data}}(\beta)$"))
    style_axis(ax1, grid=True, minor_grid=True)
    ax1.legend(loc="best", fontsize=7.6)

    ax2.plot(b16, a16, color=PALETTE["blue"], marker="o", linestyle="-", label=r"$N=16, M=80$")
    ax2.plot(b24, a24, color=PALETTE["green"], marker="s", linestyle="--", label=r"$N=24, M=120$")
    ax2.axhline(THETA_TRUE["alpha"], color=PALETTE["charcoal"], linestyle=":", label=rf"$\alpha^\dagger = {THETA_TRUE['alpha']:.2f}$")
    ax2.axvline(THETA_TRUE["beta"], color=PALETTE["charcoal"], linestyle=":")
    ax2.set_xlabel(r"Viscoelastic order $\beta$")
    ax2.set_ylabel(r"Compensating $\widehat{\alpha}(\beta)$")
    ax2.set_title(format_panel_title("b", r"Mesh Sensitivity of $\widehat{\alpha}(\beta)$"))
    style_axis(ax2, grid=True)
    ax2.legend(loc="best", fontsize=7.8)

    ax3.plot(b16, d16, color=PALETTE["blue"], marker="o", linestyle="-", label=r"$N=16, M=80$")
    ax3.plot(b24, d24, color=PALETTE["green"], marker="s", linestyle="--", label=r"$N=24, M=120$")
    ax3.axhline(THETA_TRUE["D"], color=PALETTE["charcoal"], linestyle=":", label=rf"$D^\dagger = {THETA_TRUE['D']:.2f}$")
    ax3.axvline(THETA_TRUE["beta"], color=PALETTE["charcoal"], linestyle=":")
    ax3.set_xlabel(r"Viscoelastic order $\beta$")
    ax3.set_ylabel(r"Compensating $\widehat{D}(\beta)$")
    ax3.set_title(format_panel_title("c", r"Mesh Sensitivity of $\widehat{D}(\beta)$"))
    style_axis(ax3, grid=True)
    ax3.legend(loc="best", fontsize=7.8)

    ax4.plot(b16, e16, color=PALETTE["blue"], marker="o", linestyle="-", label=r"$N=16, M=80$")
    ax4.plot(b24, e24, color=PALETTE["green"], marker="s", linestyle="--", label=r"$N=24, M=120$")
    ax4.axhline(THETA_TRUE["E"], color=PALETTE["charcoal"], linestyle=":", label=rf"$E^\dagger = {THETA_TRUE['E']:.2f}$")
    ax4.axvline(THETA_TRUE["beta"], color=PALETTE["charcoal"], linestyle=":")
    ax4.set_xlabel(r"Viscoelastic order $\beta$")
    ax4.set_ylabel(r"Compensating $\widehat{E}(\beta)$")
    ax4.set_title(format_panel_title("d", r"Mesh Sensitivity of $\widehat{E}(\beta)$"))
    style_axis(ax4, grid=True)
    ax4.legend(loc="best", fontsize=7.8)

    fig.tight_layout()
    paths = export_figure(fig, "10_beta_mesh_sensitivity", "fig_beta_mesh_sensitivity")
    plt.close(fig)
    return paths


def run_all() -> List[Tuple[Path, Path]]:
    """Generates all Parameter Profile and Landscape figures."""
    return [
        replot_beta_conditional_landscape(),
        replot_beta_optimized_profiles(),
        replot_beta_mesh_sensitivity(),
    ]


if __name__ == "__main__":
    generated = run_all()
    for pdf_p, png_p in generated:
        print(f"[parameter_profiles] Generated: {pdf_p} | {png_p}")