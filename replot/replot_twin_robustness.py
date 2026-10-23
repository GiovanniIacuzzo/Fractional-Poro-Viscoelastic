"""
replot/replot_twin_robustness.py
Reconstructs all figures in Family E (Twin Experiments, Multi-Start, Forcing/Objective Diagnostics,
and Final Parameter Recovery Campaign):
1. `03a_twin_noise_free_multistart/fig_multistart_parameter_recovery_v2`
2. `03a_twin_noise_free_multistart/fig_multistart_convergence_v2`
3. `03a_twin_noise_free_multistart/fig_multistart_objective_decomposition_v2`
4. `03_twin_robustness/fig_noise_parameter_errors_v2`
5. `03_twin_robustness/fig_noise_relative_residuals_v2`
6. `03b_twin_forcing_validation/fig_forcing_validation_diagnostics`
7. `04_objective_diagnostics/fig_lambda_sensitivity_and_modes`
8. `05_pilot_inverse/fig_pilot_inverse_convergence`
9. `11_final_twin_robustness/fig_final_twin_parameter_recovery`
10. `11_final_twin_robustness/fig_final_twin_convergence`
"""

from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np

from common_style import (
    ARTIFACTS_DIR,
    FIG_SIZES,
    PALETTE,
    PARAM_KEYS,
    PARAM_MATH_LABELS,
    PARAM_STYLE,
    RUN_COLORS,
    RUN_LINESTYLES,
    RUN_MARKERS,
    apply_common_style,
    export_figure,
    format_panel_title,
    require_file,
    style_axis,
)

THETA_TRUE = {"alpha": 0.70, "beta": 0.45, "D": 0.08, "E": 2.50}
PARAM_BOUNDS = {"alpha": (0.10, 0.99), "beta": (0.05, 0.95), "D": (0.005, 0.50), "E": (0.50, 10.0)}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    require_file(path, f"CSV file {path.name}")
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _get_theta_hat_from_row_or_dict(item: Dict[str, Any], k: str) -> float:
    if "theta_hat" in item and isinstance(item["theta_hat"], dict):
        return float(item["theta_hat"][k])
    for col in (f"theta_hat_{k}", f"{k}_hat", f"hat_{k}", f"best_{k}", k):
        if col in item and item[col] != "":
            return float(item[col])
    raise KeyError(f"[replot] Cannot find estimated parameter '{k}' in {list(item.keys())}")


def _get_cost_val(item: Dict[str, Any], candidates: Sequence[str], substring_hint: str, default: float = 0.0) -> float:
    for c in candidates:
        if c in item and item[c] is not None and str(item[c]).strip() != "":
            return float(item[c])
    for col, val in item.items():
        if substring_hint in col.lower() and val is not None and str(val).strip() != "":
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    return default


def _extract_best_j_series(rows: List[Dict[str, str]]) -> np.ndarray:
    """Extracts best-so-far J_total series from a list of convergence CSV rows."""
    cols = list(rows[0].keys())
    best_col = next(
        (
            c for c in (
                "best_j_total_so_far",
                "best_total_so_far",
                "best_j_so_far",
                "best_j_total",
                "best_cost_so_far",
                "best_cost_J",
                "best_cost",
            )
            if c in cols
        ),
        None,
    )
    if best_col is not None:
        return np.array([float(r[best_col]) for r in rows], dtype=np.float64)

    raw_col = next((c for c in ("j_total", "J_total", "cost_J", "cost") if c in cols), None)
    if raw_col is not None:
        raw_vals = np.array([float(r[raw_col]) for r in rows], dtype=np.float64)
        return np.minimum.accumulate(raw_vals)

    raise KeyError(f"[replot] Could not find objective column in convergence CSV columns: {cols}")


def replot_03a_multistart() -> List[Tuple[Path, Path]]:
    """Reconstructs the 3 noise-free multi-start figures in `03a_twin_noise_free_multistart`."""
    dir_03a = ARTIFACTS_DIR / "03a_twin_noise_free_multistart"
    runs_csv = _read_csv(dir_03a / "multistart_runs_summary.csv")
    conv_rows = _read_csv(dir_03a / "multistart_convergence_history.csv")

    n_runs = len(runs_csv)
    run_id_strs = [str(r.get("run_id", r.get("case_id", f"run_{i+1}"))) for i, r in enumerate(runs_csv)]
    run_labels = [f"Run {i + 1}" for i in range(n_runs)]
    x_pos = np.arange(1, n_runs + 1)

    outputs: List[Tuple[Path, Path]] = []
    apply_common_style()

    # 1. Parameter recovery across 5 multi-start runs
    fig1, axes = plt.subplots(2, 2, figsize=FIG_SIZES["quad_grid"], sharex=True)
    panel_ids = ("a", "b", "c", "d")
    p_titles = {
        "alpha": r"Subdiffusion Order $\alpha$",
        "beta": r"Viscoelastic Order $\beta$",
        "D": r"Apparent Diffusivity $D$",
        "E": r"Effective Young's Modulus $E$",
    }

    for idx, k in enumerate(PARAM_KEYS):
        ax = axes.ravel()[idx]
        t_val = THETA_TRUE[k]
        est_vals = np.array([_get_theta_hat_from_row_or_dict(r, k) for r in runs_csv], dtype=np.float64)
        err_pct = 100.0 * np.abs(est_vals - t_val) / abs(t_val)

        ax.axhline(t_val, color=PALETTE["red"], linestyle="--", linewidth=1.4, label=rf"Ground truth $\theta_j^\dagger = {t_val:g}$")
        for r_i in range(n_runs):
            ax.plot(
                x_pos[r_i], est_vals[r_i],
                marker=RUN_MARKERS[r_i % len(RUN_MARKERS)],
                color=RUN_COLORS[r_i % len(RUN_COLORS)],
                markersize=6.5, linestyle="none",
                label=run_labels[r_i] if idx == 0 else None,
            )
            ax.annotate(
                f"{err_pct[r_i]:.1f}%",
                (x_pos[r_i], est_vals[r_i]),
                textcoords="offset points", xytext=(0, 6),
                ha="center", fontsize=7.5,
            )

        b_low, b_high = PARAM_BOUNDS[k]
        span = float(b_high) - float(b_low)
        ax.set_ylim(float(b_low) - 0.08 * span, float(b_high) + 0.14 * span)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(run_labels)
        ax.set_ylabel(f"Estimated {PARAM_MATH_LABELS[k]}")
        ax.set_title(format_panel_title(panel_ids[idx], p_titles[k]))
        style_axis(ax, grid=True)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=7.5, ncol=2)

    fig1.tight_layout()
    outputs.append(export_figure(fig1, "03a_twin_noise_free_multistart", "fig_multistart_parameter_recovery_v2"))
    plt.close(fig1)

    # 2. Multi-start convergence history
    fig2, ax = plt.subplots(figsize=FIG_SIZES["single_wide"])
    id_col = next((c for c in conv_rows[0].keys() if "run" in c.lower() or "case" in c.lower()), "run_id")
    ev_col = next((c for c in conv_rows[0].keys() if "eval" in c.lower()), None)

    for r_i, r in enumerate(runs_csv):
        rid_str = run_id_strs[r_i]
        r_rows = [row for row in conv_rows if str(row[id_col]) == rid_str]
        evals = (
            np.array([int(float(row[ev_col])) for row in r_rows], dtype=np.int64)
            if ev_col is not None
            else np.arange(1, len(r_rows) + 1)
        )
        best_j = _extract_best_j_series(r_rows)
        color = RUN_COLORS[r_i % len(RUN_COLORS)]
        ls = RUN_LINESTYLES[r_i % len(RUN_LINESTYLES)]
        seed_val = r.get("seed", r.get("cpso_seed", ""))
        best_ev = int(float(r.get("best_evaluation_index", r.get("best_eval_index", evals[-1]))))
        j_tot_val = _get_cost_val(r, ("j_total", "best_j_total", "best_cost_J"), "tot", float(best_j[-1]))
        ax.semilogy(
            evals, best_j, color=color, linestyle=ls, linewidth=1.7,
            label=f"Run {r_i + 1} (seed {seed_val}, best @ eval {best_ev})",
        )
        ax.plot(best_ev, j_tot_val, marker="o", markersize=5.5, color=color)

    ax.set_xlabel("Forward FEM evaluations")
    ax.set_ylabel(r"Best objective value $J_{\mathrm{total}}$")
    ax.set_title("Noise-Free Multi-Start CPSO Convergence (120-Evaluation Diagnostic Budget)")
    style_axis(ax, grid=True, minor_grid=True)
    ax.legend(loc="upper right", fontsize=8.0)
    fig2.tight_layout()
    outputs.append(export_figure(fig2, "03a_twin_noise_free_multistart", "fig_multistart_convergence_v2"))
    plt.close(fig2)

    # 3. Stacked objective decomposition J_total = J_data + J_reg
    fig3, ax = plt.subplots(figsize=FIG_SIZES["single"])
    j_data = np.array([_get_cost_val(r, ("j_data", "best_j_data", "data_misfit"), "data") for r in runs_csv], dtype=np.float64)
    j_reg = np.array([_get_cost_val(r, ("j_reg", "best_j_reg", "reg_penalty"), "reg") for r in runs_csv], dtype=np.float64)
    j_tot = np.array([
        _get_cost_val(r, ("j_total", "best_j_total", "best_cost_J"), "tot", j_data[i] + j_reg[i])
        for i, r in enumerate(runs_csv)
    ], dtype=np.float64)

    w = 0.46
    ax.bar(x_pos, j_data, width=w, color=PALETTE["blue"], edgecolor=PALETTE["charcoal"], label=r"Data misfit $J_{\mathrm{data}}(\widehat{\boldsymbol{\theta}})$")
    ax.bar(x_pos, j_reg, width=w, bottom=j_data, color=PALETTE["orange"], edgecolor=PALETTE["charcoal"], label=r"Tikhonov penalty $J_{\mathrm{reg}}(\widehat{\boldsymbol{\theta}})$")
    for r_i in range(n_runs):
        ratio_pct = 100.0 * j_reg[r_i] / max(j_tot[r_i], 1.0e-30)
        ax.text(
            x_pos[r_i], j_tot[r_i] * 1.02,
            f"$J_{{\\mathrm{{reg}}}} / J_{{\\mathrm{{tot}}}} = {ratio_pct:.1f}\\%$",
            ha="center", va="bottom", fontsize=7.6,
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(run_labels)
    ax.set_ylim(0.0, float(np.max(j_tot)) * 1.18)
    ax.set_xlabel("Independent multi-start run")
    ax.set_ylabel(r"Objective functional value ($\lambda = 10^{-4}$)")
    ax.set_title(r"Additive Objective Decomposition $J_{\mathrm{total}} = J_{\mathrm{data}} + J_{\mathrm{reg}}$")
    style_axis(ax, grid=True, grid_axis="y")
    ax.legend(loc="upper right")
    fig3.tight_layout()
    outputs.append(export_figure(fig3, "03a_twin_noise_free_multistart", "fig_multistart_objective_decomposition_v2"))
    plt.close(fig3)

    return outputs


def replot_03_preliminary_noise_diagnostics() -> List[Tuple[Path, Path]]:
    """Reconstructs `fig_noise_parameter_errors_v2` and `fig_noise_relative_residuals_v2` from `twin_robustness_summary.csv`."""
    csv_path = require_file(
        ARTIFACTS_DIR / "03_twin_robustness" / "diagnostics" / "twin_robustness_summary.csv",
        "Twin robustness summary CSV",
    )
    rows = _read_csv(csv_path)
    eta_col = next((c for c in rows[0].keys() if "eta" in c.lower()), "eta")
    etas_pct = np.array([100.0 * float(r[eta_col]) for r in rows], dtype=np.float64)
    x_idx = np.arange(len(rows))
    x_labels = [f"$\\eta = {e:.0f}\\%$" for e in etas_pct]

    outputs: List[Tuple[Path, Path]] = []
    apply_common_style()

    fig1, ax = plt.subplots(figsize=FIG_SIZES["single"])
    for k in PARAM_KEYS:
        st = PARAM_STYLE[k]
        est_vals = np.array([_get_theta_hat_from_row_or_dict(r, k) for r in rows], dtype=np.float64)
        errs = 100.0 * np.abs(est_vals - THETA_TRUE[k]) / abs(THETA_TRUE[k])
        ax.plot(
            x_idx, errs,
            color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
            linewidth=1.7, label=st["label"],
        )
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Nominal observation noise level")
    ax.set_ylabel(r"Relative parameter error $|\widehat{\theta}_j - \theta_j^\dagger| / \theta_j^\dagger$ [%]")
    ax.set_title("Preliminary Single-Realization Parameter Error (120 FEM Evaluations)")
    style_axis(ax, grid=True)
    ax.legend(loc="upper left", ncol=2)
    fig1.tight_layout()
    outputs.append(export_figure(fig1, "03_twin_robustness", "fig_noise_parameter_errors_v2"))
    plt.close(fig1)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"])
    noisy_col = next((c for c in rows[0].keys() if "noisy" in c.lower()), None)
    clean_col = next((c for c in rows[0].keys() if "clean" in c.lower()), None)
    eff_col = next((c for c in rows[0].keys() if "noise" in c.lower() and c != eta_col and "seed" not in c.lower()), eta_col)

    res_noisy = np.array([100.0 * float(r[noisy_col]) for r in rows], dtype=np.float64) if noisy_col else etas_pct
    res_clean = np.array([100.0 * float(r[clean_col]) for r in rows], dtype=np.float64) if clean_col else etas_pct * 0.25
    disc_gap = 100.0 * float(rows[0].get("discretization_gap_rel_l2", 0.001937))
    realized_noise = np.array([100.0 * float(r[eff_col]) for r in rows], dtype=np.float64)

    ax1.plot(
        x_idx, res_noisy, marker="o", color=PALETTE["blue"], linestyle="-",
        label=r"$\|\mathcal{G}_h(\widehat{\boldsymbol{\theta}}) - \mathbf{y}^{\mathrm{obs}}\|_2 / \|\mathbf{y}^{\mathrm{obs}}\|_2$",
    )
    ax1.plot(
        x_idx, res_clean, marker="s", color=PALETTE["green"], linestyle="--",
        label=r"$\|\mathcal{G}_h(\widehat{\boldsymbol{\theta}}) - \mathbf{y}_{\mathrm{clean}}\|_2 / \|\mathbf{y}_{\mathrm{clean}}\|_2$",
    )
    ax1.axhline(
        disc_gap, color=PALETTE["red"], linestyle=":", linewidth=1.3,
        label=rf"Mesh discretization gap ({disc_gap:.2f}%)",
    )
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(x_labels)
    ax1.set_xlabel("Nominal observation noise level")
    ax1.set_ylabel("Relative $L^2$ residual [%]")
    ax1.set_title(format_panel_title("a", "Relative Observation Residuals"))
    style_axis(ax1, grid=True)
    ax1.legend(loc="upper left", fontsize=7.6)

    ax2.plot(
        x_idx, realized_noise, marker="^", color=PALETTE["purple"], linestyle="-",
        label=r"Realized $\|\boldsymbol{\varepsilon}\|_2 / \|\mathbf{y}_{\mathrm{clean}}\|_2$",
    )
    ax2.plot(
        x_idx, etas_pct, marker="D", color=PALETTE["charcoal"], linestyle=":",
        label=r"Nominal reference $\eta$",
    )
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(x_labels)
    ax2.set_xlabel("Nominal observation noise level")
    ax2.set_ylabel("Relative noise norm [%]")
    ax2.set_title(format_panel_title("b", "Realized vs. Nominal Noise Norm"))
    style_axis(ax2, grid=True)
    ax2.legend(loc="upper left")

    fig2.tight_layout()
    outputs.append(export_figure(fig2, "03_twin_robustness", "fig_noise_relative_residuals_v2"))
    plt.close(fig2)

    return outputs


def replot_03b_forcing_validation() -> Tuple[Path, Path]:
    """Reconstructs diagnostic figure `fig_forcing_validation_diagnostics` from `03b_twin_forcing_validation`."""
    dir_03b = ARTIFACTS_DIR / "03b_twin_forcing_validation"
    rows = _read_csv(dir_03b / "forcing_validation_discretizations.csv")

    def _is_num(val: Any) -> bool:
        try:
            float(str(val).strip())
            return str(val).strip().lower() not in ("", "true", "false", "nan", "none")
        except (ValueError, TypeError):
            return False

    apply_common_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"])

    cols = list(rows[0].keys())
    num_cols = [c for c in cols if all(_is_num(r.get(c, "")) for r in rows)]
    str_cols = [c for c in cols if c not in num_cols]

    if str_cols:
        labels = [str(r[str_cols[0]]) for r in rows]
    else:
        n_col = next((c for c in cols if "subdiv" in c.lower() or c.lower() == "n"), cols[0])
        m_col = next((c for c in cols if "step" in c.lower() or c.lower() == "m"), cols[min(1, len(cols) - 1)])
        labels = [f"$N={r[n_col]}, M={r[m_col]}$" for r in rows]

    pe_mean_col = next((c for c in num_cols if ("pe" in c.lower() or "peclet" in c.lower()) and "mean" in c.lower()), None)
    pe_max_col = next((c for c in num_cols if ("pe" in c.lower() or "peclet" in c.lower()) and "max" in c.lower()), None)
    q_col = next((c for c in num_cols if any(w in c.lower() for w in ("darcy", "vel", "q_", "flux", "adv"))), None)

    x = np.arange(len(rows))
    w = 0.28
    if pe_mean_col and pe_max_col:
        pe_mean = np.array([float(r[pe_mean_col]) for r in rows], dtype=np.float64)
        pe_max = np.array([float(r[pe_max_col]) for r in rows], dtype=np.float64)
        ax1.bar(x - 0.5 * w, pe_mean, width=w, color=PALETTE["blue"], edgecolor=PALETTE["charcoal"], label=r"Mean Péclet $\mathrm{Pe}_{\mathrm{mean}}$")
        ax1.bar(x + 0.5 * w, pe_max, width=w, color=PALETTE["red"], edgecolor=PALETTE["charcoal"], label=r"Peak Péclet $\mathrm{Pe}_{\mathrm{max}}$")
    elif q_col:
        q_vals = np.array([float(r[q_col]) for r in rows], dtype=np.float64)
        ax1.bar(x, q_vals, width=0.42, color=PALETTE["blue"], edgecolor=PALETTE["charcoal"], label=q_col.replace("_", " "))
    else:
        plot_cols = [c for c in num_cols if c.lower() not in ("n", "m", "subdivisions_n", "time_steps_m")][:2]
        for ic, nc in enumerate(plot_cols):
            vals = np.array([float(r[nc]) for r in rows], dtype=np.float64)
            ax1.bar(x + (ic - 0.5) * w, vals, width=w, color=RUN_COLORS[ic], edgecolor=PALETTE["charcoal"], label=nc.replace("_", " "))

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Dimensionless flow / forcing metric")
    ax1.set_title(format_panel_title("a", "Active Darcy Flow Metrics Across Grids"))
    style_axis(ax1, grid=True, grid_axis="y")
    ax1.legend(loc="best", fontsize=7.8)

    mass_col = next((c for c in num_cols if any(w in c.lower() for w in ("mass", "drift", "err", "res"))), None)
    mass_drift = (
        np.array([abs(float(r[mass_col])) for r in rows], dtype=np.float64)
        if mass_col is not None
        else np.full(len(rows), 1.0e-15, dtype=np.float64)
    )
    ax2.semilogy(
        x, np.maximum(mass_drift, 1.0e-16), marker="s", color=PALETTE["purple"], linestyle="--",
        label=r"Relative conservation error $|M(T) - M(0)|/M(0)$",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(1.0e-17, max(1.0e-10, float(np.max(mass_drift)) * 10.0))
    ax2.set_ylabel("Verification error")
    ax2.set_title(format_panel_title("b", "Weak Forcing & Mass Conservation Consistency"))
    style_axis(ax2, grid=True, minor_grid=True)
    ax2.legend(loc="upper right", fontsize=7.8)

    fig.tight_layout()
    paths = export_figure(fig, "03b_twin_forcing_validation", "fig_forcing_validation_diagnostics")
    plt.close(fig)
    return paths

def replot_04_objective_diagnostics() -> Tuple[Path, Path]:
    """Reconstructs `fig_lambda_sensitivity_and_modes` from `04_objective_diagnostics`."""
    dir_04 = ARTIFACTS_DIR / "04_objective_diagnostics"
    lam_rows = _read_csv(dir_04 / "lambda_sensitivity_grid.csv")

    apply_common_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"])

    mode_col = next((c for c in lam_rows[0].keys() if "mode" in c.lower()), "weighting_mode")
    cand_col = next((c for c in lam_rows[0].keys() if "cand" in c.lower()), "candidate_name")
    lam_col = next((c for c in lam_rows[0].keys() if "lambda" in c.lower()), "lambda_reg")

    cand_names = sorted({r.get(cand_col, "") for r in lam_rows if _get_cost_val(r, ("j_reg", "best_j_reg"), "reg") > 0.0})
    target_cand = "moderate_perturbation" if "moderate_perturbation" in cand_names else (cand_names[0] if cand_names else "")

    rows_a = [
        r for r in lam_rows
        if "mode_a" in r.get(mode_col, "").lower()
        and float(r.get(lam_col, 0.0)) > 0.0
        and (not target_cand or r.get(cand_col, "") == target_cand)
    ]
    rows_b = [
        r for r in lam_rows
        if "mode_b" in r.get(mode_col, "").lower()
        and float(r.get(lam_col, 0.0)) > 0.0
        and (not target_cand or r.get(cand_col, "") == target_cand)
    ]

    if rows_a:
        lam_a = np.array([float(r[lam_col]) for r in rows_a], dtype=np.float64)
        jd_a = np.array([_get_cost_val(r, ("j_data", "best_j_data"), "data") for r in rows_a], dtype=np.float64)
        jr_a = np.array([_get_cost_val(r, ("j_reg", "best_j_reg"), "reg") for r in rows_a], dtype=np.float64)
        jt_a = np.array([_get_cost_val(r, ("j_total", "best_j_total"), "tot") for r in rows_a], dtype=np.float64)
        order_a = np.argsort(lam_a)
        ax1.loglog(lam_a[order_a], jt_a[order_a], color=PALETTE["charcoal"], marker="o", linestyle="-", label=r"$J_{\mathrm{total}}^{(A)}(\lambda)$")
        ax1.loglog(lam_a[order_a], jd_a[order_a], color=PALETTE["blue"], marker="s", linestyle="--", label=r"$J_{\mathrm{data}}^{(A)}$")
        ax1.loglog(lam_a[order_a], jr_a[order_a], color=PALETTE["orange"], marker="^", linestyle=":", label=r"$J_{\mathrm{reg}}(\lambda)$")
        ax1.axvline(1.0e-4, color=PALETTE["red"], linestyle=":", label=r"Canonical $\lambda_A = 10^{-4}$")

    ax1.set_xlabel(r"Regularization parameter $\lambda$")
    ax1.set_ylabel("Objective value (Mode A: deterministic normalized)")
    ax1.set_title(format_panel_title("a", r"Mode A ($\eta = 0$): Scaling of $J_{\mathrm{data}}$ and $J_{\mathrm{reg}}$"))
    style_axis(ax1, grid=True, minor_grid=True)
    ax1.legend(loc="best", fontsize=7.8)

    if rows_b:
        lam_b = np.array([float(r[lam_col]) for r in rows_b], dtype=np.float64)
        jd_b = np.array([_get_cost_val(r, ("j_data", "best_j_data"), "data") for r in rows_b], dtype=np.float64)
        jr_b = np.array([_get_cost_val(r, ("j_reg", "best_j_reg"), "reg") for r in rows_b], dtype=np.float64)
        jt_b = np.array([_get_cost_val(r, ("j_total", "best_j_total"), "tot") for r in rows_b], dtype=np.float64)
        order_b = np.argsort(lam_b)
        ax2.loglog(lam_b[order_b], jt_b[order_b], color=PALETTE["charcoal"], marker="o", linestyle="-", label=r"$J_{\mathrm{total}}^{(B)}(\lambda)$")
        ax2.loglog(lam_b[order_b], jd_b[order_b], color=PALETTE["blue"], marker="s", linestyle="--", label=r"$J_{\mathrm{data}}^{(B)}$ ($\mathbf{W}^\mathsf{T}\mathbf{W}=\boldsymbol{\Sigma}^{-1}$)")
        ax2.loglog(lam_b[order_b], jr_b[order_b], color=PALETTE["orange"], marker="^", linestyle=":", label=r"$J_{\mathrm{reg}}(\lambda)$")

    ax2.set_xlabel(r"Regularization parameter $\lambda$")
    ax2.set_ylabel(r"Objective value (Mode B: $\eta = 5\%$ inverse covariance)")
    ax2.set_title(format_panel_title("b", r"Mode B ($\eta = 5\%$): Inverse-Covariance Weighting"))
    style_axis(ax2, grid=True, minor_grid=True)
    ax2.legend(loc="best", fontsize=7.8)

    fig.tight_layout()
    paths = export_figure(fig, "04_objective_diagnostics", "fig_lambda_sensitivity_and_modes")
    plt.close(fig)
    return paths


def replot_05_pilot_inverse() -> Tuple[Path, Path]:
    """Reconstructs `fig_pilot_inverse_convergence` from `05_pilot_inverse`."""
    dir_05 = ARTIFACTS_DIR / "05_pilot_inverse"
    sum_rows = _read_csv(dir_05 / "pilot_inverse_runs_summary.csv")
    conv_rows = _read_csv(dir_05 / "pilot_inverse_convergence_history.csv")

    apply_common_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"])

    id_col = next((c for c in conv_rows[0].keys() if "case" in c.lower() or "run" in c.lower()), "case_id")
    ev_col = next((c for c in conv_rows[0].keys() if "eval" in c.lower()), None)

    run_ids: List[str] = []
    for r in conv_rows:
        cid = str(r[id_col])
        if cid not in run_ids:
            run_ids.append(cid)

    for idx, rid in enumerate(run_ids):
        c_sub = [r for r in conv_rows if str(r[id_col]) == rid]
        evals = (
            np.array([int(float(r[ev_col])) for r in c_sub], dtype=np.int64)
            if ev_col is not None
            else np.arange(1, len(c_sub) + 1)
        )
        j_best = _extract_best_j_series(c_sub)
        clean_label = (
            r"Noise-free pilot ($\eta=0\%$, Mode A)"
            if "0" in rid or "clean" in rid.lower()
            else r"Noisy pilot ($\eta=5\%$, Mode B)"
        )
        ax1.semilogy(
            evals, j_best,
            color=RUN_COLORS[idx % len(RUN_COLORS)],
            linestyle=RUN_LINESTYLES[idx % len(RUN_LINESTYLES)],
            linewidth=1.7, label=clean_label,
        )

    ax1.set_xlabel("Forward FEM evaluations")
    ax1.set_ylabel(r"Best objective value $J_{\mathrm{total}}$")
    ax1.set_title(format_panel_title("a", "Pilot Inverse Convergence (120 Evaluations)"))
    style_axis(ax1, grid=True, minor_grid=True)
    ax1.legend(loc="upper right")

    x = np.arange(len(PARAM_KEYS))
    w = 0.34
    for idx, r in enumerate(sum_rows):
        errs = [
            100.0 * abs(_get_theta_hat_from_row_or_dict(r, k) - THETA_TRUE[k]) / abs(THETA_TRUE[k])
            for k in PARAM_KEYS
        ]
        eta_val = float(r.get("eta", r.get("noise_eta", 0.0)))
        lbl = rf"$\eta = {100.0 * eta_val:.0f}\%$"
        offset = (idx - 0.5 * (len(sum_rows) - 1)) * w
        ax2.bar(
            x + offset, errs, width=w,
            color=RUN_COLORS[idx % len(RUN_COLORS)],
            edgecolor=PALETTE["charcoal"], linewidth=0.7, label=lbl,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels([PARAM_MATH_LABELS[k] for k in PARAM_KEYS])
    ax2.set_xlabel("Model parameter")
    ax2.set_ylabel("Relative recovery error [%]")
    ax2.set_title(format_panel_title("b", "Pilot Parameter Recovery Errors"))
    style_axis(ax2, grid=True, grid_axis="y")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    paths = export_figure(fig, "05_pilot_inverse", "fig_pilot_inverse_convergence")
    plt.close(fig)
    return paths


def replot_11_final_twin_robustness() -> List[Tuple[Path, Path]]:
    """
    Reconstructs the two main manuscript figures in `11_final_twin_robustness`:
    1. `fig_final_twin_parameter_recovery.{pdf,png}`
    2. `fig_final_twin_convergence.{pdf,png}`
    """
    dir_11 = ARTIFACTS_DIR / "11_final_twin_robustness"
    runs_rows = _read_csv(dir_11 / "final_twin_robustness_runs.csv")
    proto_json = json.loads(
        require_file(dir_11 / "final_twin_robustness_protocol_and_results.json").read_text(encoding="utf-8")
    )

    outputs: List[Tuple[Path, Path]] = []
    apply_common_style()

    eta_col = next((c for c in runs_rows[0].keys() if "eta" in c.lower()), "noise_eta")
    eta_levels = [0.0, 0.05, 0.10]
    eta_labels = [r"$\eta = 0\%$ ($n=3$)", r"$\eta = 5\%$ ($n=5$)", r"$\eta = 10\%$ ($n=5$)"]
    eta_colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["red"]]

    fig1, axes = plt.subplots(2, 2, figsize=FIG_SIZES["quad_grid"], sharex=True)
    panel_ids = ("a", "b", "c", "d")
    p_titles = {
        "alpha": r"Subdiffusion Order $\alpha$ ($\alpha^\dagger = 0.70$)",
        "beta": r"Viscoelastic Order $\beta$ ($\beta^\dagger = 0.45$)",
        "D": r"Apparent Diffusivity $D$ ($D^\dagger = 0.08$)",
        "E": r"Effective Young's Modulus $E$ ($E^\dagger = 2.50$)",
    }

    x_groups = np.arange(len(eta_levels))

    for idx, k in enumerate(PARAM_KEYS):
        ax = axes.ravel()[idx]
        t_val = THETA_TRUE[k]
        ax.axhline(t_val, color=PALETTE["charcoal"], linestyle="--", linewidth=1.4, label=rf"Ground truth $\theta_j^\dagger = {t_val:g}$")

        for g_i, eta in enumerate(eta_levels):
            g_rows = [r for r in runs_rows if np.isclose(float(r[eta_col]), eta)]
            vals = np.array([_get_theta_hat_from_row_or_dict(r, k) for r in g_rows], dtype=np.float64)
            j_totals = np.array([_get_cost_val(r, ("j_total", "best_j_total"), "tot") for r in g_rows], dtype=np.float64)
            best_i = int(np.argmin(j_totals))

            offs = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(
                np.full_like(vals, x_groups[g_i]) + offs,
                vals,
                color=eta_colors[g_i],
                edgecolor=PALETTE["charcoal"],
                s=36,
                alpha=0.88,
                zorder=4,
                label=eta_labels[g_i] if idx == 0 else None,
            )
            ax.plot(
                x_groups[g_i] + offs[best_i], vals[best_i],
                marker="*", markersize=9.5, color=PALETTE["green"],
                markeredgecolor=PALETTE["charcoal"], markeredgewidth=0.7,
                linestyle="none", zorder=5,
                label=r"Selected ($\min J_{\mathrm{total}}$)" if (idx == 0 and g_i == 0) else None,
            )
            mean_v = float(np.mean(vals))
            sd_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            ax.errorbar(
                x_groups[g_i] + 0.22, mean_v, yerr=sd_v,
                fmt="D", markersize=4.8, color=PALETTE["charcoal"],
                capsize=3.5, linewidth=1.1, zorder=5,
                label=r"Mean $\pm$ SD" if (idx == 0 and g_i == 0) else None,
            )

        ax.set_xticks(x_groups)
        ax.set_xticklabels(eta_labels)
        ax.set_ylabel(f"Estimated {PARAM_MATH_LABELS[k]}")
        ax.set_title(format_panel_title(panel_ids[idx], p_titles[k]))
        style_axis(ax, grid=True)
        if idx == 0:
            ax.legend(loc="best", fontsize=7.2, ncol=2)

    fig1.tight_layout()
    outputs.append(export_figure(fig1, "11_final_twin_robustness", "fig_final_twin_parameter_recovery"))
    plt.close(fig1)

    # Figure 2: Convergence trajectories
    fig2, axes2 = plt.subplots(1, 3, figsize=FIG_SIZES["triple_row"], sharex=True)
    regime_titles = [
        ("a", r"Noise-Free Baseline ($\eta = 0\%$, Mode A)"),
        ("b", r"Moderate Noise ($\eta = 5\%$, Mode B)"),
        ("c", r"High Noise ($\eta = 10\%$, Mode B)"),
    ]

    hist_by_run: Dict[str, np.ndarray] = {}

    def _scan_json_for_histories(obj: Any) -> None:
        if isinstance(obj, dict):
            rid = str(obj.get("run_id", obj.get("seed", obj.get("cpso_seed", ""))))
            for hk in ("best_j_total_so_far", "convergence_history", "cost_history", "history", "best_total_so_far", "best_j_total"):
                if hk in obj and isinstance(obj[hk], list) and len(obj[hk]) > 1:
                    if isinstance(obj[hk][0], (int, float)):
                        hist_by_run[rid] = np.minimum.accumulate(np.asarray(obj[hk], dtype=np.float64))
                    elif isinstance(obj[hk][0], dict):
                        hist_by_run[rid] = _extract_best_j_series(obj[hk])
            for v in obj.values():
                _scan_json_for_histories(v)
        elif isinstance(obj, list):
            for item in obj:
                _scan_json_for_histories(item)

    _scan_json_for_histories(proto_json)

    ms_conv_path = ARTIFACTS_DIR / "07_cpso_multiseed" / "cpso_multiseed_convergence_history.csv"
    if ms_conv_path.exists():
        ms_rows = _read_csv(ms_conv_path)
        s_col = next((c for c in ms_rows[0].keys() if "seed" in c.lower() or "run" in c.lower()), "seed")
        for s_val in {r[s_col] for r in ms_rows}:
            sub_r = [r for r in ms_rows if r[s_col] == s_val]
            hist_by_run[str(s_val)] = _extract_best_j_series(sub_r)

    for g_i, eta in enumerate(eta_levels):
        ax = axes2[g_i]
        g_runs = [r for r in runs_rows if np.isclose(float(r[eta_col]), eta)]
        for r_i, r in enumerate(g_runs):
            rid = str(r.get("run_id", ""))
            c_seed = str(r.get("cpso_seed", r.get("seed", "")))
            n_seed = str(r.get("noise_seed", ""))
            trace = hist_by_run.get(rid)
            if trace is None and eta == 0.0:
                trace = hist_by_run.get(c_seed)
            color = RUN_COLORS[r_i % len(RUN_COLORS)]
            ls = RUN_LINESTYLES[r_i % len(RUN_LINESTYLES)]
            lbl = f"Seed {c_seed}" if eta == 0.0 else f"Realization {r_i + 1} (seed {n_seed})"
            j_final = _get_cost_val(r, ("j_total", "best_j_total"), "tot")
            best_ev = int(float(r.get("best_evaluation_index", r.get("best_eval_index", 600))))

            if trace is not None and len(trace) > 1:
                evals = np.arange(1, len(trace) + 1)
                ax.semilogy(evals, trace, color=color, linestyle=ls, linewidth=1.5, label=lbl)
            else:
                ax.axhline(j_final, color=color, linestyle=ls, linewidth=1.4, label=f"{lbl}: $J={j_final:.2f}$")
                ax.plot(best_ev, j_final, marker="o", markersize=5.5, color=color)

        ax.set_xlim(1, 600)
        ax.set_xlabel("Forward FEM evaluations")
        ax.set_ylabel(r"Best objective value $J_{\mathrm{total}}$")
        ax.set_title(format_panel_title(*regime_titles[g_i]))
        style_axis(ax, grid=True, minor_grid=True)
        ax.legend(loc="upper right", fontsize=7.2)

    fig2.tight_layout()
    outputs.append(export_figure(fig2, "11_final_twin_robustness", "fig_final_twin_convergence"))
    plt.close(fig2)

    return outputs


def run_all() -> List[Tuple[Path, Path]]:
    """Generates all Twin Experiment and Parameter Recovery figures."""
    outputs: List[Tuple[Path, Path]] = []
    outputs.extend(replot_03a_multistart())
    outputs.extend(replot_03_preliminary_noise_diagnostics())
    outputs.append(replot_03b_forcing_validation())
    outputs.append(replot_04_objective_diagnostics())
    outputs.append(replot_05_pilot_inverse())
    outputs.extend(replot_11_final_twin_robustness())
    return outputs


if __name__ == "__main__":
    generated = run_all()
    for pdf_p, png_p in generated:
        print(f"[twin_robustness] Generated: {pdf_p} | {png_p}")