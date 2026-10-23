"""
replot/replot_optimizer_benchmarks.py
Reconstructs all figures in Family F (CPSO Budget Diagnostics, Multi-Seed Variability,
Equal-Budget SOTA Optimizer Benchmark, and Noisy Regularization/Optimizer Controls):
1. `06_cpso_budget_diagnostic/fig_cpso_budget_120_vs_600`
2. `07_cpso_multiseed/fig_cpso_multiseed_convergence`
3. `12_optimizer_benchmark/fig_optimizer_benchmark_cost_and_errors`
4. `12_optimizer_benchmark/fig_optimizer_benchmark_convergence`
5. `14_reviewer_inverse_controls/fig_noisy_controls_lambda_and_optimizer`
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
    OPTIMIZER_STYLE,
    PALETTE,
    PARAM_KEYS,
    PARAM_MATH_LABELS,
    PARAM_STYLE,
    RUN_COLORS,
    RUN_LINESTYLES,
    apply_common_style,
    export_figure,
    format_panel_title,
    require_file,
    style_axis,
)

THETA_TRUE = {"alpha": 0.70, "beta": 0.45, "D": 0.08, "E": 2.50}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    require_file(path, f"Benchmark CSV {path.name}")
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pick_col(row: Dict[str, str], candidates: Sequence[str], fallback_substrings: Sequence[str]) -> str:
    for c in candidates:
        if c in row:
            return c
    for col in row.keys():
        cl = col.lower()
        if all(sub in cl for sub in fallback_substrings):
            return col
    raise KeyError(f"[replot] Could not find column matching {candidates} in {list(row.keys())}")


def _get_optimizer_name(row: Dict[str, str]) -> str:
    """Normalizes the optimizer label ('CPSO', 'DE', 'Powell') from any column or `run_id`."""
    for c in ("optimizer", "method", "algorithm", "optimizer_name", "solver", "run_id", "case_id"):
        if c in row and row[c]:
            val = str(row[c]).strip()
            vl = val.lower()
            if "cpso" in vl:
                return "CPSO"
            if vl == "de" or "differential" in vl or "_de_" in vl or vl.startswith("de_"):
                return "DE"
            if "powell" in vl:
                return "Powell"
    for val in row.values():
        vl = str(val).lower()
        if "cpso" in vl:
            return "CPSO"
        if "powell" in vl:
            return "Powell"
        if "differential" in vl or "_de_" in vl:
            return "DE"
    return ""

def _extract_param_error_pct(row: Dict[str, str], k: str, suffix: str = "") -> float:
    """Extracts relative parameter error in percent [%] for parameter `k` (supporting scalar or vector columns)."""
    import json as _json

    pct_candidates = [
        f"err_rel_{k}_pct{suffix}",
        f"err_{k}_pct{suffix}",
        f"err_pct_{k}{suffix}",
        f"rel_err_{k}_pct{suffix}",
        f"rel_error_{k}_pct{suffix}",
    ]
    for c in pct_candidates:
        if c in row and row[c] != "":
            return float(row[c])

    frac_candidates = [
        f"rel_error_{k}{suffix}",
        f"rel_err_{k}{suffix}",
        f"err_rel_{k}{suffix}",
        f"err_{k}{suffix}",
    ]
    for c in frac_candidates:
        if c in row and row[c] != "":
            return 100.0 * float(row[c])

    for hat_c in (f"{k}_hat{suffix}", f"theta_hat_{k}{suffix}", f"{k}_hat", f"theta_hat_{k}", f"best_{k}{suffix}", f"best_{k}"):
        if hat_c in row and row[hat_c] != "":
            return 100.0 * abs(float(row[hat_c]) - THETA_TRUE[k]) / abs(THETA_TRUE[k])

    # Support vector columns like 'individual_relative_errors_pct_vector' or 'theta_hat_vector'
    p_idx = PARAM_KEYS.index(k)
    if "individual_relative_errors_pct_vector" in row and row["individual_relative_errors_pct_vector"]:
        vec = _json.loads(row["individual_relative_errors_pct_vector"])
        return float(vec[p_idx])

    if "theta_hat_vector" in row and row["theta_hat_vector"]:
        vec = _json.loads(row["theta_hat_vector"])
        return 100.0 * abs(float(vec[p_idx]) - THETA_TRUE[k]) / abs(THETA_TRUE[k])

    raise KeyError(f"[replot] Could not find error or estimate column for parameter '{k}' in {list(row.keys())}")
def replot_06_cpso_budget_diagnostic() -> Tuple[Path, Path]:
    """Reconstructs `06_cpso_budget_diagnostic/fig_cpso_budget_120_vs_600.{pdf,png}`."""
    dir_06 = ARTIFACTS_DIR / "06_cpso_budget_diagnostic"
    conv_rows = _read_csv(dir_06 / "cpso_600_convergence_history.csv")
    mile_rows = _read_csv(dir_06 / "cpso_600_budget_milestones.csv")

    ev_col = _pick_col(conv_rows[0], ("evaluation_index", "eval_index", "evaluation", "fem_evaluation"), ("eval",))
    jt_col = _pick_col(
        conv_rows[0],
        ("best_j_total_so_far", "best_total_so_far", "best_j_so_far", "best_j_total", "j_total"),
        ("best",),
    )
    jd_col = _pick_col(
        conv_rows[0],
        ("best_j_data_so_far", "best_data_so_far", "best_j_data", "j_data"),
        ("dat",),
    )
    jr_col = _pick_col(
        conv_rows[0],
        ("best_j_reg_so_far", "best_reg_so_far", "best_j_reg", "j_reg"),
        ("reg",),
    )

    evals = np.array([int(float(r[ev_col])) for r in conv_rows], dtype=np.int64)
    j_tot = np.minimum.accumulate(np.array([float(r[jt_col]) for r in conv_rows], dtype=np.float64))
    j_dat = np.array([float(r[jd_col]) for r in conv_rows], dtype=np.float64)
    j_reg = np.array([float(r[jr_col]) for r in conv_rows], dtype=np.float64)

    apply_common_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"])

    ax1.semilogy(evals, j_tot, color=PALETTE["charcoal"], linestyle="-", linewidth=1.8, label=r"Best $J_{\mathrm{total}}$")
    ax1.semilogy(evals, j_dat, color=PALETTE["blue"], linestyle="--", linewidth=1.5, label=r"Corresponding $J_{\mathrm{data}}$")
    ax1.semilogy(evals, np.maximum(j_reg, 1.0e-10), color=PALETTE["orange"], linestyle=":", linewidth=1.5, label=r"Corresponding $J_{\mathrm{reg}}$")
    ax1.axvline(120, color=PALETTE["red"], linestyle="-.", linewidth=1.2, label="120-evaluation cutoff")
    ax1.set_xlabel("Forward FEM evaluations")
    ax1.set_ylabel("Objective functional value")
    ax1.set_title(format_panel_title("a", "CPSO Objective Decay (120 vs. 600 Evaluations)"))
    style_axis(ax1, grid=True, minor_grid=True)
    ax1.legend(loc="upper right", fontsize=7.8)

    m_col = _pick_col(
        mile_rows[0],
        ("fem_evaluation_milestone", "milestone_budget", "milestone_eval", "budget_milestone"),
        ("milestone",),
    )
    m_evals = np.array([int(float(r[m_col])) for r in mile_rows], dtype=np.int64)
    for k in PARAM_KEYS:
        st = PARAM_STYLE[k]
        errs = np.array([_extract_param_error_pct(r, k) for r in mile_rows], dtype=np.float64)
        ax2.plot(
            m_evals, errs,
            color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
            linewidth=1.7, label=st["label"],
        )
    ax2.axvline(120, color=PALETTE["red"], linestyle="-.", linewidth=1.2, label="120-eval mark")
    ax2.set_xlabel("Evaluation budget milestone")
    ax2.set_ylabel("Relative parameter error [%]")
    ax2.set_title(format_panel_title("b", "Parameter Error Across Budget Milestones"))
    style_axis(ax2, grid=True)
    ax2.legend(loc="upper right", ncol=2)

    fig.tight_layout()
    paths = export_figure(fig, "06_cpso_budget_diagnostic", "fig_cpso_budget_120_vs_600")
    plt.close(fig)
    return paths

def replot_07_cpso_multiseed() -> Tuple[Path, Path]:
    """Reconstructs `07_cpso_multiseed/fig_cpso_multiseed_convergence.{pdf,png}`."""
    dir_07 = ARTIFACTS_DIR / "07_cpso_multiseed"
    conv_rows = _read_csv(dir_07 / "cpso_multiseed_convergence_history.csv")
    comp_rows = _read_csv(dir_07 / "cpso_multiseed_comparison.csv")

    apply_common_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"])

    seed_key = _pick_col(conv_rows[0], ("seed", "cpso_seed", "run_id"), ("seed",))
    ev_col = _pick_col(conv_rows[0], ("evaluation_index", "eval_index", "evaluation", "fem_evaluation"), ("eval",))
    jt_col = _pick_col(
        conv_rows[0],
        ("best_j_total_so_far", "best_total_so_far", "best_j_so_far", "best_j_total", "j_total"),
        ("best",),
    )

    seeds: List[str] = []
    for r in conv_rows:
        if r[seed_key] not in seeds:
            seeds.append(r[seed_key])

    for idx, s_val in enumerate(seeds):
        s_sub = [r for r in conv_rows if r[seed_key] == s_val]
        evals = np.array([int(float(r[ev_col])) for r in s_sub], dtype=np.int64)
        j_best = np.minimum.accumulate(np.array([float(r[jt_col]) for r in s_sub], dtype=np.float64))
        ax1.semilogy(
            evals, j_best,
            color=RUN_COLORS[idx % len(RUN_COLORS)],
            linestyle=RUN_LINESTYLES[idx % len(RUN_LINESTYLES)],
            linewidth=1.7,
            label=f"Seed {idx + 1} ({s_val})",
        )

    ax1.axvline(120, color=PALETTE["slate"], linestyle=":", linewidth=1.2, label="120-evaluation mark")
    ax1.set_xlabel("Forward FEM evaluations")
    ax1.set_ylabel(r"Best objective value $J_{\mathrm{total}}$")
    ax1.set_title(format_panel_title("a", "Multi-Seed CPSO Convergence (600 Evaluations)"))
    style_axis(ax1, grid=True, minor_grid=True)
    ax1.legend(loc="upper right")

    b_col = next((c for c in comp_rows[0].keys() if "budget" in c.lower()), None)
    rows_600 = (
        [r for r in comp_rows if "600" in str(r[b_col])]
        if b_col is not None and any("600" in str(r[b_col]) for r in comp_rows)
        else comp_rows
    )

    x = np.arange(len(PARAM_KEYS))
    w = 0.25
    for idx, r in enumerate(rows_600[:3]):
        errs = [
            _extract_param_error_pct(r, k, "_600")
            if any(k in col and "600" in col for col in r.keys())
            else _extract_param_error_pct(r, k)
            for k in PARAM_KEYS
        ]
        s_id = r.get("seed", r.get("cpso_seed", f"Seed {idx + 1}"))
        ax2.bar(
            x + (idx - 1) * w, errs, width=w,
            color=RUN_COLORS[idx % len(RUN_COLORS)],
            edgecolor=PALETTE["charcoal"], linewidth=0.7,
            label=f"Seed {idx + 1} ({s_id})",
        )

    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels([PARAM_MATH_LABELS[k] for k in PARAM_KEYS])
    ax2.set_xlabel("Model parameter")
    ax2.set_ylabel("Relative recovery error at 600 evaluations [%] (log scale)")
    ax2.set_title(format_panel_title("b", "Multi-Seed Parameter Recovery Dispersion"))
    style_axis(ax2, grid=True, grid_axis="y", minor_grid=True)
    ax2.legend(loc="upper left")

    fig.tight_layout()
    paths = export_figure(fig, "07_cpso_multiseed", "fig_cpso_multiseed_convergence")
    plt.close(fig)
    return paths


def replot_12_optimizer_benchmark() -> List[Tuple[Path, Path]]:
    """
    Reconstructs the two SOTA optimizer benchmark figures in `12_optimizer_benchmark`:
    1. `fig_optimizer_benchmark_cost_and_errors.{pdf,png}`
    2. `fig_optimizer_benchmark_convergence.{pdf,png}`
    """
    dir_12 = ARTIFACTS_DIR / "12_optimizer_benchmark"
    runs_rows = _read_csv(dir_12 / "optimizer_benchmark_runs.csv")
    conv_rows = _read_csv(dir_12 / "optimizer_benchmark_convergence_histories.csv")

    optimizers = ["CPSO", "DE", "Powell"]
    outputs: List[Tuple[Path, Path]] = []
    apply_common_style()

    jt_run_col = _pick_col(runs_rows[0], ("j_total", "best_j_total", "best_j_so_far", "best_cost_J"), ("j",))

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"])

    x_opt = np.arange(len(optimizers))
    for o_i, opt_name in enumerate(optimizers):
        o_rows = [r for r in runs_rows if _get_optimizer_name(r) == opt_name]
        j_tots = np.array([float(r[jt_run_col]) for r in o_rows], dtype=np.float64)
        st = OPTIMIZER_STYLE[opt_name]
        offs = np.linspace(-0.08, 0.08, len(j_tots))
        ax1.scatter(
            np.full_like(j_tots, x_opt[o_i]) + offs, j_tots,
            color=st["color"], marker=st["marker"], s=45, edgecolor=PALETTE["charcoal"],
            zorder=4, label=st["label"],
        )
        mean_j = float(np.mean(j_tots))
        ax1.hlines(mean_j, x_opt[o_i] - 0.18, x_opt[o_i] + 0.18, colors=st["color"], linestyles="-", linewidth=2.0)

    ax1.set_yscale("log")
    ax1.set_xticks(x_opt)
    ax1.set_xticklabels(optimizers)
    ax1.set_xlabel("Optimization algorithm (600 FEM evaluations, $n=3$ seeds)")
    ax1.set_ylabel(r"Final objective value $J_{\mathrm{total}}$")
    ax1.set_title(format_panel_title("a", r"Final Objective Cost $J_{\mathrm{total}}$ by Optimizer"))
    style_axis(ax1, grid=True, minor_grid=True)
    ax1.legend(loc="upper left", fontsize=8.0)

    x_param = np.arange(len(PARAM_KEYS))
    w = 0.25
    for o_i, opt_name in enumerate(optimizers):
        o_rows = [r for r in runs_rows if _get_optimizer_name(r) == opt_name]
        st = OPTIMIZER_STYLE[opt_name]
        means = [float(np.mean([_extract_param_error_pct(r, k) for r in o_rows])) for k in PARAM_KEYS]
        mins = [float(np.min([_extract_param_error_pct(r, k) for r in o_rows])) for k in PARAM_KEYS]
        maxs = [float(np.max([_extract_param_error_pct(r, k) for r in o_rows])) for k in PARAM_KEYS]
        pos = x_param + (o_i - 1) * w
        ax2.bar(
            pos, means, width=w, color=st["color"], edgecolor=PALETTE["charcoal"],
            linewidth=0.7, alpha=0.88, label=st["label"],
        )
        yerr_low = np.maximum(np.array(means) - np.array(mins), 0.0)
        yerr_high = np.maximum(np.array(maxs) - np.array(means), 0.0)
        ax2.errorbar(
            pos, means, yerr=[yerr_low, yerr_high],
            fmt="none", ecolor=PALETTE["charcoal"], elinewidth=1.0, capsize=2.8, zorder=5,
        )

    ax2.set_yscale("log")
    ax2.set_xticks(x_param)
    ax2.set_xticklabels([PARAM_MATH_LABELS[k] for k in PARAM_KEYS])
    ax2.set_xlabel("Model parameter")
    ax2.set_ylabel("Relative parameter error [%] (mean and range, log scale)")
    ax2.set_title(format_panel_title("b", "Parameter Recovery Accuracy Across Optimizers"))
    style_axis(ax2, grid=True, grid_axis="y", minor_grid=True)
    ax2.legend(loc="upper left", fontsize=8.0)

    fig1.tight_layout()
    outputs.append(export_figure(fig1, "12_optimizer_benchmark", "fig_optimizer_benchmark_cost_and_errors"))
    plt.close(fig1)

    fig2, ax = plt.subplots(figsize=FIG_SIZES["single_wide"])
    id_run_col = _pick_col(runs_rows[0], ("run_id", "case_id"), ("id",))
    id_conv_col = _pick_col(conv_rows[0], ("run_id", "case_id"), ("id",))
    ev_col = _pick_col(conv_rows[0], ("eval_index", "evaluation_index", "evaluation", "fem_evaluation"), ("eval",))
    jt_col = _pick_col(
        conv_rows[0],
        ("best_j_so_far", "best_j_total_so_far", "best_total_so_far", "best_j_total", "j_total"),
        ("best",),
    )

    for o_i, opt_name in enumerate(optimizers):
        st = OPTIMIZER_STYLE[opt_name]
        o_runs = [r for r in runs_rows if _get_optimizer_name(r) == opt_name]
        for s_i, r in enumerate(o_runs):
            rid = r[id_run_col]
            c_sub = [row for row in conv_rows if row[id_conv_col] == rid]
            if not c_sub:
                continue
            evals = np.array([int(float(row[ev_col])) for row in c_sub], dtype=np.int64)
            best_j = np.minimum.accumulate(np.array([float(row[jt_col]) for row in c_sub], dtype=np.float64))
            seed_val = r.get("optimizer_seed", r.get("seed", ""))
            ax.semilogy(
                evals, best_j,
                color=st["color"],
                linestyle=RUN_LINESTYLES[s_i % len(RUN_LINESTYLES)],
                linewidth=1.55,
                alpha=0.90,
                label=f"{st['label']} (3 seeds, e.g. {seed_val})" if s_i == 0 else None,
            )

    ax.set_xlim(1, 600)
    ax.set_xlabel("Forward FEM evaluations (strictly capped at 600)")
    ax.set_ylabel(r"Best objective value $J_{\mathrm{total}}$")
    ax.set_title("Equal-Budget Optimizer Benchmark Convergence ($n = 3$ Seeds per Algorithm)")
    style_axis(ax, grid=True, minor_grid=True)
    ax.legend(loc="upper right")

    fig2.tight_layout()
    outputs.append(export_figure(fig2, "12_optimizer_benchmark", "fig_optimizer_benchmark_convergence"))
    plt.close(fig2)

    return outputs


def replot_14_noisy_controls() -> Tuple[Path, Path]:
    """Reconstructs `14_reviewer_inverse_controls/fig_noisy_controls_lambda_and_optimizer.{pdf,png}`."""
    dir_14 = ARTIFACTS_DIR / "14_reviewer_inverse_controls"
    runs_rows = _read_csv(dir_14 / "reviewer_inverse_controls_runs.csv")
    runs_11 = _read_csv(ARTIFACTS_DIR / "11_final_twin_robustness" / "final_twin_robustness_runs.csv")

    eta_col_14 = next((c for c in runs_rows[0].keys() if "eta" in c.lower()), "noise_eta")
    eta_col_11 = next((c for c in runs_11[0].keys() if "eta" in c.lower()), "noise_eta")

    apply_common_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"])

    cpso_eta05 = [r for r in runs_11 if np.isclose(float(r[eta_col_11]), 0.05)]
    powell_eta05 = [
        r for r in runs_rows
        if _get_optimizer_name(r) == "Powell" and np.isclose(float(r[eta_col_14]), 0.05)
    ]

    x_param = np.arange(len(PARAM_KEYS))
    w = 0.34
    for idx_g, (grp_rows, lbl, col) in enumerate([
        (cpso_eta05, r"CPSO ($\eta=5\%, \lambda_B=7.2$)", OPTIMIZER_STYLE["CPSO"]["color"]),
        (powell_eta05, r"Powell ($\eta=5\%, \lambda_B=7.2$)", OPTIMIZER_STYLE["Powell"]["color"]),
    ]):
        means = [float(np.mean([_extract_param_error_pct(r, k) for r in grp_rows])) for k in PARAM_KEYS]
        sds = [float(np.std([_extract_param_error_pct(r, k) for r in grp_rows], ddof=1)) for k in PARAM_KEYS]
        pos = x_param + (idx_g - 0.5) * w
        ax1.bar(pos, means, width=w, color=col, edgecolor=PALETTE["charcoal"], linewidth=0.7, alpha=0.88, label=lbl)
        ax1.errorbar(pos, means, yerr=sds, fmt="none", ecolor=PALETTE["charcoal"], elinewidth=1.0, capsize=3.0)

    ax1.set_xticks(x_param)
    ax1.set_xticklabels([PARAM_MATH_LABELS[k] for k in PARAM_KEYS])
    ax1.set_xlabel(r"Model parameter ($\eta = 5\%$, $n = 5$ realizations)")
    ax1.set_ylabel(r"Relative parameter error [%] (mean $\pm$ SD)")
    ax1.set_title(format_panel_title("a", r"Moderate Noise ($\eta = 5\%$): CPSO vs. Powell"))
    style_axis(ax1, grid=True, grid_axis="y")
    ax1.legend(loc="upper left", fontsize=7.8)

    cpso_eta10_lam7p2 = [r for r in runs_11 if np.isclose(float(r[eta_col_11]), 0.10)]
    cpso_eta10_lam1p8 = [
        r for r in runs_rows
        if _get_optimizer_name(r) == "CPSO" and np.isclose(float(r[eta_col_14]), 0.10)
    ]
    powell_eta10_lam1p8 = [
        r for r in runs_rows
        if _get_optimizer_name(r) == "Powell" and np.isclose(float(r[eta_col_14]), 0.10)
    ]

    w3 = 0.25
    for idx_g, (grp_rows, lbl, col) in enumerate([
        (cpso_eta10_lam7p2, r"CPSO ($\lambda_B = 7.2$)", PALETTE["cyan"]),
        (cpso_eta10_lam1p8, r"CPSO ($\lambda_B = 1.8$ eq.)", OPTIMIZER_STYLE["CPSO"]["color"]),
        (powell_eta10_lam1p8, r"Powell ($\lambda_B = 1.8$ eq.)", OPTIMIZER_STYLE["Powell"]["color"]),
    ]):
        means = [float(np.mean([_extract_param_error_pct(r, k) for r in grp_rows])) for k in PARAM_KEYS]
        sds = [float(np.std([_extract_param_error_pct(r, k) for r in grp_rows], ddof=1)) for k in PARAM_KEYS]
        pos = x_param + (idx_g - 1) * w3
        ax2.bar(pos, means, width=w3, color=col, edgecolor=PALETTE["charcoal"], linewidth=0.7, alpha=0.88, label=lbl)
        ax2.errorbar(pos, means, yerr=sds, fmt="none", ecolor=PALETTE["charcoal"], elinewidth=1.0, capsize=2.8)

    ax2.set_xticks(x_param)
    ax2.set_xticklabels([PARAM_MATH_LABELS[k] for k in PARAM_KEYS])
    ax2.set_xlabel(r"Model parameter ($\eta = 10\%$, $n = 5$ realizations)")
    ax2.set_ylabel(r"Relative parameter error [%] (mean $\pm$ SD)")
    ax2.set_title(format_panel_title("b", r"High Noise ($\eta = 10\%$): $\lambda_B$ Scaling & Optimizer Control"))
    style_axis(ax2, grid=True, grid_axis="y")
    ax2.legend(loc="upper left", fontsize=7.6)

    fig.tight_layout()
    paths = export_figure(fig, "14_reviewer_inverse_controls", "fig_noisy_controls_lambda_and_optimizer")
    plt.close(fig)
    return paths


def run_all() -> List[Tuple[Path, Path]]:
    """Generates all CPSO diagnostic and Optimizer Benchmark figures."""
    outputs: List[Tuple[Path, Path]] = [
        replot_06_cpso_budget_diagnostic(),
        replot_07_cpso_multiseed(),
    ]
    outputs.extend(replot_12_optimizer_benchmark())
    outputs.append(replot_14_noisy_controls())
    return outputs


if __name__ == "__main__":
    generated = run_all()
    for pdf_p, png_p in generated:
        print(f"[optimizer_benchmarks] Generated: {pdf_p} | {png_p}")