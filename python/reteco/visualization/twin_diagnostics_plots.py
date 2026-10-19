"""
python/reteco/visualization/twin_diagnostics_plots.py
Publication-grade plotting utilities and standalone artifact loaders for the
RETECO v1 synthetic twin experiments (03a noise-free multistart and 03 noise study).

Visual style matches `verification_plots.py`, `identifiability_plots.py`, and
`benchmark_plots.py`:
- English labels and titles throughout;
- Sober typography (no bold titles, compact legends, subtle dotted grids);
- Print- and colorblind-friendly palette with distinct markers;
- Dual export to high-resolution PNG (300 dpi) and vector PDF.
"""

from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PARAM_KEYS: Tuple[str, str, str, str] = ("alpha", "beta", "D", "E")

PARAM_PANEL_TITLES: Tuple[str, str, str, str] = (
    r"(a) Fractional Order $\alpha$",
    r"(b) Viscoelastic Order $\beta$",
    r"(c) Apparent Diffusivity $D$",
    r"(d) Effective Modulus $E$",
)

PARAM_YLABELS: Tuple[str, str, str, str] = (
    r"Parameter $\alpha$",
    r"Parameter $\beta$",
    r"Parameter $D$",
    r"Parameter $E$",
)

PARAM_LEGEND_LABELS: Tuple[str, str, str, str] = (
    r"$\alpha$",
    r"$\beta$",
    r"$D$",
    r"$E$",
)

# Palette and markers consistent with existing repository figures
RUN_COLORS: Tuple[str, ...] = (
    "#1f77b4",  # muted navy blue
    "#d62728",  # brick red
    "#2ca02c",  # forest green
    "#ff7f0e",  # ochre orange
    "#9467bd",  # muted purple
)
RUN_LINESTYLES: Tuple[str, ...] = ("-", "--", "-.", ":", "-")
PARAM_COLORS: Tuple[str, str, str, str] = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
)
PARAM_MARKERS: Tuple[str, str, str, str] = ("o", "s", "^", "D")


def _Apply_paper_rc_style() -> Dict[str, Any]:
    """Returns matplotlib rcParams dictionary aligned with existing repository plots."""
    return {
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "normal",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.titlesize": 11.0,
        "figure.titleweight": "normal",
        "axes.linewidth": 0.8,
        "grid.linestyle": ":",
        "grid.linewidth": 0.65,
        "grid.alpha": 0.45,
        "savefig.dpi": 300,
    }


def save_figure_png_and_pdf(
    fig: plt.Figure,
    base_or_png_path: str | Path,
    dpi: int = 300,
) -> Tuple[Path, Path]:
    """
    Exports `fig` to both PNG (300 dpi) and vector PDF.
    If `base_or_png_path` ends with `.png` or `.pdf`, the companion extension is
    generated automatically in the same directory.
    """
    p = Path(base_or_png_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in (".png", ".pdf"):
        stem_path = p.with_suffix("")
    else:
        stem_path = p

    png_path = stem_path.with_suffix(".png")
    pdf_path = stem_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path


def _extract_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_run_dict(rec: Any, idx: int) -> Dict[str, Any]:
    """Normalizes either a SingleRunDiagnosticRecord or a serialized JSON dict."""
    if isinstance(rec, dict):
        theta_true_raw = rec.get("theta_true_vector") or [
            rec["theta_true"][k] for k in PARAM_KEYS
        ]
        theta_hat_raw = rec.get("theta_hat_vector") or [
            rec["theta_hat"][k] for k in PARAM_KEYS
        ]
        err_raw = rec.get("individual_relative_errors_pct_vector") or [
            rec["individual_relative_errors_pct"][k] for k in PARAM_KEYS
        ]
        conv = rec.get("convergence_history", {})
        eval_index = conv.get("eval_index", [])
        best_j_so_far = conv.get("best_j_so_far", [])
        return {
            "run_index": idx + 1,
            "run_label": f"Run {idx + 1}",
            "run_id": str(rec.get("run_id", f"run_{idx + 1:02d}")),
            "optimizer_seed": int(rec.get("optimizer_seed", 0)),
            "data_seed": int(rec.get("data_seed", rec.get("seed", 0))),
            "eta_level": float(rec.get("eta_level", 0.0)),
            "theta_true": np.asarray(theta_true_raw, dtype=np.float64).ravel(),
            "theta_hat": np.asarray(theta_hat_raw, dtype=np.float64).ravel(),
            "errors_pct": np.asarray(err_raw, dtype=np.float64).ravel(),
            "best_cost_J_total": float(
                rec.get("best_cost_J_total", rec.get("best_cost_J", 0.0))
            ),
            "best_cost_J_data": float(rec.get("best_cost_J_data", 0.0)),
            "best_cost_J_reg": float(rec.get("best_cost_J_reg", 0.0)),
            "nominal_noise_std": float(rec.get("nominal_noise_std", 0.0)),
            "effective_relative_noise_norm": float(
                rec.get("effective_relative_noise_norm", 0.0)
            ),
            "relative_residual_vs_noisy_data": float(
                rec.get("relative_residual_vs_noisy_data", 0.0)
            ),
            "relative_residual_vs_clean_target": float(
                rec.get("relative_residual_vs_clean_target", 0.0)
            ),
            "planned_fem_budget": int(rec.get("planned_fem_budget", 120)),
            "optimizer_fem_evaluations": int(
                rec.get("optimizer_fem_evaluations", rec.get("num_fem_evaluations", 120))
            ),
            "eval_index": np.asarray(eval_index, dtype=np.int64),
            "best_j_so_far": np.asarray(best_j_so_far, dtype=np.float64),
        }

    # SingleRunDiagnosticRecord object
    history = getattr(rec, "history", [])
    return {
        "run_index": idx + 1,
        "run_label": f"Run {idx + 1}",
        "run_id": str(rec.run_id),
        "optimizer_seed": int(rec.optimizer_seed),
        "data_seed": int(rec.data_seed),
        "eta_level": float(rec.eta_level),
        "theta_true": np.asarray(rec.theta_true, dtype=np.float64).ravel(),
        "theta_hat": np.asarray(rec.theta_hat, dtype=np.float64).ravel(),
        "errors_pct": np.asarray(rec.individual_relative_errors_pct, dtype=np.float64).ravel(),
        "best_cost_J_total": float(rec.best_cost_J_total),
        "best_cost_J_data": float(rec.best_cost_J_data),
        "best_cost_J_reg": float(rec.best_cost_J_reg),
        "nominal_noise_std": float(rec.nominal_noise_std),
        "effective_relative_noise_norm": float(rec.effective_relative_noise_norm),
        "relative_residual_vs_noisy_data": float(rec.relative_residual_vs_noisy_data),
        "relative_residual_vs_clean_target": float(rec.relative_residual_vs_clean_target),
        "planned_fem_budget": int(rec.planned_fem_budget),
        "optimizer_fem_evaluations": int(rec.optimizer_fem_evaluations),
        "eval_index": np.asarray([r.eval_index for r in history], dtype=np.int64),
        "best_j_so_far": np.asarray([r.best_j_so_far for r in history], dtype=np.float64),
    }


def plot_multistart_parameter_recovery_v2(
    records: Sequence[Any],
    bounds_dict: Optional[Dict[str, Tuple[float, float]]],
    out_path: str | Path,
) -> Tuple[Path, Path]:
    """
    Figure 1 (v2): 4-panel parameter recovery comparison for alpha, beta, D, E.
    - Uses Run 1..Run 5 on the horizontal axis;
    - Single shared legend above the panels;
    - Clean reference line for ground truth without overlapping percentage text or background fills;
    - Balanced vertical limits around the true value and estimates.
    """
    runs = [_normalize_run_dict(r, i) for i, r in enumerate(records)]
    n_runs = len(runs)
    x_idx = np.arange(n_runs)
    run_labels = [r["run_label"] for r in runs]

    with plt.rc_context(_Apply_paper_rc_style()):
        fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.8))
        axes_flat = axes.ravel()

        for p_idx, p_key in enumerate(PARAM_KEYS):
            ax = axes_flat[p_idx]
            true_val = float(runs[0]["theta_true"][p_idx])
            estimates = np.array([float(r["theta_hat"][p_idx]) for r in runs], dtype=np.float64)

            ref_line = ax.axhline(
                true_val,
                color="#b22222",
                linestyle="--",
                linewidth=1.4,
                label=r"Ground truth $\theta^\dagger_j$",
                zorder=2,
            )
            pts = ax.plot(
                x_idx,
                estimates,
                linestyle="none",
                marker="o",
                markersize=6.0,
                color="#1f77b4",
                markeredgecolor="#0b3c68",
                markeredgewidth=0.8,
                label=r"CPSO estimate $\widehat{\theta}_j$",
                zorder=3,
            )[0]

            # Calibrate vertical range around true value and estimates without distortion
            max_dev = float(np.max(np.abs(estimates - true_val)))
            min_half_span = max(0.06 * abs(true_val), 1.0e-6)
            half_span = max(1.35 * max_dev, min_half_span)
            y_low = true_val - half_span
            y_high = true_val + half_span

            if bounds_dict is not None and p_key in bounds_dict:
                b_min, b_max = bounds_dict[p_key]
                y_low = max(y_low, b_min - 0.03 * (b_max - b_min))
                y_high = min(y_high, b_max + 0.03 * (b_max - b_min))
                if y_high <= y_low:
                    y_low, y_high = b_min, b_max

            ax.set_ylim(y_low, y_high)
            ax.set_xlim(-0.45, n_runs - 0.55)
            ax.set_xticks(x_idx)
            ax.set_xticklabels(run_labels)
            ax.set_ylabel(PARAM_YLABELS[p_idx])
            ax.set_title(PARAM_PANEL_TITLES[p_idx])
            ax.grid(True)

            if p_key == "D":
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

        handles = [pts, ref_line]
        labels = [r"CPSO estimate $\widehat{\theta}_j$", r"Ground truth $\theta^\dagger_j$"]
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=2,
            frameon=True,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
        paths = save_figure_png_and_pdf(fig, out_path)
        plt.close(fig)
        return paths


def plot_multistart_convergence_v2(
    records: Sequence[Any],
    out_path: str | Path,
) -> Tuple[Path, Path]:
    """
    Figure 2 (v2): Best objective functional history J_total vs. forward FEM evaluations.
    - Labels simplified to Run 1..Run 5;
    - Logarithmic vertical scale;
    - Visible terminal markers at evaluation 120, uncovered by the legend;
    - Explicit note that all runs exhausted the 120-evaluation budget.
    """
    runs = [_normalize_run_dict(r, i) for i, r in enumerate(records)]

    with plt.rc_context(_Apply_paper_rc_style()):
        fig, ax = plt.subplots(figsize=(6.8, 4.4))

        all_positive = all(
            len(r["best_j_so_far"]) > 0 and np.all(r["best_j_so_far"] > 0.0)
            for r in runs
        )

        max_eval_seen = 120
        for i, r in enumerate(runs):
            evals = r["eval_index"]
            best_j = r["best_j_so_far"]
            if evals.size == 0 or best_j.size == 0:
                continue
            max_eval_seen = max(max_eval_seen, int(evals[-1]))
            color = RUN_COLORS[i % len(RUN_COLORS)]
            ls = RUN_LINESTYLES[i % len(RUN_LINESTYLES)]

            ax.step(
                evals,
                best_j,
                where="post",
                color=color,
                linestyle=ls,
                linewidth=1.6,
                label=r["run_label"],
                zorder=3,
            )
            # Highlight final point at budget exhaustion so it remains clearly visible
            ax.plot(
                evals[-1],
                best_j[-1],
                marker="o",
                markersize=4.5,
                color=color,
                zorder=4,
            )

        if all_positive:
            ax.set_yscale("log")

        ax.set_xlim(1, max_eval_seen + 6)
        ax.set_xlabel("Number of Forward FEM Evaluations")
        ax.set_ylabel(r"Best Objective Functional $J_{\mathrm{total}}(\boldsymbol{\theta})$")
        ax.set_title("Best Objective Functional History Across Independent CPSO Runs")
        ax.grid(True, which="major", ls=":", alpha=0.45)
        ax.grid(True, which="minor", ls=":", alpha=0.20)
        ax.legend(loc="upper right", ncol=1, frameon=True)

        fig.text(
            0.5,
            0.01,
            f"Note: All runs terminated upon exhausting the planned budget of {max_eval_seen} forward FEM evaluations.",
            ha="center",
            va="bottom",
            fontsize=8.0,
            style="italic",
            color="#333333",
        )
        fig.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])
        paths = save_figure_png_and_pdf(fig, out_path)
        plt.close(fig)
        return paths


def plot_multistart_objective_decomposition_v2(
    records: Sequence[Any],
    out_path: str | Path,
) -> Tuple[Path, Path]:
    """
    Figure 3 (v2): Single-panel stacked bar decomposition of J_total = J_data + J_reg
    at the best candidate of each run on a single linear vertical scale.
    """
    runs = [_normalize_run_dict(r, i) for i, r in enumerate(records)]
    n_runs = len(runs)
    x_idx = np.arange(n_runs)
    run_labels = [r["run_label"] for r in runs]

    j_data = np.array([r["best_cost_J_data"] for r in runs], dtype=np.float64)
    j_reg = np.array([r["best_cost_J_reg"] for r in runs], dtype=np.float64)
    j_tot = j_data + j_reg

    with plt.rc_context(_Apply_paper_rc_style()):
        fig, ax = plt.subplots(figsize=(6.8, 4.4))

        width = 0.48
        ax.bar(
            x_idx,
            j_data,
            width=width,
            color="#1f77b4",
            edgecolor="#0b3c68",
            linewidth=0.7,
            label=r"Data fidelity $J_{\mathrm{data}}(\widehat{\boldsymbol{\theta}})$",
            zorder=3,
        )
        ax.bar(
            x_idx,
            j_reg,
            width=width,
            bottom=j_data,
            color="#ff7f0e",
            edgecolor="#8c4600",
            linewidth=0.7,
            label=r"Tikhonov penalty $J_{\mathrm{reg}}(\widehat{\boldsymbol{\theta}})$ ($\lambda = 10^{-4}$)",
            zorder=3,
        )

        y_max = float(np.max(j_tot)) if j_tot.size > 0 else 1.0
        if y_max <= 0.0:
            y_max = 1.0
        ax.set_ylim(0.0, y_max * 1.22)

        for i, tot_val in enumerate(j_tot):
            ax.annotate(
                f"{tot_val:.2e}",
                xy=(x_idx[i], tot_val),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.0,
            )

        ax.set_xticks(x_idx)
        ax.set_xticklabels(run_labels)
        ax.set_ylabel(r"Objective Functional $J_{\mathrm{total}} = J_{\mathrm{data}} + J_{\mathrm{reg}}$")
        ax.set_title(r"Objective Functional Decomposition at Best Candidate ($\eta = 0\%$)")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.grid(True, axis="y")
        ax.legend(loc="upper right", frameon=True)

        # If one component is <3% of total across runs, add a concise quantitative note
        reg_share_pct = 100.0 * np.mean(j_reg / np.maximum(j_tot, 1.0e-30))
        fig.text(
            0.5,
            0.01,
            f"Bar height equals $J_{{\\mathrm{{total}}}}$ (mean regularization share $J_{{\\mathrm{{reg}}}}/J_{{\\mathrm{{total}}}} = {reg_share_pct:.1f}\\%$).",
            ha="center",
            va="bottom",
            fontsize=8.0,
            style="italic",
            color="#333333",
        )
        fig.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])
        paths = save_figure_png_and_pdf(fig, out_path)
        plt.close(fig)
        return paths


def plot_noise_study_parameter_errors_v2(
    records: Sequence[Any],
    out_path: str | Path,
) -> Tuple[Path, Path]:
    """
    Figure 4 (v2): Relative parameter recovery errors [%] across nominal noise levels
    eta in {0%, 1%, 5%, 10%}.
    - Uses grouped discrete markers (no interpolating lines, no statistical error bars);
    - X-axis shows only the noise levels (seeds remain in CSV/report);
    - Discrete caption notes exploratory single-realization protocol.
    """
    runs = [_normalize_run_dict(r, i) for i, r in enumerate(records)]
    n_levels = len(runs)
    x_pos = np.arange(n_levels)
    eta_labels = [f"{r['eta_level'] * 100:.0f}%" for r in runs]
    offsets = (-0.18, -0.06, 0.06, 0.18)

    with plt.rc_context(_Apply_paper_rc_style()):
        fig, ax = plt.subplots(figsize=(6.8, 4.4))

        max_err = 0.0
        for p_idx, p_key in enumerate(PARAM_KEYS):
            errs = np.array([float(r["errors_pct"][p_idx]) for r in runs], dtype=np.float64)
            if errs.size > 0:
                max_err = max(max_err, float(np.max(errs)))
            x_shifted = x_pos + offsets[p_idx]

            ax.vlines(
                x_shifted,
                0.0,
                errs,
                colors=PARAM_COLORS[p_idx],
                linestyles=":",
                linewidth=1.0,
                alpha=0.65,
                zorder=2,
            )
            ax.plot(
                x_shifted,
                errs,
                linestyle="none",
                marker=PARAM_MARKERS[p_idx],
                markersize=6.0,
                color=PARAM_COLORS[p_idx],
                label=PARAM_LEGEND_LABELS[p_idx],
                zorder=3,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(eta_labels)
        ax.set_xlim(-0.45, n_levels - 0.55)
        ax.set_ylim(0.0, max(max_err * 1.15, 5.0))
        ax.set_xlabel(r"Additive Noise Level $\eta$ [%]")
        ax.set_ylabel(r"Relative Parameter Error $|\widehat{\theta}_j - \theta^\dagger_j| / |\theta^\dagger_j|$ [%]")
        ax.set_title("Parameter Recovery Error Across Nominal Noise Levels")
        ax.grid(True)
        ax.legend(title="Parameter", loc="upper left", ncol=2, frameon=True)

        fig.text(
            0.5,
            0.01,
            "Exploratory comparison: single CPSO realization (120 FEM evaluations) per noise level.",
            ha="center",
            va="bottom",
            fontsize=8.0,
            style="italic",
            color="#333333",
        )
        fig.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])
        paths = save_figure_png_and_pdf(fig, out_path)
        plt.close(fig)
        return paths


def plot_noise_study_relative_residuals_v2(
    records: Sequence[Any],
    discretization_rel_gap: float,
    out_path: str | Path,
) -> Tuple[Path, Path]:
    """
    Figure 5 (v2): Two-panel diagnostic separation of residuals and realized noise:
    - Panel (a): Relative L2 residuals vs. observed noisy data and vs. clean target,
      with the global discretization gap (h_tar vs. h_inv) shown as a dashed reference line.
    - Panel (b): Realized relative L2 noise norm ||epsilon||_2 / ||y_clean||_2 across
      nominal noise levels eta.
    """
    runs = [_normalize_run_dict(r, i) for i, r in enumerate(records)]
    n_levels = len(runs)
    x_pos = np.arange(n_levels)
    eta_labels = [f"{r['eta_level'] * 100:.0f}%" for r in runs]

    res_noisy = np.array(
        [r["relative_residual_vs_noisy_data"] for r in runs], dtype=np.float64
    )
    res_clean = np.array(
        [r["relative_residual_vs_clean_target"] for r in runs], dtype=np.float64
    )
    noise_eff = np.array(
        [r["effective_relative_noise_norm"] for r in runs], dtype=np.float64
    )

    with plt.rc_context(_Apply_paper_rc_style()):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

        width = 0.32
        ax1.bar(
            x_pos - width / 2,
            res_noisy,
            width=width,
            color="#ff7f0e",
            edgecolor="#8c4600",
            linewidth=0.7,
            label=r"Vs. noisy data $\mathbf{y}_{\mathrm{noisy}}$",
            zorder=3,
        )
        ax1.bar(
            x_pos + width / 2,
            res_clean,
            width=width,
            color="#1f77b4",
            edgecolor="#0b3c68",
            linewidth=0.7,
            label=r"Vs. clean target $\mathbf{y}_{\mathrm{clean}}$",
            zorder=3,
        )
        ax1.axhline(
            discretization_rel_gap,
            color="k",
            linestyle="--",
            linewidth=1.4,
            label=r"Discretization gap ($h_{\mathrm{tar}}$ vs. $h_{\mathrm{inv}}$)",
            zorder=4,
        )
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(eta_labels)
        ax1.set_xlabel(r"Additive Noise Level $\eta$ [%]")
        ax1.set_ylabel(r"Relative $L^2$ Residual Norm")
        ax1.set_title("(a) Relative Observation Residuals")
        ax1.grid(True, axis="y")
        ax1.legend(loc="upper left", frameon=True)

        ax2.bar(
            x_pos,
            noise_eff,
            width=0.42,
            color="#5c6b73",
            edgecolor="#253237",
            linewidth=0.7,
            label=r"Realized $\|\boldsymbol{\varepsilon}\|_2 / \|\mathbf{y}_{\mathrm{clean}}\|_2$",
            zorder=3,
        )
        y2_max = float(np.max(noise_eff)) if noise_eff.size > 0 else 0.1
        ax2.set_ylim(0.0, max(y2_max * 1.20, 1.0e-3))
        for i, val in enumerate(noise_eff):
            ax2.annotate(
                f"{val:.3f}",
                xy=(x_pos[i], val),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.0,
            )

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(eta_labels)
        ax2.set_xlabel(r"Nominal Noise Level $\eta$ [%]")
        ax2.set_ylabel(r"Realized Noise Norm $\|\boldsymbol{\varepsilon}\|_2 / \|\mathbf{y}_{\mathrm{clean}}\|_2$")
        ax2.set_title(r"(b) Realized Relative Noise Norm")
        ax2.grid(True, axis="y")
        ax2.legend(loc="upper left", frameon=True)

        fig.tight_layout()
        paths = save_figure_png_and_pdf(fig, out_path)
        plt.close(fig)
        return paths


def verify_and_load_multistart_artifacts(
    multistart_dir: str | Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Loads existing JSON and CSV artifacts from `artifacts/03a_twin_noise_free_multistart/`
    without modifying any file, and checks numerical consistency between JSON and CSV.
    """
    d = Path(multistart_dir)
    json_path = d / "multistart_summary.json"
    runs_csv_path = d / "multistart_runs_summary.csv"
    hist_csv_path = d / "multistart_convergence_history.csv"

    if not json_path.exists():
        raise FileNotFoundError(f"Missing multistart JSON summary: {json_path}")

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    runs = summary.get("runs", [])

    if runs_csv_path.exists():
        with open(runs_csv_path, "r", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        if len(csv_rows) != len(runs):
            warnings.warn(
                f"Row count mismatch between {json_path.name} ({len(runs)}) and "
                f"{runs_csv_path.name} ({len(csv_rows)}).",
                RuntimeWarning,
            )
        else:
            for r_json, r_csv in zip(runs, csv_rows):
                j_json = float(r_json["best_cost_J_total"])
                j_csv = float(r_csv["best_cost_J_total"])
                if not np.isclose(j_json, j_csv, rtol=1e-10, atol=1e-12):
                    warnings.warn(
                        f"Inconsistency in best_cost_J_total for {r_json['run_id']}: "
                        f"JSON={j_json} vs CSV={j_csv}",
                        RuntimeWarning,
                    )

    # If convergence history is missing in JSON but present in CSV, reconstruct from CSV
    if hist_csv_path.exists():
        with open(hist_csv_path, "r", encoding="utf-8") as f:
            hist_rows = list(csv.DictReader(f))
        by_run: Dict[str, List[Dict[str, str]]] = {}
        for row in hist_rows:
            by_run.setdefault(row["run_id"], []).append(row)
        for r_json in runs:
            rid = r_json["run_id"]
            conv = r_json.setdefault("convergence_history", {})
            if not conv.get("eval_index") and rid in by_run:
                conv["eval_index"] = [int(x["eval_index"]) for x in by_run[rid]]
                conv["best_j_so_far"] = [float(x["best_j_so_far"]) for x in by_run[rid]]

    return summary, runs


def verify_and_load_noise_diagnostics_artifacts(
    noise_diag_dir: str | Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Loads existing JSON and CSV artifacts from `artifacts/03_twin_robustness/diagnostics/`
    without modifying any file, and checks numerical consistency between JSON and CSV.
    """
    d = Path(noise_diag_dir)
    json_path = d / "twin_robustness_diagnostics.json"
    runs_csv_path = d / "twin_robustness_summary.csv"

    if not json_path.exists():
        raise FileNotFoundError(f"Missing noise diagnostics JSON: {json_path}")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    runs = payload.get("detailed_runs", [])

    if runs_csv_path.exists():
        with open(runs_csv_path, "r", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        if len(csv_rows) != len(runs):
            warnings.warn(
                f"Row count mismatch between {json_path.name} ({len(runs)}) and "
                f"{runs_csv_path.name} ({len(csv_rows)}).",
                RuntimeWarning,
            )
        else:
            for r_json, r_csv in zip(runs, csv_rows):
                j_json = float(r_json["best_cost_J_total"])
                j_csv = float(r_csv["best_cost_J_total"])
                if not np.isclose(j_json, j_csv, rtol=1e-10, atol=1e-12):
                    warnings.warn(
                        f"Inconsistency in best_cost_J_total for {r_json['run_id']}: "
                        f"JSON={j_json} vs CSV={j_csv}",
                        RuntimeWarning,
                    )

    return payload, runs


def regenerate_all_twin_figures_from_artifacts(
    multistart_dir: str | Path = "artifacts/03a_twin_noise_free_multistart",
    noise_diag_dir: str | Path = "artifacts/03_twin_robustness/diagnostics",
    bounds_dict: Optional[Dict[str, Tuple[float, float]]] = None,
    suffix: str = "_v2",
) -> Dict[str, Tuple[Path, Path]]:
    """
    Regenerates all 5 twin-experiment diagnostic figures (both 300 dpi PNG and vector PDF)
    directly from the existing JSON/CSV files, without invoking the FEM forward solver
    and without overwriting the original v1 figures or numerical artifacts.
    """
    generated: Dict[str, Tuple[Path, Path]] = {}

    m_dir = Path(multistart_dir)
    if (m_dir / "multistart_summary.json").exists():
        _, m_runs = verify_and_load_multistart_artifacts(m_dir)
        generated["multistart_parameter_recovery"] = plot_multistart_parameter_recovery_v2(
            m_runs,
            bounds_dict=bounds_dict,
            out_path=m_dir / f"fig_multistart_parameter_recovery{suffix}.png",
        )
        generated["multistart_convergence"] = plot_multistart_convergence_v2(
            m_runs,
            out_path=m_dir / f"fig_multistart_convergence{suffix}.png",
        )
        generated["multistart_objective_decomposition"] = plot_multistart_objective_decomposition_v2(
            m_runs,
            out_path=m_dir / f"fig_multistart_objective_decomposition{suffix}.png",
        )

    n_dir = Path(noise_diag_dir)
    if (n_dir / "twin_robustness_diagnostics.json").exists():
        n_payload, n_runs = verify_and_load_noise_diagnostics_artifacts(n_dir)
        disc_gap = float(n_payload.get("discretization_relative_gap", 0.0))
        generated["noise_parameter_errors"] = plot_noise_study_parameter_errors_v2(
            n_runs,
            out_path=n_dir / f"fig_noise_parameter_errors{suffix}.png",
        )
        generated["noise_relative_residuals"] = plot_noise_study_relative_residuals_v2(
            n_runs,
            discretization_rel_gap=disc_gap,
            out_path=n_dir / f"fig_noise_relative_residuals{suffix}.png",
        )

    return generated