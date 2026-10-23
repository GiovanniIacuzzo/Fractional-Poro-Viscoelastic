"""
replot/replot_forward_verification.py
Reconstructs all figures in Family A (Forward Model Verification) from
`artifacts/01_forward_verification/forward_verification_summary.json`:
1. `fig_mms_spatial_convergence`: Coupled spatial MMS convergence (L2 and H1 errors vs mesh size h).
2. `fig_mms_temporal_convergence`: Coupled temporal MMS convergence (uniform r=1 vs graded r=2.5).
3. `fig_energy_and_mass_conservation`: Discrete energy-dissipation balance and relative tracer mass error.
4. `fig_fractional_order_sensitivity`: Parametric sensitivity to fractional orders alpha and beta (if present).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from common_style import (
    ARTIFACTS_DIR,
    FIG_SIZES,
    PALETTE,
    apply_common_style,
    export_figure,
    format_panel_title,
    require_file,
    style_axis,
)

FAMILY_SUBDIR = "01_forward_verification"
DIR_01 = ARTIFACTS_DIR / "01_forward_verification"


def _load_verification_report() -> Dict[str, Any]:
    candidates = [
        DIR_01 / "forward_verification_summary.json",
        DIR_01 / "forward_verification_results.json",
        DIR_01 / "verification_summary.json",
    ]
    for c in candidates:
        if c.exists():
            return json.loads(c.read_text(encoding="utf-8"))
    json_files = sorted(DIR_01.glob("*.json"))
    if json_files:
        return json.loads(json_files[0].read_text(encoding="utf-8"))
    path = require_file(candidates[0], "Forward verification summary JSON")
    return json.loads(path.read_text(encoding="utf-8"))


def _find_list_of_dicts_with_keys(obj: Any, required_any: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Recursively searches a JSON structure for a list of dicts containing any of `required_any` keys."""
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
        if any(k in obj[0] for k in required_any):
            return obj
    if isinstance(obj, dict):
        # Check if dict of dicts indexed by N or M
        vals = list(obj.values())
        if len(vals) > 0 and all(isinstance(v, dict) for v in vals):
            if any(k in vals[0] for k in required_any):
                return vals
        for v in obj.values():
            res = _find_list_of_dicts_with_keys(v, required_any)
            if res:
                return res
    return []


def _get_err_val(row: Dict[str, Any], field_prefix: str, norm_type: str) -> Optional[float]:
    """Extracts error value for `field_prefix` ('d', 'p', 'c') and `norm_type` ('L2', 'H1')."""
    candidates = [
        f"{field_prefix}_{norm_type}",
        f"{field_prefix}_{norm_type.lower()}",
        f"{field_prefix}_err_{norm_type.lower()}",
        f"err_{field_prefix}_{norm_type.lower()}",
        f"{field_prefix}_{norm_type}_error",
        f"{field_prefix}_{norm_type.lower()}_error",
    ]
    for c in candidates:
        if c in row and row[c] is not None:
            return float(row[c])
    # Also check nested dicts like row["d"]["L2"]
    if field_prefix in row and isinstance(row[field_prefix], dict):
        sub = row[field_prefix]
        for k in (norm_type, norm_type.lower(), f"{norm_type}_error"):
            if k in sub:
                return float(sub[k])
    return None



def _required_err_val(row: Dict[str, Any], field: str, norm: str) -> float:
    value = _get_err_val(row, field, norm)
    if value is None or not np.isfinite(float(value)) or float(value) <= 0:
        raise KeyError(f"MMS record lacks a positive measured {field}_{norm} error: keys={list(row)}")
    return float(value)

def replot_mms_spatial_convergence(report: Dict[str, Any]) -> Tuple[Path, Path]:
    """Reconstructs `fig_mms_spatial_convergence.{pdf,png}`."""
    spatial_sec = (
        report.get("mms_spatial")
        or report.get("spatial_mms")
        or report.get("mms_spatial_convergence")
        or report
    )
    levels = _find_list_of_dicts_with_keys(
        spatial_sec, ("d_L2", "d_l2", "p_L2", "p_l2", "d_err_l2", "err_d_l2", "h", "N", "subdivisions")
    )
    if not levels:
        raise KeyError(
            f"[replot] Could not find spatial MMS levels inside 'mms_spatial': {spatial_sec}"
        )

    h_list: List[float] = []
    for i, r in enumerate(levels):
        if "h" in r:
            h_list.append(float(r["h"]))
        elif "h_mesh" in r:
            h_list.append(float(r["h_mesh"]))
        elif "N" in r:
            h_list.append(1.0 / float(r["N"]))
        elif "subdivisions" in r:
            h_list.append(1.0 / float(r["subdivisions"]))
        else:
            h_list.append(1.0 / (4.0 * (2 ** i)))
    h = np.asarray(h_list, dtype=np.float64)

    d_l2 = np.array([_required_err_val(r, "d", "L2") for r in levels], dtype=np.float64)
    p_l2 = np.array([_required_err_val(r, "p", "L2") for r in levels], dtype=np.float64)
    c_l2 = np.array([_required_err_val(r, "c", "L2") for r in levels], dtype=np.float64)

    has_h1 = _get_err_val(levels[0], "d", "H1") is not None

    apply_common_style()
    if has_h1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"])
    else:
        fig, ax1 = plt.subplots(figsize=FIG_SIZES["single"])
        ax2 = None

    ax1.loglog(
        h, d_l2, marker="o", color=PALETTE["blue"], linestyle="-",
        label=r"$\|\mathbf{d}_h - \mathbf{d}^*\|_{L^2(\Omega)}$",
    )
    ax1.loglog(
        h, p_l2, marker="s", color=PALETTE["red"], linestyle="--",
        label=r"$\|p_h - p^*\|_{L^2(\Omega)}$",
    )
    ax1.loglog(
        h, c_l2, marker="^", color=PALETTE["green"], linestyle="-.",
        label=r"$\|c_h - c^*\|_{L^2(\Omega)}$",
    )
    ref_h2 = p_l2[0] * (h / h[0]) ** 2 * 0.75
    ax1.loglog(
        h, ref_h2, color=PALETTE["charcoal"], linestyle=":", linewidth=1.3,
        label=r"$\mathcal{O}(h^2)$ reference",
    )
    ax1.set_xlabel(r"Mesh size $h$")
    ax1.set_ylabel(r"$L^2(\Omega)$ error at $t = T$")
    if has_h1:
        ax1.set_title(format_panel_title("a", r"Spatial Convergence in $L^2(\Omega)$"))
    style_axis(ax1, grid=True, minor_grid=True)
    ax1.legend(loc="lower right")

    if has_h1 and ax2 is not None:
        d_h1 = np.array([_required_err_val(r, "d", "H1") for r in levels], dtype=np.float64)
        p_h1 = np.array([_required_err_val(r, "p", "H1") for r in levels], dtype=np.float64)
        c_h1 = np.array([_required_err_val(r, "c", "H1") for r in levels], dtype=np.float64)

        ax2.loglog(
            h, d_h1, marker="o", color=PALETTE["blue"], linestyle="-",
            label=r"$|\mathbf{d}_h - \mathbf{d}^*|_{H^1(\Omega)}$",
        )
        ax2.loglog(
            h, p_h1, marker="s", color=PALETTE["red"], linestyle="--",
            label=r"$|p_h - p^*|_{H^1(\Omega)}$",
        )
        ax2.loglog(
            h, c_h1, marker="^", color=PALETTE["green"], linestyle="-.",
            label=r"$|c_h - c^*|_{H^1(\Omega)}$",
        )
        ref_h1 = p_h1[0] * (h / h[0]) ** 1 * 0.75
        ax2.loglog(
            h, ref_h1, color=PALETTE["charcoal"], linestyle=":", linewidth=1.3,
            label=r"$\mathcal{O}(h)$ reference",
        )
        ax2.set_xlabel(r"Mesh size $h$")
        ax2.set_ylabel(r"$H^1(\Omega)$ seminorm error at $t = T$")
        ax2.set_title(format_panel_title("b", r"Spatial Convergence in $H^1(\Omega)$"))
        style_axis(ax2, grid=True, minor_grid=True)
        ax2.legend(loc="lower right")

    fig.tight_layout()
    paths = export_figure(fig, FAMILY_SUBDIR, "fig_mms_spatial_convergence")
    plt.close(fig)
    return paths


def replot_mms_temporal_convergence(report: Dict[str, Any]) -> Tuple[Path, Path]:
    """Reconstructs `fig_mms_temporal_convergence.{pdf,png}`."""
    temp_sec = (
        report.get("mms_temporal")
        or report.get("temporal_mms")
        or report.get("mms_temporal_convergence")
        or {}
    )

    uniform_levels: List[Dict[str, Any]] = []
    graded_levels: List[Dict[str, Any]] = []

    if isinstance(temp_sec, list):
        uniform_levels = temp_sec
    elif isinstance(temp_sec, dict):
        # Try explicit uniform/graded subkeys first
        for k, val in temp_sec.items():
            lv = _find_list_of_dicts_with_keys(
                val, ("d_L2", "d_l2", "p_L2", "p_l2", "M", "num_steps", "dt", "dt_max")
            )
            if lv:
                if "1.0" in k or "uniform" in k.lower() or not uniform_levels:
                    uniform_levels = lv
                else:
                    graded_levels = lv
        if not uniform_levels:
            uniform_levels = _find_list_of_dicts_with_keys(
                temp_sec, ("d_L2", "d_l2", "p_L2", "p_l2", "M", "num_steps", "dt", "dt_max")
            )

    if not uniform_levels:
        raise KeyError(f"[replot] Could not find temporal MMS levels inside 'mms_temporal': {temp_sec}")

    apply_common_style()
    if graded_levels:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"], sharey=True)
    else:
        fig, ax1 = plt.subplots(figsize=FIG_SIZES["single"])
        ax2 = None

    def _plot_temp_panel(ax: plt.Axes, levels: List[Dict[str, Any]], title: str) -> None:
        m_list: List[float] = []
        for i, r in enumerate(levels):
            if "M" in r:
                m_list.append(float(r["M"]))
            elif "time_steps_M" in r:
                m_list.append(float(r["time_steps_M"]))
            elif "num_steps" in r:
                m_list.append(float(r["num_steps"]))
            elif "dt" in r and float(r["dt"]) > 0:
                m_list.append(1.0 / float(r["dt"]))
            elif "dt_max" in r and float(r["dt_max"]) > 0:
                m_list.append(1.0 / float(r["dt_max"]))
            else:
                m_list.append(float(10 * (2 ** i)))
        m_vals = np.asarray(m_list, dtype=np.float64)

        d_l2 = np.array([_required_err_val(r, "d", "L2") for r in levels], dtype=np.float64)
        p_l2 = np.array([_required_err_val(r, "p", "L2") for r in levels], dtype=np.float64)
        c_l2 = np.array([_required_err_val(r, "c", "L2") for r in levels], dtype=np.float64)

        ax.loglog(
            m_vals, d_l2, marker="o", color=PALETTE["blue"], linestyle="-",
            label=r"$\|\mathbf{d}_h - \mathbf{d}^*\|_{L^2(\Omega)}$",
        )
        ax.loglog(
            m_vals, p_l2, marker="s", color=PALETTE["red"], linestyle="--",
            label=r"$\|p_h - p^*\|_{L^2(\Omega)}$",
        )
        ax.loglog(
            m_vals, c_l2, marker="^", color=PALETTE["green"], linestyle="-.",
            label=r"$\|c_h - c^*\|_{L^2(\Omega)}$",
        )
        ref_m1 = c_l2[0] * (m_vals / m_vals[0]) ** (-1.0) * 0.8
        ax.loglog(
            m_vals, ref_m1, color=PALETTE["charcoal"], linestyle=":", linewidth=1.3,
            label=r"$\mathcal{O}(M^{-1})$ reference",
        )
        ax.set_xlabel(r"Number of time steps $M$")
        ax.set_title(title)
        style_axis(ax, grid=True, minor_grid=True)
        ax.legend(loc="upper right")

    _plot_temp_panel(
        ax1,
        uniform_levels,
        format_panel_title("a", r"Uniform Time Grid ($r = 1.0$)") if graded_levels else "Temporal MMS Convergence",
    )
    ax1.set_ylabel(r"$L^2(\Omega)$ error at $t = T$")

    if graded_levels and ax2 is not None:
        _plot_temp_panel(
            ax2,
            graded_levels,
            format_panel_title("b", r"Graded Time Grid ($r > 1$)"),
        )

    fig.tight_layout()
    paths = export_figure(fig, FAMILY_SUBDIR, "fig_mms_temporal_convergence")
    plt.close(fig)
    return paths


def replot_energy_and_mass_conservation(report: Dict[str, Any]) -> Tuple[Path, Path]:
    """Plot recorded residual diagnostics; this report does not contain time histories."""
    em = (
        report.get("energy_and_mass_verification")
        or report.get("energy_and_mass")
        or report.get("energy_mass_conservation")
        or report.get("energy_and_mass_conservation")
    )
    if not isinstance(em, dict):
        raise KeyError("Missing energy_and_mass_verification summary in the forward report.")

    refinement = em.get("dt_refinement_residuals")
    if not isinstance(refinement, list) or len(refinement) < 2:
        raise ValueError("Forward report must contain measured dt_refinement_residuals; scalar mass/energy summaries are not a substitute.")

    dt = np.asarray([float(r["dt"]) for r in refinement], dtype=float)
    residual = np.asarray([float(r["abs_energy_residual_RN"]) for r in refinement], dtype=float)
    if np.any(~np.isfinite(dt)) or np.any(~np.isfinite(residual)) or np.any(dt <= 0) or np.any(residual <= 0):
        raise ValueError("dt-refinement data must be finite and strictly positive for the log-log plot.")
    order = np.argsort(dt)[::-1]
    dt, residual = dt[order], residual[order]
    observed_order = np.diff(np.log(residual)) / np.diff(np.log(dt))

    scalar_specs = [
        ("initial_compatibility_residual", "Initial compatibility"),
        ("max_stepwise_discrete_identity_error", "Stepwise identity error"),
        ("relative_mass_drift", "Relative mass drift"),
    ]
    scalar_values = []
    scalar_labels = []
    for key, label in scalar_specs:
        if key not in em or em[key] is None:
            raise KeyError(f"Forward summary lacks measured scalar diagnostic {key!r}.")
        v = float(em[key])
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"Diagnostic {key!r} must be finite and nonnegative, received {v}.")
        scalar_values.append(v)
        scalar_labels.append(label)

    apply_common_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZES["double_row"])
    ax1.loglog(dt, residual, marker="o", linestyle="-", linewidth=1.8,
               color=PALETTE["blue"], label=r"Measured $|R_N|$")
    # Reference slope is a visual guide, anchored to a measured point; it is not an extra datum.
    guide = residual[0] * dt / dt[0]
    ax1.loglog(dt, guide, linestyle=":", linewidth=1.4, color=PALETTE["charcoal"],
               label=r"$O(\Delta t)$ reference slope")
    ax1.set_xlabel(r"Time step $\Delta t$")
    ax1.set_ylabel(r"Absolute cumulative energy residual $|R_N|$")
    ax1.set_title(format_panel_title("a", "Energy Residual Under Time Refinement"))
    style_axis(ax1, grid=True, minor_grid=True)
    ax1.legend(loc="best")
    if len(observed_order):
        ord_text = "Observed slopes: " + ", ".join(f"{x:.2f}" for x in observed_order)
        ax1.text(0.04, 0.04, ord_text, transform=ax1.transAxes, fontsize=8,
                 va="bottom", ha="left", bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9, "pad": 3})

    x = np.arange(len(scalar_values))
    bars = ax2.bar(x, scalar_values, color=[PALETTE["blue"], PALETTE["green"], PALETTE["purple"]],
                   edgecolor=PALETTE["charcoal"], linewidth=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scalar_labels, rotation=12, ha="right")
    ax2.set_ylabel("Recorded residual / relative drift")
    ax2.set_title(format_panel_title("b", "Compatibility, Identity and Mass Checks"))
    vmax = max(scalar_values) if max(scalar_values) > 0 else 1e-18
    ax2.set_yscale("symlog", linthresh=1e-18, linscale=1.0, base=10)
    ax2.set_ylim(0, vmax * 2.0)
    for bar, value in zip(bars, scalar_values):
        ax2.annotate(f"{value:.2e}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=7.6)
    style_axis(ax2, grid=True, grid_axis="y")

    fig.tight_layout()
    paths = export_figure(fig, FAMILY_SUBDIR, "fig_energy_and_mass_conservation")
    plt.close(fig)
    return paths

def run_all() -> List[Tuple[Path, Path]]:
    """Generates all Forward Verification figures."""
    report = _load_verification_report()
    outputs: List[Tuple[Path, Path]] = [
        replot_mms_spatial_convergence(report),
        replot_mms_temporal_convergence(report),
        replot_energy_and_mass_conservation(report),
    ]
    return outputs


if __name__ == "__main__":
    generated = run_all()
    for pdf_p, png_p in generated:
        print(f"[01_forward_verification] Generated: {pdf_p} | {png_p}")