"""
replot/common_style.py
Shared publication-grade visual standard for all scientific figures in the manuscript.
Designed for Q1 journals in applied mathematics, computational mechanics, and numerical analysis.

Strictly standalone: never imports the C++ FEM core (`reteco._core`) or any experiment runner.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Project root resolution assuming this file resides in `<repo_root>/replot/common_style.py`
REPLOT_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = REPLOT_DIR.parent
ARTIFACTS_DIR: Path = REPO_ROOT / "artifacts"
OUTPUT_ROOT: Path = REPLOT_DIR / "output"

# Unified colorblind-safe and print-friendly scientific palette (Okabe-Ito / Tol inspired)
PALETTE: Dict[str, str] = {
    "blue": "#0072B2",       # Primary data / CPSO / displacement d_h / alpha
    "red": "#D55E00",        # Ground truth / DE / pressure p_h / beta
    "green": "#009E73",      # Powell / tracer c_h / diffusivity D
    "purple": "#CC79A7",     # Effective modulus E / secondary metric
    "orange": "#E69F00",     # Regularization / intermediate resolution / eta=5%
    "cyan": "#56B4E9",       # Fine resolution / secondary trajectory
    "yellow": "#F0E442",     # Highlight
    "charcoal": "#2B2D42",   # Dark neutral for reference lines and total cost
    "slate": "#5C6B73",      # Muted grey-blue for zero-flow / coarse baseline
    "light_grey": "#D9D9D9", # Subtle background / grid / bar edges
}

# Consistent algorithm styling across all inverse and benchmark figures
OPTIMIZER_STYLE: Dict[str, Dict[str, Any]] = {
    "CPSO": {
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
        "label": "CPSO",
        "zorder": 4,
    },
    "DE": {
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "--",
        "label": "Differential Evolution (DE)",
        "zorder": 3,
    },
    "Differential Evolution": {
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "--",
        "label": "Differential Evolution (DE)",
        "zorder": 3,
    },
    "Powell": {
        "color": "#009E73",
        "marker": "^",
        "linestyle": "-.",
        "label": "Powell",
        "zorder": 3,
    },
}

# Consistent parameter styling across all identifiability and recovery figures
PARAM_KEYS: Tuple[str, str, str, str] = ("alpha", "beta", "D", "E")

PARAM_MATH_LABELS: Dict[str, str] = {
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
    "D": r"$D$",
    "E": r"$E$",
}

PARAM_FULL_LABELS: Dict[str, str] = {
    "alpha": r"Subdiffusion Order $\alpha$",
    "beta": r"Viscoelastic Order $\beta$",
    "D": r"Apparent Diffusivity $D$",
    "E": r"Effective Young's Modulus $E$",
}

PARAM_STYLE: Dict[str, Dict[str, Any]] = {
    "alpha": {"color": "#0072B2", "marker": "o", "linestyle": "-", "label": r"$\alpha$"},
    "beta": {"color": "#D55E00", "marker": "s", "linestyle": "--", "label": r"$\beta$"},
    "D": {"color": "#009E73", "marker": "^", "linestyle": "-.", "label": r"$D$"},
    "E": {"color": "#CC79A7", "marker": "D", "linestyle": ":", "label": r"$E$"},
}

# Distinct run trajectories (e.g., Run 1..Run 5 in multi-start / multi-seed)
RUN_COLORS: Tuple[str, ...] = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
)
RUN_LINESTYLES: Tuple[str, ...] = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)), "-")
RUN_MARKERS: Tuple[str, ...] = ("o", "s", "^", "D", "v", "P")

# Standard figure dimensions (in inches) tailored to 1-column and 2-column LaTeX layouts
FIG_SIZES: Dict[str, Tuple[float, float]] = {
    "single": (6.2, 4.1),
    "single_wide": (7.0, 4.3),
    "double_row": (10.4, 4.1),
    "triple_row": (12.6, 3.9),
    "quad_grid": (9.4, 6.0),
    "six_grid": (11.2, 6.8),
    "tall_grid": (9.6, 7.6),
}


def get_rc_params() -> Dict[str, Any]:
    """Returns unified matplotlib rcParams compatible with Computer Modern / serif math."""
    return {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 9.5,
        "axes.titlesize": 10.0,
        "axes.titleweight": "normal",
        "axes.labelsize": 9.5,
        "axes.labelweight": "normal",
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.2,
        "legend.title_fontsize": 8.5,
        "figure.titlesize": 10.5,
        "figure.titleweight": "normal",
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#222222",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "lines.linewidth": 1.6,
        "lines.markersize": 5.2,
        "grid.color": "#B0B0B0",
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.45,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#CCCCCC",
        "legend.fancybox": False,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def apply_common_style() -> None:
    """Applies the global publication style to matplotlib."""
    plt.rcParams.update(get_rc_params())


def style_axis(
    ax: plt.Axes,
    *,
    grid: bool = True,
    minor_grid: bool = False,
    grid_axis: str = "both",
    despine_top_right: bool = False,
) -> None:
    """Applies consistent grid and spine formatting to a matplotlib Axes."""
    if grid:
        ax.grid(True, which="major", axis=grid_axis, linestyle=":", linewidth=0.6, alpha=0.45)
    else:
        ax.grid(False)
    if minor_grid:
        ax.grid(True, which="minor", axis=grid_axis, linestyle=":", linewidth=0.45, alpha=0.22)
    if despine_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def export_figure(
    fig: plt.Figure,
    family_subdir: str,
    figure_stem: str,
    dpi: int = 300,
) -> Tuple[Path, Path]:
    """
    Exports `fig` to both vector PDF and high-resolution PNG (300 dpi) inside
    `replot/output/<family_subdir>/<figure_stem>.{pdf,png}`.
    Never overwrites original artifacts.
    """
    out_dir = OUTPUT_ROOT / family_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    stem_clean = figure_stem
    for suffix in (".pdf", ".png", ".svg"):
        if stem_clean.lower().endswith(suffix):
            stem_clean = stem_clean[: -len(suffix)]

    pdf_path = out_dir / f"{stem_clean}.pdf"
    png_path = out_dir / f"{stem_clean}.png"

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight")
    return pdf_path, png_path


def require_file(path: Path, description: str = "Required data file") -> Path:
    """Verifies that a source file exists; raises a descriptive FileNotFoundError otherwise."""
    if not path.exists():
        raise FileNotFoundError(
            f"[replot] {description} not found at '{path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}'."
        )
    return path


def require_columns(
    available_columns: Iterable[str],
    required: Sequence[str],
    source_path: Path,
) -> None:
    """Verifies that all required columns are present in a CSV or DataFrame."""
    avail_set = set(available_columns)
    missing = [c for c in required if c not in avail_set]
    if missing:
        raise KeyError(
            f"[replot] Missing required columns {missing} in '{source_path}'. "
            f"Available columns: {sorted(avail_set)}"
        )


def format_panel_title(panel_letter: str, text: str) -> str:
    """Formats a subplot panel title consistently as `(a) Description`."""
    letter = panel_letter.strip().strip("()")
    return f"({letter}) {text}"