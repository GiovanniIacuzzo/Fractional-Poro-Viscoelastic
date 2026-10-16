"""
python/reteco/visualization/identifiability_plots.py
Grafici dello spettro dei valori singolari di J^G(theta_tilde), delle norme colonnari
e della matrice di correlazione locale di Fisher.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from ..identifiability.svd_conditioning import LocalIdentifiabilityReport


def plot_svd_identifiability_summary(
    report_active: LocalIdentifiabilityReport,
    report_zero_q: LocalIdentifiabilityReport,
    output_path: str | Path,
) -> None:
    param_labels = [r"$\widetilde{\alpha}$", r"$\widetilde{\beta}$", r"$\widetilde{D}$", r"$\widetilde{E}$"]
    x = np.arange(4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150)

    width = 0.35
    ax1.bar(x - width / 2, np.maximum(report_active.column_norms, 1e-16), width, label=r"Active Darcy Flow ($\mathbf{q}\neq\mathbf{0}$)")
    ax1.bar(x + width / 2, np.maximum(report_zero_q.column_norms, 1e-16), width, label=r"Zero Flow ($\mathbf{q}\equiv\mathbf{0}$, Prop. 2)")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(param_labels)
    ax1.set_ylabel(r"Jacobian Column Norm $\|\partial \mathcal{G}_h / \partial \widetilde{\theta}_j\|_2$")
    ax1.set_title("Parameter Sensitivity Norms")
    ax1.grid(True, which="both", ls=":", alpha=0.6)
    ax1.legend()

    modes = np.arange(1, 5)
    ax2.semilogy(modes, np.maximum(report_active.singular_values, 1e-16), "bo-", lw=2, label=r"Active Flow ($\mathbf{q}\neq\mathbf{0}$)")
    ax2.semilogy(modes, np.maximum(report_zero_q.singular_values, 1e-16), "rs--", lw=2, label=r"Zero Flow ($\mathbf{q}\equiv\mathbf{0}$)")
    ax2.set_xticks(modes)
    ax2.set_xlabel("Singular Value Index $k$")
    ax2.set_ylabel(r"Singular Value $\sigma_k$")
    ax2.set_title("Singular Value Spectrum of $J^{\mathcal{G}}(\widetilde{\boldsymbol{\theta}})$")
    ax2.grid(True, which="both", ls=":", alpha=0.6)
    ax2.legend()

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)