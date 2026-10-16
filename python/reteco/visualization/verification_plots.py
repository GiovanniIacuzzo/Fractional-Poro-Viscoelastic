"""
python/reteco/visualization/verification_plots.py
Generazione dei grafici di verifica del problema diretto:
- Tassi di convergenza MMS spaziali e temporali per (d, p, c)
- Bilancio energetico discreto (Proposizione 1) e conservazione della massa M(t) = M(0)
"""

from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt

from .. import _core


def plot_mms_spatial_convergence(
    reports: Sequence[_core.MMSErrorReport],
    output_path: str | Path,
) -> None:
    h = np.array([r.h_mesh for r in reports])
    d_l2 = np.array([r.d_L2 for r in reports])
    p_l2 = np.array([r.p_L2 for r in reports])
    c_l2 = np.array([r.c_L2 for r in reports])

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    ax.loglog(h, d_l2, "o-", label=r"$\|\mathbf{d}_h - \mathbf{d}^*\|_{L^2}$")
    ax.loglog(h, p_l2, "s-", label=r"$\|p_h - p^*\|_{L^2}$")
    ax.loglog(h, c_l2, "^-", label=r"$\|c_h - c^*\|_{L^2}$")

    ref_h2 = p_l2[0] * (h / h[0]) ** 2
    ax.loglog(h, ref_h2, "k--", alpha=0.6, label=r"$\mathcal{O}(h^2)$ reference")

    ax.set_xlabel(r"Mesh size $h$")
    ax.set_ylabel(r"$L^2(\Omega)$ Error at $t=T$")
    ax.set_title("RETECO v1 Coupled MMS Spatial Convergence")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_energy_and_mass_diagnostics(
    diagnostics: Sequence[_core.StepDiagnostics],
    output_path: str | Path,
) -> None:
    t = np.array([d.time for d in diagnostics])
    e_tot = np.array([d.total_energy for d in diagnostics])
    d_visco = np.array([d.cumulative_visco_dissip for d in diagnostics])
    d_darcy = np.array([d.cumulative_darcy_dissip for d in diagnostics])
    mass = np.array([d.tracer_total_mass for d in diagnostics])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=150)

    ax1.plot(t, e_tot, "b-", lw=2, label=r"$\mathcal{E}(t_n)$")
    ax1.plot(t, d_visco, "r--", lw=1.8, label=r"$\mathcal{D}_\beta(t_n)$")
    ax1.plot(t, d_darcy, "g-.", lw=1.8, label=r"$\int_0^{t_n} a_h(p,p)\,ds$")
    ax1.plot(t, e_tot + d_visco + d_darcy, "k:", lw=2, label=r"Total $\mathcal{E}+\mathcal{D}_\beta+\mathcal{D}_h$")
    ax1.set_xlabel("Dimensionless Time $t$")
    ax1.set_ylabel("Energy / Dissipation")
    ax1.set_title("Discrete Energy Balance (Homogeneous Case)")
    ax1.grid(True, ls=":", alpha=0.6)
    ax1.legend()

    rel_mass_err = np.abs(mass - mass[0]) / max(abs(mass[0]), 1.0e-15)
    ax2.semilogy(t, np.maximum(rel_mass_err, 1.0e-16), "m-o", markersize=3)
    ax2.set_xlabel("Dimensionless Time $t$")
    ax2.set_ylabel(r"$|M(t_n) - M(0)| / M(0)$")
    ax2.set_title("Discrete Tracer Mass Conservation")
    ax2.grid(True, which="both", ls=":", alpha=0.6)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)