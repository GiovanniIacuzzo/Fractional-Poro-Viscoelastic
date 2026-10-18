"""
python/reteco/visualization/identifiability_plots.py
Figure scientifiche da pubblicazione per lo studio di identificabilità RETECO v1:
- Figura A: Sensibilità colonnare, spettro SVD, soglia del rango e modi singolari destri;
- Figura B: Heatmap della matrice di correlazione locale con gestione trasparente del rango deficiente;
- Figura C: Sezioni unidimensionali del funzionale di costo (One-at-a-Time Cost Slices);
- Figura D: Profili dell'obiettivo ottimizzati (Optimized Objective Profiles) con distinzione
  dei vincoli attivi ai bordi.
Salva ogni figura sia in PDF vettoriale (.pdf) sia in PNG ad alta risoluzione (>= 300 dpi).
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np

from ..config_schema import ParameterCoordinateTransform
from ..identifiability.profile_likelihood import (
    OneAtATimeCostSlicesReport,
    OptimizedObjectiveProfilesReport,
)
from ..identifiability.svd_conditioning import LocalIdentifiabilityReport

PARAM_LATEX_NORM: Tuple[str, ...] = (
    r"$\widetilde{\alpha}$",
    r"$\widetilde{\beta}$",
    r"$\widetilde{D}$",
    r"$\widetilde{E}$",
)
PARAM_LATEX_PHYS: Dict[str, str] = {
    "alpha": r"$\alpha$ (Caputo tracer order) [-]",
    "beta": r"$\beta$ (Kelvin-Voigt viscoelastic order) [-]",
    "D": r"$D$ (Dimensionless diffusivity) [-]",
    "E": r"$E$ (Dimensionless drained Young modulus) [-]",
}


def _save_pdf_and_png(fig: plt.Figure, base_or_pdf_path: str | Path, dpi: int = 300) -> Dict[str, str]:
    p = Path(base_or_pdf_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = p.with_suffix(".pdf")
    png_path = p.with_suffix(".png")
    fig.savefig(pdf_path, format="pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf_path), "png": str(png_path)}


# =============================================================================
# Figura A — Sensibilità e spettro SVD
# =============================================================================

def plot_figure_a_sensitivity_and_svd(
    rep_active: LocalIdentifiabilityReport,
    rep_zero: LocalIdentifiabilityReport,
    output_path: str | Path,
) -> Dict[str, str]:
    """
    Figura A (4 pannelli):
    (a) Norme delle colonne del Jacobiano per flusso nullo (q=0) vs flusso attivo (q!=0),
        distinguendo esplicitamente i valori esattamente nulli dal floor grafico (1e-16);
    (b) Spettro dei valori singolari su scala logaritmica con soglia del rango e rango rilevato;
    (c) Matrice dei pesi parametrici nei vettori singolari destri |V_{jk}|^2 (flusso attivo)
        con evidenziazione del modo meno informativo (Modo 4);
    (d) Componenti segnate v_{4,j} del modo singolare meno informativo (flusso attivo)
        a confronto con il nucleo esatto span{beta~, E~} del caso a flusso nullo.
    """
    visual_floor = 1.0e-16
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.6), constrained_layout=True)
    fig.suptitle(
        "Figura A — Analisi Locale di Sensibilità e Spettro SVD del Jacobiano Normalizzato\n"
        f"(Osservazioni: N_obs={rep_active.jacobian_tilde.shape[0]} | "
        f"Modalità pesatura: {rep_active.weighting_metadata.get('weighting_mode', 'unweighted')})",
        fontsize=11,
        fontweight="bold",
    )

    # --- Pannello (a): Norme delle colonne del Jacobiano ---
    ax_a = axes[0, 0]
    x_idx = np.arange(4)
    width = 0.36
    cn_zero_raw = rep_zero.column_norms
    cn_act_raw = rep_active.column_norms

    cn_zero_plot = np.maximum(cn_zero_raw, visual_floor)
    cn_act_plot = np.maximum(cn_act_raw, visual_floor)

    bars_z = ax_a.bar(
        x_idx - 0.5 * width,
        cn_zero_plot,
        width=width,
        color="#d9534f",
        edgecolor="black",
        alpha=0.85,
        label=r"Zero Darcy Flow ($\mathbf{q}\equiv\mathbf{0}$, Prop. 2)",
    )
    bars_a = ax_a.bar(
        x_idx + 0.5 * width,
        cn_act_plot,
        width=width,
        color="#1f77b4",
        edgecolor="black",
        alpha=0.90,
        label=r"Active Coupled Flow ($\mathbf{q}\neq\mathbf{0}$)",
    )

    for j in range(4):
        if cn_zero_raw[j] <= 1.0e-14:
            bars_z[j].set_hatch("//")
            bars_z[j].set_Alpha(0.40) if hasattr(bars_z[j], "set_Alpha") else None
            ax_a.text(
                x_idx[j] - 0.5 * width,
                visual_floor * 3.5,
                "Exact 0\n(floor)",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#a94442",
                fontweight="bold",
                rotation=90,
            )
        else:
            ax_a.text(
                x_idx[j] - 0.5 * width,
                cn_zero_plot[j] * 1.35,
                f"{cn_zero_raw[j]:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        ax_a.text(
            x_idx[j] + 0.5 * width,
            cn_act_plot[j] * 1.35,
            f"{cn_act_raw[j]:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#0b3c5d",
            fontweight="bold",
        )

    ax_a.set_yscale("log")
    ax_a.set_ylim(visual_floor * 0.5, max(float(np.max(cn_act_plot)), 1.0) * 35.0)
    ax_a.axhline(
        rep_active.rank_threshold_absolute,
        color="#1f77b4",
        ls=":",
        lw=1.3,
        label=rf"Active threshold $\tau_{{\mathrm{{abs}}}}={rep_active.rank_threshold_absolute:.2e}$",
    )
    ax_a.set_xticks(x_idx)
    ax_a.set_xticklabels(PARAM_LATEX_NORM, fontsize=10.5)
    ax_a.set_ylabel(r"Column Euclidean Norm $\| \partial \mathcal{G}_h / \partial \widetilde{\theta}_j \|_2$")
    ax_a.set_title("(a) Sensibilità Colonnare per Parametro", fontsize=10.5, fontweight="bold")
    ax_a.grid(True, which="both", ls=":", alpha=0.55)
    ax_a.legend(loc="upper right", fontsize=8.0)

    # --- Pannello (b): Spettro dei valori singolari ---
    ax_b = axes[0, 1]
    modes = np.arange(1, 5)
    sv_zero = rep_zero.singular_values
    sv_act = rep_active.singular_values

    # Separiamo i valori singolari positivi da quelli numericamente nulli nel caso q=0
    pos_mask_z = sv_zero > rep_zero.rank_threshold_absolute
    ax_b.semilogy(
        modes[pos_mask_z],
        sv_zero[pos_mask_z],
        "rs--",
        lw=2.0,
        markersize=7,
        label=rf"Zero Flow ($\mathrm{{rank}}={rep_zero.numerical_rank}$)",
    )
    if np.any(~pos_mask_z):
        ax_b.scatter(
            modes[~pos_mask_z],
            np.full(int(np.sum(~pos_mask_z)), visual_floor),
            marker="v",
            s=85,
            facecolors="none",
            edgecolors="#d9534f",
            linewidths=2.0,
            label=r"Zero Flow $\sigma_3=\sigma_4=0$ (plotted at floor $10^{-16}$)",
        )
        for m_idx in modes[~pos_mask_z]:
            ax_b.annotate(
                "Numerically 0\n(rank-deficient)",
                xy=(m_idx, visual_floor),
                xytext=(m_idx - 0.15, visual_floor * 25.0),
                fontsize=7.5,
                color="#a94442",
                ha="center",
                arrowprops={"arrowstyle": "->", "color": "#d9534f", "lw": 1.0},
            )

    ax_b.semilogy(
        modes,
        sv_act,
        "bo-",
        lw=2.2,
        markersize=7,
        label=(
            rf"Active Flow ($\mathrm{{rank}}={rep_active.numerical_rank}$, "
            rf"$\kappa={rep_active.condition_number:.2f}$, $\gamma={rep_active.collinearity_index:.2f}$)"
        ),
    )
    for m_idx, val in zip(modes, sv_act):
        ax_b.text(m_idx, val * 1.45, f"{val:.3f}", ha="center", va="bottom", fontsize=8.0, color="#0b3c5d")

    ax_b.axhline(
        rep_active.rank_threshold_absolute,
        color="#1f77b4",
        ls="--",
        lw=1.3,
        label=rf"Rank cutoff $\tau_{{\mathrm{{rel}}}}\sigma_1 = {rep_active.rank_threshold_absolute:.2e}$",
    )
    ax_b.set_ylim(visual_floor * 0.3, max(float(sv_act[0]), 1.0) * 30.0)
    ax_b.set_xticks(modes)
    ax_b.set_xticklabels([f"Mode {k}" for k in modes])
    ax_b.set_xlabel("Singular Mode Index $k$")
    ax_b.set_ylabel(r"Singular Value $\sigma_k(J_w)$")
    ax_b.set_title("(b) Spettro dei Valori Singolari e Soglia del Rango", fontsize=10.5, fontweight="bold")
    ax_b.grid(True, which="both", ls=":", alpha=0.55)
    ax_b.legend(loc="lower left", fontsize=7.8)

    # --- Pannello (c): Partecipazione parametrica |V_{jk}|^2 nei modi singolari destri ---
    ax_c = axes[1, 0]
    v_sq = (rep_active.right_singular_vectors_V ** 2)
    im_c = ax_c.imshow(v_sq, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax_c.set_xticks(np.arange(4))
    ax_c.set_xticklabels(
        [f"Mode {k+1}\n($\\sigma_{k+1}={sv_act[k]:.3f}$)" for k in range(4)], fontsize=8.5
    )
    ax_c.set_yticks(np.arange(4))
    ax_c.set_yticklabels(PARAM_LATEX_NORM, fontsize=10.5)
    for i in range(4):
        for k in range(4):
            val_sq = v_sq[i, k]
            signed_v = rep_active.right_singular_vectors_V[i, k]
            txt_col = "white" if val_sq > 0.52 else "black"
            ax_c.text(
                k,
                i,
                f"{val_sq:.2f}\n({signed_v:+.2f})",
                ha="center",
                va="center",
                color=txt_col,
                fontsize=8.0,
            )
    # Evidenziazione della colonna 4 (modo meno informativo)
    rect = Rectangle((2.52, -0.48), 0.96, 3.96, fill=False, edgecolor="crimson", lw=2.4, ls="--")
    ax_c.add_patch(rect)
    ax_c.set_title(
        r"(c) Modi Singolari Destri $V_{jk}^2$ ($V_{jk}$) — Flusso Attivo",
        fontsize=10.5,
        fontweight="bold",
    )
    cbar_c = fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.03)
    cbar_c.set_label(r"Participation Weight $V_{jk}^2 \in [0,1]$", fontsize=8.5)

    # --- Pannello (d): Dettaglio del modo meno informativo v_4 ---
    ax_d = axes[1, 1]
    v4_act = rep_active.right_singular_vectors_V[:, 3]
    bars_d = ax_d.bar(
        x_idx,
        v4_act,
        width=0.50,
        color=["#ff7f0e" if abs(v4_act[j]) == np.max(np.abs(v4_act)) else "#5bc0de" for j in range(4)],
        edgecolor="black",
    )
    ax_d.axhline(0.0, color="black", lw=1.0)
    ax_d.set_ylim(-1.15, 1.15)
    for j in range(4):
        val = float(v4_act[j])
        offset = 0.06 if val >= 0.0 else -0.12
        ax_d.text(x_idx[j], val + offset, f"{val:+.3f}\n({val**2:.1%})", ha="center", fontsize=8.0, fontweight="bold")

    dom_p = rep_active.least_informative_mode.get("dominant_parameter", "beta")
    ax_d.set_xticks(x_idx)
    ax_d.set_xticklabels(PARAM_LATEX_NORM, fontsize=10.5)
    ax_d.set_ylabel(r"Right Singular Vector Component $v_{4,j}$")
    ax_d.set_title(
        rf"(d) Modo Meno Informativo $\mathbf{{v}}_4$ ($\sigma_4={sv_act[3]:.4f}$, dom: {dom_p})",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_d.grid(True, ls=":", alpha=0.55)

    return _save_pdf_and_png(fig, output_path)


# =============================================================================
# Figura B — Matrice di correlazione locale
# =============================================================================

def plot_figure_b_local_correlation(
    rep_active: LocalIdentifiabilityReport,
    rep_zero: LocalIdentifiabilityReport,
    output_path: str | Path,
) -> Dict[str, str]:
    """
    Figura B (2 pannelli):
    Confronta la matrice di correlazione locale tra il caso a flusso attivo (rango pieno = 4)
    e il caso a flusso nullo (rango deficiente = 2, dove le correlazioni di beta~ ed E~
    sono matematicamente non definite e vengono mostrate in modo trasparente).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), constrained_layout=True)
    interp_tag = (
        "Approssimazione Statistica Calibrata sul Modello di Rumore Dichiarato"
        if rep_active.weighting_metadata.get("is_statistically_calibrated", False)
        else "Diagnostica Geometrica Locale di Sensibilità da F = J^T J (Nessun Modello di Rumore Assunto)"
    )
    fig.suptitle(
        f"Figura B — Matrice di Correlazione Locale dei Parametri Normalizzati\n({interp_tag})",
        fontsize=10.5,
        fontweight="bold",
    )

    panels = [
        (
            axes[0],
            rep_active,
            f"(a) Protocollo a Flusso Attivo (Rango = {rep_active.numerical_rank}/4)",
        ),
        (
            axes[1],
            rep_zero,
            f"(b) Protocollo a Flusso Nullo (Rango = {rep_zero.numerical_rank}/4 — Rango Deficiente)",
        ),
    ]

    last_im = None
    for ax, rep, subtitle in panels:
        if rep.correlation_matrix_defined:
            mat = rep.fisher_correlation_matrix
        else:
            # Per rango deficiente, mostriamo solo la sottomatrice coseno sui parametri identificabili
            # e marchiamo esplicitamente come UNDEFINED tutte le celle non definite
            mat = rep.column_cosine_similarity_matrix

        masked = np.ma.masked_invalid(mat)
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad(color="#e9ecef")
        im = ax.imshow(masked, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="equal")
        last_im = im

        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(PARAM_LATEX_NORM, fontsize=11)
        ax.set_yticks(np.arange(4))
        ax.set_yticklabels(PARAM_LATEX_NORM, fontsize=11)
        ax.set_title(subtitle, fontsize=10, fontweight="bold")

        for i in range(4):
            for j in range(4):
                val = mat[i, j]
                if np.isfinite(val) and rep.correlation_matrix_defined:
                    txt_col = "white" if abs(val) > 0.55 else "black"
                    ax.text(j, i, f"{val:+.2f}", ha="center", va="center", color=txt_col, fontsize=9.5, fontweight="bold")
                elif np.isfinite(val) and not rep.correlation_matrix_defined:
                    ax.text(
                        j,
                        i,
                        f"{val:+.2f}\n(cos sub)",
                        ha="center",
                        va="center",
                        color="#212529",
                        fontsize=7.8,
                    )
                else:
                    rect = Rectangle(
                        (j - 0.5, i - 0.5),
                        1.0,
                        1.0,
                        facecolor="#dee2e6",
                        edgecolor="#adb5bd",
                        hatch="///",
                    )
                    ax.add_patch(rect)
                    ax.text(
                        j,
                        i,
                        "UNDEFINED\n(Rank < 4)",
                        ha="center",
                        va="center",
                        color="#a94442",
                        fontsize=7.5,
                        fontweight="bold",
                    )

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.tolist(), fraction=0.035, pad=0.03)
        cbar.set_label("Local Correlation / Cosine Coupling $[-1, +1]$", fontsize=9.0)

    return _save_pdf_and_png(fig, output_path)


# =============================================================================
# Figura C — Sezioni unidimensionali del funzionale di costo (OAT Cost Slices)
# =============================================================================

def plot_figure_c_cost_slices(
    slices_active: OneAtATimeCostSlicesReport,
    slices_zero: Optional[OneAtATimeCostSlicesReport],
    transform: ParameterCoordinateTransform,
    output_path: str | Path,
) -> Dict[str, str]:
    """
    Figura C (4 pannelli):
    Rappresenta le sezioni 1D del funzionale di costo (One-at-a-Time Cost Slices)
    con gli altri 3 parametri fissati a theta_dagger, etichettate esplicitamente
    per non confonderle con profili ottimizzati o profile likelihood.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), constrained_layout=True)
    fig.suptitle(
        "Figura C — Sezioni Unidimensionali del Funzionale di Costo (One-at-a-Time Cost Slices)\n"
        "(Altri 3 Parametri Fissi a $\\boldsymbol{\\theta}^\\dagger$ — Queste Sezioni NON sono Profile Likelihood)",
        fontsize=10.5,
        fontweight="bold",
    )

    param_keys = ("alpha", "beta", "D", "E")
    for idx, pname in enumerate(param_keys):
        ax = axes[idx // 2, idx % 2]
        sl_a = slices_active.slices[pname]

        ax.plot(
            sl_a.grid_physical,
            sl_a.cost_values,
            "b-o",
            lw=2.0,
            markersize=4.2,
            label=r"Active Flow ($\mathbf{q}\neq\mathbf{0}$): OAT Slice",
        )
        if slices_zero is not None and pname in slices_zero.slices:
            sl_z = slices_zero.slices[pname]
            lbl_z = (
                r"Zero Flow ($\mathbf{q}\equiv\mathbf{0}$): Flat ($J\equiv 0$)"
                if sl_z.is_flat_within_tol
                else r"Zero Flow ($\mathbf{q}\equiv\mathbf{0}$): OAT Slice"
            )
            ax.plot(
                sl_z.grid_physical,
                sl_z.cost_values,
                "r--s",
                lw=1.7,
                markersize=3.8,
                alpha=0.85,
                label=lbl_z,
            )

        ax.axvline(
            sl_a.reference_physical,
            color="black",
            ls=":",
            lw=1.5,
            label=rf"Reference $\theta^\dagger = {sl_a.reference_physical:.4g}$",
        )
        ax.set_xlabel(PARAM_LATEX_PHYS[pname], fontsize=9.5)
        ax.set_ylabel(r"Cost Slice $J_{\mathrm{obj}}(\theta_j; \boldsymbol{\theta}_{-j}^\dagger)$ [-]", fontsize=9.0)
        ax.set_title(
            f"({chr(97 + idx)}) OAT Cost Slice — Parametro `{pname}` "
            f"( Coordinate Normalizzate $\\widetilde{{{pname}}} \\in [{sl_a.grid_normalized[0]:.2f}, {sl_a.grid_normalized[-1]:.2f}]$ )",
            fontsize=9.5,
            fontweight="bold",
        )
        ax.grid(True, ls=":", alpha=0.6)
        ax.legend(loc="best", fontsize=8.0)

    return _save_pdf_and_png(fig, output_path)


# =============================================================================
# Figura D — Profili dell'obiettivo ottimizzati (Optimized Objective Profiles)
# =============================================================================

def plot_figure_d_optimized_profiles(
    profiles_report: OptimizedObjectiveProfilesReport,
    output_path: str | Path,
) -> Dict[str, str]:
    """
    Figura D (4 pannelli):
    Confronta per ciascun parametro la sezione OAT J_obj(theta_j; theta_{-j}^dagger)
    con il profilo dell'obiettivo ottimizzato min_{theta_{-j}} J_obj(theta_j, theta_{-j}),
    evidenziando i punti in cui i parametri di disturbo raggiungono i limiti della scatola.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.0), constrained_layout=True)
    title_kind = (
        "Profile Likelihood Statistica Calibrata"
        if profiles_report.is_statistical_profile_likelihood
        else "Profili Ottimizzati del Funzionale Obiettivo (Nessuna Soglia Probabilistica Assunta)"
    )
    fig.suptitle(
        f"Figura D — {title_kind}\n"
        r"($P_j(\theta_j) = \min_{\boldsymbol{\theta}_{-j}\in\Theta_{-j}} J_{\mathrm{obj}}(\theta_j,\boldsymbol{\theta}_{-j})$ "
        f"| Metodo: {profiles_report.optimizer_method} | Valutazioni FEM totali: {profiles_report.total_forward_evaluations})",
        fontsize=10.5,
        fontweight="bold",
    )

    param_keys = ("alpha", "beta", "D", "E")
    for idx, pname in enumerate(param_keys):
        ax = axes[idx // 2, idx % 2]
        pr = profiles_report.profiles[pname]

        ax.plot(
            pr.grid_physical,
            pr.oat_slice_cost_values,
            color="#6c757d",
            ls="--",
            lw=1.6,
            marker=".",
            label=r"OAT Cost Slice (fixed $\boldsymbol{\theta}_{-j}^\dagger$)",
        )
        ax.plot(
            pr.grid_physical,
            pr.profile_cost_values,
            color="#0056b3",
            ls="-",
            lw=2.2,
            label=r"Optimized Objective Profile $\min_{\boldsymbol{\theta}_{-j}} J_{\mathrm{obj}}$",
        )

        bdy_mask = np.asarray(pr.nuisance_hit_bounds_mask, dtype=bool)
        int_mask = ~bdy_mask
        if np.any(int_mask):
            ax.scatter(
                pr.grid_physical[int_mask],
                pr.profile_cost_values[int_mask],
                color="#0056b3",
                marker="o",
                s=45,
                zorder=4,
                label="Interior nuisance optimum",
            )
        if np.any(bdy_mask):
            ax.scatter(
                pr.grid_physical[bdy_mask],
                pr.profile_cost_values[bdy_mask],
                facecolors="#fd7e14",
                edgecolors="black",
                marker="^",
                s=70,
                zorder=5,
                label="Nuisance param constrained at box bound",
            )

        ax.axvline(
            pr.reference_physical,
            color="black",
            ls=":",
            lw=1.5,
            label=rf"True $\theta^\dagger = {pr.reference_physical:.4g}$",
        )
        min_lbl = (
            f"Profile min ({pr.profile_argmin_physical:.4g}, AT BOUND)"
            if pr.profile_minimum_at_grid_boundary
            else f"Profile min ({pr.profile_argmin_physical:.4g}, interior)"
        )
        ax.scatter(
            [pr.profile_argmin_physical],
            [pr.profile_min_cost],
            color="crimson",
            marker="*",
            s=130,
            zorder=6,
            label=min_lbl,
        )

        ax.set_xlabel(PARAM_LATEX_PHYS[pname], fontsize=9.5)
        ax.set_ylabel(r"Objective Value $J_{\mathrm{obj}}$ [-]", fontsize=9.0)
        ax.set_title(
            f"({chr(97 + idx)}) Profilo Ottimizzato — `{pname}` "
            f"({pr.total_evaluations_for_parameter} evals)",
            fontsize=9.5,
            fontweight="bold",
        )
        ax.grid(True, ls=":", alpha=0.6)
        ax.legend(loc="best", fontsize=7.6)

    return _save_pdf_and_png(fig, output_path)


# Wrapper retrocompatibile
def plot_svd_identifiability_summary(
    rep_active: LocalIdentifiabilityReport,
    rep_zero: LocalIdentifiabilityReport,
    output_path: str | Path,
) -> Dict[str, str]:
    return plot_figure_a_sensitivity_and_svd(rep_active, rep_zero, output_path)