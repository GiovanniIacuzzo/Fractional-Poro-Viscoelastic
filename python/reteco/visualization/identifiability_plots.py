"""
python/reteco/visualization/identifiability_plots.py
Figure scientifiche da pubblicazione per lo studio di identificabilità RETECO v1:
- Figura A: Sensibilità colonnare (senza soglia SVD impropria), spettro SVD con soglia
  di rango, e modi singolari destri;
- Figura B: Heatmap della matrice di correlazione geometrica locale (flusso attivo)
  separata dalla matrice di similarità coseno delle colonne identificabili (flusso nullo),
  con corrispondenza numerica esatta rispetto al JSON (+1.00 sulle diagonali definite);
- Figura C: Sezioni unidimensionali del funzionale di costo (One-at-a-Time Cost Slices);
- Figura D: Profili dell'obiettivo ottimizzati (Optimized Objective Profiles) con
  distinzione esplicita tra punti convergenti, non convergenti/incompleti e vincolati ai bordi.
Salva ogni figura sia in PDF vettoriale (.pdf) sia in PNG ad alta risoluzione (>= 300 dpi).
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
# Figura A — Sensibilità e spettro SVD (Requisito 4)
# =============================================================================

def plot_figure_a_sensitivity_and_svd(
    rep_active: LocalIdentifiabilityReport,
    rep_zero: LocalIdentifiabilityReport,
    output_path: str | Path,
) -> Dict[str, str]:
    """
    Figura A (4 pannelli):
    (a) Norme delle colonne del Jacobiano per flusso nullo (q=0) vs flusso attivo (q!=0),
        senza linee di soglia spettrometriche SVD improprie;
    (b) Spettro dei valori singolari su scala logaritmica con soglia di rango SVD tau_abs;
    (c) Matrice dei pesi di partecipazione |V_{jk}|^2 nei modi singolari destri;
    (d) Componenti segnate v_{4,j} del modo singolare meno informativo.
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

    # --- Pannello (a): Norme delle colonne del Jacobiano (senza soglia SVD di rango) ---
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
    ax_a.bar(
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
            ax_a.text(
                x_idx[j] - 0.5 * width,
                visual_floor * 3.5,
                "Exact 0\n(visual floor)",
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
                fontsize=7.8,
            )
        ax_a.text(
            x_idx[j] + 0.5 * width,
            cn_act_plot[j] * 1.35,
            f"{cn_act_raw[j]:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
            color="#0b3c5d",
            fontweight="bold",
        )

    ax_a.set_yscale("log")
    ax_a.set_ylim(visual_floor * 0.5, max(float(np.max(cn_act_plot)), 1.0) * 35.0)
    ax_a.set_xticks(x_idx)
    ax_a.set_xticklabels(PARAM_LATEX_NORM, fontsize=10.5)
    ax_a.set_ylabel(r"Column Euclidean Norm $\| \partial \mathcal{G}_h / \partial \widetilde{\theta}_j \|_2$")
    ax_a.set_title("(a) Sensibilità Colonnare per Parametro", fontsize=10.5, fontweight="bold")
    ax_a.grid(True, which="both", ls=":", alpha=0.55)
    ax_a.legend(loc="upper right", fontsize=8.0)

    # --- Pannello (b): Spettro dei valori singolari e soglia di rango SVD ---
    ax_b = axes[0, 1]
    modes = np.arange(1, 5)
    sv_zero = rep_zero.singular_values
    sv_act = rep_active.singular_values

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
        label=rf"SVD Rank Cutoff $\tau_{{\mathrm{{abs}}}}=\tau_{{\mathrm{{rel}}}}\sigma_1 = {rep_active.rank_threshold_absolute:.2e}$",
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
    v_sq = rep_active.right_singular_vectors_V ** 2
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
    rect = Rectangle((2.52, -0.48), 0.96, 3.96, fill=False, edgecolor="crimson", lw=2.4, ls="--")
    ax_c.add_patch(rect)
    ax_c.set_xlim(-0.5, 3.5)
    ax_c.set_ylim(3.5, -0.5)
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
    ax_d.bar(
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
        ax_d.text(
            x_idx[j],
            val + offset,
            f"{val:+.3f}\n({val**2:.1%})",
            ha="center",
            fontsize=8.0,
            fontweight="bold",
        )

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
# Figura B — Correlazione geometrica locale vs Similarità coseno (Requisito 3)
# =============================================================================

def build_figure_b_cell_annotations(
    rep: LocalIdentifiabilityReport,
) -> List[List[Dict[str, Any]]]:
    """
    Costruisce in modo deterministico e verificabile la tabella 4x4 delle annotazioni
    della Figura B direttamente dalle matrici del report (`fisher_correlation_matrix`
    quando `correlation_matrix_defined == True`, altrimenti `column_cosine_similarity_matrix`):
    - Diagonali definite: sempre formattate come `+1.00` (o `+1.00\\n(diag)` per il coseno);
    - Celle fuori diagonale definite: formattate in modo coerente col JSON (usando 3 cifre
      decimali quando |val| >= 0.995 ma < 1.0, così che -0.996 non appaia mai come -1.00);
    - Celle non definite (NaN): marcate esplicitamente come `UNDEFINED\\n(Rank < 4)`.
    """
    mat = (
        rep.fisher_correlation_matrix
        if rep.correlation_matrix_defined
        else rep.column_cosine_similarity_matrix
    )
    table: List[List[Dict[str, Any]]] = []
    for i in range(4):
        row_list: List[Dict[str, Any]] = []
        for j in range(4):
            val = float(mat[i, j])
            if not np.isfinite(val):
                row_list.append(
                    {
                        "row": i,
                        "col": j,
                        "is_defined": False,
                        "is_diagonal": (i == j),
                        "numeric_value": None,
                        "formatted_number": None,
                        "display_text": "UNDEFINED\n(Rank < 4)",
                        "kind": "undefined_rank_deficient",
                    }
                )
            elif i == j:
                # Sulla diagonale di una matrice di correlazione o di similarità coseno
                # di una colonna attiva (non nulla), il valore teorico e JSON è +1.0.
                num_str = f"{val:+.2f}"
                disp = num_str if rep.correlation_matrix_defined else f"{num_str}\n(diag)"
                row_list.append(
                    {
                        "row": i,
                        "col": j,
                        "is_defined": True,
                        "is_diagonal": True,
                        "numeric_value": val,
                        "formatted_number": num_str,
                        "display_text": disp,
                        "kind": "diagonal_unit",
                    }
                )
            else:
                # Fuori diagonale: se il valore è molto vicino a +/-1 (es. -0.9963) ma < 0.9999,
                # usiamo 3 decimali per evitare che l'arrotondamento a 2 cifre mostri -1.00/+1.00.
                if 0.995 <= abs(val) < 0.9999:
                    num_str = f"{val:+.3f}"
                else:
                    num_str = f"{val:+.2f}"
                disp = num_str if rep.correlation_matrix_defined else f"{num_str}\n(cos sim)"
                row_list.append(
                    {
                        "row": i,
                        "col": j,
                        "is_defined": True,
                        "is_diagonal": False,
                        "numeric_value": val,
                        "formatted_number": num_str,
                        "display_text": disp,
                        "kind": (
                            "off_diagonal_geometric_correlation"
                            if rep.correlation_matrix_defined
                            else "off_diagonal_cosine_similarity"
                        ),
                    }
                )
        table.append(row_list)
    return table


def plot_figure_b_local_correlation(
    rep_active: LocalIdentifiabilityReport,
    rep_zero: LocalIdentifiabilityReport,
    output_path: str | Path,
) -> Dict[str, str]:
    """
    Figura B (2 pannelli):
    - Pannello (a): Matrice di correlazione geometrica locale derivata da F^{-1} = (J^T J)^{-1}
      per il protocollo a flusso attivo (rango pieno = 4);
    - Pannello (b): Matrice di similarità coseno C_{ij} = <j_i, j_j> / (||j_i|| ||j_j||)
      sulle sole colonne identificabili (alpha, D) per il protocollo a flusso nullo
      (rango = 2 < 4, in cui F = J^T J è singolare e la correlazione di Fisher è indefinita).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.0), constrained_layout=True)
    interp_tag = (
        "Approssimazione Statistica Calibrata sul Modello di Rumore Dichiarato"
        if rep_active.weighting_metadata.get("is_statistically_calibrated", False)
        else "Diagnostica Geometrica Locale di Sensibilità (Nessun Modello di Rumore Assunto)"
    )
    fig.suptitle(
        f"Figura B — Accoppiamento Parametrico Locale: Correlazione Geometrica vs Similarità Coseno\n({interp_tag})",
        fontsize=10.5,
        fontweight="bold",
    )

    panels = [
        (
            axes[0],
            rep_active,
            f"(a) Flusso Attivo (Rango = {rep_active.numerical_rank}/4)\n"
            r"Correlazione Geometrica Locale $R_{ij}$ da $(J^\top J)^{-1}$",
            "Local Geometric Correlation $R_{ij} \\in [-1, +1]$",
        ),
        (
            axes[1],
            rep_zero,
            f"(b) Flusso Nullo (Rango = {rep_zero.numerical_rank}/4 — $F=J^\\top J$ Singolare)\n"
            r"Similarità Coseno Colonnare $C_{ij} = \cos(\mathbf{j}_i, \mathbf{j}_j)$ su Sottoinsieme Attivo",
            "Column Cosine Similarity $C_{ij} \\in [-1, +1]$ ($R_{ij}$ Undefined)",
        ),
    ]

    for ax, rep, subtitle, cbar_label in panels:
        mat = (
            rep.fisher_correlation_matrix
            if rep.correlation_matrix_defined
            else rep.column_cosine_similarity_matrix
        )
        annot_table = build_figure_b_cell_annotations(rep)

        masked = np.ma.masked_invalid(mat)
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad(color="#e9ecef")
        im = ax.imshow(masked, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="equal")

        for i in range(4):
            for j in range(4):
                cell_info = annot_table[i][j]
                if cell_info["is_defined"]:
                    # Riquadro bianco ad alto contrasto con testo nero in grassetto
                    # per garantire leggibilità perfetta del segno '+' anche su rosso/blu scuro
                    ax.text(
                        j,
                        i,
                        cell_info["display_text"],
                        ha="center",
                        va="center",
                        color="#000000",
                        fontsize=9.0,
                        fontweight="bold",
                        bbox={
                            "boxstyle": "round,pad=0.22",
                            "facecolor": "white",
                            "edgecolor": "#343a40",
                            "alpha": 0.90,
                        },
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
                        cell_info["display_text"],
                        ha="center",
                        va="center",
                        color="#a94442",
                        fontsize=7.8,
                        fontweight="bold",
                        bbox={
                            "boxstyle": "round,pad=0.16",
                            "facecolor": "#f8f9fa",
                            "edgecolor": "#a94442",
                            "alpha": 0.92,
                        },
                    )

        # Blocco esplicito dei limiti degli assi dopo ax.add_patch per preservare l'orientamento matriciale
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(3.5, -0.5)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(PARAM_LATEX_NORM, fontsize=11)
        ax.set_yticks(np.arange(4))
        ax.set_yticklabels(PARAM_LATEX_NORM, fontsize=11)
        ax.set_title(subtitle, fontsize=9.8, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(cbar_label, fontsize=8.5)

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
            f"( $\\widetilde{{{pname}}} \\in [{sl_a.grid_normalized[0]:.2f}, {sl_a.grid_normalized[-1]:.2f}]$ )",
            fontsize=9.5,
            fontweight="bold",
        )
        ax.grid(True, ls=":", alpha=0.6)
        ax.legend(loc="best", fontsize=8.0)

    return _save_pdf_and_png(fig, output_path)


# =============================================================================
# Figura D — Optimized Objective Profiles (Requisiti 1, 2, 3, 5)
# =============================================================================

def plot_figure_d_optimized_profiles(
    profiles_report: OptimizedObjectiveProfilesReport,
    output_path: str | Path,
) -> Dict[str, str]:
    """
    Figura D (4 pannelli):
    - Traccia la curva continua blu SOLO quando tutti i punti del parametro sono convergenti;
      se un parametro contiene anche un solo punto non convergente, l'intera curva è
      presentata come PROVISIONAL (linea tratteggiata grigia + segmenti continui solo
      tra coppie adiacenti entrambe convergenti);
    - Evidenzia separatamente i punti non convergenti (`X` rossi), i punti incompleti per
      esaurimento del budget globale (rombi vuoti) e i parametri di disturbo al bordo (`^` arancioni);
    - Riporta il conteggio esatto delle valutazioni (setup + punti) e verifica che
      `max(nfev) <= max_evals_per_point`.
    """
    tot_conv = profiles_report.total_converged_points
    tot_pts = profiles_report.total_grid_points
    tot_incomp = profiles_report.total_incomplete_points
    is_complete = profiles_report.all_profiles_complete_and_converged

    status_banner = (
        f"COMPLETE & CONVERGED ({tot_conv}/{tot_pts} grid optimizations converged)"
        if is_complete
        else f"PROVISIONAL / PARTIALLY CONVERGED ({tot_conv}/{tot_pts} converged, {tot_incomp} incomplete)"
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.8), constrained_layout=True)
    fig.suptitle(
        f"Figura D — Optimized Objective Profiles [{status_banner}]\n"
        r"($P_j(\theta_j) = \min_{\boldsymbol{\theta}_{-j}\in\Theta_{-j}} J_{\mathrm{obj}}(\theta_j,\boldsymbol{\theta}_{-j})$ "
        f"| Per-point cap: nfev <= {profiles_report.max_evals_per_point} | "
        f"Total FEM evals: {profiles_report.total_forward_evaluations} "
        f"[{profiles_report.setup_reference_evaluations} ref + {profiles_report.total_point_evaluations} pts] "
        "— NO Statistical Profile Likelihood Assumed)",
        fontsize=10.0,
        fontweight="bold",
    )

    param_keys = ("alpha", "beta", "D", "E")
    for idx, pname in enumerate(param_keys):
        ax = axes[idx // 2, idx % 2]
        pr = profiles_report.profiles[pname]

        x_phys = pr.grid_physical
        y_oat = pr.oat_slice_cost_values
        y_prof = pr.profile_cost_values

        conv_mask = np.asarray(pr.converged_flags, dtype=bool)
        incomp_mask = np.asarray(pr.incomplete_points_mask, dtype=bool)
        bdy_mask = np.asarray(pr.nuisance_hit_bounds_mask, dtype=bool)
        non_conv_mask = (~conv_mask) & (~incomp_mask)

        # 1. Curva di riferimento One-at-a-Time (OAT)
        ax.plot(
            x_phys,
            y_oat,
            color="#6c757d",
            ls="--",
            lw=1.5,
            marker=".",
            alpha=0.80,
            label=r"OAT Cost Slice (fixed $\boldsymbol{\theta}_{-j}^\dagger$)",
        )

        # 2. Rappresentazione rigorosa della curva del profilo:
        #    presentata come profilo completo solo se `pr.all_points_converged` è True.
        if pr.all_points_converged:
            ax.plot(
                x_phys,
                y_prof,
                color="#0056b3",
                ls="-",
                lw=2.2,
                label="Optimized Objective Profile (complete, all pts converged)",
            )
        else:
            ax.plot(
                x_phys,
                y_prof,
                color="#adb5bd",
                ls=":",
                lw=1.6,
                label="Provisional profile trajectory (contains non-converged pts)",
            )
            # Disegna tratti continui esclusivamente tra punti adiacenti entrambi convergenti
            added_seg_label = False
            for k_seg in range(len(x_phys) - 1):
                if conv_mask[k_seg] and conv_mask[k_seg + 1]:
                    ax.plot(
                        x_phys[k_seg : k_seg + 2],
                        y_prof[k_seg : k_seg + 2],
                        color="#0056b3",
                        ls="-",
                        lw=2.1,
                        label="Converged profile sub-segments" if not added_seg_label else None,
                    )
                    added_seg_label = True

        # 3. Marker distinti per stato di convergenza e vincoli al bordo
        conv_interior = conv_mask & (~bdy_mask)
        conv_boundary = conv_mask & bdy_mask

        if np.any(conv_interior):
            ax.scatter(
                x_phys[conv_interior],
                y_prof[conv_interior],
                color="#0056b3",
                marker="o",
                s=52,
                zorder=4,
                label="Converged (interior nuisance optimum)",
            )
        if np.any(conv_boundary):
            ax.scatter(
                x_phys[conv_boundary],
                y_prof[conv_boundary],
                facecolors="#fd7e14",
                edgecolors="black",
                marker="^",
                s=75,
                zorder=5,
                label="Converged (nuisance param at box bound)",
            )
        if np.any(non_conv_mask):
            ax.scatter(
                x_phys[non_conv_mask],
                y_prof[non_conv_mask],
                color="#dc3545",
                marker="X",
                s=90,
                zorder=6,
                label="Provisional / Non-converged point",
            )
        if np.any(incomp_mask):
            ax.scatter(
                x_phys[incomp_mask],
                y_prof[incomp_mask],
                facecolors="none",
                edgecolors="#dc3545",
                marker="D",
                s=75,
                linewidths=2.0,
                zorder=6,
                label="Incomplete (global budget exhausted)",
            )

        ax.axvline(
            pr.reference_physical,
            color="black",
            ls=":",
            lw=1.5,
            label=rf"True $\theta^\dagger = {pr.reference_physical:.4g}$",
        )

        if np.isfinite(pr.profile_min_cost):
            min_idx = int(np.nanargmin(y_prof))
            min_is_conv = bool(conv_mask[min_idx])
            min_state_str = "converged" if min_is_conv else "PROVISIONAL"
            bdy_str = ", AT GRID BOUND" if pr.profile_minimum_at_grid_boundary else ", interior"
            ax.scatter(
                [pr.profile_argmin_physical],
                [pr.profile_min_cost],
                color="crimson" if min_is_conv else "#dc3545",
                marker="*" if min_is_conv else "P",
                s=135,
                zorder=7,
                label=f"Min ({pr.profile_argmin_physical:.4g}: {min_state_str}{bdy_str})",
            )

        conv_tag = (
            f"COMPLETE: {pr.num_converged_points}/{len(x_phys)} conv"
            if pr.all_points_converged
            else f"PROVISIONAL: {pr.num_converged_points}/{len(x_phys)} conv"
        )
        ax.set_xlabel(PARAM_LATEX_PHYS[pname], fontsize=9.5)
        ax.set_ylabel(r"Objective Value $J_{\mathrm{obj}}$ [-]", fontsize=9.0)
        ax.set_title(
            f"({chr(97 + idx)}) `{pname}` — {conv_tag} "
            f"(sum nfev={pr.total_evaluations_for_parameter}, max pt nfev={pr.max_point_evaluations_observed}/{profiles_report.max_evals_per_point})",
            fontsize=9.1,
            fontweight="bold",
        )
        ax.grid(True, ls=":", alpha=0.6)
        ax.legend(loc="best", fontsize=7.3)

    caption_text = (
        "Methodological Caption: Curves represent deterministic Optimized Objective Profiles in dimensionless observation space "
        "(NOT statistical Profile Likelihoods). "
        + (
            f"All {tot_conv}/{tot_pts} constrained nuisance optimizations satisfied genuine optimality criteria "
            f"with strict per-point cap nfev <= {profiles_report.max_evals_per_point}."
            if is_complete
            else f"PROVISIONAL STATUS: {tot_conv}/{tot_pts} points satisfied optimality criteria; "
            f"{profiles_report.total_non_converged_points} non-converged points (red X) are shown as provisional upper bounds."
        )
    )
    fig.text(
        0.5,
        -0.02,
        caption_text,
        ha="center",
        va="top",
        fontsize=8.5,
        style="italic",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8f9fa", "edgecolor": "#ced4da"},
    )

    return _save_pdf_and_png(fig, output_path)


def plot_svd_identifiability_summary(
    rep_active: LocalIdentifiabilityReport,
    rep_zero: LocalIdentifiabilityReport,
    output_path: str | Path,
) -> Dict[str, str]:
    return plot_figure_a_sensitivity_and_svd(rep_active, rep_zero, output_path)
