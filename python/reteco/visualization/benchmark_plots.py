"""
python/reteco/visualization/benchmark_plots.py
Generazione di grafici comparativi e tabelle LaTeX riproducibili per il confronto
tra CPSO e ottimizzatori SOTA a parità di budget di valutazioni FEM (Contratto Sez. 8-9).
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt

from ..inverse.cpso import OptimizationRunResult


def compute_individual_relative_errors_pct(
    theta_hat: np.ndarray,
    theta_dagger: np.ndarray,
) -> np.ndarray:
    """Calcola gli errori relativi percentuali componente per componente: 100 * |hat - true| / |true|."""
    return 100.0 * np.abs(theta_hat - theta_dagger) / np.abs(theta_dagger)


def export_sota_latex_table(
    results_by_alg: Dict[str, List[OptimizationRunResult]],
    theta_dagger: np.ndarray,
    output_tex_path: str | Path,
) -> str:
    """
    Genera una tabella LaTeX con mediana e scarto interquartile (IQR) degli errori relativi
    individuali su (alpha, beta, D, E), del costo finale J_h e del numero esatto di valutazioni FEM.
    """
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & $e_\alpha$ [\%] & $e_\beta$ [\%] & $e_D$ [\%] & $e_E$ [\%] & Final $J_h(\widehat{\boldsymbol{\theta}})$ & FEM Evals \\",
        r"\midrule",
    ]

    for alg_name, runs in results_by_alg.items():
        errs = np.array(
            [compute_individual_relative_errors_pct(r.theta_hat, theta_dagger) for r in runs]
        )
        med_err = np.median(errs, axis=0)
        costs = np.array([r.best_cost_J for r in runs])
        evals = int(np.median([r.num_fem_evaluations for r in runs]))

        row = (
            f"{alg_name} & "
            f"{med_err[0]:.2f} & {med_err[1]:.2f} & {med_err[2]:.2f} & {med_err[3]:.2f} & "
            f"{np.median(costs):.3e} & {evals} \\\\"
        )
        lines.append(row)

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    tex_content = "\n".join(lines)
    out = Path(output_tex_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex_content, encoding="utf-8")
    return tex_content


def plot_sota_convergence_by_fem_evaluations(
    results_by_alg: Dict[str, List[OptimizationRunResult]],
    output_path: str | Path,
) -> None:
    """
    Traccia il decadimento mediano di J_h(theta) in funzione del numero esatto di
    valutazioni del problema diretto FEM (e non del numero astratto di generazioni).
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=150)

    for alg_name, runs in results_by_alg.items():
        min_len = min(len(r.convergence_eval_trace) for r in runs)
        mat = np.array([r.convergence_eval_trace[:min_len] for r in runs])
        med_trace = np.median(mat, axis=0)
        eval_axis = np.arange(1, min_len + 1)
        ax.semilogy(eval_axis, med_trace, lw=2, label=alg_name)

    ax.set_xlabel("Number of Forward FEM Evaluations")
    ax.set_ylabel(r"Best Objective Functional $J_h(\boldsymbol{\theta})$")
    ax.set_title("Fair-Budget Convergence Comparison")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)