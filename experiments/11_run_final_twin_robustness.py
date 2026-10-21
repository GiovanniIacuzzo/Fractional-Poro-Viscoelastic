"""
experiments/11_run_final_twin_robustness.py
RETECO — Campagna scientifica finale: recupero parametrico e robustezza al rumore.

Rispetta integralmente il protocollo congelato:
1. Parametri veri: theta_dagger = (0.70, 0.45, 0.08, 2.50)
2. Target sintetico: (N=32, M=160), SHA-256 = 1bca9e29c60d77c1450fd2fcea845d2320c9e44945dd7d3ccda52aa33ef261af
3. Inversione: (N=16, M=80), N_obs = 180
4. CPSO: 10 particelle, 60 iterazioni, 600 valutazioni FEM
5. Baseline senza rumore (eta=0): riusa le 3 run CPSO a 600 valutazioni già disponibili in artifacts/
   senza rieseguirle, indicando la soluzione selezionata dal criterio del minimo di J_total.
6. Campagna rumorosa: 5 realizzazioni indipendenti per eta=0.05 e 5 per eta=0.10, con seed del
   rumore espliciti/distinti e seed del CPSO fisso, modalità B `exact_inverse_covariance` e lambda_B = 7.2.
7. Artefatti salvati in artifacts/11_final_twin_robustness/ senza sovrascrivere fasi precedenti.
"""

from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

from reteco import _core, load_master_config
from reteco.config_schema import ParameterCoordinateTransform
from reteco.forward.target_generator import build_standard_driving_Callbacks
from reteco.forward.wrapper import RetecoForwardProblem2D
from reteco.inverse.cpso import run_cpso_optimization
from reteco.inverse.diagnostics import (
    PARAM_NAMES,
    SingleRunDiagnosticRecord,
    extract_run_diagnostics_from_history,
    serialize_run_record,
)
from reteco.inverse.objective_functional import InverseObjectiveFunctional
from reteco.inverse.regularization import TikhonovRegularizer
from reteco.observation.noise_model import generate_gaussian_noisy_observations
from reteco.observation.operator import DiscreteObservationOperator

# =============================================================================
# 1. PARAMETRI CONGELATI DEL PROTOCOLLO
# =============================================================================
EXPECTED_THETA_TRUE = np.array([0.70, 0.45, 0.08, 2.50], dtype=np.float64)
EXPECTED_TARGET_SHA256 = "1bca9e29c60d77c1450fd2fcea845d2320c9e44945dd7d3ccda52aa33ef261af"
TARGET_N, TARGET_M = 32, 160
INVERSE_N, INVERSE_M = 16, 80
EXPECTED_N_OBS = 180

CPSO_NUM_PARTICLES = 10
CPSO_MAX_ITERATIONS = 60
PLANNED_FEM_BUDGET = CPSO_NUM_PARTICLES * CPSO_MAX_ITERATIONS  # 600

LAMBDA_A = 1.0e-4
ETA_REF = 0.05
LAMBDA_B = (EXPECTED_N_OBS / (ETA_REF ** 2)) * LAMBDA_A  # 7.2

NOISY_LEVELS: Tuple[float, float] = (0.05, 0.10)
# Seed espliciti, distinti e registrati per le 5 realizzazioni a eta=0.05 e 5 a eta=0.10
DATA_SEEDS_BY_ETA: Dict[float, Tuple[int, ...]] = {
    0.05: (2026110501, 2026110502, 2026110503, 2026110504, 2026110505),
    0.10: (2026111001, 2026111002, 2026111003, 2026111004, 2026111005),
}
DEFAULT_FIXED_CPSO_SEED = 20261017

OUT_DIR = Path("artifacts/11_final_twin_robustness")


def compute_array_sha256(arr: np.ndarray) -> str:
    """Calcola l'impronta SHA-256 canonica di un array float64 little-endian."""
    contiguous = np.ascontiguousarray(arr, dtype="<f8").ravel()
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def verify_objective_normalization_and_lambda_b_contract(
    y_clean: np.ndarray,
    transform: ParameterCoordinateTransform,
    prior_theta_tilde_0: np.ndarray,
) -> Dict[str, Any]:
    """
    Controllo bloccante richiesto al Punto 2:
    Verifica che in `objective_functional.py` la Modalità B con `exact_inverse_covariance`
    soddisfi esattamente:
        1) W^T W = Sigma^{-1} = sigma_nom^{-2} I_{N_obs}
        2) J_data^(B,cov)(theta; eta_ref) / J_data^(A)(theta) == N_obs / eta_ref^2 == 72000.0
        3) J_reg^(B)(theta) / J_data^(B,cov)(theta; eta_ref) == J_reg^(A)(theta) / J_data^(A)(theta)
           quando lambda_B = (N_obs / eta_ref^2) * lambda_A = 7.2
        4) Isolamento rigoroso di y_clean nella valutazione di Modalità B.
    In caso di discrepanza interrompe immediatamente la campagna e documenta il problema.
    """
    n_obs = int(y_clean.size)
    if n_obs != EXPECTED_N_OBS:
        raise RuntimeError(f"N_obs = {n_obs} diverso dal valore del contratto ({EXPECTED_N_OBS}).")

    reg_a = TikhonovRegularizer(transform, lambda_reg=LAMBDA_A, theta_tilde_0=prior_theta_tilde_0)
    reg_b = TikhonovRegularizer(transform, lambda_reg=LAMBDA_B, theta_tilde_0=prior_theta_tilde_0)

    sample_ref = generate_gaussian_noisy_observations(
        y_clean,
        eta_level=ETA_REF,
        seed=999001,
        weighting_mode="mode_b_noise_weighted",
        reference_scale_type="linf",
        covariance_scaling="exact_inverse_covariance",
    )

    dummy_residual = np.linspace(-0.05, 0.07, n_obs, dtype=np.float64)
    theta_probe = np.array([0.62, 0.51, 0.11, 3.10], dtype=np.float64)

    obj_a = InverseObjectiveFunctional.from_mode_a_deterministic(
        forward_operator=lambda _: y_clean + dummy_residual,
        y_observed=y_clean,
        transform=transform,
        regularizer=reg_a,
        y_reference=y_clean,
        scale_type="linf",
        max_evaluations=10,
    )
    obj_b = InverseObjectiveFunctional.from_mode_b_noise_model(
        forward_operator=lambda _: sample_ref.y_noisy + dummy_residual,
        sample=sample_ref,
        transform=transform,
        regularizer=reg_b,
        covariance_scaling="exact_inverse_covariance",
        reference_scale_type="linf",
        max_evaluations=10,
    )

    bd_a = obj_a.evaluate_breakdown(theta_probe)
    bd_b = obj_b.evaluate_breakdown(theta_probe)

    expected_scale_factor = float(n_obs / (ETA_REF ** 2))  # 72000.0
    observed_j_data_ratio = float(bd_b.j_data / bd_a.j_data)
    observed_j_reg_ratio = float(bd_b.j_reg / bd_a.j_reg)
    observed_j_total_ratio = float(bd_b.j_total / bd_a.j_total)

    cov_identity_ok = bool(obj_b.verify_exact_inverse_covariance_identity())
    isolation_report = InverseObjectiveFunctional.verify_clean_target_isolation_mode_b(
        sample_ref,
        transform,
        reg_b,
        covariance_scaling="exact_inverse_covariance",
    )
    isolation_ok = bool(isolation_report["clean_target_isolation_verified"])

    ratios_match = bool(
        np.isclose(observed_j_data_ratio, expected_scale_factor, rtol=1e-12, atol=1e-12)
        and np.isclose(observed_j_reg_ratio, expected_scale_factor, rtol=1e-12, atol=1e-12)
        and np.isclose(observed_j_total_ratio, expected_scale_factor, rtol=1e-12, atol=1e-12)
        and np.isclose(bd_a.reg_share_of_total, bd_b.reg_share_of_total, rtol=1e-12, atol=1e-14)
    )

    report = {
        "N_obs": n_obs,
        "eta_ref": ETA_REF,
        "lambda_A": LAMBDA_A,
        "lambda_B": LAMBDA_B,
        "expected_scale_factor_Nobs_over_etaref2": expected_scale_factor,
        "observed_j_data_ratio_ModeB_over_ModeA": observed_j_data_ratio,
        "observed_j_reg_ratio_ModeB_over_ModeA": observed_j_reg_ratio,
        "observed_j_total_ratio_ModeB_over_ModeA": observed_j_total_ratio,
        "mode_a_reg_share": float(bd_a.reg_share_of_total),
        "mode_b_reg_share_at_eta_ref": float(bd_b.reg_share_of_total),
        "exact_inverse_covariance_identity_verified": cov_identity_ok,
        "clean_target_isolation_report": isolation_report,
        "all_contract_checks_passed": bool(cov_identity_ok and isolation_ok and ratios_match),
    }

    if not report["all_contract_checks_passed"]:
        err_file = OUT_DIR / "BLOCKING_CONTRACT_DISCREPANCY.json"
        err_file.parent.mkdir(parents=True, exist_ok=True)
        err_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise RuntimeError(
            f"INTERRUZIONE CAMPAGNA (Punto 2): Discrepanza nella verifica di lambda_B / J_data. "
            f"Dettagli salvati in {err_file}."
        )

    return report


def _normalize_baseline_run_dict(raw: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Normalizza i campi di una run baseline caricata da JSON pregressi."""
    theta_true = np.asarray(
        raw.get("theta_true_vector", raw.get("theta_true", EXPECTED_THETA_TRUE)),
        dtype=np.float64,
    ).ravel()
    theta_hat = np.asarray(
        raw.get("theta_hat_vector", raw.get("theta_hat")),
        dtype=np.float64,
    ).ravel()
    if "individual_relative_errors_pct_vector" in raw:
        rel_errs = [float(x) for x in raw["individual_relative_errors_pct_vector"]]
    elif "individual_relative_errors_pct" in raw:
        ie = raw["individual_relative_errors_pct"]
        if isinstance(ie, dict):
            rel_errs = [float(ie[k]) for k in PARAM_NAMES]
        else:
            rel_errs = [float(x) for x in ie]
    else:
        rel_errs = [
            float(abs(theta_hat[i] - theta_true[i]) / abs(theta_true[i]) * 100.0)
            for i in range(4)
        ]

    j_data = float(raw.get("best_cost_J_data", raw.get("J_data", raw.get("j_data"))))
    j_reg = float(raw.get("best_cost_J_reg", raw.get("J_reg", raw.get("j_reg"))))
    j_total = float(raw.get("best_cost_J_total", raw.get("J_total", raw.get("j_total", j_data + j_reg))))
    w_norm = float(raw.get("weighted_residual_l2_norm", np.sqrt(max(0.0, 2.0 * j_data))))

    rel_noisy = float(
        raw.get(
            "relative_residual_vs_noisy_data",
            raw.get("relative_residual_l2", raw.get("relative_residual_vs_clean_target", 0.0)),
        )
    )
    rel_clean = float(raw.get("relative_residual_vs_clean_target", rel_noisy))

    conv_hist = raw.get("convergence_history", {})
    if not conv_hist and "history" in raw and isinstance(raw["history"], list):
        conv_hist = {
            "eval_index": [int(h.get("eval_index", i + 1)) for i, h in enumerate(raw["history"])],
            "best_j_so_far": [float(h["best_j_so_far"]) for h in raw["history"]],
            "j_total": [float(h["j_total"]) for h in raw["history"]],
            "j_data": [float(h["j_data"]) for h in raw["history"]],
            "j_reg": [float(h["j_reg"]) for h in raw["history"]],
        }
    elif not conv_hist and "convergence_eval_trace" in raw:
        trace = [float(x) for x in raw["convergence_eval_trace"]]
        conv_hist = {
            "eval_index": list(range(1, len(trace) + 1)),
            "best_j_so_far": trace,
        }

    term_cond = str(raw.get("termination_condition", "budget_exhausted_max_iterations_reached"))
    if term_cond.lower() == "converged":
        term_cond = "budget_exhausted_max_iterations_reached"

    return {
        "run_id": str(raw.get("run_id", f"baseline_eta0_run_{idx + 1:02d}")),
        "eta_level": 0.0,
        "data_seed": int(raw.get("data_seed", 20261016)),
        "optimizer_seed": int(raw["optimizer_seed"]),
        "objective_mode": str(raw.get("objective_mode", "mode_a_deterministic_normalized_linf")),
        "regularization_lambda": float(raw.get("regularization_lambda", LAMBDA_A)),
        "nominal_noise_std": 0.0,
        "effective_relative_noise_norm": 0.0,
        "theta_true_vector": theta_true.tolist(),
        "theta_hat_vector": theta_hat.tolist(),
        "individual_relative_errors_pct_vector": rel_errs,
        "best_cost_J_data": j_data,
        "best_cost_J_reg": j_reg,
        "best_cost_J_total": j_total,
        "relative_residual_vs_noisy_data": rel_noisy,
        "relative_residual_vs_clean_target": rel_clean,
        "weighted_residual_l2_norm": w_norm,
        "planned_fem_budget": int(raw.get("planned_fem_budget", PLANNED_FEM_BUDGET)),
        "optimizer_fem_evaluations": int(
            raw.get("optimizer_fem_evaluations", raw.get("num_fem_evaluations", PLANNED_FEM_BUDGET))
        ),
        "best_eval_index": int(raw.get("best_eval_index", -1)),
        "termination_condition": term_cond,
        "wall_time_seconds": float(raw.get("wall_time_seconds", 0.0)),
        "convergence_history": conv_hist,
    }


def load_existing_noise_free_baseline_runs() -> Tuple[List[Dict[str, Any]], str]:
    """
    Cerca in `artifacts/` i risultati delle 3 run CPSO già disponibili per il caso senza rumore
    (eta=0, 600 valutazioni FEM, lambda_A=1e-4) senza rieseguirle (Punto 3).
    """
    artifacts_root = Path("artifacts")
    if not artifacts_root.exists():
        raise FileNotFoundError("La cartella artifacts/ non esiste.")

    candidate_files = sorted(
        [p for p in artifacts_root.rglob("*.json") if "11_final_twin_robustness" not in str(p)],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for json_file in candidate_files:
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        lists_to_inspect: List[List[Any]] = []
        if isinstance(payload, dict):
            for k, v in payload.items():
                if isinstance(v, list) and len(v) >= 3 and all(isinstance(x, dict) for x in v):
                    lists_to_inspect.append(v)
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, list) and len(vv) >= 3 and all(isinstance(x, dict) for x in vv):
                            lists_to_inspect.append(vv)

        for cand_list in lists_to_inspect:
            matched: List[Dict[str, Any]] = []
            for item in cand_list:
                eta_val = float(item.get("eta_level", 0.0))
                evals = int(
                    item.get(
                        "optimizer_fem_evaluations",
                        item.get("num_fem_evaluations", item.get("planned_fem_budget", 0)),
                    )
                )
                has_theta = "theta_hat_vector" in item or "theta_hat" in item
                has_cost = "best_cost_J_total" in item or "J_total" in item or "j_total" in item
                if np.isclose(eta_val, 0.0) and evals == PLANNED_FEM_BUDGET and has_theta and has_cost:
                    matched.append(item)

            if len(matched) == 3:
                normalized = [_normalize_baseline_run_dict(r, i) for i, r in enumerate(matched)]
                return normalized, str(json_file)

    raise FileNotFoundError(
        "Impossibile trovare automaticamente in artifacts/ il file JSON con le 3 run CPSO "
        "senza rumore (eta=0.0, 600 valutazioni FEM). Specifica il percorso con --baseline-json."
    )


def inspect_sota_configuration() -> Dict[str, Any]:
    """
    Esamina la configurazione SOTA presente nel repository senza avviare alcuna nuova campagna (Punto 6).
    """
    sota_cfg_path = Path("configs/sota_comparison.yaml")
    info: Dict[str, Any] = {
        "sota_config_path": str(sota_cfg_path),
        "sota_config_exists": sota_cfg_path.exists(),
        "new_comparative_campaign_launched": False,
        "methodological_note": (
            "Nessuna campagna comparativa SOTA aggiuntiva è stata avviata in questa fase. "
            "Una baseline indipendente a budget equivalente (600 valutazioni FEM con protocollo "
            "congelato e funzione obiettivo identica) è necessaria soltanto qualora il paper "
            "rivendichi la superiorità del CPSO rispetto ad altri ottimizzatori. "
            "In assenza di tale confronto a budget equivalente, il manoscritto non deve formulare "
            "affermazioni di superiorità del CPSO."
        ),
    }
    if sota_cfg_path.exists():
        try:
            with open(sota_cfg_path, "r", encoding="utf-8") as fh:
                sota_yaml = yaml.safe_load(fh)
            info["configured_optimizers"] = sota_yaml.get("optimizers", sota_yaml.get("algorithms", []))
            info["configured_budget"] = sota_yaml.get("max_fem_evaluations", sota_yaml.get("budget", None))
        except Exception as exc:
            info["parse_warning"] = str(exc)
    return info


def generate_figures(
    baseline_runs: Sequence[Dict[str, Any]],
    noisy_runs_serialized: Sequence[Dict[str, Any]],
    bounds_dict: Dict[str, Tuple[float, float]],
    out_dir: Path,
) -> Tuple[Tuple[Path, Path], Tuple[Path, Path]]:
    """
    Genera le due figure richieste al Punto 5 in PNG (300 dpi) e PDF vettoriale:
    1) Recupero parametrico a eta=0, eta=0.05, eta=0.10 distinguendo i quattro parametri;
    2) Convergenza del CPSO in funzione delle valutazioni FEM (pannelli separati per caso pulito
       in Modalità A e casi rumorosi in Modalità B, date le diverse scale del funzionale).
    """
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
        }
    )

    best_base_idx = int(np.argmin([float(r["best_cost_J_total"]) for r in baseline_runs]))

    # --- Figura 1: Recupero Parametrico ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(11.2, 8.2), dpi=300)
    axes_flat = axes1.ravel()
    panel_tags = ["(a)", "(b)", "(c)", "(d)"]

    eta_005_runs = [r for r in noisy_runs_serialized if np.isclose(float(r["eta_level"]), 0.05)]
    eta_010_runs = [r for r in noisy_runs_serialized if np.isclose(float(r["eta_level"]), 0.10)]

    x_ticks = [0, 1, 2]
    x_labels = [
        r"$\eta = 0\%$ ($n=3$)" + "\n[Modalità A]",
        r"$\eta = 5\%$ ($n=5$)" + "\n[Modalità B]",
        r"$\eta = 10\%$ ($n=5$)" + "\n[Modalità B]",
    ]

    for p_idx, p_name in enumerate(PARAM_NAMES):
        ax = axes_flat[p_idx]
        true_val = float(EXPECTED_THETA_TRUE[p_idx])
        ax.axhline(
            true_val,
            color="#111111",
            linestyle="--",
            linewidth=1.4,
            label=r"Valore vero $\theta^\dagger$" if p_idx == 0 else None,
            zorder=2,
        )

        # Gruppo eta = 0 (3 run, distinguendo la soluzione selezionata da min J_total)
        vals_0 = np.array([float(r["theta_hat_vector"][p_idx]) for r in baseline_runs], dtype=np.float64)
        offsets_0 = np.linspace(-0.10, 0.10, vals_0.size)
        for i_b, v_b in enumerate(vals_0):
            is_sel = i_b == best_base_idx
            ax.scatter(
                x_ticks[0] + offsets_0[i_b],
                v_b,
                marker="*" if is_sel else "o",
                s=110 if is_sel else 48,
                color="#0072B2" if is_sel else "#56B4E9",
                edgecolors="black",
                linewidths=0.8,
                zorder=5 if is_sel else 4,
                label=(
                    r"Baseline $\eta=0\%$ ($\min J_{\mathrm{total}}$)"
                    if (p_idx == 0 and is_sel)
                    else (r"Baseline $\eta=0\%$ (altre run)" if (p_idx == 0 and i_b == (1 if best_base_idx == 0 else 0)) else None)
                ),
            )
        med_0 = float(np.median(vals_0))
        ax.errorbar(
            x_ticks[0],
            med_0,
            yerr=[[med_0 - float(np.min(vals_0))], [float(np.max(vals_0)) - med_0]],
            fmt="none",
            ecolor="#0072B2",
            elinewidth=1.8,
            capsize=5,
            zorder=3,
        )

        # Gruppi rumorosi eta = 0.05 e eta = 0.10
        for g_idx, (sub_runs, col, lbl) in enumerate(
            [
                (eta_005_runs, "#E69F00", r"Realizzazioni $\eta=5\%$ ($n=5$)"),
                (eta_010_runs, "#CC79A7", r"Realizzazioni $\eta=10\%$ ($n=5$)"),
            ],
            start=1,
        ):
            vals = np.array([float(r["theta_hat_vector"][p_idx]) for r in sub_runs], dtype=np.float64)
            offsets = np.linspace(-0.12, 0.12, vals.size)
            ax.scatter(
                np.full(vals.size, x_ticks[g_idx]) + offsets,
                vals,
                marker="s" if g_idx == 1 else "^",
                s=52,
                color=col,
                edgecolors="black",
                linewidths=0.7,
                alpha=0.9,
                zorder=4,
                label=lbl if p_idx == 0 else None,
            )
            med_v = float(np.median(vals))
            min_v, max_v = float(np.min(vals)), float(np.max(vals))
            ax.errorbar(
                x_ticks[g_idx],
                med_v,
                yerr=[[med_v - min_v], [max_v - med_v]],
                fmt="D",
                 markersize=5,
                color="black",
                ecolor=col,
                elinewidth=2.0,
                capsize=5,
                zorder=5,
                label="Mediana e range [min, max]" if (p_idx == 0 and g_idx == 1) else None,
            )

        if p_name in bounds_dict:
            b_low, b_high = bounds_dict[p_name]
            span = b_high - b_low
            ax.set_ylim(b_low - 0.06 * span, b_high + 0.06 * span)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_title(f"{panel_tags[p_idx]} Parametro `{p_name}` (vero = {true_val:g})")
        ax.set_ylabel(f"Stima `{p_name}`")
        ax.grid(True, linestyle=":", alpha=0.55)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig1.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig1.suptitle(
        "RETECO — Recupero parametrico per livello di rumore (singole realizzazioni, mediana e min–max)",
        fontsize=12.5,
        y=0.995,
    )
    fig1.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
    fig1_png = out_dir / "fig_final_parameter_recovery.png"
    fig1_pdf = out_dir / "fig_final_parameter_recovery.pdf"
    fig1.savefig(fig1_png, dpi=300)
    fig1.savefig(fig1_pdf)
    plt.close(fig1)

    # --- Figura 2: Convergenza CPSO (pannelli separati per scale non direttamente comparabili) ---
    fig2, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=300)

    # Pannello (a): Caso senza rumore (Modalità A, lambda_A = 1e-4)
    for idx, r in enumerate(baseline_runs):
        is_sel = idx == best_base_idx
        hist = r.get("convergence_history", {})
        best_j = hist.get("best_j_so_far", [])
        evals = hist.get("eval_index", list(range(1, len(best_j) + 1)))
        if best_j:
            lbl = (
                f"Run {idx + 1} (seed {r['optimizer_seed']}) — Selezionata"
                if is_sel
                else f"Run {idx + 1} (seed {r['optimizer_seed']})"
            )
            ax_a.plot(
                evals,
                best_j,
                linewidth=2.2 if is_sel else 1.3,
                linestyle="-" if is_sel else "--",
                color="#0072B2" if is_sel else ("#56B4E9" if idx == 1 else "#009E73"),
                label=lbl,
            )
            ax_a.scatter([evals[-1]], [best_j[-1]], s=30, zorder=5)

    ax_a.set_yscale("log")
    ax_a.set_xlim(1, PLANNED_FEM_BUDGET + 20)
    ax_a.set_xlabel("Valutazioni FEM")
    ax_a.set_ylabel(r"Miglior costo $J_{\mathrm{total}}^{(A)}$ (scala log)")
    ax_a.set_title(r"(a) Baseline senza rumore ($\eta=0\%$, $\lambda_A=10^{-4}$)")
    ax_a.grid(True, which="both", linestyle=":", alpha=0.55)
    ax_a.legend(loc="upper right", frameon=True)

    # Pannello (b): Campagna rumorosa (Modalità B covarianza inversa esatta, lambda_B = 7.2)
    for sub_runs, col, lbl in [
        (eta_005_runs, "#E69F00", r"$\eta = 5\%$"),
        (eta_010_runs, "#CC79A7", r"$\eta = 10\%$"),
    ]:
        traces = []
        evals_vec = None
        for r_idx, r in enumerate(sub_runs):
            hist = r.get("convergence_history", {})
            best_j = hist.get("best_j_so_far", [])
            evals = hist.get("eval_index", list(range(1, len(best_j) + 1)))
            if best_j:
                traces.append(best_j)
                evals_vec = evals
                ax_b.plot(
                    evals,
                    best_j,
                    color=col,
                    alpha=0.32,
                    linewidth=1.0,
                    label=f"Realizzazioni {lbl} ($n=5$)" if r_idx == 0 else None,
                )
        if traces and evals_vec is not None:
            arr_t = np.asarray(traces, dtype=np.float64)
            med_trace = np.median(arr_t, axis=0)
            ax_b.plot(
                evals_vec,
                med_trace,
                color=col,
                linewidth=2.3,
                linestyle="-",
                label=f"Mediana {lbl}",
            )
            ax_b.scatter([evals_vec[-1]], [med_trace[-1]], color=col, edgecolors="black", s=36, zorder=5)

    ax_b.set_yscale("log")
    ax_b.set_xlim(1, PLANNED_FEM_BUDGET + 20)
    ax_b.set_xlabel("Valutazioni FEM")
    ax_b.set_ylabel(r"Miglior costo $J_{\mathrm{total}}^{(B,\mathrm{cov})}$ (scala log)")
    ax_b.set_title(r"(b) Campagna rumorosa ($\eta \in \{5\%, 10\%\}$, $\lambda_B=7.2$)")
    ax_b.grid(True, which="both", linestyle=":", alpha=0.55)
    ax_b.legend(loc="upper right", frameon=True)

    fig2.suptitle(
        "RETECO — Cronologia del miglior costo CPSO (600 valutazioni FEM per run; arresto per esaurimento budget)",
        fontsize=12,
    )
    fig2.tight_layout()
    fig2_png = out_dir / "fig_final_cpso_convergence.png"
    fig2_pdf = out_dir / "fig_final_cpso_convergence.pdf"
    fig2.savefig(fig2_png, dpi=300)
    fig2.savefig(fig2_pdf)
    plt.close(fig2)

    return (fig1_png, fig1_pdf), (fig2_png, fig2_pdf)


def write_csv_and_markdown_report(
    baseline_runs: Sequence[Dict[str, Any]],
    noisy_runs_serialized: Sequence[Dict[str, Any]],
    norm_report: Dict[str, Any],
    sota_info: Dict[str, Any],
    baseline_source_path: str,
    fixed_cpso_seed: int,
    csv_path: Path,
    md_path: Path,
) -> Dict[str, Any]:
    """
    Genera il CSV con una riga per realizzazione (3 baseline + 10 rumorose = 13 righe)
    e il report Markdown con tutte le tabelle, la dispersione (mediana, min, max) e le note metodologiche.
    """
    fieldnames = [
        "run_id",
        "campaign_group",
        "selected_by_functional_criterion",
        "objective_mode",
        "regularization_lambda",
        "eta_level",
        "data_seed",
        "optimizer_seed",
        "nominal_noise_std",
        "effective_relative_noise_norm",
        "alpha_true",
        "beta_true",
        "D_true",
        "E_true",
        "alpha_hat",
        "beta_hat",
        "D_hat",
        "E_hat",
        "err_rel_alpha_pct",
        "err_rel_beta_pct",
        "err_rel_D_pct",
        "err_rel_E_pct",
        "J_data",
        "J_reg",
        "J_total",
        "relative_residual_vs_noisy_data",
        "relative_residual_vs_clean_target",
        "weighted_residual_l2_norm",
        "planned_fem_budget",
        "effective_fem_evaluations",
        "best_eval_index",
        "termination_condition",
        "wall_time_seconds",
    ]

    best_baseline_idx = int(np.argmin([float(r["best_cost_J_total"]) for r in baseline_runs]))

    rows: List[Dict[str, Any]] = []
    for idx, b in enumerate(baseline_runs):
        t_true = b["theta_true_vector"]
        t_hat = b["theta_hat_vector"]
        errs = b["individual_relative_errors_pct_vector"]
        rows.append(
            {
                "run_id": b["run_id"],
                "campaign_group": "baseline_eta_0.00",
                "selected_by_functional_criterion": bool(idx == best_baseline_idx),
                "objective_mode": b["objective_mode"],
                "regularization_lambda": float(b["regularization_lambda"]),
                "eta_level": 0.0,
                "data_seed": int(b["data_seed"]),
                "optimizer_seed": int(b["optimizer_seed"]),
                "nominal_noise_std": 0.0,
                "effective_relative_noise_norm": 0.0,
                "alpha_true": float(t_true[0]),
                "beta_true": float(t_true[1]),
                "D_true": float(t_true[2]),
                "E_true": float(t_true[3]),
                "alpha_hat": float(t_hat[0]),
                "beta_hat": float(t_hat[1]),
                "D_hat": float(t_hat[2]),
                "E_hat": float(t_hat[3]),
                "err_rel_alpha_pct": float(errs[0]),
                "err_rel_beta_pct": float(errs[1]),
                "err_rel_D_pct": float(errs[2]),
                "err_rel_E_pct": float(errs[3]),
                "J_data": float(b["best_cost_J_data"]),
                "J_reg": float(b["best_cost_J_reg"]),
                "J_total": float(b["best_cost_J_total"]),
                "relative_residual_vs_noisy_data": float(b["relative_residual_vs_noisy_data"]),
                "relative_residual_vs_clean_target": float(b["relative_residual_vs_clean_target"]),
                "weighted_residual_l2_norm": float(b["weighted_residual_l2_norm"]),
                "planned_fem_budget": int(b["planned_fem_budget"]),
                "effective_fem_evaluations": int(b["optimizer_fem_evaluations"]),
                "best_eval_index": int(b["best_eval_index"]),
                "termination_condition": str(b["termination_condition"]),
                "wall_time_seconds": float(b["wall_time_seconds"]),
            }
        )

    for r in noisy_runs_serialized:
        t_true = r["theta_true_vector"]
        t_hat = r["theta_hat_vector"]
        errs = r["individual_relative_errors_pct_vector"]
        rows.append(
            {
                "run_id": r["run_id"],
                "campaign_group": f"noisy_eta_{float(r['eta_level']):.2f}",
                "selected_by_functional_criterion": False,
                "objective_mode": r["objective_mode"],
                "regularization_lambda": LAMBDA_B,
                "eta_level": float(r["eta_level"]),
                "data_seed": int(r["data_seed"]),
                "optimizer_seed": int(r["optimizer_seed"]),
                "nominal_noise_std": float(r["nominal_noise_std"]),
                "effective_relative_noise_norm": float(r["effective_relative_noise_norm"]),
                "alpha_true": float(t_true[0]),
                "beta_true": float(t_true[1]),
                "D_true": float(t_true[2]),
                "E_true": float(t_true[3]),
                "alpha_hat": float(t_hat[0]),
                "beta_hat": float(t_hat[1]),
                "D_hat": float(t_hat[2]),
                "E_hat": float(t_hat[3]),
                "err_rel_alpha_pct": float(errs[0]),
                "err_rel_beta_pct": float(errs[1]),
                "err_rel_D_pct": float(errs[2]),
                "err_rel_E_pct": float(errs[3]),
                "J_data": float(r["best_cost_J_data"]),
                "J_reg": float(r["best_cost_J_reg"]),
                "J_total": float(r["best_cost_J_total"]),
                "relative_residual_vs_noisy_data": float(r["relative_residual_vs_noisy_data"]),
                "relative_residual_vs_clean_target": float(r["relative_residual_vs_clean_target"]),
                "weighted_residual_l2_norm": float(r["weighted_residual_l2_norm"]),
                "planned_fem_budget": int(r["planned_fem_budget"]),
                "effective_fem_evaluations": int(r["optimizer_fem_evaluations"]),
                "best_eval_index": int(r["best_eval_index"]),
                "termination_condition": str(r["termination_condition"]),
                "wall_time_seconds": float(r["wall_time_seconds"]),
            }
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Calcolo statistiche descrittive (mediana, min, max) per livello
    dispersion_summary: Dict[str, Any] = {}
    for eta_val in (0.0, 0.05, 0.10):
        sub = [row for row in rows if np.isclose(float(row["eta_level"]), eta_val)]
        eta_key = f"eta_{eta_val:.2f}"
        dispersion_summary[eta_key] = {"n_runs": len(sub), "parameters": {}}
        for p_name in PARAM_NAMES:
            hats = np.array([float(row[f"{p_name}_hat"]) for row in sub], dtype=np.float64)
            errs = np.array([float(row[f"err_rel_{p_name}_pct"]) for row in sub], dtype=np.float64)
            dispersion_summary[eta_key]["parameters"][p_name] = {
                "estimate_median": float(np.median(hats)),
                "estimate_min": float(np.min(hats)),
                "estimate_max": float(np.max(hats)),
                "rel_error_pct_median": float(np.median(errs)),
                "rel_error_pct_min": float(np.min(errs)),
                "rel_error_pct_max": float(np.max(errs)),
            }

    # Costruzione del Report Markdown
    md = [
        "# RETECO — Campagna Finale: Recupero Parametrico e Robustezza al Rumore (`11`)",
        "",
        "## 1. Protocollo Scientifico Congelato e Verifica del Contratto",
        f"- **Parametri veri (`theta_dagger`)**: $(\\alpha^\\dagger, \\beta^\\dagger, D^\\dagger, E^\\dagger) = ({EXPECTED_THETA_TRUE[0]:.2f}, {EXPECTED_THETA_TRUE[1]:.2f}, {EXPECTED_THETA_TRUE[2]:.2f}, {EXPECTED_THETA_TRUE[3]:.2f})$",
        f"- **Target sintetico**: $(N={TARGET_N}, M={TARGET_M})$, $N_{{\\mathrm{{obs}}}} = {EXPECTED_N_OBS}$, SHA-256 verificato `{EXPECTED_TARGET_SHA256}`",
        f"- **Discretizzazione di inversione**: $(N={INVERSE_N}, M={INVERSE_M})$",
        f"- **Configurazione CPSO**: `{CPSO_NUM_PARTICLES}` particelle, `{CPSO_MAX_ITERATIONS}` iterazioni, `{PLANNED_FEM_BUDGET}` valutazioni FEM per run",
        f"- **Seed CPSO fisso per le 10 nuove run rumorose**: `{fixed_cpso_seed}` (mantenuto fisso per isolare l'effetto delle realizzazioni di rumore)",
        f"- **Conversione della regolarizzazione verificata**: $\\lambda_A = {LAMBDA_A:.1e}$ ($\\eta=0$), $\\lambda_B = \\frac{{N_{{\\mathrm{{obs}}}}}}{{\\eta_{{\\mathrm{{ref}}}}^2}}\\lambda_A = {LAMBDA_B:.1f}$ ($\\eta \\in \\{{0.05, 0.10\\}}$, rapporto verificato `J_data^(B,cov)/J_data^(A) = {norm_report['observed_j_data_ratio_ModeB_over_ModeA']:.1f}`)",
        "",
        "## 2. Baseline Senza Rumore ($\\eta = 0$, 3 Run Riutilizzate)",
        f"Sorgente riutilizzata senza riesecuzione: `{baseline_source_path}`. La run con il costo totale $J_{{\\mathrm{{total}}}}$ più basso è indicata come soluzione selezionata dal criterio del funzionale.",
        "",
        "| Run ID | Seed CPSO | Selezionata ($\\min J_{\\mathrm{total}}$) | $\\hat{\\alpha}$ (Err %) | $\\hat{\\beta}$ (Err %) | $\\hat{D}$ (Err %) | $\\hat{E}$ (Err %) | $J_{\\mathrm{data}}$ | $J_{\\mathrm{reg}}$ | $J_{\\mathrm{total}}$ | Res. Rel. Pulito | Norma Pesata | Best Eval | Stato Arresto |",
        "|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for r in rows[:3]:
        sel = "**SÌ**" if r["selected_by_functional_criterion"] else "No"
        md.append(
            f"| `{r['run_id']}` | `{r['optimizer_seed']}` | {sel} | "
            f"`{r['alpha_hat']:.4f}` (`{r['err_rel_alpha_pct']:.2f}%`) | "
            f"`{r['beta_hat']:.4f}` (`{r['err_rel_beta_pct']:.2f}%`) | "
            f"`{r['D_hat']:.5f}` (`{r['err_rel_D_pct']:.2f}%`) | "
            f"`{r['E_hat']:.4f}` (`{r['err_rel_E_pct']:.2f}%`) | "
            f"`{r['J_data']:.4e}` | `{r['J_reg']:.4e}` | `{r['J_total']:.4e}` | "
            f"`{r['relative_residual_vs_clean_target']:.4e}` | `{r['weighted_residual_l2_norm']:.4e}` | "
            f"`{r['best_eval_index']}/{r['effective_fem_evaluations']}` | `{r['termination_condition']}` |"
        )

    md.extend(
        [
            "",
            "## 3. Campagna Rumorosa ($\\eta \\in \\{0.05, 0.10\\}$, 10 Nuove Run con Covarianza Inversa Esatta)",
            "| Run ID | $\\eta$ | Seed Rumore | Seed CPSO | $\\|\\epsilon\\|_2/\\|y_{\\mathrm{clean}}\\|_2$ | $\\hat{\\alpha}$ (Err %) | $\\hat{\\beta}$ (Err %) | $\\hat{D}$ (Err %) | $\\hat{E}$ (Err %) | $J_{\\mathrm{data}}$ | $J_{\\mathrm{reg}}$ | $J_{\\mathrm{total}}$ | Res. Rel. Noisy | Res. Rel. Clean | Norma Pesata | Best Eval | Tempo [s] | Stato Arresto |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    for r in rows[3:]:
        md.append(
            f"| `{r['run_id']}` | `{r['eta_level']:.2f}` | `{r['data_seed']}` | `{r['optimizer_seed']}` | "
            f"`{r['effective_relative_noise_norm']:.4e}` | "
            f"`{r['alpha_hat']:.4f}` (`{r['err_rel_alpha_pct']:.2f}%`) | "
            f"`{r['beta_hat']:.4f}` (`{r['err_rel_beta_pct']:.2f}%`) | "
            f"`{r['D_hat']:.5f}` (`{r['err_rel_D_pct']:.2f}%`) | "
            f"`{r['E_hat']:.4f}` (`{r['err_rel_E_pct']:.2f}%`) | "
            f"`{r['J_data']:.4e}` | `{r['J_reg']:.4e}` | `{r['J_total']:.4e}` | "
            f"`{r['relative_residual_vs_noisy_data']:.4e}` | `{r['relative_residual_vs_clean_target']:.4e}` | "
            f"`{r['weighted_residual_l2_norm']:.4e}` | "
            f"`{r['best_eval_index']}/{r['effective_fem_evaluations']}` | `{r['wall_time_seconds']:.1f}` | "
            f"`{r['termination_condition']}` |"
        )

    md.extend(
        [
            "",
            "## 4. Dispersione Osservata per Parametro e Livello di Rumore (Mediana, Minimo, Massimo)",
            "> **Nota metodologica**: Con 3 realizzazioni per il caso senza rumore e 5 realizzazioni indipendenti per ciascun livello di rumore ($\\eta=0.05$ e $\\eta=0.10$), vengono riportati esclusivamente mediana, minimo e massimo osservati per descrivere la dispersione empirica del campione, senza intervalli di confidenza parametrici stretti né conclusioni statistiche forti.",
            "",
            "| Livello $\\eta$ | $n$ | Parametro | Valore Vero | Stima Mediana `[Min, Max]` | Errore Relativo [%] Mediana `[Min, Max]` |",
            "|---:|---:|---|---:|---:|---:|",
        ]
    )

    for eta_val in (0.0, 0.05, 0.10):
        eta_key = f"eta_{eta_val:.2f}"
        n_r = dispersion_summary[eta_key]["n_runs"]
        for p_idx, p_name in enumerate(PARAM_NAMES):
            p_stat = dispersion_summary[eta_key]["parameters"][p_name]
            md.append(
                f"| `{eta_val:.2f}` | `{n_r}` | `{p_name}` | `{EXPECTED_THETA_TRUE[p_idx]:g}` | "
                f"`{p_stat['estimate_median']:.4f} [{p_stat['estimate_min']:.4f}, {p_stat['estimate_max']:.4f}]` | "
                f"`{p_stat['rel_error_pct_median']:.2f}% [{p_stat['rel_error_pct_min']:.2f}%, {p_stat['rel_error_pct_max']:.2f}%]` |"
            )

    md.extend(
        [
            "",
            "## 5. Nota sul Confronto con Altri Ottimizzatori (Punto 6)",
            f"- {sota_info['methodological_note']}",
        ]
    )

    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return dispersion_summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="RETECO — Campagna finale recupero parametrico e robustezza al rumore."
    )
    parser.add_argument(
        "--baseline-json",
        type=str,
        default=None,
        help="Percorso opzionale al JSON delle 3 run baseline senza rumore a 600 valutazioni.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Rigenera CSV, Markdown e figure a partire dal JSON già salvato in artifacts/11_final_twin_robustness/.",
    )
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "final_twin_robustness_protocol_and_results.json"
    csv_path = OUT_DIR / "final_twin_robustness_runs.csv"
    md_path = OUT_DIR / "final_twin_robustness_report.md"

    master = load_master_config("configs/scales_and_physics.yaml")
    raw_bounds = master.parameter_space_theta.bounds
    bounds_dict = {
        name: (float(getattr(raw_bounds, name)[0]), float(getattr(raw_bounds, name)[1]))
        if hasattr(raw_bounds, name)
        else (float(raw_bounds[name][0]), float(raw_bounds[name][1]))
        for name in PARAM_NAMES
    }

    if args.plots_only:
        if not json_path.exists():
            raise FileNotFoundError(f"Impossibile usare --plots-only: {json_path} non esiste.")
        saved = json.loads(json_path.read_text(encoding="utf-8"))
        baseline_runs = saved["noise_free_baseline_runs"]
        noisy_runs_serialized = saved["noisy_campaign_runs"]
        norm_report = saved["normalization_contract_verification"]
        sota_info = saved["sota_comparison_inspection"]
        baseline_source_path = saved["noise_free_baseline_source"]
        fixed_cpso_seed = int(saved["frozen_protocol"]["cpso"]["fixed_optimizer_seed_noisy_campaign"])

        write_csv_and_markdown_report(
            baseline_runs=baseline_runs,
            noisy_runs_serialized=noisy_runs_serialized,
            norm_report=norm_report,
            sota_info=sota_info,
            baseline_source_path=baseline_source_path,
            fixed_cpso_seed=fixed_cpso_seed,
            csv_path=csv_path,
            md_path=md_path,
        )
        generate_figures(
            baseline_runs=baseline_runs,
            noisy_runs_serialized=noisy_runs_serialized,
            bounds_dict=bounds_dict,
            out_dir=OUT_DIR,
        )
        print(f"[11] Figure e tabelle rigenerate da {json_path} senza rieseguire il solver FEM.")
        return 0

    with open("configs/twin_experiment_synthetic.yaml", "r", encoding="utf-8") as fh:
        twin_cfg = yaml.safe_load(fh)

    theta_dagger = master.parameter_space_theta.theta_dagger_array()
    if not np.allclose(theta_dagger, EXPECTED_THETA_TRUE, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Parametri veri {theta_dagger} diversi dal protocollo congelato {EXPECTED_THETA_TRUE}."
        )

    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    prior_tilde_0 = np.array(
        twin_cfg["noise_and_regularization"]["prior_theta_tilde_0"], dtype=np.float64
    )

    obs_cfg = twin_cfg["observation_protocol"]
    obs_op = DiscreteObservationOperator.from_regular_grid(
        nx=int(obs_cfg["sensor_grid_nx"]),
        ny=int(obs_cfg["sensor_grid_ny"]),
        margin=float(obs_cfg["sensor_margin"]),
        observation_times=obs_cfg["observation_times"],
        observed_fields=obs_cfg["observed_fields"],
    )

    # 1. Generazione e verifica SHA-256 del target pulito a (N=32, M=160)
    print("[11] 1/5 Generazione target sintetico pulito a (N=32, M=160) e verifica SHA-256...")
    t_tar0 = time.perf_counter()
    tar_cfg = twin_cfg["target_discretization"]
    if int(tar_cfg["subdivisions_N"]) != TARGET_N or int(tar_cfg["time_steps_M"]) != TARGET_M:
        raise ValueError("La discretizzazione target in YAML non coincide con (N=32, M=160).")

    grid_tar = _core.TimeGrid.make_graded(
        float(tar_cfg["final_time_T"]), TARGET_M, float(tar_cfg["grading_r"])
    )
    prob_tar = RetecoForwardProblem2D(
        subdivisions_N=TARGET_N,
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid_tar,
    )
    c0_tar, F_tar, Sf_tar = build_standard_driving_Callbacks(
        prob_tar, twin_cfg["driving_protocol"]
    )
    traj_tar = prob_tar.solve_trajectory(
        theta_dagger, c0=c0_tar, F_mech_func=F_tar, S_fluid_func=Sf_tar
    )
    y_target_clean = np.asarray(obs_op.apply(prob_tar, traj_tar), dtype=np.float64).ravel()
    target_wall_s = time.perf_counter() - t_tar0

    target_sha256 = compute_array_sha256(y_target_clean)
    if target_sha256 != EXPECTED_TARGET_SHA256:
        raise RuntimeError(
            f"SHA-256 del target pulito ({target_sha256}) non coincide con l'hash congelato "
            f"({EXPECTED_TARGET_SHA256})."
        )
    print(f"     Target SHA-256 verificato: {target_sha256} ({target_wall_s:.2f} s)")

    # 2. Verifica bloccante della normalizzazione di objective_functional.py e lambda_B = 7.2
    print("[11] 2/5 Verifica del contratto di normalizzazione (lambda_B = 7.2)...")
    norm_report = verify_objective_normalization_and_lambda_b_contract(
        y_clean=y_target_clean,
        transform=transform,
        prior_theta_tilde_0=prior_tilde_0,
    )
    print("     Contratto matematico e isolamento target pulito in Modalità B verificati.")

    # 3. Caricamento delle 3 run baseline senza rumore senza rieseguirle
    print("[11] 3/5 Caricamento delle 3 run CPSO baseline (eta=0) esistenti...")
    if args.baseline_json is not None:
        p_base = Path(args.baseline_json)
        payload = json.loads(p_base.read_text(encoding="utf-8"))
        raw_list = payload.get("runs", payload.get("detailed_runs", payload.get("noise_free_baseline_runs", [])))
        baseline_runs = [_normalize_baseline_run_dict(r, i) for i, r in enumerate(raw_list[:3])]
        baseline_source_path = str(p_base)
    else:
        baseline_runs, baseline_source_path = load_existing_noise_free_baseline_runs()

    fixed_cpso_seed = int(baseline_runs[0].get("optimizer_seed", DEFAULT_FIXED_CPSO_SEED))
    print(
        f"     Caricate 3 run baseline da {baseline_source_path}. "
        f"Seed CPSO fisso per la campagna rumorosa: {fixed_cpso_seed}"
    )

    # 4. Inizializzazione problema inverso a (N=16, M=80) ed esecuzione delle 10 run rumorose
    inv_cfg = twin_cfg["inverse_discretization"]
    if int(inv_cfg["subdivisions_N"]) != INVERSE_N or int(inv_cfg["time_steps_M"]) != INVERSE_M:
        raise ValueError("La discretizzazione inversa in YAML non coincide con (N=16, M=80).")

    grid_inv = _core.TimeGrid.make_graded(
        float(inv_cfg["final_time_T"]), INVERSE_M, float(inv_cfg["grading_r"])
    )
    prob_inv = RetecoForwardProblem2D(
        subdivisions_N=INVERSE_N,
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid_inv,
    )
    c0_inv, F_inv, Sf_inv = build_standard_driving_Callbacks(
        prob_inv, twin_cfg["driving_protocol"]
    )

    def G_inv(theta: np.ndarray) -> np.ndarray:
        tr = prob_inv.solve_trajectory(
            theta, c0=c0_inv, F_mech_func=F_inv, S_fluid_func=Sf_inv
        )
        return np.asarray(obs_op.apply(prob_inv, tr), dtype=np.float64).ravel()

    reg_b = TikhonovRegularizer(
        transform,
        lambda_reg=LAMBDA_B,
        theta_tilde_0=prior_tilde_0,
    )

    noisy_records: List[SingleRunDiagnosticRecord] = []
    noisy_runs_serialized: List[Dict[str, Any]] = []
    mandatory_verifications: Dict[str, Any] = {}
    seen_noise_hashes: Dict[str, str] = {}
    failed_runs: List[Dict[str, Any]] = []

    print("[11] 4/5 Avvio delle 10 run rumorose (5 per eta=0.05, 5 per eta=0.10)...")
    run_counter = 0
    for eta in NOISY_LEVELS:
        for rep_idx, data_seed in enumerate(DATA_SEEDS_BY_ETA[eta], start=1):
            run_counter += 1
            run_id = f"final_eta_{int(round(eta * 100)):02d}pct_rep_{rep_idx:02d}_dseed_{data_seed}"
            print(
                f"  -> [{run_counter:02d}/10] Esecuzione {run_id} "
                f"(eta={eta:.2f}, data_seed={data_seed}, cpso_seed={fixed_cpso_seed})..."
            )
            try:
                sample = generate_gaussian_noisy_observations(
                    y_clean=y_target_clean,
                    eta_level=eta,
                    seed=data_seed,
                    weighting_mode="mode_b_noise_weighted",
                    reference_scale_type="linf",
                    covariance_scaling="exact_inverse_covariance",
                )
                sample_repro = generate_gaussian_noisy_observations(
                    y_clean=y_target_clean,
                    eta_level=eta,
                    seed=data_seed,
                    weighting_mode="mode_b_noise_weighted",
                    reference_scale_type="linf",
                    covariance_scaling="exact_inverse_covariance",
                )
                if not np.array_equal(sample.noise_vector, sample_repro.noise_vector):
                    raise RuntimeError(f"Il seed del rumore {data_seed} non è riproducibile.")

                clean_hash_run = compute_array_sha256(sample.y_clean)
                if clean_hash_run != EXPECTED_TARGET_SHA256:
                    raise RuntimeError("Il target pulito è mutato tra le realizzazioni.")

                noise_hash = compute_array_sha256(sample.noise_vector)
                if noise_hash in seen_noise_hashes:
                    raise RuntimeError(
                        f"Vettore di rumore duplicato tra {run_id} e {seen_noise_hashes[noise_hash]}."
                    )
                seen_noise_hashes[noise_hash] = run_id

                obj = InverseObjectiveFunctional.from_mode_b_noise_model(
                    forward_operator=G_inv,
                    sample=sample,
                    transform=transform,
                    regularizer=reg_b,
                    covariance_scaling="exact_inverse_covariance",
                    reference_scale_type="linf",
                    max_evaluations=PLANNED_FEM_BUDGET,
                )
                if not obj.verify_exact_inverse_covariance_identity():
                    raise RuntimeError(f"{run_id}: W^T W != Sigma^(-1).")

                t0 = time.perf_counter()
                opt_res = run_cpso_optimization(
                    obj,
                    seed=fixed_cpso_seed,
                    num_particles=CPSO_NUM_PARTICLES,
                    max_iterations=CPSO_MAX_ITERATIONS,
                )
                wall_s = time.perf_counter() - t0

                hist_checks = obj.verify_history_and_best_candidate_consistency(
                    opt_res, raise_on_failure=True
                )

                rec = extract_run_diagnostics_from_history(
                    run_id=run_id,
                    obj=obj,
                    opt_result=opt_res,
                    theta_true=theta_dagger,
                    y_target_clean=y_target_clean,
                    y_noisy=sample.y_noisy,
                    nominal_noise_std=sample.sigma_std,
                    eta_level=eta,
                    data_seed=data_seed,
                    optimizer_seed=fixed_cpso_seed,
                    planned_fem_budget=PLANNED_FEM_BUDGET,
                    num_particles=CPSO_NUM_PARTICLES,
                    max_iterations=CPSO_MAX_ITERATIONS,
                    wall_time_seconds=wall_s,
                    forward_operator_inv=G_inv,
                    transform=transform,
                    regularizer=reg_b,
                    seed_reproducibility_verified=True,
                )

                if str(rec.termination_condition).lower() == "converged":
                    raise RuntimeError(
                        f"{run_id}: Dichiarata convergente pur avendo raggiunto il limite delle valutazioni."
                    )

                rec_dict = serialize_run_record(rec)
                noisy_records.append(rec)
                noisy_runs_serialized.append(rec_dict)
                mandatory_verifications[run_id] = {
                    "clean_target_sha256_matches": True,
                    "noise_vector_sha256": noise_hash,
                    "noise_seed_reproducible": True,
                    "exact_inverse_covariance_verified": True,
                    "history_and_monotone_best_cost_checks": hist_checks,
                    "termination_condition": rec.termination_condition,
                }
                print(
                    f"     Completata in {wall_s:.1f}s | BestEval={rec.best_eval_index}/600 | "
                    f"J_tot={rec.best_cost_J_total:.4e} (J_data={rec.best_cost_J_data:.4e}, J_reg={rec.best_cost_J_reg:.4e}) | "
                    f"theta_hat={[round(float(x), 4) for x in rec.theta_hat]}"
                )
            except Exception as run_exc:
                # Punto 7: interrompi soltanto la run coinvolta e documenta il problema
                err_entry = {
                    "run_id": run_id,
                    "eta_level": eta,
                    "data_seed": data_seed,
                    "optimizer_seed": fixed_cpso_seed,
                    "error_type": type(run_exc).__name__,
                    "error_message": str(run_exc),
                }
                failed_runs.append(err_entry)
                print(f"  [ERRORE RUN {run_id}]: {run_exc}")

    # 5. Generazione artefatti finali (CSV, JSON, Markdown, Figure)
    print("[11] 5/5 Generazione CSV, JSON, report Markdown e figure in artifacts/11_final_twin_robustness/...")
    sota_info = inspect_sota_configuration()
    dispersion_summary = write_csv_and_markdown_report(
        baseline_runs=baseline_runs,
        noisy_runs_serialized=noisy_runs_serialized,
        norm_report=norm_report,
        sota_info=sota_info,
        baseline_source_path=baseline_source_path,
        fixed_cpso_seed=fixed_cpso_seed,
        csv_path=csv_path,
        md_path=md_path,
    )
    fig_rec_paths, fig_conv_paths = generate_figures(
        baseline_runs=baseline_runs,
        noisy_runs_serialized=noisy_runs_serialized,
        bounds_dict=bounds_dict,
        out_dir=OUT_DIR,
    )

    json_payload = {
        "experiment_id": "RETECO-11-FINAL-TWIN-ROBUSTNESS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_protocol": {
            "theta_true": {n: float(EXPECTED_THETA_TRUE[i]) for i, n in enumerate(PARAM_NAMES)},
            "target_discretization": {"N": TARGET_N, "M": TARGET_M},
            "target_clean_sha256": target_sha256,
            "target_generation_wall_time_seconds": target_wall_s,
            "inverse_discretization": {"N": INVERSE_N, "M": INVERSE_M},
            "num_observations": EXPECTED_N_OBS,
            "cpso": {
                "num_particles": CPSO_NUM_PARTICLES,
                "max_iterations": CPSO_MAX_ITERATIONS,
                "planned_fem_budget": PLANNED_FEM_BUDGET,
                "fixed_optimizer_seed_noisy_campaign": fixed_cpso_seed,
                "fixed_seed_rationale": (
                    "Mantenuto fisso per tutte le 10 nuove run rumorose (e coincidente con il primo "
                    "seed del protocollo baseline) per isolare l'effetto delle realizzazioni del rumore."
                ),
            },
            "regularization": {
                "lambda_A_noise_free": LAMBDA_A,
                "eta_ref": ETA_REF,
                "lambda_B_noisy_weighted": LAMBDA_B,
                "prior_theta_tilde_0": prior_tilde_0.tolist(),
            },
            "data_seeds_by_eta": {f"{k:.2f}": list(v) for k, v in DATA_SEEDS_BY_ETA.items()},
        },
        "normalization_contract_verification": norm_report,
        "noise_free_baseline_source": baseline_source_path,
        "noise_free_baseline_runs": baseline_runs,
        "noisy_campaign_runs": noisy_runs_serialized,
        "failed_runs": failed_runs,
        "dispersion_summary_median_min_max": dispersion_summary,
        "mandatory_run_verifications": mandatory_verifications,
        "sota_comparison_inspection": sota_info,
        "generated_artifacts": {
            "csv": str(csv_path),
            "json": str(json_path),
            "markdown_report": str(md_path),
            "figure_parameter_recovery": [str(fig_rec_paths[0]), str(fig_rec_paths[1])],
            "figure_cpso_convergence": [str(fig_conv_paths[0]), str(fig_conv_paths[1])],
        },
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    print(f"[11] Campagna finale completata. Artefatti salvati in {OUT_DIR}/")
    return 0 if not failed_runs else 1


if __name__ == "__main__":
    raise SystemExit(main())