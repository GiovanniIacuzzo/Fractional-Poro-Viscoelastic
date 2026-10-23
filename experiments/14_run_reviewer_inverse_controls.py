#!/usr/bin/env python3
"""
experiments/14_run_reviewer_inverse_controls.py

Esegue i controlli sperimentali sul problema inverso rumoroso (senza sovrascrivere
artifacts/11_final_twin_robustness/):
1. Confronto a eta = 0.10 tra:
   - Protocollo a prior fisso (lambda_B = 7.2, caricato da artifacts/11_final_twin_robustness/)
   - Protocollo a rapporto dati/regolarizzazione costante (lambda_B_eq = 1.8, 5 run CPSO seed 20261017)
2. Controllo della sensibilità al seed CPSO (seed 20261018, ottimo baseline) e confronto
   con l'ottimizzatore deterministico Powell a budget identico (600 valutazioni FEM)
   sulle stesse 10 realizzazioni di rumore (5 a eta = 0.05, 5 a eta = 0.10).
"""

from __future__ import annotations
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT_DIR / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from reteco import _core, load_master_config
from reteco.config_schema import ParameterCoordinateTransform
from reteco.forward.target_generator import build_standard_driving_Callbacks
from reteco.forward.wrapper import RetecoForwardProblem2D
from reteco.inverse.cpso import run_cpso_optimization
from reteco.inverse.diagnostics import PARAM_NAMES
from reteco.inverse.objective_functional import InverseObjectiveFunctional
from reteco.inverse.regularization import TikhonovRegularizer
from reteco.inverse.sota_runners import run_scipy_baseline
from reteco.observation.noise_model import generate_gaussian_noisy_observations
from reteco.observation.operator import DiscreteObservationOperator

EXPECTED_THETA_TRUE = np.array([0.70, 0.45, 0.08, 2.50], dtype=np.float64)
EXPECTED_TARGET_SHA256 = (
    "1bca9e29c60d77c1450fd2fcea845d2320c9e44945dd7d3ccda52aa33ef261af"
)
TARGET_N, TARGET_M = 32, 160
INVERSE_N, INVERSE_M = 16, 80
EXPECTED_N_OBS = 180
PLANNED_FEM_BUDGET = 600
CPSO_NUM_PARTICLES = 10
CPSO_MAX_ITERATIONS = 60

LAMBDA_A = 1.0e-4
LAMBDA_B_FIXED = 7.2
LAMBDA_B_EQ_010 = 1.8  # (180 / 0.10^2) * 1e-4 = 1.8

DATA_SEEDS_BY_ETA: Dict[float, Tuple[int, ...]] = {
    0.05: (2026110501, 2026110502, 2026110503, 2026110504, 2026110505),
    0.10: (2026111001, 2026111002, 2026111003, 2026111004, 2026111005),
}

OUT_DIR = Path("artifacts/14_reviewer_inverse_controls")


def _summarize_group(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"n_runs": len(runs), "parameters": {}}
    for p_idx, p_name in enumerate(PARAM_NAMES):
        hats = np.array([float(r["theta_hat_vector"][p_idx]) for r in runs], dtype=np.float64)
        errs = np.array(
            [
                100.0 * abs(float(r["theta_hat_vector"][p_idx]) - EXPECTED_THETA_TRUE[p_idx])
                / EXPECTED_THETA_TRUE[p_idx]
                for r in runs
            ],
            dtype=np.float64,
        )
        summary["parameters"][p_name] = {
            "estimate_median": float(np.median(hats)),
            "estimate_min": float(np.min(hats)),
            "estimate_max": float(np.max(hats)),
            "rel_error_pct_median": float(np.median(errs)),
            "rel_error_pct_min": float(np.min(errs)),
            "rel_error_pct_max": float(np.max(errs)),
        }
    return summary


def _execute_single_noisy_run(
    *,
    run_id: str,
    optimizer_name: str,
    optimizer_seed: int,
    eta: float,
    data_seed: int,
    lambda_b: float,
    y_target_clean: np.ndarray,
    G_inv: Any,
    transform: ParameterCoordinateTransform,
    prior_tilde_0: np.ndarray,
) -> Dict[str, Any]:
    sample = generate_gaussian_noisy_observations(
        y_clean=y_target_clean,
        eta_level=eta,
        seed=data_seed,
        weighting_mode="mode_b_noise_weighted",
        reference_scale_type="linf",
        covariance_scaling="exact_inverse_covariance",
    )
    reg = TikhonovRegularizer(transform, lambda_reg=lambda_b, theta_tilde_0=prior_tilde_0)
    obj = InverseObjectiveFunctional.from_mode_b_noise_model(
        forward_operator=G_inv,
        sample=sample,
        transform=transform,
        regularizer=reg,
        covariance_scaling="exact_inverse_covariance",
        reference_scale_type="linf",
        max_evaluations=PLANNED_FEM_BUDGET,
    )

    t0 = time.perf_counter()
    if optimizer_name == "CPSO":
        opt_res = run_cpso_optimization(
            obj,
            seed=optimizer_seed,
            num_particles=CPSO_NUM_PARTICLES,
            max_iterations=CPSO_MAX_ITERATIONS,
        )
    else:
        opt_res = run_scipy_baseline(optimizer_name, obj, seed=optimizer_seed)
    wall_s = time.perf_counter() - t0

    theta_hat = np.asarray(opt_res.theta_hat, dtype=np.float64).ravel()
    rel_errs = [
        float(100.0 * abs(theta_hat[i] - EXPECTED_THETA_TRUE[i]) / EXPECTED_THETA_TRUE[i])
        for i in range(4)
    ]

    # Estrae J_data e J_reg dal miglior record nella history senza rivalutare il solutore
    best_idx = int(np.argmin([r.j_total for r in obj.history]))
    best_rec = obj.history[best_idx]
    y_hat = np.asarray(G_inv(theta_hat), dtype=np.float64).ravel()
    rel_res_clean = float(np.linalg.norm(y_hat - y_target_clean) / np.linalg.norm(y_target_clean))
    eff_noise = float(
        np.linalg.norm(sample.y_noisy - y_target_clean) / np.linalg.norm(y_target_clean)
    )

    return {
        "run_id": run_id,
        "optimizer": optimizer_name,
        "optimizer_seed": int(optimizer_seed),
        "eta_level": float(eta),
        "data_seed": int(data_seed),
        "regularization_lambda": float(lambda_b),
        "effective_relative_noise_norm": eff_noise,
        "theta_hat_vector": theta_hat.tolist(),
        "individual_relative_errors_pct_vector": rel_errs,
        "best_cost_J_total": float(best_rec.j_total),
        "best_cost_J_data": float(best_rec.j_data),
        "best_cost_J_reg": float(best_rec.j_reg),
        "relative_residual_vs_clean_target": rel_res_clean,
        "optimizer_fem_evaluations": int(obj.num_fem_evaluations),
        "best_eval_index": int(best_rec.eval_index),
        "wall_time_seconds": float(wall_s),
    }


def plot_lambda_and_optimizer_comparison(
    archived_005_cpso20261017: Sequence[Dict[str, Any]],
    archived_010_l72_cpso20261017: Sequence[Dict[str, Any]],
    new_010_l18_cpso20261017: Sequence[Dict[str, Any]],
    powell_005_l72: Sequence[Dict[str, Any]],
    powell_010_l18: Sequence[Dict[str, Any]],
    out_base: Path,
) -> Tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.0), dpi=300)
    axes_flat = axes.ravel()
    panel_tags = ["(a)", "(b)", "(c)", "(d)"]

    groups = [
        ("CPSO (seed 17)\n$\\eta=5\\%,\\,\\lambda_B=7.2$", archived_005_cpso20261017, "#E69F00", "s"),
        ("CPSO (seed 17)\n$\\eta=10\\%,\\,\\lambda_B=7.2$\n[fixed prior]", archived_010_l72_cpso20261017, "#CC79A7", "^"),
        ("CPSO (seed 17)\n$\\eta=10\\%,\\,\\lambda_B^{\\mathrm{eq}}=1.8$\n[balanced ratio]", new_010_l18_cpso20261017, "#D55E00", "v"),
        ("Powell\n$\\eta=5\\%,\\,\\lambda_B=7.2$", powell_005_l72, "#0072B2", "o"),
        ("Powell\n$\\eta=10\\%,\\,\\lambda_B^{\\mathrm{eq}}=1.8$", powell_010_l18, "#009E73", "D"),
    ]

    x_ticks = np.arange(len(groups))

    for p_idx, p_name in enumerate(PARAM_NAMES):
        ax = axes_flat[p_idx]
        true_val = float(EXPECTED_THETA_TRUE[p_idx])
        ax.axhline(
            true_val,
            color="#111111",
            linestyle="--",
            linewidth=1.4,
            label=r"True parameter $\theta^\dagger$" if p_idx == 0 else None,
        )

        for g_idx, (lbl, runs, col, mk) in enumerate(groups):
            vals = np.array([float(r["theta_hat_vector"][p_idx]) for r in runs], dtype=np.float64)
            offsets = np.linspace(-0.14, 0.14, vals.size)
            ax.scatter(
                np.full(vals.size, x_ticks[g_idx]) + offsets,
                vals,
                marker=mk,
                s=46,
                color=col,
                edgecolors="black",
                linewidths=0.7,
                alpha=0.88,
                zorder=4,
            )
            med_v = float(np.median(vals))
            min_v, max_v = float(np.min(vals)), float(np.max(vals))
            ax.errorbar(
                x_ticks[g_idx],
                med_v,
                yerr=[[med_v - min_v], [max_v - med_v]],
                fmt="o",
                markersize=4,
                color="black",
                ecolor=col,
                elinewidth=2.0,
                capsize=5,
                zorder=5,
            )

        ax.set_xticks(x_ticks)
        ax.set_xticklabels([g[0] for g in groups])
        ax.set_title(f"{panel_tags[p_idx]} Parameter `{p_name}` (true = {true_val:g})")
        ax.set_ylabel(f"Estimated `{p_name}`")
        ax.grid(True, linestyle=":", alpha=0.55)

    fig.suptitle(
        "Noisy Twin Recovery: Effect of Regularization Scaling ($\\lambda_B=7.2$ vs $1.8$) and Optimizer Choice",
        fontsize=12,
    )
    fig.tight_layout()
    png_path = out_base.with_suffix(".png")
    pdf_path = out_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    archived_json_path = Path(
        "artifacts/11_final_twin_robustness/final_twin_robustness_protocol_and_results.json"
    )
    if not archived_json_path.exists():
        raise FileNotFoundError(
            f"File archiviato non trovato: {archived_json_path}. "
            "Assicurati che artifacts/11_final_twin_robustness/ sia presente."
        )

    archived_data = json.loads(archived_json_path.read_text(encoding="utf-8"))
    archived_noisy = archived_data["noisy_campaign_runs"]
    archived_005_l72 = [r for r in archived_noisy if np.isclose(float(r["eta_level"]), 0.05)]
    archived_010_l72 = [r for r in archived_noisy if np.isclose(float(r["eta_level"]), 0.10)]

    master = load_master_config("configs/scales_and_physics.yaml")
    with open("configs/twin_experiment_synthetic.yaml", "r", encoding="utf-8") as fh:
        twin_cfg = yaml.safe_load(fh)

    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_dagger = master.parameter_space_theta.theta_dagger_array()
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

    # 1. Target sintetico pulito (N=32, M=160) e verifica SHA-256
    print("[14] 1/4 Generazione e verifica SHA-256 del target pulito (N=32, M=160)...")
    tar_cfg = twin_cfg["target_discretization"]
    grid_tar = _core.TimeGrid.make_graded(
        float(tar_cfg["final_time_T"]), TARGET_M, float(tar_cfg["grading_r"])
    )
    prob_tar = RetecoForwardProblem2D(
        subdivisions_N=TARGET_N,
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid_tar,
    )
    c0_tar, F_tar, Sf_tar = build_standard_driving_Callbacks(prob_tar, twin_cfg["driving_protocol"])
    traj_tar = prob_tar.solve_trajectory(
        theta_dagger, c0=c0_tar, F_mech_func=F_tar, S_fluid_func=Sf_tar
    )
    y_target_clean = np.asarray(obs_op.apply(prob_tar, traj_tar), dtype=np.float64).ravel()
    sha = hashlib.sha256(np.ascontiguousarray(y_target_clean, dtype="<f8").tobytes()).hexdigest()
    if sha != EXPECTED_TARGET_SHA256:
        raise RuntimeError(f"SHA-256 target non corrispondente: {sha}")

    # 2. Solutore inverso (N=16, M=80)
    inv_cfg = twin_cfg["inverse_discretization"]
    grid_inv = _core.TimeGrid.make_graded(
        float(inv_cfg["final_time_T"]), INVERSE_M, float(inv_cfg["grading_r"])
    )
    prob_inv = RetecoForwardProblem2D(
        subdivisions_N=INVERSE_N,
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid_inv,
    )
    c0_inv, F_inv, Sf_inv = build_standard_driving_Callbacks(prob_inv, twin_cfg["driving_protocol"])

    def G_inv(theta: np.ndarray) -> np.ndarray:
        tr = prob_inv.solve_trajectory(theta, c0=c0_inv, F_mech_func=F_inv, S_fluid_func=Sf_inv)
        return np.asarray(obs_op.apply(prob_inv, tr), dtype=np.float64).ravel()

    # 3. Controllo A: 5 run CPSO (seed 20261017) a eta = 0.10 con lambda_B_eq = 1.8
    print("[14] 2/4 Esecuzione 5 run di controllo CPSO (seed 20261017) a eta=0.10 con lambda_B_eq=1.8...")
    cpso_s17_eta010_l18: List[Dict[str, Any]] = []
    for rep_idx, dseed in enumerate(DATA_SEEDS_BY_ETA[0.10], start=1):
        rid = f"ctrl_cpso_s20261017_eta10_lambda1p8_rep{rep_idx:02d}"
        rec = _execute_single_noisy_run(
            run_id=rid,
            optimizer_name="CPSO",
            optimizer_seed=20261018,
            eta=0.10,
            data_seed=dseed,
            lambda_b=LAMBDA_B_EQ_010,
            y_target_clean=y_target_clean,
            G_inv=G_inv,
            transform=transform,
            prior_tilde_0=prior_tilde_0,
        )
        cpso_s17_eta010_l18.append(rec)
        print(
            f"  -> {rid}: theta_hat={[round(x, 4) for x in rec['theta_hat_vector']]} "
            f"| errs={[round(x, 2) for x in rec['individual_relative_errors_pct_vector']]}%"
        )

    # 4. Controllo B: 10 run con Powell (budget 600 eval) a eta=0.05 (lambda=7.2) e eta=0.10 (lambda=1.8)
    print("[14] 3/4 Esecuzione 10 run rumorose con Powell (600 eval, lambda_B bilanciato)...")
    powell_eta005_l72: List[Dict[str, Any]] = []
    powell_eta010_l18: List[Dict[str, Any]] = []

    for rep_idx, dseed in enumerate(DATA_SEEDS_BY_ETA[0.05], start=1):
        rid = f"ctrl_powell_eta05_lambda7p2_rep{rep_idx:02d}"
        rec = _execute_single_noisy_run(
            run_id=rid,
            optimizer_name="Powell",
            optimizer_seed=2026120201,
            eta=0.05,
            data_seed=dseed,
            lambda_b=LAMBDA_B_FIXED,
            y_target_clean=y_target_clean,
            G_inv=G_inv,
            transform=transform,
            prior_tilde_0=prior_tilde_0,
        )
        powell_eta005_l72.append(rec)
        print(
            f"  -> {rid}: theta_hat={[round(x, 4) for x in rec['theta_hat_vector']]} "
            f"| errs={[round(x, 2) for x in rec['individual_relative_errors_pct_vector']]}%"
        )

    for rep_idx, dseed in enumerate(DATA_SEEDS_BY_ETA[0.10], start=1):
        rid = f"ctrl_powell_eta10_lambda1p8_rep{rep_idx:02d}"
        rec = _execute_single_noisy_run(
            run_id=rid,
            optimizer_name="Powell",
            optimizer_seed=2026120201,
            eta=0.10,
            data_seed=dseed,
            lambda_b=LAMBDA_B_EQ_010,
            y_target_clean=y_target_clean,
            G_inv=G_inv,
            transform=transform,
            prior_tilde_0=prior_tilde_0,
        )
        powell_eta010_l18.append(rec)
        print(
            f"  -> {rid}: theta_hat={[round(x, 4) for x in rec['theta_hat_vector']]} "
            f"| errs={[round(x, 2) for x in rec['individual_relative_errors_pct_vector']]}%"
        )

    # Salvataggio CSV completo di tutte le nuove run di controllo
    all_new_runs = cpso_s17_eta010_l18 + powell_eta005_l72 + powell_eta010_l18
    csv_path = OUT_DIR / "reviewer_inverse_controls_runs.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_new_runs[0].keys()))
        writer.writeheader()
        writer.writerows(all_new_runs)

    fig_png, fig_pdf = plot_lambda_and_optimizer_comparison(
        archived_005_cpso20261017=archived_005_l72,
        archived_010_l72_cpso20261017=archived_010_l72,
        new_010_l18_cpso20261017=cpso_s17_eta010_l18,
        powell_005_l72=powell_eta005_l72,
        powell_010_l18=powell_eta010_l18,
        out_base=OUT_DIR / "fig_noisy_controls_lambda_and_optimizer",
    )

    summary_payload = {
        "group_summaries": {
            "cpso_seed20261017_eta005_lambda7p2_archived": _summarize_group(archived_005_l72),
            "cpso_seed20261017_eta010_lambda7p2_archived_fixed_prior": _summarize_group(
                archived_010_l72
            ),
            "cpso_seed20261017_eta010_lambda1p8_balanced_ratio": _summarize_group(
                cpso_s17_eta010_l18
            ),
            "powell_eta005_lambda7p2": _summarize_group(powell_eta005_l72),
            "powell_eta010_lambda1p8_balanced_ratio": _summarize_group(powell_eta010_l18),
        },
        "new_runs": all_new_runs,
        "generated_files": {
            "csv": str(csv_path),
            "figure_png": str(fig_png),
            "figure_pdf": str(fig_pdf),
        },
    }

    json_path = OUT_DIR / "reviewer_inverse_controls_summary.json"
    json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("\n=== RIEPILOGO VERIFICHE BLOCCO 2 (MEDIANA E [MIN, MAX]) ===")
    print(json.dumps(summary_payload["group_summaries"], indent=2))
    print(f"\nArtefatti salvati in: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())