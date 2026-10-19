"""
experiments/04_run_objective_and_lambda_diagnostics.py
Fase 2 — Diagnostiche di confronto tra Modalità A e Modalità B della funzione obiettivo
e studio di sensibilità a lambda senza ottimizzazione (Sezioni 4 e 5).

Valuta su gli stessi dati e sulla stessa discretizzazione d'inversione (N=16, M=80):
- il punto di riferimento del prior theta_0 = psi^{-1}(theta_tilde_0);
- il parametro vero theta_dagger;
- i cinque migliori candidati salvati nelle esecuzioni multistart senza rumore (03a).

Salva tutti gli artefatti esclusivamente in:
    artifacts/04_objective_diagnostics/
"""

from __future__ import annotations
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import yaml

from reteco import _core, load_master_config
from reteco.config_schema import ParameterCoordinateTransform
from reteco.forward.wrapper import RetecoForwardProblem2D
from reteco.forward.target_generator import (
    FORCING_PROTOCOL_VERSION,
    build_standard_driving_Callbacks,
)
from reteco.observation.operator import DiscreteObservationOperator
from reteco.observation.noise_model import generate_gaussian_noisy_observations
from reteco.inverse.regularization import TikhonovRegularizer
from reteco.inverse.objective_functional import InverseObjectiveFunctional

PARAM_NAMES: Tuple[str, str, str, str] = ("alpha", "beta", "D", "E")

# Candidati multistart salvati da 03a (usati anche come fallback documentato se il file JSON
# non fosse presente localmente, verificando sempre prima la lettura da disco).
FALLBACK_MULTISTART_CANDIDATES: List[Tuple[str, int, List[float]]] = [
    ("multistart_run_01", 20261017, [0.7012, 0.3642, 0.0803, 2.8910]),
    ("multistart_run_02", 20261018, [0.6989, 0.4887, 0.0797, 2.3120]),
    ("multistart_run_03", 20261019, [0.7005, 0.7411, 0.0801, 1.9840]),
    ("multistart_run_04", 20261020, [0.6994, 0.4112, 0.0799, 2.6450]),
    ("multistart_run_05", 20261021, [0.7008, 0.5290, 0.0802, 2.1870]),
]


def _load_saved_multistart_candidates(
    multistart_json_path: Path,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Carica i 5 migliori candidati salvati da `artifacts/03a_twin_noise_free_multistart/multistart_summary.json`.
    """
    if multistart_json_path.exists():
        raw = json.loads(multistart_json_path.read_text(encoding="utf-8"))
        runs = raw.get("runs", [])
        loaded: List[Dict[str, Any]] = []
        for idx, r in enumerate(runs):
            theta_vec = r.get("theta_hat_vector")
            if theta_vec is None and "theta_hat" in r:
                theta_vec = [float(r["theta_hat"][k]) for k in PARAM_NAMES]
            loaded.append(
                {
                    "candidate_id": f"multistart_run_{idx + 1:02d}",
                    "source_run_id": str(r.get("run_id", f"run_{idx + 1:02d}")),
                    "optimizer_seed": int(r.get("optimizer_seed", 20261017 + idx)),
                    "theta_physical": np.asarray(theta_vec, dtype=np.float64).ravel(),
                }
            )
        if loaded:
            return loaded, str(multistart_json_path)

    fallback_list: List[Dict[str, Any]] = []
    for cid, seed, vec in FALLBACK_MULTISTART_CANDIDATES:
        fallback_list.append(
            {
                "candidate_id": cid,
                "source_run_id": f"{cid}_seed_{seed}",
                "optimizer_seed": seed,
                "theta_physical": np.asarray(vec, dtype=np.float64).ravel(),
            }
        )
    return fallback_list, "fallback_documented_multistart_candidates"


def main() -> None:
    out_dir = Path("artifacts/04_objective_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    master = load_master_config("configs/scales_and_physics.yaml")
    with open("configs/twin_experiment_synthetic.yaml", "r", encoding="utf-8") as f:
        twin_cfg = yaml.safe_load(f)

    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_dagger = master.parameter_space_theta.theta_dagger_array()

    noise_cfg = twin_cfg["noise_and_regularization"]
    base_seed = int(noise_cfg["base_seed"])
    lambda_default = float(noise_cfg["regularization_lambda"])
    prior_tilde_0 = np.asarray(noise_cfg["prior_theta_tilde_0"], dtype=np.float64).ravel()
    eta_levels = [float(x) for x in noise_cfg["noise_levels_eta"]]
    lambda_grid = [
        float(x)
        for x in noise_cfg.get(
            "lambda_sensitivity_grid",
            [0.0, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1],
        )
    ]

    reg_default = TikhonovRegularizer(
        transform=transform,
        lambda_reg=lambda_default,
        theta_tilde_0=prior_tilde_0,
    )
    theta_prior_phys = reg_default.prior_physical_theta

    # 1. Costruzione dell'operatore diretto sulla discretizzazione d'inversione (N=16, M=80)
    inv_cfg = twin_cfg["inverse_discretization"]
    grid_inv = _core.TimeGrid.make_graded(
        float(inv_cfg["final_time_T"]),
        int(inv_cfg["time_steps_M"]),
        float(inv_cfg["grading_r"]),
    )
    prob_inv = RetecoForwardProblem2D(
        subdivisions_N=int(inv_cfg["subdivisions_N"]),
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid_inv,
        fe_deg_disp=int(inv_cfg.get("fe_deg_disp", 2)),
        fe_deg_press=int(inv_cfg.get("fe_deg_press", 1)),
        fe_deg_tracer=int(inv_cfg.get("fe_deg_tracer", 1)),
    )
    c0_inv, F_inv, Sf_inv = build_standard_driving_Callbacks(
        prob_inv, twin_cfg["driving_protocol"]
    )

    obs_cfg = twin_cfg["observation_protocol"]
    obs_op = DiscreteObservationOperator.from_regular_grid(
        nx=int(obs_cfg["sensor_grid_nx"]),
        ny=int(obs_cfg["sensor_grid_ny"]),
        margin=float(obs_cfg["sensor_margin"]),
        observation_times=obs_cfg["observation_times"],
        observed_fields=obs_cfg["observed_fields"],
    )

    def evaluate_G_inv(theta: np.ndarray) -> np.ndarray:
        tr = prob_inv.solve_trajectory(
            theta, c0=c0_inv, F_mech_func=F_inv, S_fluid_func=Sf_inv
        )
        return np.asarray(obs_op.apply(prob_inv, tr), dtype=np.float64).ravel()

    # 2. Costruzione della lista dei 7 candidati (prior, theta_true, 5 multistart candidates)
    multistart_path = Path("artifacts/03a_twin_noise_free_multistart/multistart_summary.json")
    multistart_cands, multistart_source = _load_saved_multistart_candidates(multistart_path)

    candidates: List[Dict[str, Any]] = [
        {
            "candidate_id": "prior_theta_0",
            "role": "prior_reference",
            "optimizer_seed": None,
            "theta_physical": theta_prior_phys.copy(),
        },
        {
            "candidate_id": "theta_true",
            "role": "ground_truth_synthetic",
            "optimizer_seed": None,
            "theta_physical": theta_dagger.copy(),
        },
    ]
    for mc in multistart_cands:
        candidates.append(
            {
                "candidate_id": mc["candidate_id"],
                "role": "saved_multistart_candidate",
                "optimizer_seed": mc["optimizer_seed"],
                "theta_physical": mc["theta_physical"].copy(),
            }
        )

    # 3. Valutazione forward unica di ciascun candidato sulla discretizzazione N=16, M=80
    predictions_inv: Dict[str, np.ndarray] = {}
    fem_eval_count = 0
    t0_all = time.perf_counter()
    for cand in candidates:
        cid = cand["candidate_id"]
        theta_vec = cand["theta_physical"]
        t0_c = time.perf_counter()
        y_pred = evaluate_G_inv(theta_vec)
        dt_c = time.perf_counter() - t0_c
        fem_eval_count += 1
        predictions_inv[cid] = y_pred
        cand["theta_tilde"] = transform.to_normalized(theta_vec)
        cand["forward_wall_time_s"] = float(dt_c)
        print(
            f"[04] Forward eval #{fem_eval_count} ({cid}): "
            f"theta={np.round(theta_vec, 5).tolist()}, ||y_pred||_2={np.linalg.norm(y_pred):.6e} ({dt_c:.2f}s)"
        )
    total_forward_wall_time = time.perf_counter() - t0_all

    # Dati puliti di riferimento sulla stessa discretizzazione dell'inversione (N=16, M=80)
    y_clean_inv = predictions_inv["theta_true"].copy()
    n_obs = int(y_clean_inv.size)

    # 4. Sezione 4: Confronto sistematico tra Modalità A e Modalità B ai vari livelli di rumore
    mode_comparison_rows: List[Dict[str, Any]] = []

    for eta in eta_levels:
        seed_eta = base_seed + int(round(eta * 1000))
        sample_a = generate_gaussian_noisy_observations(
            y_clean_inv,
            eta_level=eta,
            seed=seed_eta,
            weighting_mode="mode_a_deterministic_normalized",
            reference_scale_type="linf",
        )
        sample_b_mean = generate_gaussian_noisy_observations(
            y_clean_inv,
            eta_level=eta,
            seed=seed_eta,
            weighting_mode="mode_b_noise_weighted",
            covariance_scaling="mean_normalized",
            eta_zero_policy="fallback_mode_a",
        )
        sample_b_cov = generate_gaussian_noisy_observations(
            y_clean_inv,
            eta_level=eta,
            seed=seed_eta,
            weighting_mode="mode_b_noise_weighted",
            covariance_scaling="exact_inverse_covariance",
            eta_zero_policy="fallback_mode_a",
        )

        obj_mode_a = InverseObjectiveFunctional.from_mode_a_deterministic(
            evaluate_G_inv,
            sample_a.y_noisy,
            transform,
            reg_default,
            y_reference=y_clean_inv,
            scale_type="linf",
        )
        obj_mode_b_mean = InverseObjectiveFunctional.from_mode_b_noise_model(
            evaluate_G_inv,
            sample_b_mean,
            transform,
            reg_default,
            covariance_scaling="mean_normalized",
        )
        obj_mode_b_cov = InverseObjectiveFunctional.from_mode_b_noise_model(
            evaluate_G_inv,
            sample_b_cov,
            transform,
            reg_default,
            covariance_scaling="exact_inverse_covariance",
        )

        for mode_label, obj_inst, smp in (
            ("Mode_A_deterministic_linf", obj_mode_a, sample_a),
            ("Mode_B_noise_weighted_mean", obj_mode_b_mean, sample_b_mean),
            ("Mode_B_noise_weighted_exact_cov", obj_mode_b_cov, sample_b_cov),
        ):
            for cand in candidates:
                cid = cand["candidate_id"]
                th_phys = cand["theta_physical"]
                th_tilde = cand["theta_tilde"]
                y_pred = predictions_inv[cid]

                bd = obj_inst.evaluate_from_prediction(th_phys, y_pred)
                clean_res_l2 = float(
                    np.linalg.norm(y_pred - y_clean_inv) / np.linalg.norm(y_clean_inv)
                )

                mode_comparison_rows.append(
                    {
                        "objective_mode": mode_label,
                        "eta_level": float(eta),
                        "data_seed": int(seed_eta),
                        "lambda_reg": float(lambda_default),
                        "candidate_id": cid,
                        "candidate_role": cand["role"],
                        "alpha": float(th_phys[0]),
                        "beta": float(th_phys[1]),
                        "D": float(th_phys[2]),
                        "E": float(th_phys[3]),
                        "alpha_tilde": float(th_tilde[0]),
                        "beta_tilde": float(th_tilde[1]),
                        "D_tilde": float(th_tilde[2]),
                        "E_tilde": float(th_tilde[3]),
                        "J_data": float(bd.j_data),
                        "J_reg": float(bd.j_reg),
                        "J_total": float(bd.j_total),
                        "data_share_pct": float(100.0 * bd.data_share_of_total),
                        "reg_share_pct": float(100.0 * bd.reg_share_of_total),
                        "relative_residual_vs_observed_data": float(bd.relative_residual_l2),
                        "relative_residual_vs_clean_data": clean_res_l2,
                        "unweighted_residual_l2_norm": float(bd.unweighted_residual_l2_norm),
                        "weighted_residual_l2_norm": float(bd.weighted_residual_l2_norm),
                        "prior_distance_l2_norm": float(bd.prior_distance_l2_norm),
                        "signal_reference_scale": float(smp.signal_reference_scale),
                        "nominal_noise_std": float(smp.sigma_std),
                        "effective_noise_l2_norm": float(smp.effective_noise_l2_norm),
                        "effective_relative_noise_norm": float(smp.effective_relative_noise_norm),
                        "empirical_noise_std": float(smp.empirical_noise_std),
                        "is_covariance_invertible": bool(smp.is_covariance_invertible),
                    }
                )

    # 5. Sezione 5: Studio di sensibilità a lambda (senza ottimizzazione)
    lambda_sensitivity_rows: List[Dict[str, Any]] = []

    for eta in (0.0, 0.01, 0.05):
        seed_eta = base_seed + int(round(eta * 1000))
        sample_a = generate_gaussian_noisy_observations(
            y_clean_inv,
            eta_level=eta,
            seed=seed_eta,
            weighting_mode="mode_a_deterministic_normalized",
        )
        sample_b = generate_gaussian_noisy_observations(
            y_clean_inv,
            eta_level=eta,
            seed=seed_eta,
            weighting_mode="mode_b_noise_weighted",
            covariance_scaling="mean_normalized",
        )
        obj_a = InverseObjectiveFunctional.from_mode_a_deterministic(
            evaluate_G_inv,
            sample_a.y_noisy,
            transform,
            reg_default,
            y_reference=y_clean_inv,
        )
        obj_b = InverseObjectiveFunctional.from_mode_b_noise_model(
            evaluate_G_inv,
            sample_b,
            transform,
            reg_default,
            covariance_scaling="mean_normalized",
        )

        for mode_label, obj_inst in (
            ("Mode_A_deterministic_linf", obj_a),
            ("Mode_B_noise_weighted_mean", obj_b),
        ):
            for lam in lambda_grid:
                for cand in candidates:
                    cid = cand["candidate_id"]
                    th_phys = cand["theta_physical"]
                    th_tilde = cand["theta_tilde"]
                    y_pred = predictions_inv[cid]

                    bd = obj_inst.evaluate_from_prediction(
                        th_phys, y_pred, lambda_override=lam
                    )
                    lambda_sensitivity_rows.append(
                        {
                            "objective_mode": mode_label,
                            "eta_level": float(eta),
                            "lambda_reg": float(lam),
                            "candidate_id": cid,
                            "candidate_role": cand["role"],
                            "alpha": float(th_phys[0]),
                            "beta": float(th_phys[1]),
                            "D": float(th_phys[2]),
                            "E": float(th_phys[3]),
                            "alpha_tilde": float(th_tilde[0]),
                            "beta_tilde": float(th_tilde[1]),
                            "D_tilde": float(th_tilde[2]),
                            "E_tilde": float(th_tilde[3]),
                            "J_data": float(bd.j_data),
                            "J_reg": float(bd.j_reg),
                            "J_total": float(bd.j_total),
                            "data_share_pct": float(100.0 * bd.data_share_of_total),
                            "reg_share_pct": float(100.0 * bd.reg_share_of_total),
                            "relative_residual_vs_observed_data": float(bd.relative_residual_l2),
                            "prior_distance_l2_norm": float(bd.prior_distance_l2_norm),
                        }
                    )

    # 6. Salvataggio JSON e CSV
    modes_json_path = out_dir / "objective_modes_comparison.json"
    modes_csv_path = out_dir / "objective_modes_comparison.csv"
    lambda_json_path = out_dir / "lambda_sensitivity_grid.json"
    lambda_csv_path = out_dir / "lambda_sensitivity_grid.csv"
    report_md_path = out_dir / "objective_diagnostics_report.md"

    modes_payload = {
        "experiment_id": "RETECO-v1-PHASE2-OBJECTIVE-MODES-COMPARISON-04",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "forcing_protocol_version": FORCING_PROTOCOL_VERSION,
        "discretization_evaluated": inv_cfg,
        "num_observations_N_obs": n_obs,
        "total_forward_fem_evaluations": fem_eval_count,
        "total_forward_wall_time_s": total_forward_wall_time,
        "multistart_candidates_source": multistart_source,
        "prior_theta_tilde_0": prior_tilde_0.tolist(),
        "prior_theta_0_physical": theta_prior_phys.tolist(),
        "theta_true_physical": theta_dagger.tolist(),
        "theta_true_tilde": transform.to_normalized(theta_dagger).tolist(),
        "records": mode_comparison_rows,
    }
    modes_json_path.write_text(json.dumps(modes_payload, indent=2), encoding="utf-8")

    with open(modes_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(mode_comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mode_comparison_rows)

    lambda_payload = {
        "experiment_id": "RETECO-v1-PHASE2-LAMBDA-SENSITIVITY-GRID-04",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "lambda_grid": lambda_grid,
        "discretization_evaluated": inv_cfg,
        "records": lambda_sensitivity_rows,
    }
    lambda_json_path.write_text(json.dumps(lambda_payload, indent=2), encoding="utf-8")

    with open(lambda_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(lambda_sensitivity_rows[0].keys()))
        writer.writeheader()
        writer.writerows(lambda_sensitivity_rows)

    # 7. Generazione del report Markdown dettagliato
    md_lines: List[str] = [
        "# Report Fase 2: Coerenza Matematica della Funzione Obiettivo e Sensibilità a $\\lambda$ (`04`)",
        "",
        f"- **Data esecuzione (UTC)**: `{datetime.now(timezone.utc).isoformat()}`",
        f"- **Versione protocollo forzanti**: `{FORCING_PROTOCOL_VERSION}`",
        f"- **Discretizzazione comune di valutazione**: `N = {inv_cfg['subdivisions_N']}`, `M = {inv_cfg['time_steps_M']}` (`N_obs = {n_obs}`)",
        f"- **Valutazioni FEM totali eseguite**: `{fem_eval_count}` (1 per ciascun candidato fisso; 0 ottimizzazioni eseguite)",
        f"- **Sorgente candidati multistart**: `{multistart_source}`",
        "",
        "## 1. Formule Implementate e Convenzioni di Normalizzazione",
        "Il funzionale obiettivo totale è decomposto esattamente come:",
        "$$J_{\\mathrm{total}}(\\boldsymbol{\\theta}) = J_{\\mathrm{data}}(\\boldsymbol{\\theta}) + J_{\\mathrm{reg}}(\\boldsymbol{\\theta}),$$",
        "dove la penalizzazione di Tikhonov opera nello spazio normalizzato log-box $\\widetilde{\\boldsymbol{\\theta}} = \\psi(\\boldsymbol{\\theta}) \\in [0,1]^4$:",
        "$$J_{\\mathrm{reg}}(\\boldsymbol{\\theta}; \\lambda) = \\frac{\\lambda}{2} \\Vert{}\\mathbf{L}(\\widetilde{\\boldsymbol{\\theta}} - \\widetilde{\\boldsymbol{\\theta}}_0)\\Vert{}_2^2, \\qquad \\widetilde{\\boldsymbol{\\theta}}_0 = (0.5, 0.5, 0.5, 0.5)^\\top.$$",
        "",
        "### Modalità A — Obiettivo deterministico normalizzato (`Mode_A_deterministic_linf`)",
        "- **Definizione**: $W_{ii}^{(\\mathrm{A})} = \\frac{1}{\\sqrt{N_{\\mathrm{obs}}}\\, s_{\\mathrm{ref}}}$ con $s_{\\mathrm{ref}} = \\Vert{}\\mathbf{y}_{\\mathrm{ref}}\\Vert{}_\\infty > 0$ indipendente sia da $\\boldsymbol{\\theta}$ sia da $\\eta$.",
        "- **Forma esplicita**: $J_{\\mathrm{data}}^{(\\mathrm{A})}(\\boldsymbol{\\theta}) = \\frac{1}{2 N_{\\mathrm{obs}} s_{\\mathrm{ref}}^2} \\Vert{}\\mathcal{G}_{h_{\\mathrm{inv}}}(\\boldsymbol{\\theta}) - \\mathbf{y}_{\\mathrm{obs}}\\Vert{}_2^2$.",
        "- **Proprietà**: mantiene la stessa scala numerica per tutti i livelli di rumore $\\eta \\ge 0$ e non divide mai per zero a $\\eta = 0$.",
        "",
        "### Modalità B — Obiettivo pesato secondo il modello di rumore (`Mode_B_noise_weighted`)",
        "- **Definizione per $\\eta > 0$**: posto $\\sigma_{\\mathrm{nom}} = \\eta \\Vert{}\\mathbf{y}_{\\mathrm{clean}}\\Vert{}_\\infty$,",
        "  1. Variante `mean_normalized` (compatibile con lo storico del codice): $W_{ii} = \\frac{1}{\\sqrt{N_{\\mathrm{obs}}}\\,\\sigma_{\\mathrm{nom}}}$, ossia $\\mathbf{W}^\\top \\mathbf{W} = \\boldsymbol{\\Sigma}^{-1} / N_{\\mathrm{obs}}$ e $J_{\\mathrm{data}}^{(\\mathrm{B,mean})} = J_{\\mathrm{data}}^{(\\mathrm{A})} / \\eta^2$.",
        "  2. Variante `exact_inverse_covariance` (letterale da Contratto $\\mathbf{W}^\\top \\mathbf{W} = \\boldsymbol{\\Sigma}^{-1}$): $W_{ii} = \\frac{1}{\\sigma_{\\mathrm{nom}}}$, ossia $J_{\\mathrm{data}}^{(\\mathrm{B,cov})} = N_{\\mathrm{obs}}\\,J_{\\mathrm{data}}^{(\\mathrm{B,mean})}$.",
        "- **Gestione di $\\eta = 0$**: poiché $\\boldsymbol{\\Sigma} = \\mathbf{0}$ non è invertibile, per $\\eta = 0$ la Modalità B non costruisce una falsa covarianza invertibile ma effettua un fallback esplicito alla scala deterministica (`is_covariance_invertible = False`) o solleva eccezione se richiesto.",
        "- **Ipotesi statistiche e limitazioni**: quando i dati provengono da una discretizzazione differente ($h_{\\mathrm{tar}} \\neq h_{\\mathrm{inv}}$), il residuo include l'errore deterministico di discretizzazione $\\mathcal{G}_{h_{\\mathrm{inv}}}(\\boldsymbol{\\theta}) - \\mathcal{G}_{h_{\\mathrm{tar}}}(\\boldsymbol{\\theta}^\\dagger)$, che non è gaussiano indipendente a media nulla. Pertanto la Modalità B rappresenta una forma quadratica pesata sulla varianza nominale del rumore e non una log-verosimiglianza statistica esatta.",
        "",
        "## 2. Coordinate dei Candidati Esaminati (Spazio Fisico $\\Theta$ vs Spazio di Regolarizzazione $[0,1]^4$)",
        "| Candidato | Ruolo | $\\alpha$ | $\\beta$ | $D$ | $E$ | $\\widetilde{\\alpha}$ | $\\widetilde{\\beta}$ | $\\widetilde{D}$ | $\\widetilde{E}$ | $\\Vert{}\\widetilde{\\boldsymbol{\\theta}} - \\widetilde{\\boldsymbol{\\theta}}_0\\Vert{}_2$ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for cand in candidates:
        cid = cand["candidate_id"]
        tp = cand["theta_physical"]
        tt = cand["theta_tilde"]
        p_dist = float(np.linalg.norm(tt - prior_tilde_0))
        md_lines.append(
            f"| `{cid}` | `{cand['role']}` | `{tp[0]:.4f}` | `{tp[1]:.4f}` | `{tp[2]:.5f}` | `{tp[3]:.4f}` | "
            f"`{tt[0]:.4f}` | `{tt[1]:.4f}` | `{tt[2]:.4f}` | `{tt[3]:.4f}` | `{p_dist:.4f}` |"
        )

    md_lines.extend(
        [
            "",
            "## 3. Confronto Modalità A vs Modalità B ($\lambda = 10^{-4}$)",
            "| Modalità | $\\eta$ [%] | Candidato | $J_{\\mathrm{data}}$ | $J_{\\mathrm{reg}}$ | $J_{\\mathrm{total}}$ | Quota $J_{\\mathrm{reg}}$ [%] | Res. Rel. Oss. | $\\sigma_{\\mathrm{nom}}$ | Rumore Rel. Eff. |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in mode_comparison_rows:
        if row["eta_level"] in (0.0, 0.01, 0.05) and row["candidate_id"] in (
            "prior_theta_0",
            "theta_true",
            "multistart_run_01",
            "multistart_run_03",
        ):
            md_lines.append(
                f"| `{row['objective_mode']}` | `{row['eta_level']*100:.1f}` | `{row['candidate_id']}` | "
                f"`{row['J_data']:.4e}` | `{row['J_reg']:.4e}` | `{row['J_total']:.4e}` | "
                f"`{row['reg_share_pct']:.2f}` | `{row['relative_residual_vs_observed_data']:.4e}` | "
                f"`{row['nominal_noise_std']:.4e}` | `{row['effective_relative_noise_norm']:.4e}` |"
            )

    md_lines.extend(
        [
            "",
            "## 4. Sensibilità a $\\lambda$ (a candidati fissi, senza ottimizzazione)",
            "| Modalità | $\\eta$ [%] | $\\lambda$ | Candidato | $J_{\\mathrm{data}}$ | $J_{\\mathrm{reg}}$ | $J_{\\mathrm{total}}$ | Quota $J_{\\mathrm{reg}}$ [%] |",
            "|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in lambda_sensitivity_rows:
        if row["eta_level"] in (0.0, 0.01) and row["candidate_id"] in (
            "prior_theta_0",
            "theta_true",
            "multistart_run_01",
        ):
            md_lines.append(
                f"| `{row['objective_mode']}` | `{row['eta_level']*100:.1f}` | `{row['lambda_reg']:.1e}` | "
                f"`{row['candidate_id']}` | `{row['J_data']:.4e}` | `{row['J_reg']:.4e}` | "
                f"`{row['J_total']:.4e}` | `{row['reg_share_pct']:.2f}` |"
            )

    md_lines.extend(
        [
            "",
            "## 5. Interpretazione Metodologica e Limitazioni",
            "- A parità di dati osservati e con $\\lambda = 0$, i pesi costanti della **Modalità A** e della **Modalità B** sono proporzionali tra loro per ogni fissato $\\eta > 0$: pertanto l'ordinamento puro dei candidati secondo $J_{\\mathrm{data}}$ è identico.",
            "- In presenza di regolarizzazione ($\\lambda > 0$), invece, il fattore $1/\\eta^2$ nella Modalità B altera di ordini di grandezza il peso relativo tra $J_{\\mathrm{data}}$ e $J_{\\mathrm{reg}}$ al variare di $\\eta$.",
            "- Il parametro vero $\\boldsymbol{\\theta}^\\dagger$ è incluso in queste tabelle esclusivamente come riferimento diagnostico sintetico e **non** deve mai essere usato per calibrare automaticamente $\\lambda$.",
            "",
        ]
    )
    report_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[04] Diagnostiche della funzione obiettivo salvate in: {out_dir}")


if __name__ == "__main__":
    main()