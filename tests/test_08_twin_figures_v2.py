"""
tests/test_08_twin_figures_v2.py
Verifica della rigenerazione grafica standalone (PNG 300 dpi + PDF vettoriale)
dai soli artefatti JSON e CSV, senza invocare il solutore FEM e preservando
invariati i file numerici originali.
"""

from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

from reteco.visualization.twin_diagnostics_plots import (
    regenerate_all_twin_figures_from_artifacts,
)


def _create_mock_artifacts(base_dir: Path) -> tuple[Path, Path]:
    m_dir = base_dir / "03a_twin_noise_free_multistart"
    n_dir = base_dir / "03_twin_robustness" / "diagnostics"
    m_dir.mkdir(parents=True, exist_ok=True)
    n_dir.mkdir(parents=True, exist_ok=True)

    theta_true = [0.75, 0.45, 1.5e-3, 2.5]
    m_runs = []
    for i in range(5):
        seed = 20261017 + i
        evals = list(range(1, 121))
        best_j = [float(1.0e-2 * np.exp(-0.04 * k) + (1.2 + 0.1 * i) * 1.0e-4) for k in evals]
        j_data = best_j[-1] * 0.72
        j_reg = best_j[-1] * 0.28
        theta_hat = [
            0.75 + 0.005 * (i - 2),
            0.45 + 0.04 * (i - 2),
            1.5e-3 * (1.0 + 0.02 * (i - 2)),
            2.5 * (1.0 + 0.08 * (i - 2)),
        ]
        errs = [
            100.0 * abs(theta_hat[j] - theta_true[j]) / abs(theta_true[j])
            for j in range(4)
        ]
        m_runs.append(
            {
                "run_id": f"multistart_run_{i + 1:02d}_seed_{seed}",
                "optimizer_seed": seed,
                "data_seed": 20261016,
                "eta_level": 0.0,
                "theta_true_vector": theta_true,
                "theta_hat_vector": theta_hat,
                "individual_relative_errors_pct_vector": errs,
                "best_cost_J_total": best_j[-1],
                "best_cost_J_data": j_data,
                "best_cost_J_reg": j_reg,
                "nominal_noise_std": 0.0,
                "effective_relative_noise_norm": 0.0,
                "relative_residual_vs_noisy_data": 2.4e-3 + 1.0e-4 * i,
                "relative_residual_vs_clean_target": 2.4e-3 + 1.0e-4 * i,
                "planned_fem_budget": 120,
                "optimizer_fem_evaluations": 120,
                "diagnostic_fem_evaluations": 1,
                "convergence_history": {
                    "eval_index": evals,
                    "best_j_so_far": best_j,
                },
            }
        )

    m_json = m_dir / "multistart_summary.json"
    m_json.write_text(
        json.dumps({"discretization_relative_gap": 1.8e-3, "runs": m_runs}, indent=2),
        encoding="utf-8",
    )

    m_csv = m_dir / "multistart_runs_summary.csv"
    with open(m_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "best_cost_J_total"])
        w.writeheader()
        for r in m_runs:
            w.writerow({"run_id": r["run_id"], "best_cost_J_total": r["best_cost_J_total"]})

    etas = [0.0, 0.01, 0.05, 0.10]
    n_runs = []
    for i, eta in enumerate(etas):
        errs = [0.8 + 1.5 * i, 6.0 + 5.0 * i, 2.5 + 3.0 * i, 9.0 + 7.5 * i]
        n_runs.append(
            {
                "run_id": f"noise_eta_{int(eta * 100):02d}pct",
                "optimizer_seed": 20261017 + int(eta * 1000),
                "data_seed": 20261016 + int(eta * 1000),
                "eta_level": eta,
                "theta_true_vector": theta_true,
                "theta_hat_vector": theta_true,
                "individual_relative_errors_pct_vector": errs,
                "best_cost_J_total": 1.0e-3 * (i + 1),
                "best_cost_J_data": 0.8e-3 * (i + 1),
                "best_cost_J_reg": 0.2e-3 * (i + 1),
                "nominal_noise_std": eta * 10.0,
                "effective_relative_noise_norm": eta * 1.42,
                "relative_residual_vs_noisy_data": 2.4e-3 + eta * 1.38,
                "relative_residual_vs_clean_target": 2.4e-3 + eta * 0.35,
                "planned_fem_budget": 120,
                "optimizer_fem_evaluations": 120,
                "diagnostic_fem_evaluations": 1,
                "convergence_history": {"eval_index": [1, 120], "best_j_so_far": [1.0, 0.1]},
            }
        )

    n_json = n_dir / "twin_robustness_diagnostics.json"
    n_json.write_text(
        json.dumps(
            {"discretization_relative_gap": 1.8e-3, "detailed_runs": n_runs},
            indent=2,
        ),
        encoding="utf-8",
    )

    n_csv = n_dir / "twin_robustness_summary.csv"
    with open(n_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "best_cost_J_total"])
        w.writeheader()
        for r in n_runs:
            w.writerow({"run_id": r["run_id"], "best_cost_J_total": r["best_cost_J_total"]})

    return m_dir, n_dir


def test_standalone_regeneration_creates_png_and_pdf_without_modifying_data(tmp_path: Path) -> None:
    m_dir, n_dir = _create_mock_artifacts(tmp_path)

    data_files = [
        m_dir / "multistart_summary.json",
        m_dir / "multistart_runs_summary.csv",
        n_dir / "twin_robustness_diagnostics.json",
        n_dir / "twin_robustness_summary.csv",
    ]
    hashes_before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in data_files}

    bounds = {
        "alpha": (0.3, 0.95),
        "beta": (0.1, 0.9),
        "D": (1.0e-4, 1.0e-2),
        "E": (0.5, 8.0),
    }
    generated = regenerate_all_twin_figures_from_artifacts(
        multistart_dir=m_dir,
        noise_diag_dir=n_dir,
        bounds_dict=bounds,
        suffix="_v2",
    )

    assert len(generated) == 5
    for key, (png_path, pdf_path) in generated.items():
        assert png_path.exists() and png_path.stat().st_size > 0, f"Missing PNG for {key}"
        assert pdf_path.exists() and pdf_path.stat().st_size > 0, f"Missing PDF for {key}"
        assert png_path.name.endswith("_v2.png")
        assert pdf_path.name.endswith("_v2.pdf")

    hashes_after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in data_files}
    assert hashes_before == hashes_after