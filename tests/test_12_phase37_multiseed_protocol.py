"""
tests/test_12_phase37_multiseed_protocol.py
RETECO — Fase 3.7: Test unitari per:
1. Verifica tassativa dell'hash SHA-256 del target e interruzione in caso di mismatch;
2. Controlli di coerenza sulle 3 run a 600 valutazioni (seed 20261017, 20261018, 20261019);
3. Generazione del CSV comparativo e della figura in scala logaritmica con il riferimento
   diagnostico J_total(theta_dagger) separato dalla storia del CPSO.
"""

from __future__ import annotations
from pathlib import Path
import importlib.util
import numpy as np
import pytest

from reteco import load_master_config
from reteco.config_schema import ParameterCoordinateTransform
from reteco.inverse.regularization import TikhonovRegularizer


def _load_exp07_module():
    spec = importlib.util.spec_from_file_location(
        "exp07_multiseed", "experiments/07_cpso_multiseed_variability.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_target_sha256_verification_and_mismatch_abort() -> None:
    exp07 = _load_exp07_module()
    vec = np.linspace(0.1, 2.0, 180, dtype=np.float64)
    expected_hash = exp07.compute_sha256_vector(vec)

    assert (
        exp07.verify_target_hash_or_abort(vec, expected_sha256=expected_hash)
        == expected_hash
    )

    corrupted = vec.copy()
    corrupted[0] += 1.0e-12
    with pytest.raises(RuntimeError, match="Hash SHA-256 del target non corrispondente"):
        exp07.verify_target_hash_or_abort(corrupted, expected_sha256=expected_hash)


def test_multiseed_600_consistency_and_log_figure_generation(tmp_path: Path) -> None:
    exp07 = _load_exp07_module()
    master = load_master_config("configs/scales_and_physics.yaml")
    transform = ParameterCoordinateTransform(master.parameter_space_theta.bounds)
    theta_true = master.parameter_space_theta.theta_dagger_array()

    grid_s = np.linspace(0.1, 1.0, 180, dtype=np.float64)

    def surrogate_G(theta: np.ndarray) -> np.ndarray:
        th = np.asarray(theta, dtype=np.float64).ravel()
        return (
            th[0] * np.exp(-th[2] * grid_s)
            + 0.20 * th[1] * np.sin(2.0 * np.pi * grid_s)
            + 0.12 * th[3] * (grid_s ** 2)
        )

    y_clean = surrogate_G(theta_true)
    sha = exp07.compute_sha256_vector(y_clean)
    reg = TikhonovRegularizer(
        transform=transform,
        lambda_reg=1.0e-4,
        theta_tilde_0=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64),
    )

    records = []
    for seed in (20261017, 20261018, 20261019):
        rec = exp07.execute_single_cpso_600_run(
            optimizer_seed=seed,
            data_seed=20261016,
            y_target_clean=y_clean,
            expected_target_sha256=sha,
            forward_operator_inv=surrogate_G,
            transform=transform,
            regularizer=reg,
            theta_dagger=theta_true,
        )
        assert len(rec.history) == 600
        assert rec.optimizer_fem_evaluations == 600
        assert rec.consistency_checks["all_checks_passed"] is True
        records.append(rec)

    assert len({r.optimizer_seed for r in records}) == 3

    png_path, pdf_path = exp07.plot_multiseed_convergence_figure(
        records, j_total_at_theta_dagger=4.984481939398449e-06, out_dir=tmp_path
    )
    assert png_path.exists() and png_path.stat().st_size > 0
    assert pdf_path.exists() and pdf_path.stat().st_size > 0

    csv_path = tmp_path / "cpso_multiseed_comparison.csv"
    rows = exp07.write_multiseed_comparison_csv(
        records=records,
        target_sha256=sha,
        j_total_at_true=4.984481939398449e-06,
        j_data_at_true=6.23e-07,
        j_reg_at_true=4.361e-06,
        csv_path=csv_path,
    )
    assert len(rows) == 3
    assert csv_path.exists()