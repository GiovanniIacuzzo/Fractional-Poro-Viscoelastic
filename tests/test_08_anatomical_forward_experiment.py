"""
tests/test_08_anatomical_forward_experiment.py
Verifica automatica per `experiments/02_run_anatomical_forward.py`:
- Punto di ingresso CLI `main()` e sottoprocesso;
- Norma L^2(Omega) vettoriale dello spostamento tramite M_d vs seminorma di rigidezza K0;
- Creazione della figura energetica anatomica dedicata;
- Arresto in presenza di una mesh che non supera i controlli di qualità;
- Identificazione della mesh più fine indipendentemente dall'ordine in YAML.
"""

from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import yaml

from reteco import _core, load_master_config
from reteco.forward.wrapper import RetecoForwardProblem2D
from reteco.visualization.anatomical_forward_plots import (
    plot_anatomical_energy_and_mass_diagnostics,
)
from tests.synthetic_brats_fixture import create_synthetic_brats_case


def _load_exp02_module():
    exp_file = Path("experiments/02_run_anatomical_forward.py").resolve()
    spec = importlib.util.spec_from_file_location("exp02_anatomical", exp_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_01_vector_displacement_l2_norm_on_known_fields() -> None:
    """
    Requisito 3 & 5: Verifica il calcolo di ||d_h||_{L^2(Omega)} = sqrt(d^T M_d d)
    su campi vettoriali a valore analitico noto e lo distingue da sqrt(d^T K_0 d).
    """
    master = load_master_config("configs/scales_and_physics.yaml")
    grid = _core.TimeGrid.make_graded(0.2, 4, 1.0)
    problem = RetecoForwardProblem2D(
        subdivisions_N=8,
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid,
    )

    # Caso A: Campo costante d(x,y) = (3.0, -4.0) su [0,1]^2 (|Omega| = 1.0)
    # ||d||_{L^2([0,1]^2)} = sqrt(3^2 + (-4)^2) = 5.0, mentre K_0 * d_const = 0 a precisione di macchina
    d_const = np.zeros(problem.num_disp_dofs, dtype=np.float64)
    d_const[0::2] = 3.0
    d_const[1::2] = -4.0

    l2_const = problem.compute_disp_L2_norm(d_const)
    stiff_const = problem.compute_disp_stiffness_seminorm(d_const)
    k0_residual_inf = float(np.max(np.abs(problem.K0_csr @ d_const)))

    assert np.isclose(l2_const, 5.0, rtol=1.0e-12, atol=1.0e-12)
    assert k0_residual_inf < 1.0e-12
    assert stiff_const < 1.0e-5  # sqrt(d^T K_0 d) ~ sqrt(eps_mach) ~ 1e-7

    # Caso B: Campo lineare d(x,y) = (a * x, b * y) su [0,1]^2
    # int_0^1 int_0^1 (a^2 x^2 + b^2 y^2) dx dy = (a^2 + b^2) / 3
    a_val, b_val = 2.0, 1.5
    pts_d = problem.disp_pts
    d_lin = np.zeros(problem.num_disp_dofs, dtype=np.float64)
    d_lin[0::2] = a_val * pts_d[0::2, 0]
    d_lin[1::2] = b_val * pts_d[1::2, 1]

    expected_l2_lin = float(np.sqrt((a_val ** 2 + b_val ** 2) / 3.0))
    l2_lin = problem.compute_disp_L2_norm(d_lin)
    stiff_lin = problem.compute_disp_stiffness_seminorm(d_lin)
    assert np.isclose(l2_lin, expected_l2_lin, rtol=1.0e-12, atol=1.0e-12)
    assert stiff_lin > 0.5


def test_02_dedicated_anatomical_energy_figure_creation(tmp_path: Path) -> None:
    """
    Requisito 2 & 5: Verifica che `plot_anatomical_energy_and_mass_diagnostics(diagnostics, output_path)`
    generi correttamente il PDF su una singola traiettoria forzata.
    """
    master = load_master_config("configs/scales_and_physics.yaml")
    grid = _core.TimeGrid.make_graded(0.3, 10, 1.2)
    problem = RetecoForwardProblem2D(
        subdivisions_N=6,
        physics_model=master.fixed_physics,
        scales=master.characteristic_scales,
        time_grid=grid,
    )
    c0 = problem.build_gaussian_initial_tracer()
    s_fluid = lambda t: 0.2 * problem.M_p_csr.dot(np.ones(problem.num_press_dofs))
    traj = problem.solve_trajectory(
        theta_vec=master.parameter_space_theta.theta_dagger_array(),
        c0=c0,
        S_fluid_func=s_fluid,
        compute_diagnostics=True,
    )

    out_pdf = tmp_path / "test_anatomical_energy_diag.pdf"
    plot_anatomical_energy_and_mass_diagnostics(traj.diagnostics, out_pdf)
    assert out_pdf.exists() and out_pdf.stat().st_size > 0


def test_03_cli_entrypoint_and_unordered_resolutions(tmp_path: Path) -> None:
    """
    Requisiti 1, 4 & 5:
    - Verifica che `main(["--config", ...])` funzioni senza TypeError;
    - Verifica che la mesh più fine sia identificata tramite min(target_element_size_mm)
      anche quando `resolutions_mm` in YAML non è ordinata ([4.0, 3.0, 5.0]);
    - Verifica la presenza delle nuove chiavi `final_disp_l2_norm` e `final_disp_stiffness_seminorm`.
    """
    mod = _load_exp02_module()
    subj_dir = tmp_path / "BraTS20_TestSubj"
    fnames = create_synthetic_brats_case(subj_dir, subject_id="BraTS20_TestSubj", use_dot_nii=True)

    with open("configs/brats_anatomical_forward.yaml", "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    cfg["subject_id"] = "BraTS20_TestSubj"
    cfg["subject_dir"] = str(subj_dir)
    cfg["modality_filenames"] = fnames
    cfg["output_dir"] = str(tmp_path / "phase1_out")
    cfg["experiment_output_dir"] = str(tmp_path / "02_anatomical_forward_test")
    cfg["provenance"] = {
        "expected_source_data_kind": "synthetic_test_fixture",
        "require_explicit_modality_filenames": True,
        "expected_shape_3d": [160, 192, 24],
        "expected_spacing_mm": [1.0, 1.0, 2.0],
        "spacing_atol_mm": 1.0e-3,
        "forbid_implicit_realignment": True,
    }
    cfg["forward_simulation"]["time_steps_M"] = 8
    cfg["mesh_sensitivity_study"]["resolutions_mm"] = [4.0, 3.0, 5.0]
    cfg["fractional_sensitivity_study"]["alpha_values"] = [0.40, 0.70, 0.90]
    cfg["fractional_sensitivity_study"]["beta_values"] = [0.25, 0.45, 0.75]

    tmp_cfg_path = tmp_path / "test_brats_anatomical_forward.yaml"
    with tmp_cfg_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)

    rc = mod.main(["--config", str(tmp_cfg_path)])
    assert rc == 0

    summary_path = tmp_path / "02_anatomical_forward_test" / "anatomical_forward_experiment_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["all_mandatory_checks_passed"] is True
    assert np.isclose(summary["mesh_sensitivity_study"]["finest_reference_h_mm"], 3.0)

    recs_by_h = {r["target_element_size_mm"]: r for r in summary["mesh_sensitivity_study"]["resolutions_evaluated"]}
    assert np.isclose(recs_by_h[3.0]["rel_diff_final_c_l2_vs_finest"], 0.0, atol=1.0e-15)
    assert np.isclose(recs_by_h[3.0]["rel_diff_final_disp_l2_vs_finest"], 0.0, atol=1.0e-15)
    assert "final_disp_l2_norm" in recs_by_h[3.0]
    assert "final_disp_stiffness_seminorm" in recs_by_h[3.0]

    for name, fpath in summary["generated_artifacts"].items():
        p = Path(fpath)
        assert p.exists() and p.stat().st_size > 0, f"Artefatto mancante o vuoto: {name} -> {fpath}"


def test_04_halt_on_mesh_quality_failure_in_sensitivity_study(tmp_path: Path) -> None:
    """
    Requisito 4 & 5: Verifica che l'esperimento si interrompa con RuntimeError se
    una delle mesh nello studio di sensibilità non supera i controlli di qualità.
    """
    mod = _load_exp02_module()
    subj_dir = tmp_path / "BraTS20_FailMesh"
    fnames = create_synthetic_brats_case(subj_dir, subject_id="BraTS20_FailMesh", use_dot_nii=True)

    with open("configs/brats_anatomical_forward.yaml", "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    cfg["subject_id"] = "BraTS20_FailMesh"
    cfg["subject_dir"] = str(subj_dir)
    cfg["modality_filenames"] = fnames
    cfg["output_dir"] = str(tmp_path / "phase1_fail")
    cfg["experiment_output_dir"] = str(tmp_path / "02_fail_out")
    cfg["provenance"] = {
        "expected_source_data_kind": "synthetic_test_fixture",
        "require_explicit_modality_filenames": True,
        "expected_shape_3d": [160, 192, 24],
        "expected_spacing_mm": [1.0, 1.0, 2.0],
        "spacing_atol_mm": 1.0e-3,
        "forbid_implicit_realignment": True,
    }
    cfg["forward_simulation"]["time_steps_M"] = 4
    # Soglia stringente superata dalla mesh di riferimento (h=3.0 mm, Dice~0.988, err_area~0.0018),
    # ma violata dalla mesh grossolana h=16.0 mm nello studio di sensibilità
    cfg["quality_thresholds"] = {
        "max_relative_area_error": 0.005,
        "min_dice_similarity": 0.985,
        "min_corner_jacobian_mm2": 1.0e-5,
    }
    cfg["mesh_sensitivity_study"]["resolutions_mm"] = [3.0, 16.0, 4.0]

    tmp_cfg_path = tmp_path / "test_fail_mesh_quality.yaml"
    with tmp_cfg_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)

    with pytest.raises(RuntimeError, match="Controlli di qualità geometrica FALLITI"):
        mod.run_anatomical_forward_experiment(tmp_cfg_path)