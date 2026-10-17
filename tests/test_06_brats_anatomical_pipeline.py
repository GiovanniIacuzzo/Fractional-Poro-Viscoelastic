"""
tests/test_06_brats_anatomical_pipeline.py
Suite di test automatici per la Fase 1 (Pipeline Anatomica BraTS 2020, Provenienza
SHA-256, Separazione Clinico vs Sintetico e Mesh Quadrilatera 2D).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
import pytest
import yaml

import scripts.run_brats_anatomical_pipeline as cli_pipeline_mod
from reteco.anatomy import brats_geometry as bg_mod
from reteco.anatomy.brats_geometry import (
    BOUNDARY_ID_INNER_HOLES,
    BOUNDARY_ID_OUTER_CONTOUR,
    AnatomicalCoordinateTransform2D,
    BraTSPipelineConfig,
    MeshGenerationConfig,
    ProvenanceConfig,
    build_anatomical_masks_2d,
    load_brats_pipeline_config,
    load_brats_subject,
    load_l0_from_physics_config,
    run_brats_anatomical_pipeline,
    select_anatomical_slice,
)
from tests.synthetic_brats_fixture import (
    SYNTHETIC_HEADER_MARKER,
    create_synthetic_brats_case,
)


def _build_synthetic_nifti_subject(
    tmp_dir: Path,
    subject_id: str = "SubjTest01",
    misalign_flair: bool = False,
    inject_invalid_label: bool = False,
    include_wm_mask: bool = False,
) -> Tuple[Path, float]:
    subj_dir = tmp_dir / subject_id
    subj_dir.mkdir(parents=True, exist_ok=True)

    shape = (140, 168, 12)
    affine_lps = np.array(
        [
            [-1.0, 0.0, 0.0, 70.0],
            [0.0, -1.0, 0.0, 84.0],
            [0.0, 0.0, 2.5, -15.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    ii, jj, kk = np.meshgrid(
        np.arange(shape[0], dtype=np.float64),
        np.arange(shape[1], dtype=np.float64),
        np.arange(shape[2], dtype=np.float64),
        indexing="ij",
    )
    x_mm = affine_lps[0, 0] * ii + affine_lps[0, 3]
    y_mm = affine_lps[1, 1] * jj + affine_lps[1, 3]
    z_mm = affine_lps[2, 2] * kk + affine_lps[2, 3]

    outer_ell = (x_mm / 55.0) ** 2 + (y_mm / 70.0) ** 2 <= 1.0
    inner_hole = ((x_mm + 8.0) / 10.0) ** 2 + ((y_mm - 5.0) / 14.0) ** 2 <= 1.0
    domain_mask = outer_ell & (~inner_hole)

    z_factor = np.maximum(0.2, 1.0 - (z_mm / 15.0) ** 2)
    r2_tumor = (x_mm - 20.0) ** 2 + (y_mm + 15.0) ** 2
    edema = (r2_tumor <= (18.0 * z_factor) ** 2) & domain_mask
    enhancing = (r2_tumor <= (11.0 * z_factor) ** 2) & domain_mask
    necrotic = (r2_tumor <= (5.5 * z_factor) ** 2) & domain_mask

    seg = np.zeros(shape, dtype=np.int16)
    seg[edema] = 2
    seg[enhancing] = 4
    seg[necrotic] = 1
    if inject_invalid_label:
        seg[70, 84, 6] = 7

    t1 = np.where(domain_mask, 700.0, 0.0).astype(np.float32)
    t1ce = np.where(enhancing, 1200.0, t1).astype(np.float32)
    t2 = np.where(edema, 1000.0, np.where(domain_mask, 500.0, 0.0)).astype(np.float32)
    flair = np.where(edema, 1100.0, np.where(domain_mask, 420.0, 0.0)).astype(np.float32)

    nib.save(nib.Nifti1Image(t1, affine_lps), str(subj_dir / f"{subject_id}_t1.nii.gz"))
    nib.save(nib.Nifti1Image(t1ce, affine_lps), str(subj_dir / f"{subject_id}_t1gd.nii"))
    nib.save(nib.Nifti1Image(t2, affine_lps), str(subj_dir / f"{subject_id}_t2.nii.gz"))

    if misalign_flair:
        affine_shifted = affine_lps.copy()
        affine_shifted[0, 3] += 1.0
        nib.save(nib.Nifti1Image(flair, affine_shifted), str(subj_dir / f"{subject_id}_flair.nii.gz"))
        nib.save(nib.Nifti1Image(seg, affine_shifted), str(subj_dir / f"{subject_id}_seg.nii"))
    else:
        nib.save(nib.Nifti1Image(flair, affine_lps), str(subj_dir / f"{subject_id}_flair.nii.gz"))
        nib.save(nib.Nifti1Image(seg, affine_lps), str(subj_dir / f"{subject_id}_seg.nii"))

    if include_wm_mask:
        wm_arr = (domain_mask & (~edema)).astype(np.uint8)
        nib.save(nib.Nifti1Image(wm_arr, affine_lps), str(subj_dir / "independent_wm_mask.nii.gz"))

    analytical_area_mm2 = float(np.pi * (55.0 * 70.0 - 10.0 * 14.0))
    return subj_dir, analytical_area_mm2


def _synthetic_provenance_cfg(forbid_realign: bool = False) -> ProvenanceConfig:
    return ProvenanceConfig(
        expected_source_data_kind="synthetic_test_fixture",
        require_explicit_modality_filenames=False,
        expected_shape_3d=None,
        expected_spacing_mm=None,
        forbid_implicit_realignment=forbid_realign,
    )


def test_01_loading_orientation_realignment_and_labels(tmp_path: Path) -> None:
    subj_dir, _ = _build_synthetic_nifti_subject(
        tmp_path, subject_id="BraTS_Realign", misalign_flair=True
    )
    volumes = load_brats_subject(
        subj_dir,
        subject_id="BraTS_Realign",
        allow_grid_realignment=True,
        provenance_cfg=_synthetic_provenance_cfg(forbid_realign=False),
    )

    assert volumes.orientation_convention == "RAS+"
    assert volumes.source_data_kind == "synthetic_test_fixture"
    assert set(volumes.modalities.keys()) == {"T1", "T1ce", "T2", "FLAIR"}
    assert set(volumes.seg_labels_after_preprocessing).issubset({0, 1, 2, 4})
    assert len(volumes.realignment_log) == 2

    with pytest.raises(ValueError, match="Disallineamento di griglia"):
        load_brats_subject(
            subj_dir,
            subject_id="BraTS_Realign",
            allow_grid_realignment=False,
            provenance_cfg=_synthetic_provenance_cfg(forbid_realign=True),
        )


def test_02_invalid_segmentation_label_rejected(tmp_path: Path) -> None:
    subj_dir, _ = _build_synthetic_nifti_subject(
        tmp_path, subject_id="BraTS_BadLabel", inject_invalid_label=True
    )
    with pytest.raises(ValueError, match="Label non ammesse"):
        load_brats_subject(
            subj_dir,
            subject_id="BraTS_BadLabel",
            provenance_cfg=_synthetic_provenance_cfg(),
        )


def test_03_slice_selection_manual_and_auto(tmp_path: Path) -> None:
    subj_dir, _ = _build_synthetic_nifti_subject(tmp_path, subject_id="BraTS_Slice")
    volumes = load_brats_subject(
        subj_dir, subject_id="BraTS_Slice", provenance_cfg=_synthetic_provenance_cfg()
    )

    auto_sel = select_anatomical_slice(volumes, mode="auto_max_tumor")
    assert auto_sel.slice_index == 6
    assert np.isclose(auto_sel.physical_z_mm, 0.0, atol=1.0e-6)

    man_sel = select_anatomical_slice(volumes, mode="manual", manual_slice_index=4)
    assert man_sel.slice_index == 4
    assert np.isclose(man_sel.physical_z_mm, -5.0, atol=1.0e-6)


def test_04_anatomical_masks_and_wm_distinction(tmp_path: Path) -> None:
    subj_dir, _ = _build_synthetic_nifti_subject(
        tmp_path, subject_id="BraTS_Masks", include_wm_mask=True
    )
    vol_no_wm = load_brats_subject(
        subj_dir, subject_id="BraTS_Masks", wm_mask_path=None, provenance_cfg=_synthetic_provenance_cfg()
    )
    sel_no_wm = select_anatomical_slice(vol_no_wm, mode="auto_max_tumor")
    masks_no_wm = build_anatomical_masks_2d(sel_no_wm, spacing_xy_mm=(1.0, 1.0))

    assert masks_no_wm.white_matter_mask is None
    assert masks_no_wm.domain_representation_type == "brain_parenchyma_proxy"
    assert any("WM_MASK_UNAVAILABLE" in w for w in masks_no_wm.warnings)
    assert masks_no_wm.preserved_internal_holes_count == 1
    assert masks_no_wm.lesion_inside_support_ratio == 1.0

    vol_wm = load_brats_subject(
        subj_dir,
        subject_id="BraTS_Masks",
        wm_mask_path=subj_dir / "independent_wm_mask.nii.gz",
        provenance_cfg=_synthetic_provenance_cfg(),
    )
    sel_wm = select_anatomical_slice(vol_wm, mode="auto_max_tumor")
    masks_wm = build_anatomical_masks_2d(sel_wm, spacing_xy_mm=(1.0, 1.0))

    assert masks_wm.white_matter_mask is not None
    assert masks_wm.domain_representation_type == "white_matter_segmented"
    assert len(masks_wm.warnings) == 0


def test_05_coordinate_transform_and_l0_consistency(tmp_path: Path) -> None:
    l0_m, l0_mm = load_l0_from_physics_config("configs/scales_and_physics.yaml")
    assert np.isclose(l0_m, 1.0e-2)
    assert np.isclose(l0_mm, 10.0)

    subj_dir, _ = _build_synthetic_nifti_subject(tmp_path, subject_id="BraTS_Coords")
    volumes = load_brats_subject(
        subj_dir, subject_id="BraTS_Coords", provenance_cfg=_synthetic_provenance_cfg()
    )
    sel = select_anatomical_slice(volumes, mode="auto_max_tumor")
    masks = build_anatomical_masks_2d(sel, spacing_xy_mm=(1.0, 1.0))

    transform = AnatomicalCoordinateTransform2D.from_ras_affine_and_mask(
        volumes.ras_affine, sel.slice_index, masks.brain_support_mask, L0_m=l0_m
    )

    test_ij = np.array([[25.0, 40.0], [70.0, 84.0], [110.5, 120.25]])
    phys_mm = transform.voxel_ij_to_physical_mm(test_ij)
    adim_xy = transform.physical_mm_to_mesh_adim(phys_mm)
    back_mm = transform.mesh_adim_to_physical_mm(adim_xy)
    back_ij = transform.physical_mm_to_voxel_ij(back_mm)

    assert np.allclose(phys_mm, back_mm, atol=1.0e-12)
    assert np.allclose(test_ij, back_ij, atol=1.0e-12)
    dist_mm = float(np.linalg.norm(phys_mm[1] - phys_mm[0]))
    dist_adim = float(np.linalg.norm(adim_xy[1] - adim_xy[0]))
    assert np.isclose(dist_mm / dist_adim, l0_mm, rtol=1.0e-12)


def test_06_pure_quad_mesh_and_full_pipeline_deliverables(tmp_path: Path) -> None:
    subj_dir, analytical_area_mm2 = _build_synthetic_nifti_subject(
        tmp_path, subject_id="BraTS_QuadMesh"
    )
    out_dir = tmp_path / "pipeline_out"

    cfg = BraTSPipelineConfig(
        subject_id="BraTS_QuadMesh",
        subject_dir=subj_dir,
        physics_config_path=Path("configs/scales_and_physics.yaml"),
        output_dir=out_dir,
        provenance=_synthetic_provenance_cfg(),
        mesh=MeshGenerationConfig(target_element_size_mm=3.0),
    )

    report, mesh = run_brats_anatomical_pipeline(cfg)

    assert report.all_mandatory_checks_passed is True
    assert report.source_data_kind == "synthetic_test_fixture"
    assert report.num_triangles == 0
    assert mesh.quads.shape[1] == 4
    assert report.cell_geometry_checks["all_cells_positive_area"] is True
    assert report.cell_geometry_checks["all_corners_positive_jacobian"] is True
    assert report.cell_geometry_checks["num_inverted_or_degenerate_cells"] == 0
    assert report.relative_area_error < 0.02
    assert abs(report.mesh_area_mm2 - analytical_area_mm2) / analytical_area_mm2 < 0.02
    assert report.rasterized_dice_similarity > 0.96
    assert str(BOUNDARY_ID_OUTER_CONTOUR) in report.boundary_metrics["by_boundary_id"]
    assert str(BOUNDARY_ID_INNER_HOLES) in report.boundary_metrics["by_boundary_id"]

    for fpath in report.exported_files.values():
        p = Path(fpath)
        assert p.exists() and p.stat().st_size > 0


# =============================================================================
# Nuovi Test Obbligatori (Requisito 9a, 9b, 9c, 9d)
# =============================================================================

def test_07_provenance_9a_synthetic_generator_confined_to_test_fixtures() -> None:
    """
    Requisito 9(a): Dimostra che `create_synthetic_brats_case` è confinato
    al modulo di test `tests/synthetic_brats_fixture.py` e non è presente né
    nello script di produzione né nel modulo `brats_geometry.py`.
    """
    assert not hasattr(cli_pipeline_mod, "create_synthetic_brats_case")
    assert not hasattr(bg_mod, "create_synthetic_brats_case")
    cli_src = inspect.getsource(cli_pipeline_mod)
    assert "create_synthetic_brats_case" not in cli_src
    assert "--generate-synthetic-brats-if-missing" not in cli_src


def test_08_provenance_9b_clinical_pipeline_fails_on_missing_data_without_generating_files(
    tmp_path: Path,
) -> None:
    """
    Requisito 9(b): Se i dati clinici mancano, il comando della pipeline termina
    con un errore esplicito senza creare file NIfTI alternativi.
    """
    missing_dir = tmp_path / "missing_clinical_raw"
    cfg_dict = {
        "subject_id": "BraTS20_Training_001",
        "subject_dir": str(missing_dir),
        "physics_config_path": "configs/scales_and_physics.yaml",
        "output_dir": str(tmp_path / "out_clinical"),
        "modality_filenames": {
            "T1": "BraTS20_Training_001_t1.nii",
            "T1ce": "BraTS20_Training_001_t1ce.nii",
            "T2": "BraTS20_Training_001_t2.nii",
            "FLAIR": "BraTS20_Training_001_flair.nii",
            "seg": "BraTS20_Training_001_seg.nii",
        },
        "provenance": {
            "expected_source_data_kind": "clinical_brats2020",
            "require_explicit_modality_filenames": True,
            "expected_shape_3d": [240, 240, 155],
            "expected_spacing_mm": [1.0, 1.0, 1.0],
        },
    }
    cfg_path = tmp_path / "missing_clinical.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "scripts/run_brats_anatomical_pipeline.py", "--config", str(cfg_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "FileNotFoundError" in proc.stderr
    assert not missing_dir.exists()


def test_09_provenance_9c_9d_sha256_reporting_and_synthetic_cannot_be_labeled_clinical(
    tmp_path: Path,
) -> None:
    """
    Requisiti 9(c) e 9(d):
    - Verifica che SHA-256, percorsi assoluti e metadati NIfTI siano riportati esattamente;
    - Verifica che il caso sintetico (160, 192, 24) NON possa in alcun modo essere
      etichettato come `clinical_brats2020`.
    """
    synth_dir = tmp_path / "synth_case"
    fnames = create_synthetic_brats_case(synth_dir, subject_id="BraTS20_Training_001", use_dot_nii=True)

    # 9(d) Tentativo di caricare il caso sintetico come `clinical_brats2020`: deve fallire
    with pytest.raises(ValueError, match="Discrepanza di griglia|Violazione di provenienza"):
        load_brats_subject(
            subject_dir=synth_dir,
            subject_id="BraTS20_Training_001",
            modality_filenames=fnames,
            provenance_cfg=ProvenanceConfig(
                expected_source_data_kind="clinical_brats2020",
                require_explicit_modality_filenames=True,
                expected_shape_3d=(240, 240, 155),
                expected_spacing_mm=(1.0, 1.0, 1.0),
            ),
        )

    # Anche disabilitando il controllo 240x240x155, la firma sintetica impedisce il tag `clinical_brats2020`
    with pytest.raises(ValueError, match="Violazione di provenienza"):
        load_brats_subject(
            subject_dir=synth_dir,
            subject_id="BraTS20_Training_001",
            modality_filenames=fnames,
            provenance_cfg=ProvenanceConfig(
                expected_source_data_kind="clinical_brats2020",
                require_explicit_modality_filenames=True,
                expected_shape_3d=None,
                expected_spacing_mm=None,
            ),
        )

    # 9(c) Caricamento come `synthetic_test_fixture` e verifica SHA-256 e percorsi assoluti
    vols = load_brats_subject(
        subject_dir=synth_dir,
        subject_id="BraTS20_Training_001",
        modality_filenames=fnames,
        provenance_cfg=ProvenanceConfig(
            expected_source_data_kind="synthetic_test_fixture",
            require_explicit_modality_filenames=True,
            expected_shape_3d=None,
            expected_spacing_mm=None,
        ),
    )
    assert vols.source_data_kind == "synthetic_test_fixture"
    for k, fname in fnames.items():
        expected_abs = str((synth_dir / fname).resolve())
        expected_sha = hashlib.sha256((synth_dir / fname).read_bytes()).hexdigest()
        assert vols.file_paths[k] == expected_abs
        assert vols.volume_provenance[k].sha256 == expected_sha
        assert vols.volume_provenance[k].header_descrip == SYNTHETIC_HEADER_MARKER