"""
tests/synthetic_brats_fixture.py
Fixture sintetica confinata esclusivamente ai test automatici (Pytest).
NON deve mai essere importata o invocata dagli script di produzione clinica.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict
import nibabel as nib
import numpy as np

SYNTHETIC_HEADER_MARKER = "RETECO_SYNTHETIC_TEST_FIXTURE"


def create_synthetic_brats_case(
    subject_dir: Path,
    subject_id: str = "Synthetic_Phantom_001",
    use_dot_nii: bool = False,
) -> Dict[str, str]:
    """
    Genera un fantoccio NIfTI 3D sintetico con geometria analitica nota:
    - shape = (160, 192, 24), spacing = (1.0, 1.0, 2.0) mm;
    - affine con origine (-80, -96, -20) mm;
    - lesione centrata a (22, -12, 4) mm (sezione k = 12).
    Marca esplicitamente l'header NIfTI come `RETECO_SYNTHETIC_TEST_FIXTURE`.
    """
    subject_dir.mkdir(parents=True, exist_ok=True)
    shape = (160, 192, 24)
    spacing = (1.0, 1.0, 2.0)

    affine = np.array(
        [
            [spacing[0], 0.0, 0.0, -80.0],
            [0.0, spacing[1], 0.0, -96.0],
            [0.0, 0.0, spacing[2], -20.0],
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
    x_mm = affine[0, 0] * ii + affine[0, 3]
    y_mm = affine[1, 1] * jj + affine[1, 3]
    z_mm = affine[2, 2] * kk + affine[2, 3]

    r_ell = (x_mm / 60.0) ** 2 + (y_mm / 74.0) ** 2 + (z_mm / 45.0) ** 2
    fissure_indent = 0.10 * np.exp(-((x_mm / 7.5) ** 2)) * ((y_mm / 74.0) ** 4)
    parenchyma = (r_ell + fissure_indent) <= 1.0

    ventricle = (
        ((x_mm + 10.0) / 8.5) ** 2 + ((y_mm - 4.0) / 13.0) ** 2 + ((z_mm - 4.0) / 18.0) ** 2
    ) <= 1.0
    brain_tissue = parenchyma & (~ventricle)

    dist2_lesion = (x_mm - 22.0) ** 2 + (y_mm + 12.0) ** 2 + ((z_mm - 4.0) * 1.5) ** 2
    edema = (dist2_lesion <= 21.0 ** 2) & brain_tissue
    enhancing = (dist2_lesion <= 12.5 ** 2) & brain_tissue
    necrotic = (dist2_lesion <= 6.5 ** 2) & brain_tissue

    seg = np.zeros(shape, dtype=np.int16)
    seg[edema] = 2
    seg[enhancing] = 4
    seg[necrotic] = 1

    rng = np.random.default_rng(202604)
    t1 = np.where(brain_tissue, 650.0 + 40.0 * np.cos(x_mm / 20.0), 0.0)
    t1[edema] = 480.0
    t1[necrotic] = 310.0

    t1ce = t1.copy()
    t1ce[enhancing] = 1150.0

    t2 = np.where(brain_tissue, 520.0 + 30.0 * np.sin(y_mm / 25.0), 0.0)
    t2[edema] = 980.0
    t2[necrotic] = 890.0

    flair = np.where(brain_tissue, 450.0, 0.0)
    flair[edema] = 1050.0
    flair[enhancing] = 880.0

    for arr in (t1, t1ce, t2, flair):
        noise = rng.normal(0.0, 8.0, size=shape)
        arr[brain_tissue] = np.clip(arr[brain_tissue] + noise[brain_tissue], 50.0, None)

    ext = ".nii" if use_dot_nii else ".nii.gz"
    filenames = {
        "T1": f"{subject_id}_t1{ext}",
        "T1ce": f"{subject_id}_t1ce{ext}",
        "T2": f"{subject_id}_t2{ext}",
        "FLAIR": f"{subject_id}_flair{ext}",
        "seg": f"{subject_id}_seg{ext}",
    }
    arrays = {
        "T1": t1.astype(np.float32),
        "T1ce": t1ce.astype(np.float32),
        "T2": t2.astype(np.float32),
        "FLAIR": flair.astype(np.float32),
        "seg": seg,
    }

    for k, fname in filenames.items():
        img = nib.Nifti1Image(arrays[k], affine)
        img.header["descrip"] = SYNTHETIC_HEADER_MARKER.encode("ascii")
        nib.save(img, str(subject_dir / fname))

    return filenames