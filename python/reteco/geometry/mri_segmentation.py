"""
python/reteco/geometry/mri_segmentation.py
Estrazione rigorosa di contorni 2D da volumi NIfTI con trasformazione affine voxel -> mm -> L_0.
Non etichetta mai il contorno esterno dell'encefalo come "sostanza bianca segmentata" (Contratto Sez. 10).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class AnatomicalContour2D:
    region_type: str               # "whole_brain_parenchyma_silhouette" oppure "explicit_label_mask"
    slice_index: int
    voxel_spacing_mm: tuple[float, float, float]
    affine_matrix: np.ndarray
    contour_mm: np.ndarray         # Coordinate fisiche in millimetri (N_pts, 2)
    contour_adim: np.ndarray       # Coordinate adimensionalizzate x* = x_m / L_0 (N_pts, 2)


def extract_anatomical_contour_from_nifti(
    nifti_path: str | Path,
    L0_m: float,
    slice_index: int | None = None,
    intensity_threshold: float | None = None,
    simplify_tolerance_adim: float = 0.01,
) -> AnatomicalContour2D:
    """
    Legge un volume NIfTI (.nii / .nii.gz), applica la matrice affine reale dello scanner
    per convertire gli indici voxel (i, j, k) in coordinate fisiche millimetriche e
    infine adimensionalizza rispetto a L0_m.
    """
    import nibabel as nib
    from skimage import measure
    from skimage.filters import threshold_otsu

    path = Path(nifti_path)
    if not path.exists():
        raise FileNotFoundError(f"File NIfTI non trovato: {path}")

    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float64)
    affine = np.asarray(img.affine, dtype=np.float64)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])

    if data.ndim != 3:
        raise ValueError(f"Atteso volume NIfTI 3D, ricevuto array di dimensione {data.ndim}.")

    if slice_index is None:
        slice_index = data.shape[2] // 2
    if not (0 <= slice_index < data.shape[2]):
        raise ValueError(f"slice_index={slice_index} fuori range [0, {data.shape[2]-1}].")

    slice_2d = data[:, :, slice_index]
    if np.max(slice_2d) <= np.min(slice_2d):
        raise ValueError("La slice MRI selezionata è costante/vuota.")

    thresh = float(threshold_otsu(slice_2d)) if intensity_threshold is None else float(intensity_threshold)
    contours_ijk = measure.find_contours(slice_2d, level=thresh)
    if not contours_ijk:
        raise RuntimeError("Nessun contorno chiuso trovato nella slice MRI.")

    # Selezioniamo il contorno esterno di lunghezza massima
    longest_ijk = max(contours_ijk, key=lambda c: c.shape[0])

    # Conversione tramite matrice affine NIfTI: [i, j, slice_index, 1]^T -> [x_mm, y_mm, z_mm, 1]^T
    n_pts = longest_ijk.shape[0]
    homog_ijk = np.column_stack(
        [longest_ijk[:, 0], longest_ijk[:, 1], np.full(n_pts, slice_index), np.ones(n_pts)]
    )
    xyz_mm = (homog_ijk @ affine.T)[:, :2]

    # Centratura sul centroide e conversione in unità adimensionali tramite L0_mm = L0_m * 1000
    L0_mm = L0_m * 1000.0
    centroid_mm = np.mean(xyz_mm, axis=0)
    contour_mm = xyz_mm - centroid_mm
    contour_adim = contour_mm / L0_mm

    if simplify_tolerance_adim > 0.0:
        contour_adim = measure.approximate_polygon(contour_adim, tolerance=simplify_tolerance_adim)
        if np.allclose(contour_adim[0], contour_adim[-1]):
            contour_adim = contour_adim[:-1]
        contour_mm = contour_adim * L0_mm

    return AnatomicalContour2D(
        region_type="whole_brain_parenchyma_silhouette",
        slice_index=slice_index,
        voxel_spacing_mm=(zooms[0], zooms[1], zooms[2]),
        affine_matrix=affine,
        contour_mm=contour_mm,
        contour_adim=contour_adim,
    )