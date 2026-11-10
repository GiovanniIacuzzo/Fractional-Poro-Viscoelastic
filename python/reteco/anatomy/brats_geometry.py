"""
python/reteco/anatomy/brats_geometry.py
Pipeline anatomica 2D per BraTS 2020 con verifica rigorosa della provenienza
dei file NIfTI (SHA-256, metadati header, griglia 240x240x155 a 1 mm) e generazione
di mesh puramente quadrilatere verificate per deal.II (Triangulation<2>, FE_Q<2>).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage import measure
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import closing, disk, opening
import yaml

BRATS_VALID_LABELS: Tuple[int, ...] = (0, 1, 2, 4)
BOUNDARY_ID_OUTER_CONTOUR: int = 1
BOUNDARY_ID_INNER_HOLES: int = 2

REQUIRED_VOLUME_KEYS: Tuple[str, ...] = ("T1", "T1ce", "T2", "FLAIR", "seg")
SYNTHETIC_SHAPES_BLACKLIST: Tuple[Tuple[int, int, int], ...] = (
    (160, 192, 24),
    (140, 168, 12),
)
SYNTHETIC_HEADER_MARKER: str = "RETECO_SYNTHETIC_TEST_FIXTURE"


# =============================================================================
# 0. Configurazione, Provenienza e Caricamento di L0
# =============================================================================

@dataclass
class SupportMaskConfig:
    gaussian_sigma_mm: float = 1.0
    closing_radius_mm: float = 2.0
    opening_radius_mm: float = 1.0
    min_component_area_mm2: float = 150.0
    max_fillable_hole_area_mm2: float = 35.0
    otsu_scale_factor: float = 0.35


@dataclass
class MeshGenerationConfig:
    target_element_size_mm: float = 3.0
    # Optional conforming tensor-grid grading near pathology. Refinement follows
    # coordinate bands intersecting the ROI bounding box; interfaces are NOT
    # snapped to the voxel segmentation. No hanging nodes are introduced.
    roi_fine_element_size_mm: Optional[float] = None
    roi_refinement_margin_mm: float = 6.0
    roi_refinement_labels: Tuple[int, ...] = (1, 2, 4)
    boundary_projection_iterations: int = 2
    laplacian_smoothing_iterations: int = 6
    laplacian_relaxation: float = 0.28
    contour_simplify_tolerance_mm: float = 0.45
    min_hole_area_to_mesh_mm2: float = 40.0


@dataclass
class ProvenanceConfig:
    expected_source_data_kind: str = "clinical_brats2020"  # "clinical_brats2020" | "synthetic_test_fixture"
    require_explicit_modality_filenames: bool = True
    expected_shape_3d: Optional[Tuple[int, int, int]] = (240, 240, 155)
    expected_spacing_mm: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0)
    spacing_atol_mm: float = 1.0e-3
    forbid_implicit_realignment: bool = True


@dataclass
class VolumeProvenanceRecord:
    modality_key: str
    configured_filename: str
    absolute_path: str
    file_size_bytes: int
    sha256: str
    nifti_data_dtype: str
    raw_shape_3d: List[int]
    raw_zooms_mm: List[float]
    sform_code: int
    qform_code: int
    header_descrip: str
    original_orientation: str
    original_affine_4x4: List[List[float]]
    canonical_ras_shape_3d: List[int]
    canonical_ras_spacing_mm: List[float]
    canonical_ras_affine_4x4: List[List[float]]


@dataclass
class BraTSPipelineConfig:
    subject_id: str
    subject_dir: Path
    physics_config_path: Path
    output_dir: Path
    modality_filenames: Optional[Dict[str, str]] = None
    wm_mask_path: Optional[Path] = None
    slice_selection_mode: str = "auto_max_tumor"
    manual_slice_index: Optional[int] = None
    allow_grid_realignment: bool = False
    remap_label_3_to_4: bool = False
    provenance: ProvenanceConfig = field(default_factory=ProvenanceConfig)
    support_mask: SupportMaskConfig = field(default_factory=SupportMaskConfig)
    mesh: MeshGenerationConfig = field(default_factory=MeshGenerationConfig)
    quality_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "max_relative_area_error": 0.05,
            "min_dice_similarity": 0.92,
            "min_corner_jacobian_mm2": 1.0e-5,
        }
    )


def _compute_file_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def load_l0_from_physics_config(physics_config_path: str | Path) -> Tuple[float, float]:
    """
    Carica la scala caratteristica L0 dalla configurazione fisica RETECO esistente
    (`configs/scales_and_physics.yaml`), senza introdurre alcuna copia secondaria.
    Restituisce (L0_m, L0_mm).
    """
    path = Path(physics_config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configurazione fisica RETECO non trovata: {path}")

    try:
        from ..config_schema import load_master_config
        master_cfg = load_master_config(path)
        l0_m = float(master_cfg.characteristic_scales.L0_m)
        return l0_m, l0_m * 1000.0
    except Exception:
        pass

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    scales = raw.get("characteristic_scales", {})
    if "L0_m" in scales:
        l0_m = float(scales["L0_m"])
        l0_mm = l0_m * 1000.0
    elif "L0_mm" in scales:
        l0_mm = float(scales["L0_mm"])
        l0_m = l0_mm * 1.0e-3
    else:
        raise KeyError(
            f"Impossibile trovare 'L0_m' o 'L0_mm' in characteristic_scales di {path}."
        )

    if not np.isfinite(l0_m) or l0_m <= 0.0:
        raise ValueError(f"Scala L0 non valida: L0_m={l0_m}")
    return l0_m, l0_mm


def load_brats_pipeline_config(config_path: str | Path) -> BraTSPipelineConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"File di configurazione anatomica non trovato: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if "L0_m" in raw or "L0_mm" in raw or "characteristic_scales" in raw:
        raise ValueError(
            "Violazione di configurazione: L0 non deve essere duplicato in "
            "brats_anatomical_forward.yaml. Usa esclusivamente physics_config_path."
        )

    sup_raw = raw.get("support_mask", {})
    mesh_raw = raw.get("mesh", {})
    prov_raw = dict(raw.get("provenance", {}))
    if "expected_shape_3d" in prov_raw and prov_raw["expected_shape_3d"] is not None:
        prov_raw["expected_shape_3d"] = tuple(int(x) for x in prov_raw["expected_shape_3d"])
    if "expected_spacing_mm" in prov_raw and prov_raw["expected_spacing_mm"] is not None:
        prov_raw["expected_spacing_mm"] = tuple(float(x) for x in prov_raw["expected_spacing_mm"])

    wm_raw = raw.get("wm_mask_path", None)

    return BraTSPipelineConfig(
        subject_id=str(raw["subject_id"]),
        subject_dir=Path(raw["subject_dir"]),
        physics_config_path=Path(raw.get("physics_config_path", "configs/scales_and_physics.yaml")),
        output_dir=Path(raw.get("output_dir", "artifacts/brats_anatomical_phase1_clinical")),
        modality_filenames=raw.get("modality_filenames", None),
        wm_mask_path=Path(wm_raw) if wm_raw else None,
        slice_selection_mode=str(raw.get("slice_selection_mode", "auto_max_tumor")),
        manual_slice_index=raw.get("manual_slice_index", None),
        allow_grid_realignment=bool(raw.get("allow_grid_realignment", False)),
        remap_label_3_to_4=bool(raw.get("remap_label_3_to_4", False)),
        provenance=ProvenanceConfig(**prov_raw),
        support_mask=SupportMaskConfig(**sup_raw),
        mesh=MeshGenerationConfig(**mesh_raw),
        quality_thresholds=raw.get(
            "quality_thresholds",
            {
                "max_relative_area_error": 0.05,
                "min_dice_similarity": 0.92,
                "min_corner_jacobian_mm2": 1.0e-5,
            },
        ),
    )


# =============================================================================
# 1. Caricamento dei volumi MRI, verifica della provenienza e dell'integrità
# =============================================================================

@dataclass
class BraTSSubjectVolumes:
    subject_id: str
    source_data_kind: str                      # "clinical_brats2020" oppure "synthetic_test_fixture"
    provenance_verified: bool
    modalities: Dict[str, np.ndarray]          # "T1", "T1ce", "T2", "FLAIR" in RAS+
    seg: np.ndarray                            # Segmentazione int32 in RAS+ con label in {0,1,2,4}
    wm_mask_3d: Optional[np.ndarray]
    shape: Tuple[int, int, int]
    spacing_mm: Tuple[float, float, float]
    orientation_convention: str                # "RAS+"
    original_affines: Dict[str, np.ndarray]
    original_orientations: Dict[str, Tuple[str, str, str]]
    ras_affine: np.ndarray
    realignment_log: List[Dict[str, Any]]
    seg_labels_before_preprocessing: List[int]
    seg_labels_after_preprocessing: List[int]
    file_paths: Dict[str, str]                 # Percorsi assoluti risolti
    volume_provenance: Dict[str, VolumeProvenanceRecord]


def _resolve_nifti_file(
    subject_dir: Path,
    subject_id: str,
    modality_key: str,
    explicit_name: Optional[str] = None,
    require_explicit: bool = False,
) -> Path:
    """
    Risolve il file NIfTI per `modality_key`.
    Se `explicit_name` è fornito (o `require_explicit=True`), seleziona ESCLUSIVAMENTE
    il file `subject_dir / explicit_name`, vietando qualsiasi fallback su nomi o cartelle alternative.
    """
    subj_dir_resolved = subject_dir.resolve()
    if require_explicit and not explicit_name:
        raise ValueError(
            f"Provenienza rigorosa attiva: nome file esplicito mancante in "
            f"`modality_filenames` per il volume '{modality_key}'."
        )

    if explicit_name:
        if Path(explicit_name).name != explicit_name:
            raise ValueError(
                f"Il nome file in modality_filenames['{modality_key}']='{explicit_name}' "
                "non deve contenere sottocartelle o percorsi relativi."
            )
        candidate = (subj_dir_resolved / explicit_name).resolve()
        if candidate.parent != subj_dir_resolved:
            raise ValueError(
                f"Percorso non valido per '{modality_key}': {candidate} esce da {subj_dir_resolved}."
            )
        if not candidate.is_file():
            raise FileNotFoundError(
                f"File NIfTI richiesto in modality_filenames['{modality_key}'] non trovato: {candidate}"
            )
        return candidate

    aliases_map = {
        "T1": ["t1", "T1"],
        "T1ce": ["t1ce", "t1gd", "T1ce", "T1Gd", "t1c"],
        "T2": ["T2", "t2"],
        "FLAIR": ["flair", "FLAIR"],
        "seg": ["seg", "SEG"],
    }
    suffixes = aliases_map[modality_key]
    for suf in suffixes:
        for ext in (".nii.gz", ".nii"):
            cand1 = (subj_dir_resolved / f"{subject_id}_{suf}{ext}").resolve()
            if cand1.is_file():
                return cand1
            cand2 = (subj_dir_resolved / f"{suf}{ext}").resolve()
            if cand2.is_file():
                return cand2

    raise FileNotFoundError(
        f"Impossibile individuare il volume '{modality_key}' (.nii / .nii.gz) in {subj_dir_resolved}"
    )


def _resample_volume_to_reference_ras(
    src_data: np.ndarray,
    src_affine: np.ndarray,
    ref_shape: Tuple[int, int, int],
    ref_affine: np.ndarray,
    interpolation_order: int,
) -> np.ndarray:
    vox_ref_to_src = np.linalg.inv(src_affine) @ ref_affine
    matrix = vox_ref_to_src[:3, :3]
    offset = vox_ref_to_src[:3, 3]
    return ndi.affine_transform(
        src_data,
        matrix=matrix,
        offset=offset,
        output_shape=ref_shape,
        order=interpolation_order,
        mode="constant",
        cval=0.0,
        prefilter=(interpolation_order > 1),
    )


def load_brats_subject(
    subject_dir: str | Path,
    subject_id: Optional[str] = None,
    modality_filenames: Optional[Dict[str, str]] = None,
    wm_mask_path: Optional[str | Path] = None,
    allow_grid_realignment: bool = False,
    remap_label_3_to_4: bool = False,
    affine_atol: float = 1.0e-3,
    provenance_cfg: Optional[ProvenanceConfig] = None,
) -> BraTSSubjectVolumes:
    """
    Carica i 4 volumi MRI (T1, T1ce, T2, FLAIR) e la segmentazione `seg`,
    calcola gli hash SHA-256, verifica l'integrità della griglia e la provenienza
    (`clinical_brats2020` vs `synthetic_test_fixture`), converte in orientamento RAS+
    e valida le label BraTS {0, 1, 2, 4}.
    """
    subj_path = Path(subject_dir)
    if not subj_path.exists() or not subj_path.is_dir():
        raise FileNotFoundError(f"Directory del soggetto BraTS non trovata: {subj_path.resolve()}")

    subj_id = subject_id or subj_path.name
    mod_files = modality_filenames or {}

    if provenance_cfg is None:
        # Se modality_filenames non è fornito (es. invocazione diretta in unit test),
        # classifichiamo i dati come synthetic_test_fixture a meno che non rispettino tutti i criteri clinici.
        prov = ProvenanceConfig(
            expected_source_data_kind="clinical_brats2020" if modality_filenames else "synthetic_test_fixture",
            require_explicit_modality_filenames=bool(modality_filenames),
            expected_shape_3d=(240, 240, 155) if modality_filenames else None,
            expected_spacing_mm=(1.0, 1.0, 1.0) if modality_filenames else None,
            forbid_implicit_realignment=not allow_grid_realignment,
        )
    else:
        prov = provenance_cfg

    if prov.expected_source_data_kind not in ("clinical_brats2020", "synthetic_test_fixture"):
        raise ValueError(
            f"expected_source_data_kind non valido: '{prov.expected_source_data_kind}'. "
            "Valori ammessi: 'clinical_brats2020', 'synthetic_test_fixture'."
        )

    require_explicit = prov.require_explicit_modality_filenames or (
        prov.expected_source_data_kind == "clinical_brats2020"
    )
    if require_explicit:
        missing_keys = [k for k in REQUIRED_VOLUME_KEYS if not mod_files.get(k)]
        if missing_keys:
            raise ValueError(
                f"Per la modalità '{prov.expected_source_data_kind}' è obbligatorio specificare "
                f"tutti i 5 file in `modality_filenames`. Chiavi mancanti: {missing_keys}."
            )

    resolved_paths: Dict[str, Path] = {
        k: _resolve_nifti_file(
            subj_path,
            subj_id,
            k,
            explicit_name=mod_files.get(k),
            require_explicit=require_explicit,
        )
        for k in REQUIRED_VOLUME_KEYS
    }

    original_affines: Dict[str, np.ndarray] = {}
    original_orientations: Dict[str, Tuple[str, str, str]] = {}
    canonical_imgs: Dict[str, nib.Nifti1Image] = {}
    provenance_records: Dict[str, VolumeProvenanceRecord] = {}
    detected_synthetic_signature = False
    synthetic_reasons: List[str] = []

    for k, fpath in resolved_paths.items():
        abs_str = str(fpath.resolve())
        fsize = int(fpath.stat().st_size)
        sha256_hex = _compute_file_sha256(fpath)

        img = nib.load(abs_str)
        if img.ndim != 3:
            raise ValueError(f"Il volume '{k}' ({abs_str}) non è 3D: shape={img.shape}")

        raw_shape = tuple(int(s) for s in img.shape[:3])
        raw_zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        dtype_str = str(img.get_data_dtype())
        sform_code = int(img.header["sform_code"])
        qform_code = int(img.header["qform_code"])
        descrip_raw = bytes(img.header["descrip"]).decode("latin1", errors="ignore").strip("\x00 ").strip()

        orig_aff = np.asarray(img.affine, dtype=np.float64)
        if not np.all(np.isfinite(orig_aff)) or abs(np.linalg.det(orig_aff[:3, :3])) < 1.0e-8:
            raise ValueError(f"Matrice affine NIfTI degenere per '{k}' in {abs_str}")
        orig_axcodes = tuple(nib.aff2axcodes(orig_aff))

        c_img = nib.as_closest_canonical(img)
        c_aff = np.asarray(c_img.affine, dtype=np.float64)
        c_shape = tuple(int(s) for s in c_img.shape[:3])
        c_spacing = tuple(float(np.linalg.norm(c_aff[:3, axis])) for axis in range(3))

        # Rilevamento firma sintetica
        if SYNTHETIC_HEADER_MARKER in descrip_raw:
            detected_synthetic_signature = True
            synthetic_reasons.append(f"{k}: header descrip contiene '{SYNTHETIC_HEADER_MARKER}'")
        if raw_shape in SYNTHETIC_SHAPES_BLACKLIST or c_shape in SYNTHETIC_SHAPES_BLACKLIST:
            detected_synthetic_signature = True
            synthetic_reasons.append(f"{k}: shape {raw_shape} corrisponde alla fixture sintetica")
        if np.allclose(orig_aff[:3, 3], [-80.0, -96.0, -20.0], atol=1.0e-3) and np.allclose(
            c_spacing, [1.0, 1.0, 2.0], atol=1.0e-3
        ):
            detected_synthetic_signature = True
            synthetic_reasons.append(
                f"{k}: affine/spacing (origine [-80,-96,-20] mm, spacing [1,1,2] mm) corrisponde alla fixture sintetica"
            )

        # Controllo griglia attesa se configurata (es. 240 x 240 x 155, spacing 1.0 x 1.0 x 1.0 mm)
        if prov.expected_shape_3d is not None:
            exp_sh = tuple(int(x) for x in prov.expected_shape_3d)
            if raw_shape != exp_sh or c_shape != exp_sh:
                raise ValueError(
                    f"Discrepanza di griglia NIfTI sul volume '{k}' ({abs_str}): "
                    f"shape osservata raw={raw_shape} (RAS+={c_shape}), "
                    f"shape attesa={exp_sh}. Nessun crop o resize implicito consentito."
                )

        if prov.expected_spacing_mm is not None:
            exp_sp = np.asarray(prov.expected_spacing_mm, dtype=np.float64)
            if (not np.allclose(raw_zooms, exp_sp, atol=prov.spacing_atol_mm)) or (
                not np.allclose(c_spacing, exp_sp, atol=prov.spacing_atol_mm)
            ):
                raise ValueError(
                    f"Discrepanza di spacing voxel sul volume '{k}' ({abs_str}): "
                    f"spacing osservato={c_spacing} mm (header zooms={raw_zooms} mm), "
                    f"spacing atteso={tuple(exp_sp.tolist())} mm (tol={prov.spacing_atol_mm} mm)."
                )

        original_affines[k] = orig_aff
        original_orientations[k] = orig_axcodes
        canonical_imgs[k] = c_img

        provenance_records[k] = VolumeProvenanceRecord(
            modality_key=k,
            configured_filename=mod_files.get(k, fpath.name),
            absolute_path=abs_str,
            file_size_bytes=fsize,
            sha256=sha256_hex,
            nifti_data_dtype=dtype_str,
            raw_shape_3d=list(raw_shape),
            raw_zooms_mm=list(raw_zooms),
            sform_code=sform_code,
            qform_code=qform_code,
            header_descrip=descrip_raw,
            original_orientation="".join(orig_axcodes),
            original_affine_4x4=orig_aff.tolist(),
            canonical_ras_shape_3d=list(c_shape),
            canonical_ras_spacing_mm=list(c_spacing),
            canonical_ras_affine_4x4=c_aff.tolist(),
        )

    if prov.expected_source_data_kind == "clinical_brats2020" and detected_synthetic_signature:
        raise ValueError(
            "Violazione di provenienza: il dataset caricato presenta la firma della fixture "
            f"sintetica ({'; '.join(synthetic_reasons)}) e NON può essere classificato come "
            "'clinical_brats2020'."
        )

    # Riferimento spaziale esplicito: T1 convertito senza interpolazione in convenzione RAS+
    ref_img = canonical_imgs["T1"]
    ref_affine = np.asarray(ref_img.affine, dtype=np.float64)
    ref_axcodes = nib.aff2axcodes(ref_affine)
    if ref_axcodes != ("R", "A", "S"):
        raise RuntimeError(f"Orientamento canonico atteso ('R','A','S'), ottenuto {ref_axcodes}")

    ref_shape: Tuple[int, int, int] = tuple(int(s) for s in ref_img.shape[:3])
    spacing_mm: Tuple[float, float, float] = tuple(
        float(np.linalg.norm(ref_affine[:3, axis])) for axis in range(3)
    )
    if any(sp <= 0.0 or not np.isfinite(sp) for sp in spacing_mm):
        raise ValueError(f"Spacing voxel non valido: {spacing_mm}")

    can_realign = allow_grid_realignment and (not prov.forbid_implicit_realignment)
    modalities_data: Dict[str, np.ndarray] = {}
    realignment_log: List[Dict[str, Any]] = []

    for k in ("T1", "T1ce", "T2", "FLAIR"):
        c_img = canonical_imgs[k]
        c_aff = np.asarray(c_img.affine, dtype=np.float64)
        c_shape = tuple(int(s) for s in c_img.shape[:3])
        arr = np.asarray(c_img.get_fdata(dtype=np.float64), dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Il volume MRI '{k}' contiene valori NaN o Inf.")

        same_grid = (c_shape == ref_shape) and np.allclose(c_aff, ref_affine, atol=affine_atol)
        if not same_grid:
            if not can_realign:
                raise ValueError(
                    f"Disallineamento di griglia rilevato per '{k}' (shape={c_shape} vs {ref_shape}, "
                    f"max_affine_diff={np.max(np.abs(c_aff - ref_affine)):.3e}) "
                    "e il riallineamento implicito è disabilitato."
                )
            arr = _resample_volume_to_reference_ras(
                arr, c_aff, ref_shape, ref_affine, interpolation_order=1
            )
            realignment_log.append(
                {
                    "volume": k,
                    "original_shape_ras": list(c_shape),
                    "target_shape_ras": list(ref_shape),
                    "max_affine_diff": float(np.max(np.abs(c_aff - ref_affine))),
                    "interpolation": "trilinear_order_1",
                }
            )
        modalities_data[k] = arr

    # Elaborazione e verifica della segmentazione `seg`
    seg_img = canonical_imgs["seg"]
    seg_aff = np.asarray(seg_img.affine, dtype=np.float64)
    seg_shape = tuple(int(s) for s in seg_img.shape[:3])
    seg_raw_float = np.asarray(seg_img.get_fdata(dtype=np.float64), dtype=np.float64)
    if not np.all(np.isfinite(seg_raw_float)):
        raise ValueError("Il volume 'seg' contiene valori NaN o Inf.")
    if not np.allclose(seg_raw_float, np.round(seg_raw_float), atol=1.0e-4):
        raise ValueError("Il volume 'seg' contiene valori non interi prima del preprocessing.")

    seg_int = np.round(seg_raw_float).astype(np.int32)
    labels_before = sorted(int(v) for v in np.unique(seg_int))

    same_seg_grid = (seg_shape == ref_shape) and np.allclose(seg_aff, ref_affine, atol=affine_atol)
    if not same_seg_grid:
        if not can_realign:
            raise ValueError(
                f"Disallineamento di griglia rilevato per 'seg' (shape={seg_shape} vs {ref_shape}, "
                f"max_affine_diff={np.max(np.abs(seg_aff - ref_affine)):.3e}) "
                "e il riallineamento implicito è disabilitato."
            )
        seg_resampled = _resample_volume_to_reference_ras(
            seg_int.astype(np.float64),
            seg_aff,
            ref_shape,
            ref_affine,
            interpolation_order=0,
        )
        seg_int = np.round(seg_resampled).astype(np.int32)
        realignment_log.append(
            {
                "volume": "seg",
                "original_shape_ras": list(seg_shape),
                "target_shape_ras": list(ref_shape),
                "max_affine_diff": float(np.max(np.abs(seg_aff - ref_affine))),
                "interpolation": "nearest_neighbour_order_0",
            }
        )

    if remap_label_3_to_4 and (3 in seg_int):
        seg_int = np.where(seg_int == 3, 4, seg_int)

    labels_after = sorted(int(v) for v in np.unique(seg_int))
    invalid_labels = [lb for lb in labels_after if lb not in BRATS_VALID_LABELS]
    if invalid_labels:
        raise ValueError(
            f"Label non ammesse nella segmentazione BraTS: {invalid_labels}. "
            f"Label consentite: {BRATS_VALID_LABELS}."
        )

    # Caricamento opzionale di una maschera indipendente di sostanza bianca (WM)
    wm_mask_3d: Optional[np.ndarray] = None
    file_paths_dict = {k: str(v.resolve()) for k, v in resolved_paths.items()}
    if wm_mask_path is not None:
        wm_p = Path(wm_mask_path).resolve()
        if not wm_p.is_file():
            raise FileNotFoundError(f"Maschera WM indipendente non trovata: {wm_p}")
        file_paths_dict["wm_mask"] = str(wm_p)
        wm_img = nib.as_closest_canonical(nib.load(str(wm_p)))
        wm_aff = np.asarray(wm_img.affine, dtype=np.float64)
        wm_shape = tuple(int(s) for s in wm_img.shape[:3])
        wm_arr = np.asarray(wm_img.get_fdata(dtype=np.float64)) > 0.5
        if (wm_shape != ref_shape) or (not np.allclose(wm_aff, ref_affine, atol=affine_atol)):
            if not can_realign:
                raise ValueError("Disallineamento della maschera WM e riallineamento disabilitato.")
            wm_res = _resample_volume_to_reference_ras(
                wm_arr.astype(np.float64), wm_aff, ref_shape, ref_affine, interpolation_order=0
            )
            wm_arr = wm_res > 0.5
            realignment_log.append(
                {
                    "volume": "wm_mask",
                    "original_shape_ras": list(wm_shape),
                    "target_shape_ras": list(ref_shape),
                    "max_affine_diff": float(np.max(np.abs(wm_aff - ref_affine))),
                    "interpolation": "nearest_neighbour_order_0",
                }
            )
        wm_mask_3d = wm_arr.astype(bool)

    # Determinazione finale certificata di `source_data_kind`
    is_standard_brats_grid = (ref_shape == (240, 240, 155)) and np.allclose(
        spacing_mm, (1.0, 1.0, 1.0), atol=prov.spacing_atol_mm
    )
    clinical_checks_ok = (
        (prov.expected_source_data_kind == "clinical_brats2020")
        and (not detected_synthetic_signature)
        and is_standard_brats_grid
        and (len(realignment_log) == 0)
        and all(bool(mod_files.get(k)) for k in REQUIRED_VOLUME_KEYS)
    )

    if prov.expected_source_data_kind == "clinical_brats2020" and not clinical_checks_ok:
        raise ValueError(
            "Impossibile certificare source_data_kind='clinical_brats2020': "
            f"is_standard_brats_grid={is_standard_brats_grid}, "
            f"detected_synthetic_signature={detected_synthetic_signature}, "
            f"realignment_events={len(realignment_log)}."
        )

    source_data_kind = "clinical_brats2020" if clinical_checks_ok else "synthetic_test_fixture"

    return BraTSSubjectVolumes(
        subject_id=subj_id,
        source_data_kind=source_data_kind,
        provenance_verified=clinical_checks_ok if source_data_kind == "clinical_brats2020" else True,
        modalities=modalities_data,
        seg=seg_int,
        wm_mask_3d=wm_mask_3d,
        shape=ref_shape,
        spacing_mm=spacing_mm,
        orientation_convention="RAS+",
        original_affines=original_affines,
        original_orientations=original_orientations,
        ras_affine=ref_affine,
        realignment_log=realignment_log,
        seg_labels_before_preprocessing=labels_before,
        seg_labels_after_preprocessing=labels_after,
        file_paths=file_paths_dict,
        volume_provenance=provenance_records,
    )


# =============================================================================
# 2. Selezione riproducibile della sezione (manuale o max estensione tumorale)
# =============================================================================

@dataclass
class BraTSSliceSelection:
    slice_axis: int
    slice_index: int
    selection_mode: str
    physical_z_mm: float
    slice_plane_reference_point_mm: Tuple[float, float, float]
    tumor_voxel_count_in_slice: int
    tumor_area_in_slice_mm2: float
    modalities_2d: Dict[str, np.ndarray]
    seg_2d: np.ndarray
    wm_mask_2d: Optional[np.ndarray]


def select_anatomical_slice(
    volumes: BraTSSubjectVolumes,
    mode: str = "auto_max_tumor",
    manual_slice_index: Optional[int] = None,
) -> BraTSSliceSelection:
    n_z = volumes.shape[2]
    voxel_area_mm2 = volumes.spacing_mm[0] * volumes.spacing_mm[1]
    tumor_counts_per_slice = np.sum(volumes.seg > 0, axis=(0, 1))

    if mode == "manual":
        if manual_slice_index is None:
            raise ValueError("manual_slice_index è obbligatorio quando mode='manual'.")
        k_idx = int(manual_slice_index)
        if not (0 <= k_idx < n_z):
            raise ValueError(f"manual_slice_index={k_idx} fuori dall'intervallo [0, {n_z - 1}].")
    elif mode == "auto_max_tumor":
        max_count = int(np.max(tumor_counts_per_slice))
        if max_count == 0:
            raise ValueError(
                "Impossibile selezionare la sezione con massima estensione tumorale: "
                "il volume `seg` non contiene alcun voxel con seg > 0."
            )
        candidate_indices = np.flatnonzero(tumor_counts_per_slice == max_count)
        k_idx = int(candidate_indices[(len(candidate_indices) - 1) // 2])
    else:
        raise ValueError(f"Modalità di selezione sezione non riconosciuta: '{mode}'.")

    cx = 0.5 * (volumes.shape[0] - 1)
    cy = 0.5 * (volumes.shape[1] - 1)
    phys_pt = volumes.ras_affine @ np.array([cx, cy, float(k_idx), 1.0], dtype=np.float64)
    z_mm = float(phys_pt[2])

    mod_2d = {k: np.asarray(v[:, :, k_idx], dtype=np.float64) for k, v in volumes.modalities.items()}
    seg_2d = np.asarray(volumes.seg[:, :, k_idx], dtype=np.int32)
    wm_2d = (
        np.asarray(volumes.wm_mask_3d[:, :, k_idx], dtype=bool)
        if volumes.wm_mask_3d is not None
        else None
    )

    t_count = int(tumor_counts_per_slice[k_idx])
    return BraTSSliceSelection(
        slice_axis=2,
        slice_index=k_idx,
        selection_mode=mode,
        physical_z_mm=z_mm,
        slice_plane_reference_point_mm=(float(phys_pt[0]), float(phys_pt[1]), z_mm),
        tumor_voxel_count_in_slice=t_count,
        tumor_area_in_slice_mm2=float(t_count * voxel_area_mm2),
        modalities_2d=mod_2d,
        seg_2d=seg_2d,
        wm_mask_2d=wm_2d,
    )


# =============================================================================
# 3. Estrazione delle maschere anatomiche e controlli di coerenza
# =============================================================================

@dataclass
class AnatomicalMasks2D:
    brain_support_mask: np.ndarray
    whole_tumor_mask: np.ndarray
    edema_mask: np.ndarray
    necrotic_non_enhancing_mask: np.ndarray
    enhancing_tumor_mask: np.ndarray
    white_matter_mask: Optional[np.ndarray]
    domain_representation_type: str
    warnings: List[str]
    preserved_internal_holes_count: int
    filled_small_artifact_holes_count: int
    removed_isolated_components_count: int
    lesion_inside_support_ratio: float


def _fill_only_small_holes(
    mask: np.ndarray,
    voxel_area_mm2: float,
    max_fillable_hole_area_mm2: float,
) -> Tuple[np.ndarray, int, int]:
    filled_all = ndi.binary_fill_holes(mask)
    holes_mask = filled_all & (~mask)
    labeled_holes, num_holes = ndi.label(holes_mask)
    if num_holes == 0:
        return mask.copy(), 0, 0

    out = mask.copy()
    filled_count = 0
    preserved_count = 0
    for h_id in range(1, num_holes + 1):
        comp = labeled_holes == h_id
        area_mm2 = float(np.sum(comp)) * voxel_area_mm2
        if area_mm2 <= max_fillable_hole_area_mm2:
            out[comp] = True
            filled_count += 1
        else:
            preserved_count += 1
    return out, preserved_count, filled_count


def build_anatomical_masks_2d(
    slice_sel: BraTSSliceSelection,
    spacing_xy_mm: Tuple[float, float],
    cfg: Optional[SupportMaskConfig] = None,
) -> AnatomicalMasks2D:
    cfg = cfg or SupportMaskConfig()
    dx_mm, dy_mm = float(spacing_xy_mm[0]), float(spacing_xy_mm[1])
    voxel_area_mm2 = dx_mm * dy_mm
    mean_sp_mm = 0.5 * (dx_mm + dy_mm)

    seg = slice_sel.seg_2d
    whole_tumor = seg > 0
    necrotic_mask = seg == 1
    edema_mask = seg == 2
    enhancing_mask = seg == 4

    norm_stack = []
    nonzero_union = np.zeros_like(seg, dtype=bool)
    for mod_name in ("T1", "T1ce", "T2", "FLAIR"):
        arr = slice_sel.modalities_2d[mod_name]
        pos = arr > 0.0
        nonzero_union |= pos
        if np.any(pos):
            p99 = float(np.percentile(arr[pos], 99.0))
            scale = p99 if p99 > 1.0e-8 else float(np.max(arr))
            norm_stack.append(np.clip(arr / max(scale, 1.0e-8), 0.0, 1.5))

    if not norm_stack or not np.any(nonzero_union):
        raise ValueError("La sezione selezionata è completamente vuota in tutte le modalità MRI.")

    combined = np.mean(norm_stack, axis=0)
    sigma_vox = (
        max(cfg.gaussian_sigma_mm / dx_mm, 0.5),
        max(cfg.gaussian_sigma_mm / dy_mm, 0.5),
    )
    smoothed = gaussian(combined, sigma=sigma_vox, preserve_range=True)

    fg_vals = smoothed[nonzero_union]
    otsu_val = float(threshold_otsu(fg_vals)) if fg_vals.size > 10 else 0.1
    adaptive_thresh = max(1.0e-4, cfg.otsu_scale_factor * otsu_val)

    raw_support = ((smoothed >= adaptive_thresh) & nonzero_union) | whole_tumor

    close_rad_px = max(1, int(round(cfg.closing_radius_mm / mean_sp_mm)))
    open_rad_px = max(1, int(round(cfg.opening_radius_mm / mean_sp_mm)))

    morphed = closing(raw_support, footprint=disk(close_rad_px)).astype(bool)
    morphed = opening(morphed, footprint=disk(open_rad_px)).astype(bool)
    morphed = closing(morphed | whole_tumor, footprint=disk(max(1, close_rad_px // 2))).astype(bool)

    labeled, num_comp = ndi.label(morphed)
    if num_comp == 0:
        raise ValueError("Maschera di supporto cerebrale vuota dopo il filtraggio morfologico.")

    comp_sizes = np.bincount(labeled.ravel())[1:]
    largest_label = int(np.argmax(comp_sizes) + 1)
    main_support = labeled == largest_label
    removed_components = int(num_comp - 1)

    brain_support, preserved_holes, filled_small_holes = _fill_only_small_holes(
        main_support,
        voxel_area_mm2=voxel_area_mm2,
        max_fillable_hole_area_mm2=cfg.max_fillable_hole_area_mm2,
    )

    _, final_num_comp = ndi.label(brain_support)
    if final_num_comp != 1:
        raise ValueError(
            f"La maschera di supporto finale presenta {final_num_comp} componenti connesse (atteso 1)."
        )

    support_area_mm2 = float(np.sum(brain_support)) * voxel_area_mm2
    if support_area_mm2 < cfg.min_component_area_mm2:
        raise ValueError(
            f"Area del supporto cerebrale troppo piccola ({support_area_mm2:.2f} mm^2 < "
            f"{cfg.min_component_area_mm2:.2f} mm^2)."
        )

    tumor_voxels = int(np.sum(whole_tumor))
    if tumor_voxels > 0:
        inside_ratio = float(np.sum(whole_tumor & brain_support)) / float(tumor_voxels)
    else:
        inside_ratio = 1.0

    if inside_ratio < 0.995:
        raise ValueError(
            f"Incoerenza anatomica: la lesione tumorale non è contenuta nel "
            f"dominio cerebrale (frazione interna = {inside_ratio:.4f})."
        )

    warnings_list: List[str] = []
    wm_mask_out: Optional[np.ndarray] = None
    if slice_sel.wm_mask_2d is not None:
        wm_candidate = slice_sel.wm_mask_2d & brain_support
        if not np.any(wm_candidate):
            raise ValueError("La maschera di sostanza bianca fornita è vuota sulla sezione selezionata.")
        wm_mask_out = wm_candidate
        domain_rep = "white_matter_segmented"
    else:
        domain_rep = "brain_parenchyma_proxy"
        warnings_list.append(
            "WM_MASK_UNAVAILABLE: Le label BraTS (0, 1, 2, 4) identificano sottoregioni "
            "tumorali e non la sostanza bianca. In assenza di una segmentazione tissutale "
            "indipendente, il dominio 2D rappresenta un proxy del parenchima cerebrale "
            "(`brain_parenchyma_proxy`) e NON una maschera di sostanza bianca."
        )

    return AnatomicalMasks2D(
        brain_support_mask=brain_support,
        whole_tumor_mask=whole_tumor,
        edema_mask=edema_mask,
        necrotic_non_enhancing_mask=necrotic_mask,
        enhancing_tumor_mask=enhancing_mask,
        white_matter_mask=wm_mask_out,
        domain_representation_type=domain_rep,
        warnings=warnings_list,
        preserved_internal_holes_count=preserved_holes,
        filled_small_artifact_holes_count=filled_small_holes,
        removed_isolated_components_count=removed_components,
        lesion_inside_support_ratio=inside_ratio,
    )


# =============================================================================
# 4. Trasformazione verificabile di coordinate (Voxel <-> mm <-> Adimensionale L0)
# =============================================================================

@dataclass(frozen=True)
class AnatomicalCoordinateTransform2D:
    L0_m: float
    L0_mm: float
    slice_index: int
    physical_z_mm: float
    origin_xy_mm: Tuple[float, float]
    spacing_xy_mm: Tuple[float, float]
    ras_affine: np.ndarray
    inv_ras_affine: np.ndarray
    orientation_convention: str = "RAS+"

    @classmethod
    def from_ras_affine_and_mask(
        cls,
        ras_affine: np.ndarray,
        slice_index: int,
        support_mask: np.ndarray,
        L0_m: float,
    ) -> "AnatomicalCoordinateTransform2D":
        aff = np.asarray(ras_affine, dtype=np.float64)
        inv_aff = np.linalg.inv(aff)
        l0_mm = float(L0_m) * 1000.0

        ii, jj = np.nonzero(support_mask)
        if ii.size == 0:
            raise ValueError("Impossibile costruire la trasformazione su una maschera vuota.")

        pts_ijk = np.column_stack(
            [ii.astype(np.float64), jj.astype(np.float64), np.full(ii.size, float(slice_index)), np.ones(ii.size)]
        )
        pts_xyz_mm = (pts_ijk @ aff.T)[:, :3]
        centroid_xy_mm = (float(np.mean(pts_xyz_mm[:, 0])), float(np.mean(pts_xyz_mm[:, 1])))
        z_mm = float(np.mean(pts_xyz_mm[:, 2]))
        sp_x = float(np.linalg.norm(aff[:3, 0]))
        sp_y = float(np.linalg.norm(aff[:3, 1]))

        return cls(
            L0_m=float(L0_m),
            L0_mm=l0_mm,
            slice_index=int(slice_index),
            physical_z_mm=z_mm,
            origin_xy_mm=centroid_xy_mm,
            spacing_xy_mm=(sp_x, sp_y),
            ras_affine=aff,
            inv_ras_affine=inv_aff,
            orientation_convention="RAS+",
        )

    def voxel_ij_to_physical_mm(self, ij: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(ij, dtype=np.float64))
        if pts.shape[1] != 2:
            raise ValueError(f"Atteso array (N, 2) per coordinate voxel (i, j), ricevuto {pts.shape}")
        n = pts.shape[0]
        homog = np.column_stack([pts[:, 0], pts[:, 1], np.full(n, float(self.slice_index)), np.ones(n)])
        xyz_mm = homog @ self.ras_affine.T
        return xyz_mm[:, :2]

    def physical_mm_to_voxel_ij(self, xy_mm: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(xy_mm, dtype=np.float64))
        if pts.shape[1] != 2:
            raise ValueError(f"Atteso array (N, 2) per coordinate fisiche (x_mm, y_mm), ricevuto {pts.shape}")
        a_2x2 = self.ras_affine[:2, :2]
        rhs = pts.T - (
            self.ras_affine[:2, 2:3] * float(self.slice_index) + self.ras_affine[:2, 3:4]
        )
        return np.linalg.solve(a_2x2, rhs).T

    def physical_mm_to_mesh_adim(self, xy_mm: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(xy_mm, dtype=np.float64))
        origin = np.array(self.origin_xy_mm, dtype=np.float64).reshape(1, 2)
        return (pts - origin) / self.L0_mm

    def mesh_adim_to_physical_mm(self, xy_adim: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(xy_adim, dtype=np.float64))
        origin = np.array(self.origin_xy_mm, dtype=np.float64).reshape(1, 2)
        return pts * self.L0_mm + origin

    def voxel_ij_to_mesh_adim(self, ij: np.ndarray) -> np.ndarray:
        return self.physical_mm_to_mesh_adim(self.voxel_ij_to_physical_mm(ij))

    def mesh_adim_to_voxel_ij(self, xy_adim: np.ndarray) -> np.ndarray:
        return self.physical_mm_to_voxel_ij(self.mesh_adim_to_physical_mm(xy_adim))

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "L0_m": self.L0_m,
            "L0_mm": self.L0_mm,
            "slice_index": self.slice_index,
            "physical_z_mm": self.physical_z_mm,
            "origin_xy_mm": list(self.origin_xy_mm),
            "spacing_xy_mm": list(self.spacing_xy_mm),
            "orientation_convention": self.orientation_convention,
            "ras_affine_4x4": self.ras_affine.tolist(),
            "formula_physical_to_adim": "x_adim = (x_mm - origin_xy_mm) / L0_mm",
        }


# =============================================================================
# 5. Generazione ed esportazione della mesh 2D puramente quadrilatera
# =============================================================================

@dataclass
class QuadMesh2D:
    vertices_mm: np.ndarray
    vertices_adim: np.ndarray
    quads: np.ndarray
    boundary_edges: np.ndarray
    boundary_ids: np.ndarray
    cell_tumor_label: np.ndarray
    outer_contour_mm: np.ndarray
    inner_contours_mm: List[np.ndarray]
    target_element_size_mm: float


def _extract_physical_contours(
    mask: np.ndarray,
    transform: AnatomicalCoordinateTransform2D,
    min_hole_area_mm2: float = 40.0,
    simplify_tolerance_mm: float = 0.45,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    padded = np.pad(mask.astype(np.float64), pad_width=1, mode="constant", constant_values=0.0)
    raw_contours = measure.find_contours(padded, level=0.5)
    if not raw_contours:
        raise ValueError("Nessun contorno estratto dalla maschera di supporto.")

    def _contour_to_mm(c_pad: np.ndarray) -> np.ndarray:
        ij = c_pad - 1.0
        pts_mm = transform.voxel_ij_to_physical_mm(ij)
        if simplify_tolerance_mm > 0.0 and pts_mm.shape[0] > 12:
            simp = measure.approximate_polygon(pts_mm, tolerance=simplify_tolerance_mm)
            if simp.shape[0] >= 5:
                pts_mm = simp
        if np.allclose(pts_mm[0], pts_mm[-1]):
            pts_mm = pts_mm[:-1]
        return pts_mm

    def _polygon_abs_area(pts: np.ndarray) -> float:
        x, y = pts[:, 0], pts[:, 1]
        return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    polys_mm = [_contour_to_mm(c) for c in raw_contours]
    polys_mm = [p for p in polys_mm if p.shape[0] >= 4]
    areas = [_polygon_abs_area(p) for p in polys_mm]
    outer_idx = int(np.argmax(areas))
    outer_mm = polys_mm[outer_idx]

    outer_path = MplPath(outer_mm)
    inner_mm_list: List[np.ndarray] = []
    for idx, p in enumerate(polys_mm):
        if idx == outer_idx:
            continue
        if areas[idx] >= min_hole_area_mm2 and outer_path.contains_point(np.mean(p, axis=0)):
            inner_mm_list.append(p)

    return outer_mm, inner_mm_list


def _project_points_onto_polyline(pts: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    a = polyline
    b = np.roll(polyline, -1, axis=0)
    ab = b - a
    ab_len2 = np.sum(ab * ab, axis=1)
    ab_len2 = np.where(ab_len2 < 1.0e-12, 1.0, ab_len2)

    projected = np.empty_like(pts)
    for i in range(pts.shape[0]):
        p = pts[i]
        ap = p - a
        t = np.clip(np.sum(ap * ab, axis=1) / ab_len2, 0.0, 1.0)
        cand = a + t[:, None] * ab
        d2 = np.sum((cand - p) ** 2, axis=1)
        projected[i] = cand[int(np.argmin(d2))]
    return projected


def _quad_signed_areas(vertices: np.ndarray, quads: np.ndarray) -> np.ndarray:
    x = vertices[quads, 0]
    y = vertices[quads, 1]
    return 0.5 * (
        (x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0])
        + (x[:, 1] * y[:, 2] - x[:, 2] * y[:, 1])
        + (x[:, 2] * y[:, 3] - x[:, 3] * y[:, 2])
        + (x[:, 3] * y[:, 0] - x[:, 0] * y[:, 3])
    )


def _quad_min_corner_jacobians(vertices: np.ndarray, quads: np.ndarray) -> np.ndarray:
    v = vertices[quads]
    j_corners = np.empty((quads.shape[0], 4), dtype=np.float64)
    for k in range(4):
        v_curr = v[:, k, :]
        v_next = v[:, (k + 1) % 4, :]
        v_prev = v[:, (k + 3) % 4, :]
        e1 = v_next - v_curr
        e2 = v_prev - v_curr
        j_corners[:, k] = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    return np.min(j_corners, axis=1)


def _sample_mask_bilinear(
    mask_float: np.ndarray,
    xy_mm: np.ndarray,
    transform: AnatomicalCoordinateTransform2D,
) -> np.ndarray:
    ij = transform.physical_mm_to_voxel_ij(xy_mm)
    coords = np.vstack([ij[:, 0], ij[:, 1]])
    return ndi.map_coordinates(mask_float, coords, order=1, mode="constant", cval=0.0)


def _regularize_quad_grid_topology(
    active_grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    inner_mm_list: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Regolarizza topologicamente la griglia booleana di celle quadrilatere:
    1) Rimuove speroni di larghezza 1 cella (celle attive con <= 1 vicino 4-connesso)
       e riempie intagli isolati (celle inattive con >= 3 vicini attivi);
    2) Elimina configurazioni a scacchiera 2x2 (pinch vertices non-manifold);
    3) Mantiene esclusivamente i buchi interni di `~active_grid` che contengono
       almeno un contorno interno valido e >= 4 celle, garantendo una corrispondenza
       1-a-1 tra i cicli di bordo interni della mesh e `inner_mm_list`.
    """
    nx, ny = active_grid.shape
    struct_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    grid = active_grid.copy()

    for _ in range(12):
        changed = False
        # Conteggio dei 4-vicini attivi per ogni cella
        nbr_count = (
            np.pad(grid[1:, :], ((0, 1), (0, 0)), constant_values=False).astype(int)
            + np.pad(grid[:-1, :], ((1, 0), (0, 0)), constant_values=False).astype(int)
            + np.pad(grid[:, 1:], ((0, 0), (0, 1)), constant_values=False).astype(int)
            + np.pad(grid[:, :-1], ((0, 0), (1, 0)), constant_values=False).astype(int)
        )
        spurs = grid & (nbr_count <= 1)
        if np.any(spurs):
            grid[spurs] = False
            changed = True

        notches = (~grid) & (nbr_count >= 3)
        if np.any(notches):
            grid[notches] = True
            changed = True

        for i in range(nx - 1):
            for j in range(ny - 1):
                blk = grid[i : i + 2, j : j + 2]
                if np.array_equal(blk, [[True, False], [False, True]]) or np.array_equal(
                    blk, [[False, True], [True, False]]
                ):
                    grid[i : i + 2, j : j + 2] = True
                    changed = True

        labeled_cells, n_cc = ndi.label(grid, structure=struct_4)
        if n_cc > 1:
            cc_sizes = np.bincount(labeled_cells.ravel())[1:]
            main_cc = labeled_cells == int(np.argmax(cc_sizes) + 1)
            if not np.array_equal(main_cc, grid):
                grid = main_cc
                changed = True

        if not changed:
            break

    # Analisi delle componenti connesse del complementare (~grid) per distinguere
    # l'esterno della mesh (label_bg == ext_label) dai buchi interni validi
    bg = ~grid
    labeled_bg, n_bg = ndi.label(bg, structure=struct_4)
    border_labels = set(
        np.unique(
            np.concatenate(
                [labeled_bg[0, :], labeled_bg[-1, :], labeled_bg[:, 0], labeled_bg[:, -1]]
            )
        )
    ) - {0}

    # Costruiamo la mappa: 0 = cella attiva, 1 = esterno, >=2 = indice buco interno + 2
    bg_region_map = np.zeros_like(labeled_bg, dtype=np.int32)
    for lb in border_labels:
        bg_region_map[labeled_bg == lb] = 1

    kept_inner_contours: List[np.ndarray] = []
    next_hole_tag = 2
    for lb in range(1, n_bg + 1):
        if lb in border_labels:
            continue
        comp_mask = labeled_bg == lb
        n_hole_cells = int(np.sum(comp_mask))
        if n_hole_cells < 3 or not inner_mm_list:
            grid[comp_mask] = True
            continue

        # Centro fisico della cavità nella griglia
        hi, hj = np.nonzero(comp_mask)
        hole_center_mm = np.array(
            [float(np.mean(0.5 * (xs[hi] + xs[hi + 1]))), float(np.mean(0.5 * (ys[hj] + ys[hj + 1])))]
        )
        # Troviamo il contorno interno più vicino in inner_mm_list
        dists = [float(np.linalg.norm(np.mean(ic, axis=0) - hole_center_mm)) for ic in inner_mm_list]
        best_ic_idx = int(np.argmin(dists))
        if dists[best_ic_idx] <= 3.0 * float(xs[1] - xs[0]):
            bg_region_map[comp_mask] = next_hole_tag
            kept_inner_contours.append(inner_mm_list[best_ic_idx])
            next_hole_tag += 1
        else:
            grid[comp_mask] = True

    # Se abbiamo riempito micro-buchi sub-risoluzione, rieseguiamo un ultimo controllo anti-scacchiera
    for i in range(nx - 1):
        for j in range(ny - 1):
            blk = grid[i : i + 2, j : j + 2]
            if np.array_equal(blk, [[True, False], [False, True]]) or np.array_equal(
                blk, [[False, True], [True, False]]
            ):
                grid[i : i + 2, j : j + 2] = True
                bg_region_map[i : i + 2, j : j + 2] = 0

    return grid, bg_region_map, kept_inner_contours



def _roi_banded_axis(
    low: float, high: float, coarse_h: float, fine_h: float,
    roi_low: float, roi_high: float, margin: float,
) -> np.ndarray:
    """Conforming 1D tensor-grid lines; all adjacent 2D quads share full edges.

    The ROI is a *bounding box*, not a body-fitted tissue-interface mesh.
    Smaller cells also occur in the x/y strips crossing that box.
    """
    if not (0.0 < fine_h <= coarse_h) or not np.isfinite(fine_h):
        raise ValueError("roi_fine_element_size_mm must be in (0, target_element_size_mm].")
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("roi_refinement_margin_mm must be finite and >= 0.")
    a = max(low, min(high, roi_low - margin))
    b = max(low, min(high, roi_high + margin))
    if a >= b:
        return np.linspace(low, high, max(6, int(np.ceil((high-low)/coarse_h))) + 1)
    coords = [float(low)]
    for segment_end, step in ((a, coarse_h), (b, fine_h), (high, coarse_h)):
        start = coords[-1]
        if segment_end - start < 1.0e-9:
            continue
        n = max(1, int(np.ceil((segment_end-start)/step)))
        coords.extend(np.linspace(start, segment_end, n+1)[1:].tolist())
    out = np.asarray(coords, dtype=np.float64)
    if not np.all(np.diff(out) > 0.0):
        raise ValueError("Non-monotone coordinate grading near BraTS ROI.")
    return out

def generate_anatomical_quad_mesh_2d(
    masks: AnatomicalMasks2D,
    slice_sel: BraTSSliceSelection,
    transform: AnatomicalCoordinateTransform2D,
    mesh_cfg: Optional[MeshGenerationConfig] = None,
) -> QuadMesh2D:
    cfg = mesh_cfg or MeshGenerationConfig()
    h_mm = float(cfg.target_element_size_mm)
    if not np.isfinite(h_mm) or h_mm <= 0.0:
        raise ValueError(f"target_element_size_mm deve essere > 0, ricevuto {h_mm}")

    outer_mm, inner_mm_list = _extract_physical_contours(
        masks.brain_support_mask,
        transform=transform,
        min_hole_area_mm2=cfg.min_hole_area_to_mesh_mm2,
        simplify_tolerance_mm=cfg.contour_simplify_tolerance_mm,
    )

    xy_min = np.min(outer_mm, axis=0) - 2.0 * h_mm
    xy_max = np.max(outer_mm, axis=0) + 2.0 * h_mm

    roi_fine = cfg.roi_fine_element_size_mm
    if roi_fine is None:
        # Preserve the established uniform-grid protocol when grading is off.
        nx = max(6, int(np.ceil((xy_max[0] - xy_min[0]) / h_mm)))
        ny = max(6, int(np.ceil((xy_max[1] - xy_min[1]) / h_mm)))
        xs = xy_min[0] + np.arange(nx + 1, dtype=np.float64) * h_mm
        ys = xy_min[1] + np.arange(ny + 1, dtype=np.float64) * h_mm
    else:
        # The segmentation, not MRI intensity, defines the refinement target.
        roi_mask = np.isin(slice_sel.seg_2d, list(cfg.roi_refinement_labels))
        ij = np.argwhere(roi_mask)
        if ij.size == 0:
            raise ValueError("Cannot grade the mesh: no configured BraTS ROI labels on slice.")
        roi_mm = transform.voxel_ij_to_physical_mm(ij.astype(np.float64))
        lo, hi = roi_mm.min(axis=0), roi_mm.max(axis=0)
        xs = _roi_banded_axis(
            float(xy_min[0]), float(xy_max[0]), h_mm, float(roi_fine),
            float(lo[0]), float(hi[0]), float(cfg.roi_refinement_margin_mm))
        ys = _roi_banded_axis(
            float(xy_min[1]), float(xy_max[1]), h_mm, float(roi_fine),
            float(lo[1]), float(hi[1]), float(cfg.roi_refinement_margin_mm))
        nx, ny = len(xs)-1, len(ys)-1

    mask_float = masks.brain_support_mask.astype(np.float64)
    q_offsets = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
    q_weights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

    dx = np.diff(xs)
    dy = np.diff(ys)
    cx_grid, cy_grid = np.meshgrid(0.5*(xs[:-1]+xs[1:]), 0.5*(ys[:-1]+ys[1:]), indexing="ij")
    centers = np.column_stack([cx_grid.ravel(), cy_grid.ravel()])
    dx_cell = np.broadcast_to(dx[:, None], (nx, ny)).ravel()
    dy_cell = np.broadcast_to(dy[None, :], (nx, ny)).ravel()
    occ = np.zeros(centers.shape[0], dtype=np.float64)
    w_sum = 0.0
    for a_idx, qx in enumerate(q_offsets):
        for b_idx, qy in enumerate(q_offsets):
            w = q_weights[a_idx] * q_weights[b_idx]
            pts_q = centers + np.column_stack((0.5*qx*dx_cell, 0.5*qy*dy_cell))
            occ += w * _sample_mask_bilinear(mask_float, pts_q, transform)
            w_sum += w
    occ /= w_sum
    raw_active_grid = (occ >= 0.5).reshape(nx, ny)

    active_grid, bg_region_map, valid_inner_contours = _regularize_quad_grid_topology(
        raw_active_grid, xs, ys, inner_mm_list
    )

    if not np.any(active_grid):
        raise RuntimeError(
            f"Nessuna cella quadrilatera attiva generata con target_element_size_mm={h_mm}."
        )

    active_ij = np.argwhere(active_grid)
    vertex_map = np.full((nx + 1, ny + 1), -1, dtype=np.int64)
    verts_list: List[Tuple[float, float]] = []
    quads_list: List[Tuple[int, int, int, int]] = []
    cell_grid_ij: List[Tuple[int, int]] = []

    def _get_vid(vi: int, vj: int) -> int:
        if vertex_map[vi, vj] < 0:
            vertex_map[vi, vj] = len(verts_list)
            verts_list.append((float(xs[vi]), float(ys[vj])))
        return int(vertex_map[vi, vj])

    for ci, cj in active_ij:
        v0 = _get_vid(ci, cj)
        v1 = _get_vid(ci + 1, cj)
        v2 = _get_vid(ci + 1, cj + 1)
        v3 = _get_vid(ci, cj + 1)
        quads_list.append((v0, v1, v2, v3))
        cell_grid_ij.append((int(ci), int(cj)))

    vertices_mm = np.asarray(verts_list, dtype=np.float64)
    quads = np.asarray(quads_list, dtype=np.int64)

    # Identificazione topologica dei lati di frontiera e del relativo contorno (esterno vs interno)
    # Per ogni lato orientato della cella (ci, cj), la cella adiacente esterna è:
    # (v0,v1)->(ci,cj-1), (v1,v2)->(ci+1,cj), (v2,v3)->(ci,cj+1), (v3,v0)->(ci-1,cj)
    all_contours = [outer_mm] + valid_inner_contours
    boundary_edges_list: List[Tuple[int, int]] = []
    boundary_ids_list: List[int] = []
    node_target_contour: Dict[int, int] = {}

    edge_neighbor_offsets = [
        (0, -1),  # lato inferiore (v0 -> v1)
        (1, 0),   # lato destro    (v1 -> v2)
        (0, 1),   # lato superiore (v2 -> v3)
        (-1, 0),  # lato sinistro  (v3 -> v0)
    ]

    for q_idx, (v0, v1, v2, v3) in enumerate(quads):
        ci, cj = cell_grid_ij[q_idx]
        cell_edges = ((v0, v1), (v1, v2), (v2, v3), (v3, v0))
        for side_idx, (u, v) in enumerate(cell_edges):
            di, dj = edge_neighbor_offsets[side_idx]
            ni, nj = ci + di, cj + dj
            is_bdy = (not (0 <= ni < nx and 0 <= nj < ny)) or (not active_grid[ni, nj])
            if not is_bdy:
                continue

            boundary_edges_list.append((int(u), int(v)))
            bg_tag = int(bg_region_map[ni, nj]) if (0 <= ni < nx and 0 <= nj < ny) else 1
            if bg_tag >= 2 and (bg_tag - 1) < len(all_contours):
                bid = BOUNDARY_ID_INNER_HOLES
                c_idx = bg_tag - 1
            else:
                bid = BOUNDARY_ID_OUTER_CONTOUR
                c_idx = 0
            boundary_ids_list.append(bid)
            node_target_contour[int(u)] = c_idx
            node_target_contour[int(v)] = c_idx

    boundary_edges = np.asarray(boundary_edges_list, dtype=np.int64)
    boundary_ids = np.asarray(boundary_ids_list, dtype=np.int32)

    n_verts = vertices_mm.shape[0]
    neighbors: List[set[int]] = [set() for _ in range(n_verts)]
    for v0, v1, v2, v3 in quads:
        neighbors[v0].update((v1, v3))
        neighbors[v1].update((v0, v2))
        neighbors[v2].update((v1, v3))
        neighbors[v3].update((v2, v0))

    boundary_nodes = np.unique(boundary_edges.ravel())
    is_boundary_node = np.zeros(n_verts, dtype=bool)
    is_boundary_node[boundary_nodes] = True

    min_allowed_j = 0.04 * (float(min(np.min(dx), np.min(dy))) ** 2)
    n_proj_steps = max(1, cfg.boundary_projection_iterations)
    for step in range(n_proj_steps):
        alpha_proj = float(step + 1) / float(n_proj_steps)
        cand_verts = vertices_mm.copy()
        for bn in boundary_nodes:
            c_poly = all_contours[node_target_contour.get(int(bn), 0)]
            proj_pt = _project_points_onto_polyline(cand_verts[bn : bn + 1], c_poly)[0]
            cand_verts[bn] = (1.0 - alpha_proj) * cand_verts[bn] + alpha_proj * proj_pt

        for bn in boundary_nodes:
            old_pos = vertices_mm[bn].copy()
            vertices_mm[bn] = cand_verts[bn]
            if np.min(_quad_min_corner_jacobians(vertices_mm, quads)) < min_allowed_j:
                vertices_mm[bn] = 0.5 * (old_pos + cand_verts[bn])
                if np.min(_quad_min_corner_jacobians(vertices_mm, quads)) < min_allowed_j:
                    vertices_mm[bn] = old_pos

        for _ in range(cfg.laplacian_smoothing_iterations):
            new_verts = vertices_mm.copy()
            for v_idx in range(n_verts):
                if is_boundary_node[v_idx]:
                    continue
                nbrs = list(neighbors[v_idx])
                if nbrs:
                    avg_pos = np.mean(vertices_mm[nbrs], axis=0)
                    new_verts[v_idx] = (
                        (1.0 - cfg.laplacian_relaxation) * vertices_mm[v_idx]
                        + cfg.laplacian_relaxation * avg_pos
                    )
            if np.min(_quad_min_corner_jacobians(new_verts, quads)) >= min_allowed_j:
                vertices_mm = new_verts

    signed_areas = _quad_signed_areas(vertices_mm, quads)
    for q_idx in range(quads.shape[0]):
        if signed_areas[q_idx] < 0.0:
            quads[q_idx] = quads[q_idx, [0, 3, 2, 1]]

    cell_centers_mm = np.mean(vertices_mm[quads], axis=1)
    centers_ij = np.round(transform.physical_mm_to_voxel_ij(cell_centers_mm)).astype(np.int64)
    centers_ij[:, 0] = np.clip(centers_ij[:, 0], 0, slice_sel.seg_2d.shape[0] - 1)
    centers_ij[:, 1] = np.clip(centers_ij[:, 1], 0, slice_sel.seg_2d.shape[1] - 1)
    cell_labels = slice_sel.seg_2d[centers_ij[:, 0], centers_ij[:, 1]].astype(np.int32)

    vertices_adim = transform.physical_mm_to_mesh_adim(vertices_mm)

    return QuadMesh2D(
        vertices_mm=vertices_mm,
        vertices_adim=vertices_adim,
        quads=quads,
        boundary_edges=boundary_edges,
        boundary_ids=boundary_ids,
        cell_tumor_label=cell_labels,
        outer_contour_mm=outer_mm,
        inner_contours_mm=valid_inner_contours,
        target_element_size_mm=h_mm,
    )


def export_dealii_quad_mesh(
    mesh: QuadMesh2D,
    transform: AnatomicalCoordinateTransform2D,
    output_dir: str | Path,
    basename: str = "brats_slice_quad_mesh",
) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    msh_path = out_dir / f"{basename}.msh"
    inp_path = out_dir / f"{basename}.inp"
    npz_path = out_dir / f"{basename}.npz"

    n_v = mesh.vertices_adim.shape[0]
    n_q = mesh.quads.shape[0]
    n_b = mesh.boundary_edges.shape[0]

    with msh_path.open("w", encoding="utf-8") as fh:
        fh.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        fh.write("$PhysicalNames\n")
        has_holes = bool(np.any(mesh.boundary_ids == BOUNDARY_ID_INNER_HOLES))
        n_phys = 3 if has_holes else 2
        fh.write(f"{n_phys}\n")
        fh.write(f'1 {BOUNDARY_ID_OUTER_CONTOUR} "outer_boundary"\n')
        if has_holes:
            fh.write(f'1 {BOUNDARY_ID_INNER_HOLES} "inner_hole_boundary"\n')
        fh.write('2 10 "anatomical_domain"\n')
        fh.write("$EndPhysicalNames\n")

        fh.write(f"$Nodes\n{n_v}\n")
        for i in range(n_v):
            x, y = mesh.vertices_adim[i]
            fh.write(f"{i + 1} {x:.12e} {y:.12e} 0.000000000000e+00\n")
        fh.write("$EndNodes\n")

        total_elems = n_b + n_q
        fh.write(f"$Elements\n{total_elems}\n")
        elem_id = 1
        for b_idx in range(n_b):
            u, v = mesh.boundary_edges[b_idx]
            bid = int(mesh.boundary_ids[b_idx])
            fh.write(f"{elem_id} 1 2 {bid} {bid} {u + 1} {v + 1}\n")
            elem_id += 1
        for q_idx in range(n_q):
            v0, v1, v2, v3 = mesh.quads[q_idx]
            mat_id = int(mesh.cell_tumor_label[q_idx])
            fh.write(f"{elem_id} 3 2 {mat_id} 10 {v0 + 1} {v1 + 1} {v2 + 1} {v3 + 1}\n")
            elem_id += 1
        fh.write("$EndElements\n")

    with inp_path.open("w", encoding="utf-8") as fh:
        fh.write(f"{n_v} {n_q + n_b} 0 0 0\n")
        for i in range(n_v):
            x, y = mesh.vertices_adim[i]
            fh.write(f"{i + 1} {x:.12e} {y:.12e} 0.000000000000e+00\n")
        cell_id = 1
        for q_idx in range(n_q):
            v0, v1, v2, v3 = mesh.quads[q_idx]
            mat_id = int(mesh.cell_tumor_label[q_idx])
            fh.write(f"{cell_id} {mat_id} quad {v0 + 1} {v1 + 1} {v2 + 1} {v3 + 1}\n")
            cell_id += 1
        for b_idx in range(n_b):
            u, v = mesh.boundary_edges[b_idx]
            bid = int(mesh.boundary_ids[b_idx])
            fh.write(f"{cell_id} {bid} line {u + 1} {v + 1}\n")
            cell_id += 1

    np.savez_compressed(
        npz_path,
        vertices_mm=mesh.vertices_mm,
        vertices_adim=mesh.vertices_adim,
        quads=mesh.quads,
        boundary_edges=mesh.boundary_edges,
        boundary_ids=mesh.boundary_ids,
        cell_tumor_label=mesh.cell_tumor_label,
        L0_m=np.array(transform.L0_m),
        L0_mm=np.array(transform.L0_mm),
        origin_xy_mm=np.array(transform.origin_xy_mm),
        physical_z_mm=np.array(transform.physical_z_mm),
        slice_index=np.array(transform.slice_index),
        ras_affine=transform.ras_affine,
    )

    return {
        "gmsh_msh": str(msh_path),
        "dealii_ucd_inp": str(inp_path),
        "numpy_bundle_npz": str(npz_path),
    }


# =============================================================================
# 6. Controlli obbligatori sulla geometria e report JSON con Provenienza SHA-256
# =============================================================================

@dataclass
class QuadMeshQualityReport:
    subject_id: str
    source_data_kind: str                      # "clinical_brats2020" | "synthetic_test_fixture"
    provenance_verified: bool
    input_provenance: Dict[str, Any]           # SHA-256, percorsi assoluti e metadati NIfTI per ogni volume
    domain_representation_type: str
    orientation_convention: str
    slice_index: int
    selection_mode: str
    physical_z_mm: float
    L0_m: float
    L0_mm: float
    reference_mask_area_mm2: float
    mesh_area_mm2: float
    mesh_area_adim: float
    relative_area_error: float
    rasterized_dice_similarity: float
    rasterized_iou_jaccard: float
    num_nodes: int
    num_quad_cells: int
    num_triangles: int
    num_connected_components_mask: int
    num_connected_components_mesh: int
    mesh_resolution: Dict[str, float]
    boundary_metrics: Dict[str, Any]
    lesion_centroids: Dict[str, Any]
    cell_geometry_checks: Dict[str, Any]
    segmentation_labels_report: Dict[str, Any]
    coordinate_transform: Dict[str, Any]
    realignment_events: List[Dict[str, Any]]
    exported_files: Dict[str, str]
    warnings: List[str]
    failed_checks: List[Dict[str, str]]
    all_mandatory_checks_passed: bool


def _rasterize_quad_mesh_onto_slice_grid(
    mesh: QuadMesh2D,
    transform: AnatomicalCoordinateTransform2D,
    grid_shape: Tuple[int, int],
) -> np.ndarray:
    nx_vox, ny_vox = grid_shape
    ii, jj = np.meshgrid(np.arange(nx_vox), np.arange(ny_vox), indexing="ij")
    vox_pts = np.column_stack([ii.ravel(), jj.ravel()])
    pts_mm = transform.voxel_ij_to_physical_mm(vox_pts)

    raster = np.zeros(pts_mm.shape[0], dtype=bool)
    verts = mesh.vertices_mm
    for q_idx in range(mesh.quads.shape[0]):
        poly = verts[mesh.quads[q_idx]]
        p_min = np.min(poly, axis=0) - 1.0e-6
        p_max = np.max(poly, axis=0) + 1.0e-6
        in_bbox = (
            (pts_mm[:, 0] >= p_min[0])
            & (pts_mm[:, 0] <= p_max[0])
            & (pts_mm[:, 1] >= p_min[1])
            & (pts_mm[:, 1] <= p_max[1])
        )
        cand_indices = np.flatnonzero(in_bbox & (~raster))
        if cand_indices.size > 0:
            inside = MplPath(poly).contains_points(pts_mm[cand_indices], radius=1.0e-6)
            raster[cand_indices[inside]] = True

    return raster.reshape(grid_shape)


def _count_mesh_connected_components(quads: np.ndarray) -> int:
    n_q = quads.shape[0]
    if n_q == 0:
        return 0
    parent = list(range(n_q))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    edge_owner: Dict[Tuple[int, int], int] = {}
    for q_idx, (v0, v1, v2, v3) in enumerate(quads):
        for u, v in ((v0, v1), (v1, v2), (v2, v3), (v3, v0)):
            key = (min(int(u), int(v)), max(int(u), int(v)))
            if key in edge_owner:
                _union(q_idx, edge_owner[key])
            else:
                edge_owner[key] = q_idx

    return len({_find(i) for i in range(n_q)})


def compute_geometry_quality_report(
    volumes: BraTSSubjectVolumes,
    slice_sel: BraTSSliceSelection,
    masks: AnatomicalMasks2D,
    transform: AnatomicalCoordinateTransform2D,
    mesh: QuadMesh2D,
    exported_files: Dict[str, str],
    thresholds: Optional[Dict[str, float]] = None,
) -> QuadMeshQualityReport:
    thresh = thresholds or {
        "max_relative_area_error": 0.05,
        "min_dice_similarity": 0.92,
        "min_corner_jacobian_mm2": 1.0e-5,
    }

    voxel_area_mm2 = transform.spacing_xy_mm[0] * transform.spacing_xy_mm[1]
    ref_area_mm2 = float(np.sum(masks.brain_support_mask)) * voxel_area_mm2

    signed_areas_mm2 = _quad_signed_areas(mesh.vertices_mm, mesh.quads)
    signed_areas_adim = _quad_signed_areas(mesh.vertices_adim, mesh.quads)
    mesh_area_mm2 = float(np.sum(signed_areas_mm2))
    mesh_area_adim = float(np.sum(signed_areas_adim))
    rel_area_err = abs(mesh_area_mm2 - ref_area_mm2) / max(ref_area_mm2, 1.0e-12)

    raster_mask = _rasterize_quad_mesh_onto_slice_grid(
        mesh, transform, masks.brain_support_mask.shape
    )
    inter = float(np.sum(raster_mask & masks.brain_support_mask))
    sum_masks = float(np.sum(raster_mask) + np.sum(masks.brain_support_mask))
    union_masks = float(np.sum(raster_mask | masks.brain_support_mask))
    dice = (2.0 * inter / sum_masks) if sum_masks > 0.0 else 0.0
    iou = (inter / union_masks) if union_masks > 0.0 else 0.0

    min_j_mm2 = _quad_min_corner_jacobians(mesh.vertices_mm, mesh.quads)
    all_positive_area = bool(np.all(signed_areas_mm2 > 0.0))
    all_positive_jacobian = bool(np.all(min_j_mm2 >= thresh["min_corner_jacobian_mm2"]))
    num_invalid_cells = int(np.sum((signed_areas_mm2 <= 0.0) | (min_j_mm2 <= 0.0)))

    v_q = mesh.vertices_mm[mesh.quads]
    e0 = np.linalg.norm(v_q[:, 1] - v_q[:, 0], axis=1)
    e1 = np.linalg.norm(v_q[:, 2] - v_q[:, 1], axis=1)
    e2 = np.linalg.norm(v_q[:, 3] - v_q[:, 2], axis=1)
    e3 = np.linalg.norm(v_q[:, 0] - v_q[:, 3], axis=1)
    all_edges_mm = np.concatenate([e0, e1, e2, e3])
    diag1 = np.linalg.norm(v_q[:, 2] - v_q[:, 0], axis=1)
    diag2 = np.linalg.norm(v_q[:, 3] - v_q[:, 1], axis=1)
    diameters_mm = np.maximum(diag1, diag2)
    aspect_ratios = np.maximum.reduce([e0, e1, e2, e3]) / np.maximum(
        np.minimum.reduce([e0, e1, e2, e3]), 1.0e-12
    )

    b_vecs_mm = mesh.vertices_mm[mesh.boundary_edges[:, 1]] - mesh.vertices_mm[mesh.boundary_edges[:, 0]]
    b_lengths_mm = np.linalg.norm(b_vecs_mm, axis=1)
    per_bid_summary: Dict[str, Any] = {}
    for bid in sorted(int(x) for x in np.unique(mesh.boundary_ids)):
        sel = mesh.boundary_ids == bid
        l_mm = float(np.sum(b_lengths_mm[sel]))
        per_bid_summary[str(bid)] = {
            "description": "outer_boundary" if bid == BOUNDARY_ID_OUTER_CONTOUR else "inner_hole_boundary",
            "num_boundary_faces": int(np.sum(sel)),
            "contour_length_mm": l_mm,
            "contour_length_adim": l_mm / transform.L0_mm,
        }

    t_ii, t_jj = np.nonzero(masks.whole_tumor_mask)
    if t_ii.size > 0:
        t_ij = np.column_stack([t_ii.astype(np.float64), t_jj.astype(np.float64)])
        t_xy_mm = transform.voxel_ij_to_physical_mm(t_ij)
        lesion_center_mm = np.mean(t_xy_mm, axis=0)
        lesion_center_adim = transform.physical_mm_to_mesh_adim(lesion_center_mm.reshape(1, 2))[0]
        lesion_in_mesh = bool(
            np.any(
                [
                    MplPath(mesh.vertices_mm[mesh.quads[q]]).contains_point(lesion_center_mm, radius=1.0e-4)
                    for q in range(mesh.quads.shape[0])
                ]
            )
        )
        lesion_info = {
            "has_lesion_in_slice": True,
            "lesion_voxel_count": int(t_ii.size),
            "lesion_area_mm2": float(t_ii.size * voxel_area_mm2),
            "center_physical_mm": [
                float(lesion_center_mm[0]),
                float(lesion_center_mm[1]),
                float(transform.physical_z_mm),
            ],
            "center_mesh_adim": [float(lesion_center_adim[0]), float(lesion_center_adim[1])],
            "center_inside_quad_mesh": lesion_in_mesh,
            "lesion_voxels_inside_support_ratio": float(masks.lesion_inside_support_ratio),
        }
    else:
        lesion_info = {
            "has_lesion_in_slice": False,
            "lesion_voxel_count": 0,
            "lesion_area_mm2": 0.0,
            "center_physical_mm": None,
            "center_mesh_adim": None,
            "center_inside_quad_mesh": False,
            "lesion_voxels_inside_support_ratio": 1.0,
        }

    _, n_cc_mask = ndi.label(masks.brain_support_mask)
    n_cc_mesh = _count_mesh_connected_components(mesh.quads)

    failed_checks: List[Dict[str, str]] = []
    if not all_positive_area or not all_positive_jacobian or num_invalid_cells > 0:
        failed_checks.append(
            {
                "check": "cell_positivity_and_jacobian",
                "reason": f"Trovate {num_invalid_cells} celle degenerate o invertite.",
            }
        )
    if rel_area_err > thresh["max_relative_area_error"]:
        failed_checks.append(
            {
                "check": "relative_area_error",
                "reason": (
                    f"Errore relativo d'area {rel_area_err:.4f} superiore alla soglia "
                    f"{thresh['max_relative_area_error']:.4f}."
                ),
            }
        )
    if dice < thresh["min_dice_similarity"]:
        failed_checks.append(
            {
                "check": "rasterized_dice_similarity",
                "reason": (
                    f"Dice similarity {dice:.4f} inferiore alla soglia "
                    f"{thresh['min_dice_similarity']:.4f}."
                ),
            }
        )
    if n_cc_mesh != 1:
        failed_checks.append(
            {
                "check": "mesh_connected_components",
                "reason": f"La mesh presenta {n_cc_mesh} componenti connesse invece di 1.",
            }
        )

    slice_label_counts = {
        str(lb): int(np.sum(slice_sel.seg_2d == lb)) for lb in BRATS_VALID_LABELS
    }
    vol_label_counts = {
        str(lb): int(np.sum(volumes.seg == lb)) for lb in BRATS_VALID_LABELS
    }

    input_provenance_dict = {
        "source_data_kind": volumes.source_data_kind,
        "provenance_verified": volumes.provenance_verified,
        "volume_shape_3d_ras": list(volumes.shape),
        "voxel_spacing_mm_ras": list(volumes.spacing_mm),
        "sha256_by_volume": {k: rec.sha256 for k, rec in volumes.volume_provenance.items()},
        "absolute_paths_by_volume": dict(volumes.file_paths),
        "volumes_metadata": {k: asdict(rec) for k, rec in volumes.volume_provenance.items()},
    }

    return QuadMeshQualityReport(
        subject_id=volumes.subject_id,
        source_data_kind=volumes.source_data_kind,
        provenance_verified=volumes.provenance_verified,
        input_provenance=input_provenance_dict,
        domain_representation_type=masks.domain_representation_type,
        orientation_convention=volumes.orientation_convention,
        slice_index=slice_sel.slice_index,
        selection_mode=slice_sel.selection_mode,
        physical_z_mm=slice_sel.physical_z_mm,
        L0_m=transform.L0_m,
        L0_mm=transform.L0_mm,
        reference_mask_area_mm2=ref_area_mm2,
        mesh_area_mm2=mesh_area_mm2,
        mesh_area_adim=mesh_area_adim,
        relative_area_error=rel_area_err,
        rasterized_dice_similarity=dice,
        rasterized_iou_jaccard=iou,
        num_nodes=int(mesh.vertices_mm.shape[0]),
        num_quad_cells=int(mesh.quads.shape[0]),
        num_triangles=0,
        num_connected_components_mask=int(n_cc_mask),
        num_connected_components_mesh=int(n_cc_mesh),
        mesh_resolution={
            "target_element_size_mm": mesh.target_element_size_mm,
            "target_element_size_adim": mesh.target_element_size_mm / transform.L0_mm,
            "mean_edge_length_mm": float(np.mean(all_edges_mm)),
            "min_edge_length_mm": float(np.min(all_edges_mm)),
            "max_edge_length_mm": float(np.max(all_edges_mm)),
            "mean_cell_diameter_mm": float(np.mean(diameters_mm)),
            "mean_cell_diameter_adim": float(np.mean(diameters_mm)) / transform.L0_mm,
        },
        boundary_metrics={
            "total_boundary_faces": int(mesh.boundary_edges.shape[0]),
            "total_contour_length_mm": float(np.sum(b_lengths_mm)),
            "total_contour_length_adim": float(np.sum(b_lengths_mm)) / transform.L0_mm,
            "by_boundary_id": per_bid_summary,
        },
        lesion_centroids=lesion_info,
        cell_geometry_checks={
            "all_cells_quadrilateral": bool(mesh.quads.shape[1] == 4),
            "all_cells_positive_area": all_positive_area,
            "all_cells_ccw_oriented": all_positive_area,
            "all_corners_positive_jacobian": all_positive_jacobian,
            "num_inverted_or_degenerate_cells": num_invalid_cells,
            "min_cell_area_mm2": float(np.min(signed_areas_mm2)),
            "max_cell_area_mm2": float(np.max(signed_areas_mm2)),
            "min_corner_jacobian_mm2": float(np.min(min_j_mm2)),
            "max_aspect_ratio": float(np.max(aspect_ratios)),
            "mean_aspect_ratio": float(np.mean(aspect_ratios)),
        },
        segmentation_labels_report={
            "allowed_brats_labels": list(BRATS_VALID_LABELS),
            "observed_labels_before_preprocessing": volumes.seg_labels_before_preprocessing,
            "observed_labels_after_preprocessing": volumes.seg_labels_after_preprocessing,
            "volume_3d_voxel_counts_by_label": vol_label_counts,
            "selected_slice_voxel_counts_by_label": slice_label_counts,
            "original_orientations_by_volume": {
                k: "".join(v) for k, v in volumes.original_orientations.items()
            },
        },
        coordinate_transform=transform.to_metadata_dict(),
        realignment_events=volumes.realignment_log,
        exported_files=exported_files,
        warnings=masks.warnings,
        failed_checks=failed_checks,
        all_mandatory_checks_passed=(len(failed_checks) == 0),
    )


# =============================================================================
# 7. Diagnostica visiva (Figure PDF con provenienza, legenda, unità e scala)
# =============================================================================

def _slice_physical_extent_mm(
    shape_2d: Tuple[int, int], transform: AnatomicalCoordinateTransform2D
) -> List[float]:
    corners_ij = np.array([[0.0, 0.0], [float(shape_2d[0] - 1), float(shape_2d[1] - 1)]])
    corners_mm = transform.voxel_ij_to_physical_mm(corners_ij)
    x0, x1 = float(np.min(corners_mm[:, 0])), float(np.max(corners_mm[:, 0]))
    y0, y1 = float(np.min(corners_mm[:, 1])), float(np.max(corners_mm[:, 1]))
    return [x0, x1, y0, y1]


def generate_diagnostic_figures(
    volumes: BraTSSubjectVolumes,
    slice_sel: BraTSSliceSelection,
    masks: AnatomicalMasks2D,
    transform: AnatomicalCoordinateTransform2D,
    mesh: QuadMesh2D,
    report: QuadMeshQualityReport,
    output_dir: str | Path,
) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extent_mm = _slice_physical_extent_mm(slice_sel.seg_2d.shape, transform)

    domain_label_str = (
        "Brain Parenchyma Proxy (NO WM Mask)"
        if masks.domain_representation_type == "brain_parenchyma_proxy"
        else "Segmented White Matter Mask"
    )
    slice_desc = (
        f"Soggetto: {volumes.subject_id} [{volumes.source_data_kind}] | "
        f"Shape 3D: {volumes.shape[0]}x{volumes.shape[1]}x{volumes.shape[2]} ({volumes.orientation_convention}) | "
        f"Sezione k={slice_sel.slice_index} ({slice_sel.selection_mode}, z={slice_sel.physical_z_mm:.2f} mm) | "
        f"L0={transform.L0_mm:.1f} mm"
    )

    # Figura 1: 4 pannelli MRI (T1, T1ce, T2, FLAIR)
    fig1_path = out_dir / "fig01_mri_modalities_slice.pdf"
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
    fig.suptitle(
        f"Modalità MRI della Sezione Anatomica Selezionata\n{slice_desc}",
        fontsize=10.5,
        fontweight="bold",
    )

    for ax, mod_name in zip(axes, ("T1", "T1ce", "T2", "FLAIR")):
        arr = slice_sel.modalities_2d[mod_name].T
        im = ax.imshow(arr, cmap="gray", origin="lower", extent=extent_mm)
        ax.set_title(f"Modalità {mod_name}", fontsize=10, fontweight="bold")
        ax.set_xlabel("x RAS+ [mm]")
        ax.set_ylabel("y RAS+ [mm]")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Intensità [a.u.]", fontsize=8)

    x_bar0 = extent_mm[0] + 15.0
    y_bar0 = extent_mm[2] + 15.0
    axes[0].plot(
        [x_bar0, x_bar0 + transform.L0_mm],
        [y_bar0, y_bar0],
        color="cyan",
        lw=3.0,
    )
    axes[0].text(
        x_bar0,
        y_bar0 + 5.0,
        f"L0 = {transform.L0_mm:.0f} mm",
        color="cyan",
        fontsize=8,
        fontweight="bold",
    )
    fig.savefig(fig1_path, format="pdf", dpi=300)
    plt.close(fig)

    # Figura 2: Sovrapposizione delle label tumorali BraTS (1, 2, 4)
    fig2_path = out_dir / "fig02_tumor_labels_overlay.pdf"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    fig.suptitle(
        f"Segmentazione Tumorale BraTS (Label 1, 2, 4) — {domain_label_str}\n{slice_desc}",
        fontsize=10,
        fontweight="bold",
    )

    bg_flair = slice_sel.modalities_2d["FLAIR"].T
    bg_t1ce = slice_sel.modalities_2d["T1ce"].T

    rgba = np.zeros((slice_sel.seg_2d.shape[1], slice_sel.seg_2d.shape[0], 4), dtype=np.float64)
    rgba[masks.edema_mask.T] = [0.95, 0.80, 0.15, 0.55]
    rgba[masks.necrotic_non_enhancing_mask.T] = [0.85, 0.15, 0.15, 0.70]
    rgba[masks.enhancing_tumor_mask.T] = [0.10, 0.75, 0.95, 0.75]

    for ax, bg, title in zip(
        axes,
        (bg_flair, bg_t1ce),
        ("FLAIR + Sovrapposizione Label BraTS", "T1ce + Centroide Lesione e Contorno Supporto"),
    ):
        ax.imshow(bg, cmap="gray", origin="lower", extent=extent_mm)
        ax.imshow(rgba, origin="lower", extent=extent_mm)
        ax.plot(
            mesh.outer_contour_mm[:, 0],
            mesh.outer_contour_mm[:, 1],
            color="lime",
            lw=1.3,
            label="Contorno supporto (ID=1)",
        )
        if report.lesion_centroids["has_lesion_in_slice"]:
            cx_mm, cy_mm, _ = report.lesion_centroids["center_physical_mm"]
            ax.scatter(
                [cx_mm],
                [cy_mm],
                color="magenta",
                marker="x",
                s=80,
                linewidths=2.2,
                label=f"Centro lesione ({cx_mm:.1f}, {cy_mm:.1f}) mm",
            )
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("x RAS+ [mm]")
        ax.set_ylabel("y RAS+ [mm]")

    legend_handles = [
        Patch(facecolor=(0.85, 0.15, 0.15, 0.75), label="Label 1: Necrotic / Non-Enhancing (NCR/NET)"),
        Patch(facecolor=(0.95, 0.80, 0.15, 0.65), label="Label 2: Peritumoral Edema (ED)"),
        Patch(facecolor=(0.10, 0.75, 0.95, 0.75), label="Label 4: Enhancing Tumor (ET)"),
    ]
    axes[0].legend(handles=legend_handles, loc="lower left", fontsize=7.5, framealpha=0.85)
    axes[1].legend(loc="lower left", fontsize=8, framealpha=0.85)
    fig.savefig(fig2_path, format="pdf", dpi=300)
    plt.close(fig)

    # Figura 3: Maschera del dominio (`brain_parenchyma_proxy`) e contorni
    fig3_path = out_dir / "fig03_domain_mask_and_contours.pdf"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    fig.suptitle(
        f"Maschera del Supporto Anatomico ({domain_label_str}) e Frontiere\n{slice_desc}",
        fontsize=10,
        fontweight="bold",
    )

    axes[0].imshow(masks.brain_support_mask.T, cmap="Blues", origin="lower", extent=extent_mm)
    axes[0].plot(
        mesh.outer_contour_mm[:, 0],
        mesh.outer_contour_mm[:, 1],
        color="crimson",
        lw=1.8,
        label="Contorno esterno (boundary_id=1)",
    )
    for idx_h, hole_mm in enumerate(mesh.inner_contours_mm):
        lbl = "Contorno interno (boundary_id=2)" if idx_h == 0 else None
        axes[0].plot(hole_mm[:, 0], hole_mm[:, 1], color="darkorange", lw=1.8, linestyle="--", label=lbl)
    axes[0].set_title(
        f"Maschera Dominio ({report.reference_mask_area_mm2:.1f} mm²)", fontsize=10, fontweight="bold"
    )
    axes[0].set_xlabel("x RAS+ [mm]")
    axes[0].set_ylabel("y RAS+ [mm]")
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)

    outer_adim = transform.physical_mm_to_mesh_adim(mesh.outer_contour_mm)
    axes[1].plot(
        np.append(outer_adim[:, 0], outer_adim[0, 0]),
        np.append(outer_adim[:, 1], outer_adim[0, 1]),
        color="crimson",
        lw=1.8,
        label="Frontiera esterna (ID=1)",
    )
    for idx_h, hole_mm in enumerate(mesh.inner_contours_mm):
        hole_adim = transform.physical_mm_to_mesh_adim(hole_mm)
        lbl = "Frontiera interna (ID=2)" if idx_h == 0 else None
        axes[1].plot(
            np.append(hole_adim[:, 0], hole_adim[0, 0]),
            np.append(hole_adim[:, 1], hole_adim[0, 1]),
            color="darkorange",
            lw=1.8,
            linestyle="--",
            label=lbl,
        )
    if report.lesion_centroids["has_lesion_in_slice"]:
        cx_a, cy_a = report.lesion_centroids["center_mesh_adim"]
        axes[1].scatter(
            [cx_a],
            [cy_a],
            color="magenta",
            marker="x",
            s=80,
            linewidths=2.2,
            label=f"Centro lesione ({cx_a:.2f}, {cy_a:.2f}) L0",
        )
    axes[1].axhline(0.0, color="gray", lw=0.6, ls=":")
    axes[1].axvline(0.0, color="gray", lw=0.6, ls=":")
    axes[1].set_aspect("equal")
    axes[1].set_title(
        f"Riferimento Adimensionale Solver (L0 = {transform.L0_mm:.1f} mm)",
        fontsize=10,
        fontweight="bold",
    )
    axes[1].set_xlabel("x* = (x - x0) / L0 [-]")
    axes[1].set_ylabel("y* = (y - y0) / L0 [-]")
    axes[1].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.savefig(fig3_path, format="pdf", dpi=300)
    plt.close(fig)

    # Figura 4: Sovrapposizione della mesh quadrilatera e ingrandimento del bordo
    fig4_path = out_dir / "fig04_quad_mesh_and_boundary_zoom.pdf"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
    fig.suptitle(
        f"Mesh 2D Puramente Quadrilatera ({report.num_quad_cells} quad, 0 triangoli, "
        f"Dice={report.rasterized_dice_similarity:.3f}, ErrArea={100*report.relative_area_error:.2f}%)\n"
        f"{slice_desc} — {domain_label_str}",
        fontsize=9.5,
        fontweight="bold",
    )

    axes[0].imshow(slice_sel.modalities_2d["T1"].T, cmap="gray", origin="lower", extent=extent_mm, alpha=0.85)
    polys_mm = mesh.vertices_mm[mesh.quads]
    pc_full = PolyCollection(
        polys_mm,
        facecolors="none",
        edgecolors="deepskyblue",
        linewidths=0.45,
        alpha=0.85,
    )
    axes[0].add_collection(pc_full)

    for b_idx in range(mesh.boundary_edges.shape[0]):
        u, v = mesh.boundary_edges[b_idx]
        bid = int(mesh.boundary_ids[b_idx])
        col = "crimson" if bid == BOUNDARY_ID_OUTER_CONTOUR else "gold"
        axes[0].plot(
            [mesh.vertices_mm[u, 0], mesh.vertices_mm[v, 0]],
            [mesh.vertices_mm[u, 1], mesh.vertices_mm[v, 1]],
            color=col,
            lw=1.2,
        )

    axes[0].set_title(
        f"Mesh Quadrilatera su T1 (h ≈ {mesh.target_element_size_mm:.1f} mm)",
        fontsize=10,
        fontweight="bold",
    )
    axes[0].set_xlabel("x RAS+ [mm]")
    axes[0].set_ylabel("y RAS+ [mm]")

    idx_top = int(np.argmax(mesh.outer_contour_mm[:, 1]))
    focus_pt = mesh.outer_contour_mm[idx_top]
    half_win = max(5.0 * mesh.target_element_size_mm, 14.0)
    x_z0, x_z1 = focus_pt[0] - half_win, focus_pt[0] + half_win
    y_z0, y_z1 = focus_pt[1] - 1.4 * half_win, focus_pt[1] + 0.6 * half_win

    axes[0].plot(
        [x_z0, x_z1, x_z1, x_z0, x_z0],
        [y_z0, y_z0, y_z1, y_z1, y_z0],
        color="yellow",
        lw=1.4,
        ls="--",
        label="Regione Zoom",
    )
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)

    axes[1].imshow(masks.brain_support_mask.T, cmap="Greys", origin="lower", extent=extent_mm, alpha=0.35)
    pc_zoom = PolyCollection(
        polys_mm,
        facecolors=(0.12, 0.56, 1.0, 0.18),
        edgecolors="navy",
        linewidths=1.0,
    )
    axes[1].add_collection(pc_zoom)
    axes[1].plot(
        mesh.outer_contour_mm[:, 0],
        mesh.outer_contour_mm[:, 1],
        color="limegreen",
        lw=1.8,
        ls="--",
        label="Contorno di riferimento maschera",
    )
    for b_idx in range(mesh.boundary_edges.shape[0]):
        u, v = mesh.boundary_edges[b_idx]
        bid = int(mesh.boundary_ids[b_idx])
        col = "crimson" if bid == BOUNDARY_ID_OUTER_CONTOUR else "darkorange"
        axes[1].plot(
            [mesh.vertices_mm[u, 0], mesh.vertices_mm[v, 0]],
            [mesh.vertices_mm[u, 1], mesh.vertices_mm[v, 1]],
            color=col,
            lw=2.0,
        )
    axes[1].set_xlim(x_z0, x_z1)
    axes[1].set_ylim(y_z0, y_z1)
    axes[1].set_aspect("equal")
    axes[1].set_title("Ingrandimento del Bordo e Conformità Celle Quad", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("x RAS+ [mm]")
    axes[1].set_ylabel("y RAS+ [mm]")
    axes[1].legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.savefig(fig4_path, format="pdf", dpi=300)
    plt.close(fig)

    return {
        "fig01_mri_modalities_pdf": str(fig1_path),
        "fig02_tumor_labels_pdf": str(fig2_path),
        "fig03_domain_mask_pdf": str(fig3_path),
        "fig04_quad_mesh_zoom_pdf": str(fig4_path),
    }


# =============================================================================
# 8. Entry-point completo della pipeline e salvataggio del report JSON
# =============================================================================

def run_brats_anatomical_pipeline(
    cfg: BraTSPipelineConfig,
) -> Tuple[QuadMeshQualityReport, QuadMesh2D]:
    l0_m, _ = load_l0_from_physics_config(cfg.physics_config_path)

    volumes = load_brats_subject(
        subject_dir=cfg.subject_dir,
        subject_id=cfg.subject_id,
        modality_filenames=cfg.modality_filenames,
        wm_mask_path=cfg.wm_mask_path,
        allow_grid_realignment=cfg.allow_grid_realignment,
        remap_label_3_to_4=cfg.remap_label_3_to_4,
        provenance_cfg=cfg.provenance,
    )

    slice_sel = select_anatomical_slice(
        volumes=volumes,
        mode=cfg.slice_selection_mode,
        manual_slice_index=cfg.manual_slice_index,
    )

    masks = build_anatomical_masks_2d(
        slice_sel=slice_sel,
        spacing_xy_mm=(volumes.spacing_mm[0], volumes.spacing_mm[1]),
        cfg=cfg.support_mask,
    )

    transform = AnatomicalCoordinateTransform2D.from_ras_affine_and_mask(
        ras_affine=volumes.ras_affine,
        slice_index=slice_sel.slice_index,
        support_mask=masks.brain_support_mask,
        L0_m=l0_m,
    )

    mesh = generate_anatomical_quad_mesh_2d(
        masks=masks,
        slice_sel=slice_sel,
        transform=transform,
        mesh_cfg=cfg.mesh,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    exported_mesh_files = export_dealii_quad_mesh(
        mesh=mesh,
        transform=transform,
        output_dir=cfg.output_dir,
        basename=f"{cfg.subject_id}_slice{slice_sel.slice_index:03d}_quad_mesh",
    )

    report = compute_geometry_quality_report(
        volumes=volumes,
        slice_sel=slice_sel,
        masks=masks,
        transform=transform,
        mesh=mesh,
        exported_files=exported_mesh_files,
        thresholds=cfg.quality_thresholds,
    )

    fig_files = generate_diagnostic_figures(
        volumes=volumes,
        slice_sel=slice_sel,
        masks=masks,
        transform=transform,
        mesh=mesh,
        report=report,
        output_dir=cfg.output_dir,
    )
    report.exported_files.update(fig_files)

    report_json_path = cfg.output_dir / "geometry_quality_report.json"
    report.exported_files["geometry_quality_report_json"] = str(report_json_path)

    with report_json_path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(report), fh, indent=2, ensure_ascii=False)

    return report, mesh