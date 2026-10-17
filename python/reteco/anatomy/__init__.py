"""
python/reteco/anatomy/__init__.py
Pipeline anatomica 2D riproducibile per BraTS 2020 con verifica rigorosa
della provenienza (SHA-256, metadati NIfTI, controllo griglia 240x240x155).
"""

from __future__ import annotations

from .brats_geometry import (
    BOUNDARY_ID_INNER_HOLES,
    BOUNDARY_ID_OUTER_CONTOUR,
    BRATS_VALID_LABELS,
    AnatomicalCoordinateTransform2D,
    AnatomicalMasks2D,
    BraTSPipelineConfig,
    BraTSSliceSelection,
    BraTSSubjectVolumes,
    ProvenanceConfig,
    QuadMesh2D,
    QuadMeshQualityReport,
    VolumeProvenanceRecord,
    build_anatomical_masks_2d,
    compute_geometry_quality_report,
    export_dealii_quad_mesh,
    generate_anatomical_quad_mesh_2d,
    generate_diagnostic_figures,
    load_brats_pipeline_config,
    load_brats_subject,
    load_l0_from_physics_config,
    run_brats_anatomical_pipeline,
    select_anatomical_slice,
)

__all__ = [
    "BOUNDARY_ID_INNER_HOLES",
    "BOUNDARY_ID_OUTER_CONTOUR",
    "BRATS_VALID_LABELS",
    "AnatomicalCoordinateTransform2D",
    "AnatomicalMasks2D",
    "BraTSPipelineConfig",
    "BraTSSliceSelection",
    "BraTSSubjectVolumes",
    "ProvenanceConfig",
    "QuadMesh2D",
    "QuadMeshQualityReport",
    "VolumeProvenanceRecord",
    "build_anatomical_masks_2d",
    "compute_geometry_quality_report",
    "export_dealii_quad_mesh",
    "generate_anatomical_quad_mesh_2d",
    "generate_diagnostic_figures",
    "load_brats_pipeline_config",
    "load_brats_subject",
    "load_l0_from_physics_config",
    "run_brats_anatomical_pipeline",
    "select_anatomical_slice",
]