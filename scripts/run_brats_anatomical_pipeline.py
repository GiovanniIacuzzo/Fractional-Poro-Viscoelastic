#!/usr/bin/env python3
"""
scripts/run_brats_anatomical_pipeline.py
Comando di produzione per eseguire la pipeline anatomica BraTS 2020 di Fase 1.
NON contiene né consente la generazione automatica di volumi sintetici:
se i dati clinici specificati in configurazione mancano o non superano i controlli
di provenienza (240x240x155, 1 mm, SHA-256), termina con un errore esplicito.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT_DIR / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from reteco.anatomy.brats_geometry import (
    REQUIRED_VOLUME_KEYS,
    load_brats_pipeline_config,
    run_brats_anatomical_pipeline,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RETECO Phase 1 — Clinical BraTS 2020 2D Anatomical Quad-Mesh Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/brats_anatomical_forward.yaml",
        help="Percorso al file YAML di configurazione anatomica.",
    )
    args = parser.parse_args()

    cfg = load_brats_pipeline_config(args.config)
    subj_dir = cfg.subject_dir.resolve()
    if not subj_dir.exists() or not subj_dir.is_dir():
        raise FileNotFoundError(
            f"Directory soggetto clinico non trovata: {subj_dir}. "
            "Nessuna generazione di dati sintetici è consentita."
        )

    if not cfg.modality_filenames:
        raise ValueError(
            "Configurazione incompleta: `modality_filenames` deve specificare esplicitamente "
            "i 5 volumi NIfTI (T1, T1ce, T2, FLAIR, seg)."
        )

    for key in REQUIRED_VOLUME_KEYS:
        fname = cfg.modality_filenames.get(key)
        if not fname:
            raise ValueError(f"Chiave '{key}' mancante in `modality_filenames`.")
        fpath = (subj_dir / fname).resolve()
        if not fpath.is_file():
            raise FileNotFoundError(
                f"Volume clinico obbligatorio '{key}' non trovato: {fpath}. "
                "Esecuzione interrotta (nessun fallback sintetico)."
            )

    report, _ = run_brats_anatomical_pipeline(cfg)
    report_path = Path(report.exported_files["geometry_quality_report_json"])
    print(json.dumps(json.loads(report_path.read_text(encoding="utf-8")), indent=2, ensure_ascii=False))
    return 0 if report.all_mandatory_checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())