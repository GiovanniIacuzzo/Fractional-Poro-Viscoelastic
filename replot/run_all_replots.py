"""
replot/run_all_replots.py
Master entrypoint to regenerate all scientific figures in the repository from saved numerical artifacts.
Saves all vector PDFs and 300-dpi PNGs into `replot/output/<family_subdir>/`.

Never runs forward FEM simulations, mesh generation, or inverse optimization campaigns.
Fails explicitly with a non-zero exit code if any required numerical data file is missing.
"""

from __future__ import annotations
import sys
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

from common_style import OUTPUT_ROOT, REPO_ROOT, apply_common_style
import replot_forward_verification
import replot_anatomical
import replot_identifiability
import replot_parameter_profiles
import replot_twin_robustness
import replot_optimizer_benchmarks


def main() -> int:
    apply_common_style()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    families: List[Tuple[str, Callable[[], List[Tuple[Path, Path]]]]] = [
        ("A. Forward Model Verification (01)", replot_forward_verification.run_all),
        ("B. Anatomical BraTS 2020 Clinical Case (02_anatomical, phase1)", replot_anatomical.run_all),
        ("C. Identifiability & Sensitivity Extensions (02_identifiability, 13)", replot_identifiability.run_all),
        ("D. Parameter Landscapes, Profiles & Mesh Sensitivity (08, 09, 10)", replot_parameter_profiles.run_all),
        ("E. Twin Experiments & Final Recovery Campaign (03, 03a, 03b, 04, 05, 11)", replot_twin_robustness.run_all),
        ("F. CPSO Diagnostics, Multi-Seed & SOTA Benchmarks (06, 07, 12, 14)", replot_optimizer_benchmarks.run_all),
    ]

    all_generated: List[Tuple[Path, Path]] = []
    failed_families: List[Tuple[str, str]] = []

    print("=" * 78)
    print("SCIENTIFIC REPLOTTING SUITE — STANDALONE DATA-ONLY RECONSTRUCTION")
    print(f"Repository root : {REPO_ROOT}")
    print(f"Output directory: {OUTPUT_ROOT}")
    print("=" * 78)

    for family_title, runner in families:
        print(f"\n>>> Running family: {family_title}")
        try:
            generated = runner()
            for pdf_p, png_p in generated:
                if not pdf_p.exists() or not png_p.exists():
                    raise RuntimeError(f"Expected output files were not created: {pdf_p}, {png_p}")
                rel_pdf = pdf_p.relative_to(REPO_ROOT)
                rel_png = png_p.relative_to(REPO_ROOT)
                print(f"  [OK] {rel_pdf}  |  {rel_png}")
            all_generated.extend(generated)
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"  [ERROR] Failed in '{family_title}': {exc}", file=sys.stderr)
            print(tb, file=sys.stderr)
            failed_families.append((family_title, str(exc)))

    print("\n" + "=" * 78)
    print(f"REPLOTTING SUMMARY: {len(all_generated)} figure pairs (PDF + 300-dpi PNG) generated.")
    for notice in replot_anatomical.get_unreconstructible_anatomical_notices():
        print(f"  [SKIPPED - MISSING RAW ARRAY IN ZIP] {notice}")

    if failed_families:
        print("\n[FAILED FAMILIES]:", file=sys.stderr)
        for fam, err_msg in failed_families:
            print(f"  - {fam}: {err_msg}", file=sys.stderr)
        return 1

    print("All reconstructible scientific figures were successfully exported to `replot/output/`.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())