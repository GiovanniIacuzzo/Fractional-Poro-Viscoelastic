#!/usr/bin/env python3
"""Verifica figure e citazioni BibTeX non usate nei .tex del paper.

Uso: python check_unused.py [cartella_paper]   (default: paper)
Output: solo le figure e le citazioni non presenti in nessun .tex.
"""
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "paper")
IMG_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".eps", ".svg"}


def strip_comments(text: str) -> str:
    # rimuove i commenti LaTeX (% non preceduto da backslash)
    return re.sub(r"(?<!\\)%.*", "", text)


tex_files = sorted(ROOT.rglob("*.tex"))
tex = "\n".join(strip_comments(f.read_text(encoding="utf-8", errors="ignore")) for f in tex_files)

# --- Figure ---------------------------------------------------------------
# argomenti di \includegraphics[...]{...}, \includegraphics*, \includesvg, \subfile ecc.
used_paths = re.findall(r"\\includegraphics\*?\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}", tex)
used_paths += re.findall(r"\\includesvg\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}", tex)
used_norm = {p.strip().replace("\\", "/") for p in used_paths}
used_stems = {Path(p).stem for p in used_norm}
used_names = {Path(p).name for p in used_norm}

# \graphicspath: cartelle di ricerca (non influisce sul confronto per nome/stem)
unused_figs = []
for img in sorted((ROOT / "images").rglob("*")):
    if img.suffix.lower() not in IMG_EXT:
        continue
    rel = img.relative_to(ROOT).as_posix()
    rel_noext = Path(rel).with_suffix("").as_posix()
    hit = any(
        u == rel or u == rel_noext or u.endswith("/" + rel) or u.endswith("/" + rel_noext)
        or u == img.name or u == img.stem
        or rel.endswith("/" + u) or rel_noext.endswith("/" + u)
        for u in used_norm
    )
    if not hit:
        unused_figs.append(rel)

# --- Citazioni ------------------------------------------------------------
bib_text = "\n".join(
    f.read_text(encoding="utf-8", errors="ignore") for f in sorted(ROOT.rglob("*.bib"))
)
bib_keys = re.findall(r"@(?!comment|string|preamble)\w+\s*\{\s*([^,\s]+)\s*,", bib_text, re.I)

cited = set()
for m in re.finditer(r"\\(?:[a-zA-Z]*cite[a-zA-Z]*|nocite)\*?\s*(?:\[[^\]]*\]\s*)*\{([^}]*)\}", tex):
    for k in m.group(1).split(","):
        cited.add(k.strip())

nocite_all = "*" in cited
unused_cites = [] if nocite_all else [k for k in bib_keys if k not in cited]

# --- Report ---------------------------------------------------------------
print("Figure non presenti in nessun .tex:")
for f in unused_figs:
    print(f"  {f}")
if not unused_figs:
    print("  (nessuna)")

print("\nCitazioni non presenti in nessun .tex:")
for k in unused_cites:
    print(f"  {k}")
if not unused_cites:
    print("  (nessuna)")