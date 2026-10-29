#!/usr/bin/env python3
"""Rebuild the PHASE-TTC paper, tables, figures, and saved-prediction analyses.

The build uses the included result ledgers, without network access, training,
or sensor inference. main.tex is the modular entry point. The build also
produces PHASE_TTC_Final.tex with inlined text, styles, tables, and bibliography.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "revision"
STEM = "PHASE_TTC_Final"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: list[str], log_name: str) -> str:
    result = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (REV / log_name).write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(args)}\n"
            f"See revision/{log_name}\n{result.stdout[-4000:]}"
        )
    return result.stdout


def flatten_source() -> Path:
    """Inline styles, manuscript, tables, and bibliography; keep figure assets."""
    active: set[Path] = set()

    def expand(path: Path) -> str:
        path = path.resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(f"Source path outside project: {path}")
        if path in active:
            raise ValueError(f"Circular input: {path}")
        active.add(path)
        text = path.read_text(encoding="utf-8")

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            child = ROOT / (name if Path(name).suffix else name + ".tex")
            # Avoid adding paragraph breaks inside centered tables or math.
            return expand(child).rstrip("\r\n")

        text = re.sub(r"\\input\{([^}]+)\}", replace, text)
        active.remove(path)
        return text

    text = expand(ROOT / "main.tex")
    style = (ROOT / "paper_style.sty").read_text(encoding="utf-8")
    style = re.sub(r"^\\(?:NeedsTeXFormat|ProvidesPackage).*\n", "", style, flags=re.MULTILINE)
    style = style.replace(r"\endinput", "")
    text = text.replace(
        r"\usepackage{paper_style}",
        "% BEGIN inlined paper_style.sty\n" + style + "\n% END inlined paper_style.sty",
    )
    text = text.replace(r"\bibliographystyle{plainnat}", "")
    text = text.replace(
        r"\bibliography{references}", (ROOT / "main.bbl").read_text(encoding="utf-8")
    )
    header = (
        "% PHASE-TTC / From Motion to Time.\n"
        "% Consolidated manuscript, styles, tables, and bibliography.\n"
        "% Requires the figures/ directory from the complete source ZIP.\n"
        "% Compile with pdfLaTeX three times (no BibTeX required for this file).\n"
        "% Result ledgers and reproducible analyses accompany the source package.\n\n"
    )
    target = ROOT / f"{STEM}.tex"
    target.write_text(header + text, encoding="utf-8")
    return target


def check_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    checks = {
        "overfull_boxes": len(re.findall(r"Overfull \\[hv]box", text)),
        "missing_glyphs": text.count("Missing character:"),
        "undefined_references_or_citations": len(
            re.findall(r"(?:Reference|Citation).*undefined|There were undefined", text)
        ),
        "latex_warnings": len(re.findall(r"(?:LaTeX|Package [\w-]+) Warning:", text)),
    }
    if any(checks.values()):
        raise RuntimeError(f"Unresolved layout/reference warning in {path.name}: {checks}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-regenerate",
        action="store_true",
        help="Verify evidence and compile the existing figures and tables.",
    )
    args = parser.parse_args()
    REV.mkdir(exist_ok=True)
    tex = shutil.which("pdflatex")
    if not tex:
        raise RuntimeError("pdfLaTeX is required. Install a full TeX Live or MiKTeX distribution.")
    bib = shutil.which("bibtex.original") or shutil.which("bibtex") or shutil.which("bibtex8")
    if not bib:
        raise RuntimeError("BibTeX is required for the modular-source rebuild.")

    print("Checking public evidence ...", flush=True)
    result = run([sys.executable, "scripts/verify_public_data.py"], "public_validation.log")
    checks = json.loads(result.strip().splitlines()[-1])
    (REV / "public_validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    run(
        [sys.executable, "-X", "utf8", "scripts/make_deployment_table.py"],
        "deployment_validation.log",
    )
    run(
        [
            sys.executable,
            "-X",
            "utf8",
            "scripts/make_evidence_index.py",
            "--public",
            "--verify-only",
        ],
        "evidence_validation.log",
    )
    spec = importlib.util.spec_from_file_location(
        "competition_evidence", ROOT / "scripts/make_evttc_results.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load competition validator.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows, summary = module.load_public_data()
    derived = module.validate(rows, summary)
    competition = {
        "prediction_rows": len(rows),
        "query_identities": len(derived["by_identity"]),
        "sequence_variant_cells": len(derived["per_sequence"]),
        "stats": derived["stats"],
        "disagreements": derived["disagreements"],
        "scope": "Recomputation of distributed saved predictions; no raw inference",
    }
    (REV / "competition_validation.json").write_text(json.dumps(competition, indent=2) + "\n")

    if not args.skip_regenerate:
        print("Regenerating historical and benchmark tables and figures ...", flush=True)
        for script in [
            "make_early_tables.py",
            "make_concept_figures.py",
            "make_result_figures.py",
            "make_official_results.py",
            "make_evttc_results.py",
            "make_population_table.py",
            "make_middle_tables.py",
        ]:
            run([sys.executable, "-X", "utf8", "scripts/" + script], script + ".log")
        print("Regenerating 17 vector figures and paired saved-data analyses ...", flush=True)
        run([sys.executable, "-X", "utf8", "scripts/prepare_revision.py"], "regeneration.log")

    texargs = [tex, "-interaction=nonstopmode", "-halt-on-error"]
    print("Compiling modular manuscript ...", flush=True)
    run(texargs + ["main.tex"], "modular_pass1.log")
    run([bib, "main"], "bibliography_build.log")
    for i in range(2, 5):
        run(texargs + ["main.tex"], f"modular_pass{i}.log")
    modular_checks = check_log(ROOT / "main.log")

    print("Consolidating and compiling the delivered .tex ...", flush=True)
    flatten_source()
    for i in range(1, 4):
        run(texargs + [f"{STEM}.tex"], f"delivery_pass{i}.log")
    log_checks = check_log(ROOT / f"{STEM}.log")
    generated = json.loads((REV / "generated_figures.json").read_text())
    for name in generated:
        figure_log = (ROOT / name).with_suffix(".log")
        if figure_log.exists():
            check_log(figure_log)

    final_pdf = ROOT / f"{STEM}.pdf"
    pdf_hash = digest(final_pdf)
    tex_log = (ROOT / f"{STEM}.log").read_text(errors="replace")
    pages = re.search(r"Output written on .*?\((\d+) pages", tex_log, re.S)
    aux = (ROOT / f"{STEM}.aux").read_text()
    figure_count = len(re.findall(r"\\contentsline \{figure\}", aux))
    table_count = len(re.findall(r"\\contentsline \{table\}", aux))
    receipt = {
        "built_at_utc": datetime.now(UTC).isoformat(),
        "scope": (
            "Result-ledger validation and reproducible saved-prediction analysis; "
            "no training or sensor inference"
        ),
        "python": sys.version.split()[0],
        "numpy_version": importlib.import_module("numpy").__version__,
        "pdflatex": subprocess.check_output([tex, "--version"], text=True).splitlines()[0],
        "bibtex_executable": Path(bib).name,
        "main_tex": f"{STEM}.tex",
        "pdf": f"{STEM}.pdf",
        "pdf_sha256": pdf_hash,
        "tex_sha256": digest(ROOT / f"{STEM}.tex"),
        "pages": int(pages.group(1)) if pages else None,
        "numbered_figures": figure_count,
        "numbered_tables": table_count,
        "regenerated_figure_sources": generated,
        "source_validation": checks,
        "competition_validation": "revision/competition_validation.json",
        "new_analysis": "revision/derived_transfer_analysis.json",
        "modular_log_checks": modular_checks,
        "delivered_log_checks": log_checks,
        "visual_review": "Requires visual inspection of the exact compiled output.",
    }
    (REV / "build_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in ["pdf", "pages", "numbered_figures", "numbered_tables", "pdf_sha256"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
