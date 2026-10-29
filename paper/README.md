# PHASE-TTC: From Motion to Time

**Causal Event–RGB Fusion for Signed Time-to-Contact Estimation**

Álvaro Schwiedop Souto · Evidence snapshot: 28 October 2026

[alvaro.schwiedop@evocon-solutions.com](mailto:alvaro.schwiedop@evocon-solutions.com) ·
[LinkedIn](https://www.linkedin.com/in/%C3%A1lvaro-schwiedop-souto-92718b339/) ·
[Repository](https://github.com/Kripta-Studios/phase-ttc)

## Manuscript and source

`PHASE_TTC_Final.pdf` contains the article and technical supplement.
`main.tex` is the modular LaTeX entry point; `PHASE_TTC_Final.tex` combines the
text, style, tables, and bibliography in one file. Both use the included
`figures/` directory. The source distribution also includes result ledgers,
editable vector diagrams, table generators, and numerical validation scripts.

## Rebuild

Install Python dependencies and a full TeX Live or MiKTeX distribution with
pdfLaTeX, BibTeX, XCharter, newtxmath, TikZ/PGFPlots, standalone, tcolorbox,
booktabs, natbib, and hyperref.

```sh
python -m pip install -r requirements-paper.txt
python scripts/build_final.py
```

The build verifies public evidence hashes, recomputes numerical tables and
figures, and compiles both manuscript forms. Once dependencies are installed,
it needs no network connection. To compile the existing figures and tables:

```sh
python scripts/build_final.py --skip-regenerate
```

The consolidated source also compiles with three pdfLaTeX passes and needs no
BibTeX run. Keep it beside the included `figures/` directory.

## Evidence and scope

The public result registry identifies 99 historical, official-benchmark, and
competition/literature sources. Thirteen public ledgers are hash-verified.
Original training data, raw sensors, and model checkpoints are not included;
regenerating the paper does not rerun training or sensor inference.

The dated GarlTTC Test record reports Overall MiD 35.80 versus 45.03, a 20.49%
relative reduction, with 0% scorer-defined failures over 6,762 scored tokens.
The TRAIN40 submission declares event and RGB input with up to 650 ms of causal
history. This complete-system comparison does not isolate the event contribution
or match the methods' temporal support.

The 468-query labeled EvTTC development cohort and 434-query competition cohort
remain separate. Competition predictions have no verified ground-truth
correspondence; their coverage and disagreement are not accuracy scores.
The published Garl result of 10.60% RTE concerns three sequences and is reported
as literature evidence, not as a locally reproduced score.

R01 recomputes the labeled cohort's error distributions and paired bootstrap
from frozen saved predictions. It resamples 32 complete sequence groups for
10,000 paired replicates with seed 20261028. All mean, median, and p95 difference
intervals include zero. PHASE has a lower empirical p95 than RGB, but more errors
above ten seconds. These post-hoc analyses remain development-exposed.
`revision/derived_transfer_analysis.json` records the source hash and estimands;
`revision/bootstrap_differences.csv` retains every replicate.

R02 records the exact finite phase-output convention and the original paced-run
GPU, software, precision, initialization, and parity settings, with source
digests. It adds reproducibility metadata, not new inference or measurements.
The build verifies this projection before generating the runtime setup table.

## Attribution

Project: [Kripta-Studios/phase-ttc](https://github.com/Kripta-Studios/phase-ttc).
Historical experiment identifiers remain unchanged for traceability.
The FCWD event/RGB illustration and author-supplied logo are documented
in [ASSETS.md](ASSETS.md); third-party material retains its original rights.
`data/visual_source_manifest.json` records the illustration's checksum and
transformation. `SOURCE_MANIFEST.json` records the full SHA-256 digest of every included
file except the manifest itself.
