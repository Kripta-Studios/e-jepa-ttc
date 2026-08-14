# Scientific Recovery V8 report

Status at repository inspection: implementation in progress, zero V8 results.

The generated report lives under `artifacts/scientific_recovery_v8/report/`. Run:

```powershell
uv run --no-sync python scripts/build_scientific_recovery_v8_report.py
```

The builder reconstructs tables from signed JSON artifacts and CSV files whose SHA-256 a signed artifact declares. It records invalid signatures, digest mismatches, invalid CSV headers, failed runs and negative results. Missing evidence keeps the report blocked.

The evidence package excludes datasets, caches and checkpoint bytes. It includes a checkpoint manifest with the route and SHA-256 for each V8 checkpoint.

```powershell
uv run --no-sync python scripts/package_scientific_recovery_v8_evidence.py
```

No V8 result exists at this point. Public validation, private test, EvTTC test and CodaBench remain sealed. C1 remains closed until its mechanism gate passes. The report prohibits SOTA claims.
