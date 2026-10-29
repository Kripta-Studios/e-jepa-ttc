# Stage66–69 prospective integration

This continuation starts from 5a877f6e7560f3b537ffa1e61298a2a11d8d5dd8 in a new clean-v2 worktree. All scientific constants are the byte-identical handoff PROTOCOL.json. Historical producer and training files are unchanged. No Stage64/65 fits are performed.

The public runner is `scripts/run_scientific_recovery_v9_stage66_69.py`, invoked synchronously by `scripts/RUN_STAGE66_69.ps1`. It accepts handoff-root, local-inputs, output-root, mode preflight/full/analyze/package and explicit resume. The geometry helper routes through that same gated owner; analysis and packaging never fit. Artifacts are under `artifacts/stage66_69_clean_v2`.

Production responsibilities follow document09. Normal-flow kernels reside in `models/normal_flow_routing_v10.py` to preserve the existing geometry package. Exact statistics and numerical decisions have separate versioned modules. The owner/physical-receipt implementation is in `artifacts/risk_geometry_v10.py`; essential ZIP verification is in `artifacts/essential_bundle_v10.py`.

The CPU float64 two-output head, zero final layer, AdamW, schedule and all masks are fixed. Model initialization uses torch.manual_seed(seed); the independent repeated-permutation schedule uses 10*seed+outer_fold. All four arms share the same unmodified inner-OOF scaler for a fold/seed. Complete model, optimizer, RNG, schedule, scaler, identities and loss histories are checkpointed every100 updates. Only verified role=inner_oof enters fitting. Every phase freezes the complete prescribed endpoint set before opening its outer inference path. Decisions require physically verified diagnostic coverage.

The newer user exposure policy classifies the previous garl_ttc key-only search as LEVEL0; its original incident receipt remains intact, with a separate current classification. It cannot cancel Stage67. Protected validation/test submissions remain outside this campaign. An operational failure never advances the scientific decision tree.

Historical diagnostics are POSTHOC_NONSELECTABLE. Their consistent PAIR permutation is frozen-weight usage analysis, not a refit or mechanistic acceptance. New paired uncertainty uses identical hierarchical draws to the repository implementation and exact weighted sequence resampling. Replication averages losses only; seed7 remains the deployment head.

QA separates the handoff's reference tests from integrated production tests, including numerical/gradient parity, 10 versus5+5 synthetic and real inner-OOF smoke, corrupt receipts, role forgery, control semantics, raw timestamp/geometry cases, ownership and bundle extraction. Full-suite baseline failure IDs are compared explicitly; inherited unrelated missing historical fixtures do not waive new failures. Type checking applies to all new production modules and scripts with the existing Python environment's actual site-package paths.

Final packaging includes compact inputs, tiny endpoints, safe numerical inference arrays, row-level results, all uncertainty draws, diagnostic coverage and local resume index. It excludes raw streams, intermediate optimizer bytes (indexed), historical encoder weights and nested ZIPs. Every included member is checked after extraction, followed by CRC and outer SHA256 verification. Local analysis commits follow training; no push or PR.
