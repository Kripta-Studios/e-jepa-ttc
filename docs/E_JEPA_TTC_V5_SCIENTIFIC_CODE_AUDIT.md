# E-JEPA-TTC V5 scientific code audit

Status: in progress. Started 2026-08-13 from branch `scientific-recovery-v5-provenance-dual-transport` at audited V4 commit `cd67613c265a2784a6a11da3c167f706972c159a`.

This audit treats the local checkout and artifacts as operational evidence. The V4 plan and result bundle remain historical inputs; V5 does not rewrite their protocol or results.

## Findings

| Severity | Category | File:Line | Finding | Scientific impact | Evidence | Fix | Test | Status |
|---|---|---|---|---|---|---|---|---|
| P0 | Paired comparison | `scripts/paired_cluster_bootstrap.py:_normalize` | Optional `track_id` was not namespaced before the E-JEPA/Garl merge. | The fresh A6/Garl paired comparison crashed and left the old paired artifact active. | Exact production inputs reproduced `KeyError: 'track_id'`. | Track columns are namespaced symmetrically; external metadata is cross-checked; all inputs are bound by SHA-256. | Track present on both/either/neither; duplicates; mismatched metadata; real 5000-resample reruns. | Fixed; PASS |
| P0 | Claim readiness | `scripts/build_scientific_claim_readiness.py` | Any paired JSON with `checks.exact_sample_tokens=true` was accepted for the current candidate. | A stale A4 paired result marked A6 ready for a sealed test. | The pre-fix reproduction returned `READY_FOR_ONE_SHOT_SEALED_MATCHED_ORACLE_ROI_TEST`; its prediction identity differed from A6. | Readiness now validates signatures, artifact type/status/scope, target and sample contracts, source provenance, candidate/Garl prediction SHAs, and explicit candidate promotion. | Positive case plus candidate, Garl, type, status, scope, and signature failures. | Fixed; PASS |
| P0 | Runner control flow | `scripts/run_scientific_recovery_master_v3.ps1:step73/74` | The output path was assigned before the process succeeded; an existing file survived a failed regeneration. | A failed substep could be consumed as fresh evidence. | V4 launcher log records step73/74 failure while step80 received the old paired path. | The active variable is cleared, prior output is quarantined, and assignment occurs only after exit 0 plus input-SHA validation. Current-candidate paired failure sets the master exit code. | Failure-path regression plus PowerShell parse. | Fixed; PASS |
| P0 | Replication semantics | causal hardening runner/freezer | A6 seeds all initialize from A4 causal seed 7, but the replication call compared seed 13/23 against A4 seed 13/23. | Geometry preservation and mechanistic attribution were evaluated against the wrong parent. | Frozen A6 configs bind `scientific_recovery_a4_causal_left_seed7/model_best.pt`; primary encoder tensors are exactly equal between that parent and A6 seed 7. | The fixed parent and transport-stochasticity semantics are explicit; replication receives one A4 seed-7 base. | Freezer/runner contracts plus regenerated replication artifact. | Fixed; PASS 3/3 |
| P1 | Target-dependent sampling | `src/e_jepa_ttc/data/garlttc_sampling.py:sampling_stratum` | The historical 8192-row cache selection uses an official TTC bucket despite claiming selection without labels. | The fixed public training universe is target-stratified; this limits generalization claims even though it gives neither compared model differential access. | `sampling_stratum()` calls `signed_ttc_bucket(row['ttc'])`; E-JEPA and Garl use the exact same 8192/2048 tokens. | Correct the contract/documentation; ensure V5 fold assignment never reads targets and record the source-universe limitation. Do not rewrite historical artifacts. | Target permutation must not change grouped folds. | Confirmed; scoped fix pending |
| P1 | Comparator qualification | E-JEPA/Garl pipelines | “Matched” does not mean identical preprocessing. | An unqualified matched claim overstates parity. | Exact tokens/targets/budget/ROI privilege/metrics match; E-JEPA uses 3 endpoint tensors with 12 channels and Garl uses 2 endpoints with 40 channels and a different representation/model. | Report sample/target/budget/metric/ROI-privilege matched and preprocessing parity `PARTIAL`. | Explicit parity table and source hashes. | Confirmed; documentation fix pending |
| P1 | End-to-end causality scope | common-ROI preprocessing and model audit | Current dynamic audit proves model-prefix invariance, not a deployable non-oracle streaming pipeline. | Claiming strict end-to-end streaming causality would be unsupported. | `causal_left` passes appended-input tests; the pipeline still depends on oracle ROI preprocessing and fixed materialized windows. | Preserve `model-prefix-causal`; set strict end-to-end status to `NOT CLAIMED`. Add preprocessing/window invariants where meaningful. | Prefix tests at geometry, transport, fusion and ROI/window levels. | Confirmed; tests pending |
| P2 | Geometry freeze evidence | A4/A6 checkpoints | Existing summaries rely primarily on configuration assertions. | Weak provenance could hide accidental geometry updates. | Audit load shows all 27 `encoder.*` tensors exactly equal; tensor-state SHA is `8a579eca0b7371de195e141787d9e05017ae7db0e134ca0d92e216201b408e99`. | Emit parameter/optimizer/state/output evidence for A8 folds. | Before/after fixed-probe audit. | A6 observed PASS; A8 tooling pending |
| P2 | External architecture map | DeepWiki | The requested repository is not indexed. | No scientific impact; local source remains authoritative. | DeepWiki returned repository-not-found. | Record the unavailable auxiliary source and rely on verified local code/history. | N/A | Closed |

## Fresh P0 evidence

- A6 causal seed 7 versus Garl: E-JEPA MiD `205.839927`, Garl MiD `144.887115`, delta `+60.952812`; sequence+track clustered IC95% `[45.446970, 79.909598]`; probability E-JEPA has lower MiD `0.0`; failure delta `+7.275391` percentage points with IC95% `[5.901790, 8.724490]`.
- A5 legacy diagnostic-only versus Garl: delta `+18.323599`; IC95% `[4.528645, 38.064085]`; probability E-JEPA has lower MiD `0.0058`; failure delta `+4.931641` with IC95% `[3.936999, 5.953809]`.
- Both paired artifacts use 2048 exact samples and 108 sequence+track clusters. E-JEPA track IDs are exactly verified against external cluster metadata.
- Corrected A6 mechanistic replication: `PASS`, 3/3 seeds, fixed A4 causal seed-7 parent, MiD mean `208.620253` (sample standard deviation `2.986616`).
- Claim readiness: paired provenance `PASS`, candidate explicitly promotable `false`, `NO_PROMOTABLE_CANDIDATE`, `claims_blocked=true`, `private_test_opened=false`.

## Current contract status

These statuses are provisional until the fixes and regression suite finish.

- data leakage: PASS for the inspected 8192 train / 2048 validation token, sequence, and track intersections; broader repo audit in progress
- target leakage: FAIL for the historical cache sampling claim; no target enters the neural forward
- sequence leakage: PASS for the current public train/validation artifacts
- temporal look-ahead: PASS for model inputs at each materialized endpoint window; strict preprocessing claim remains out of scope
- model prefix causality: PASS for `causal_left` under the existing numerical audit
- strict end-to-end streaming causality: NOT CLAIMED
- oracle ROI contract: PASS, with preprocessing privilege explicitly retained
- event-only contract: PASS for the neural forward
- Garl exact-sample parity: PASS
- Garl metric parity: PASS on the shared prediction population
- Garl preprocessing parity: PARTIAL
- stale artifact safety: PASS for paired/claim master paths; broader artifact audit remains in progress
- checkpoint provenance: PARTIAL pending stronger A8 evidence
- seed semantics: PASS for current A6 mechanistic replication
- public-validation contamination for A8 selection: UNKNOWN until grouped-dev is frozen
- private/test opened: false
