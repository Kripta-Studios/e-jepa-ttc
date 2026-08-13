# E-JEPA-TTC V5 scientific code audit

Status: completed through the A8.0 gate. Started 2026-08-13 from branch
`scientific-recovery-v5-provenance-dual-transport` at audited V4 commit
`cd67613c265a2784a6a11da3c167f706972c159a`. The clean aggregate was generated at
`c55e791c563e6f463385685e8dd3b4aa62d485a7`.

This audit treats the local checkout and artifacts as operational evidence. The V4 plan and result bundle remain historical inputs; V5 does not rewrite their protocol or results.

## Findings

| Severity | Category | File:Line | Finding | Scientific impact | Evidence | Fix | Test | Status |
|---|---|---|---|---|---|---|---|---|
| P0 | Paired comparison | `scripts/paired_cluster_bootstrap.py:_normalize` | Optional `track_id` was not namespaced before the E-JEPA/Garl merge. | The fresh A6/Garl paired comparison crashed and left the old paired artifact active. | Exact production inputs reproduced `KeyError: 'track_id'`. | Track columns are namespaced symmetrically; external metadata is cross-checked; all inputs are bound by SHA-256. | Track present on both/either/neither; duplicates; mismatched metadata; real 5000-resample reruns. | Fixed; PASS |
| P0 | Claim readiness | `scripts/build_scientific_claim_readiness.py` | Any paired JSON with `checks.exact_sample_tokens=true` was accepted for the current candidate. | A stale A4 paired result marked A6 ready for a sealed test. | The pre-fix reproduction returned `READY_FOR_ONE_SHOT_SEALED_MATCHED_ORACLE_ROI_TEST`; its prediction identity differed from A6. | Readiness now validates signatures, artifact type/status/scope, target and sample contracts, source provenance, candidate/Garl prediction SHAs, and explicit candidate promotion. | Positive case plus candidate, Garl, type, status, scope, and signature failures. | Fixed; PASS |
| P0 | Runner control flow | `scripts/run_scientific_recovery_master_v3.ps1:step73/74` | The output path was assigned before the process succeeded; an existing file survived a failed regeneration. | A failed substep could be consumed as fresh evidence. | V4 launcher log records step73/74 failure while step80 received the old paired path. | The active variable is cleared, prior output is quarantined, and assignment occurs only after exit 0 plus input-SHA validation. Current-candidate paired failure sets the master exit code. | Failure-path regression plus PowerShell parse. | Fixed; PASS |
| P0 | Replication semantics | causal hardening runner/freezer | A6 seeds all initialize from A4 causal seed 7, but the replication call compared seed 13/23 against A4 seed 13/23. | Geometry preservation and mechanistic attribution were evaluated against the wrong parent. | Frozen A6 configs bind `scientific_recovery_a4_causal_left_seed7/model_best.pt`; primary encoder tensors are exactly equal between that parent and A6 seed 7. | The fixed parent and transport-stochasticity semantics are explicit; replication receives one A4 seed-7 base. | Freezer/runner contracts plus regenerated replication artifact. | Fixed; PASS 3/3 |
| P1 | Target-dependent sampling | `src/e_jepa_ttc/data/garlttc_sampling.py:sampling_stratum` | The historical 8192-row cache selection uses an official TTC bucket despite claiming selection without labels. | The fixed public training universe is target-stratified; this limits generalization claims even though it gives neither compared model differential access. | `sampling_stratum()` calls `signed_ttc_bucket(row['ttc'])`; E-JEPA and Garl use the exact same 8192/2048 tokens. | The historical artifact remains unchanged. V5 declares the limitation and freezes folds from a target-free metadata table using only sequence/token/track identities. | Target permutation does not change grouped folds; exact cache parity is checked. | Mitigated and explicitly scoped |
| P1 | Comparator qualification | E-JEPA/Garl pipelines | “Matched” does not mean identical preprocessing. | An unqualified matched claim overstates parity. | Exact tokens/targets/budget/ROI privilege/metrics match; E-JEPA uses 3 endpoint tensors with 12 channels and Garl uses 2 endpoints with 40 channels and a different representation/model. | Report sample/target/budget/metric/ROI-privilege matched and preprocessing parity `PARTIAL`. | Explicit parity table and source hashes. | Confirmed; documented |
| P1 | End-to-end causality scope | common-ROI preprocessing and model audit | Current dynamic audit proves model-prefix invariance, not a deployable non-oracle streaming pipeline. | Claiming strict end-to-end streaming causality would be unsupported. | `causal_left` passes appended-input tests; the pipeline still depends on oracle ROI preprocessing and fixed materialized windows. | Preserve `model-prefix-causal`; set strict end-to-end status to `NOT CLAIMED`. | Prefix tests cover geometry, transport, fusion and TTC output. | Model tests PASS; strict streaming NOT CLAIMED |
| P2 | Geometry freeze evidence | A4/A6/A8 checkpoints | Earlier summaries relied primarily on configuration assertions. | Weak provenance could hide accidental geometry updates. | Six fold audits compare every tensor, optimizer membership and fixed-probe outputs. | Emit parameter/optimizer/state/output evidence per child. | Before/after fixed-probe audit. | PASS for A6/A8 in all folds |
| P2 | External architecture map | DeepWiki | The requested repository is not indexed. | No scientific impact; local source remains authoritative. | DeepWiki returned repository-not-found. | Record the unavailable auxiliary source and rely on verified local code/history. | N/A | Closed |
| P0 | Grouped parent exposure | legacy V5 grouped configs/runs | A6 fold children used a global A4 trained on all nine outer-fold sequences. | The apparent fold disjointness did not hold across the A4→A6 chain; MiD 119.50 was not promotion-eligible grouped evidence. | The child loaded 42 tensors and froze 203553 parameters from the exposed parent. | Classify old runs without rewriting them; train one A4 parent per fold and initialize A6/A8 only from it. | Parent fold/token/teacher contracts and SHA checks. | Fixed; old F0 diagnostic-only, old F1 interrupted |
| P1 | Checkpoint selection privilege | `scripts/train_causal_scale_eap_screen.py` | Dev bbox auxiliary losses could enter the checkpoint-selection tuple. | A training-only privilege could influence model selection although bbox was not a neural input. | Selection record included auxiliary dev components. | Select only on sequence-macro MiD then failure; retain bbox diagnostics post-selection. | Selector regression tests. | Fixed; PASS |
| P1 | Launch revision provenance | grouped training scripts | A long run could finish after HEAD changed and report the completion-time revision. | Result→code provenance could bind to code not used at launch. | Reproduced around a guarded run. | Capture launch revision/effective config before training and fail closed on incompatible tracked edits. | Revision-binding tests. | Fixed; PASS |
| P1 | Teacher scope | grouped dataset/trainer | A global DINO cache is permissible storage, but a fold runner needed proof that only fold-train targets were retrieved. | Unchecked lookup could expose outer-dev supervision indirectly. | Cache and fold protocol have exact token identities. | Prove used teacher tokens equal fold-train and are disjoint from fold-dev; dev loads no auxiliary targets. | Token-set and teacher-scope invariants. | Fixed; PASS |
| P2 | Garl runtime environment | local `.venv` | The first Garl F0 launch stopped before training because optional OpenCV was absent. | Operational failure only; no metric or checkpoint was consumed. | `ModuleNotFoundError: cv2`; no completed artifact. | Sync declared `geometry` extra and relaunch the unchanged config from scratch. | Import preflight and signed summaries. | Fixed; all folds completed |

## Fresh P0 evidence

- A6 causal seed 7 versus Garl: E-JEPA MiD `205.839927`, Garl MiD `144.887115`, delta `+60.952812`; sequence+track clustered IC95% `[45.446970, 79.909598]`; probability E-JEPA has lower MiD `0.0`; failure delta `+7.275391` percentage points with IC95% `[5.901790, 8.724490]`.
- A5 legacy diagnostic-only versus Garl: delta `+18.323599`; IC95% `[4.528645, 38.064085]`; probability E-JEPA has lower MiD `0.0058`; failure delta `+4.931641` with IC95% `[3.936999, 5.953809]`.
- Both paired artifacts use 2048 exact samples and 108 sequence+track clusters. E-JEPA track IDs are exactly verified against external cluster metadata.
- Corrected A6 mechanistic replication: `PASS`, 3/3 seeds, fixed A4 causal seed-7 parent, MiD mean `208.620253` (sample standard deviation `2.986616`).
- Claim readiness: paired provenance `PASS`, candidate explicitly promotable `false`, `NO_PROMOTABLE_CANDIDATE`, `claims_blocked=true`, `private_test_opened=false`.

## Frozen V5 grouped-development protocol

Artifact: `configs/protocol/scientific_recovery_v5_train_only_grouped_dev.json`.

- Artifact SHA-256: `f09c688fb4991714abc9d645dda787cb27f1e02a2d1857312ce3e45519bd7a63`.
- File SHA-256: `be48917ae52d1c77d046318bd9ed284a32e8b16258257203fff439332b547874`.
- Source code commit: `f21ffc5422e4e9b0e5b3f0a2f1cba5ad5c96469c`; tracked worktree was clean at freeze time.
- Universe: 8192 unique public-train samples, nine sequences, exact token/sequence/track parity with the materialized cache.
- Fold 0 dev: `5ilM1PX2vz`, `OYgB6RGWcq`, `qGsgzl4Q8B` (2731 rows).
- Fold 1 dev: `2cyv0Oedzg`, `6h5yRW2LGc`, `mHGFBekt7X` (2731 rows).
- Fold 2 dev: `OBneIVg4Cw`, `WbCh1DRerJ`, `t79dBxj1WS` (2730 rows).
- Each sequence is dev exactly once; all folds are sequence-disjoint and exhaustive.
- `public_validation_used_for_selection=false`; `private_test_opened=false`.

## Executed fold-local evidence

The parent-exposed F0 MiD `119.500657` is retained only as an observed diagnostic;
F1 was interrupted. The clean experiment trained three A4 parents, three A6
children, three A8.0 children and three Garl comparators from scratch. All training
was sequential on CUDA with `num_workers=0`.

Nine-sequence outer-dev MiD is A4 `291.087801`, A6 `211.508513`, A8.0
`197.691399`, and Garl `144.353027`. A8.0−A6 is `−13.817114`, clustered IC95%
`[−17.724530, −10.065450]`; A8.0−Garl is `+53.338372`, IC95%
`[44.312629, 61.782673]`. A8.0 fails its absolute 175 gate and is not promoted.
No later A8 arm was executed.

The aggregate artifact SHA is
`b3f0fc484b16f5d503d20deb35b275ddaf7392b2c9484ebc4874b29c8bbb4fc5`.
See [the canonical V5 status](SCIENTIFIC_RECOVERY_V5_STATUS.md) for folds, params,
failure, Pearson, coverage, geometry and paired provenance.

## Final contract status

- data leakage: PASS for the inspected train/grouped-dev/public-validation identity contracts
- target leakage: PASS for neural inputs and fold assignment; historical 8192 target-stratified universe is a declared limitation
- sequence leakage: PASS for clean fold-local chains; old parent-exposed runs are blocked
- temporal look-ahead: PASS at materialized model input and model forward; streaming preprocessing is not claimed
- model prefix causality: PASS for A6 and A8.0
- strict end-to-end streaming causality: NOT CLAIMED
- oracle ROI contract: PASS, preprocessing privilege explicitly retained
- event-only contract: PASS for neural inference; RGB/bbox are training-only supervision
- Garl exact-sample parity: PASS
- Garl metric parity: PASS
- Garl preprocessing parity: PARTIAL
- stale artifact safety: PASS for all V5 critical consumers inspected
- checkpoint provenance: PASS for A4/A6/A8/Garl fold results
- seed semantics: PASS; fold variation is not called multiseed replication
- public-validation contamination for A8 selection: PASS
- private/test opened: false
