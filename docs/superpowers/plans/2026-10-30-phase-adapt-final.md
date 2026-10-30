# PHASE-ADAPT final refit implementation plan

> Execute inline with superpowers:executing-plans; user explicitly requests continuous execution through a trained final model.

**Goal:** Train and deliver a verified final PHASE-ADAPT F1 model from all compatible local training data.
**Spec:** User continuation on 2026-10-30 plus package chapter 07. The user now authorizes actual refitting despite unresolved independent-evaluation gates.
**Architecture:** Freeze eight upstream models and normalizers; initialize the 1,955-parameter PhaseOnlyFinal head from the sealed candidate. Reuse verified TRAIN40 feature cache and extract all eligible EvTTC-32 annotated sensor anchors with the existing causal source adapter. Local labels train a declared exposed model; no independent-test or official-score claim.
**Stack:** Existing Python 3.11, PyTorch CUDA, NumPy, HDF5, TorchScript.

## Constraints and choices made before fitting

- Preserve sealed candidate, historical receipts, original audit policy and exclusions.
- A separate explicit `all_compatible_local_refit` role authorizes training, not clean transfer certification.
- Use every supported annotated anchor across all 32 source recordings, not the historical 16-anchor subset. Competition extracts are not added again; Slider has no verified compatible supervised adapter/target contract.
- Read CSV timestamp column 1 and TTC column 4; bounded linear interpolation, maximum gap 20 ms, no extrapolation. Train only within the head's signed phase domain, with all excluded rows counted.
- TRAIN40 cache retains its original targets and lineage, verified against archived receipt. Public final-test caches remain excluded.
- F1 retains original architecture, normalizers and fusion loss. Equal domain mass, then equal recording mass within each domain; recording balance is not a claim of independent acquisitions.
- AdamW LR 1e-4, weight decay 1e-4, batch128, clip1.0; complete epochs up to30 and at most2000 updates. Seeds7,13,21. Final artifact is seed7 by declaration, no best-seed or test selection. Last checkpoint, no early stopping on exposed metrics.
- Report baseline and trained resubstitution metrics, coverage and loss curves. Do not present training-fit changes as generalization.
- Export the head to TorchScript; ship upstream components and a strict runtime loader; validate saved/reloaded predictions and raw-sensor inference.

## Task 1: all-data population and immutable feature cache

- Write tests for target boundaries, duplicate identity, scope, domain/group weighting, and complete-anchor enumeration; observe failure.
- Add optional all-anchor selection to existing builder with unchanged default.
- Implement separate refit population/cache CLI; bind sources, input data identities, original candidate, source-cache receipts. Resume per-sequence shards only with matching binding.
- Enumerate, extract and verify every eligible row; preserve reason-coded missingness. Commit after focused tests.

## Task 2: train, serialize, export and replay

- Write tests for real parameter updates, deterministic fixed-budget training, immutable upstream weights, save/load parity and tamper rejection.
- Train fixed recipe on all accepted rows for seeds7/13/21; save optimizer/history/provenance/checkpoints.
- Package seed7 plus upstream weights/normalizers, model card, reproduction commands and metrics. Verify CPU export and raw input runtime.
- Commit code after tests; review once with independent reviewer, fix material findings with regression tests.

## Review focus

- Cache resume cannot mix candidate/source/target identities.
- Label fields cannot enter upstream inference; omitted rows remain visible.
- Every accepted row is visited each complete epoch; domain/recording mass has intended totals.
- Final model selection is fixed before scores; no checkpoint silently substituted.
- Deployment package verifies exact parent graph and new head, preserving original seal separately.

## Execution ledger

- Start: branch research/evttc-transfer-v2 at65edf4ee; existing linked worktree. User opts out of stopping for research gates. These remain limits on claims, not on an explicitly exposed final refit.
- Pre-flight interfaces: source sensor adapter ->340D head inputs; archived TRAIN40 cache ->same schema. Both require original parent graph. New targets are joined only after sensor feature extraction.
- Task1 implemented: all3828 sensor-eligible anchors in32recordings enumerated;3571 have supported local targets,257 do not. Verified archived TRAIN40 cache contains88744 OOF examples/40groups.
- Ruling: source execution below initially extracted sensor states before label lookup. After observing unsupported targets consuming GPU, v2 checks supervised eligibility first, but removes label paths/values from upstream inference. No predictions or errors are consulted for eligibility. All3828 rows remain in the population and target ledger. Cost if wrong: fewer training examples; reason-coded exclusions remain auditable.
- First extraction retained as incomplete evidence under output/phase_adapt_final/data; stopped deliberately after~300queries to implement reviewer resume fixes and eligibility prefilter. New output data_v2 explicitly migrates hash-verified shards; candidate, population and all scientific source files must match except extraction orchestration. Original binding/receipts are never rewritten.
- Independent review: no Critical; four Important findings. Deterministic cuBLAS config and seed/data/parent graph cross-binding corrected. Resume now allows unrelated added sources while rejecting modified executed files. Orphan NPZ is recovered only when fresh arrays match exactly; receipt writes are atomic.
- Tests: original10new tests passed; review seed/parent/config tests reproduced three failures against the pre-fix commit in isolated modules, then passed on current code. Resume/orphan tests first failed on absent implementation; final focused regression35passed in13.04s. Ruff and targeted Pyright pass. Real deterministic CUDA forward/backward passed with CUBLAS_WORKSPACE_CONFIG=:4096:8.
- Reviewer declined independent-generalization, official-equivalence, actual final improvement, all32recording contracts, historical OOF construction and pending delivery artifacts. Ruling: retain first two as unverified; exact population/fit results and final artifacts will be measured during execution; OOF receipt is preserved without asserting new historical proof. No secondary review requested.
- Performance probes during extraction: a native query profile and isolated RGB batching probe were retained under output/phase_adapt_profile.log and output/phase_adapt_batch_probe.log. Batched head inputs differed by at most7.99e-6 in the single tested query. Ruling: do not adopt this new execution path during the campaign; one concurrent timing probe is insufficient to establish a stable speed gain, and existing shards retain the original arithmetic path.
- Subsequent user steering explicitly requested bottleneck analysis and CUDA graphs, cache, prefill/preload and compression. The serial v2 extractor and waiting trainer were stopped deliberately at about1461/3828 population rows. Existing shards remain immutable. The earlier decision against RGB batching remains: native batch shapes are retained.
- Optimization: reuse the existing PersistentSensorInputs bounded raw-event/frame caches and four RGB crop workers, plus ProducerRuntime CUDA graphs. Graphs warm up and validate exact producer-core output before replay. Keep the original context/head forward so all340 features and RGB width logits remain available through the original hook. This is execution-only adaptation, with unchanged weights, clocks, ROI and targets. No lossy compression, quantization, mixed precision or context truncation.
- Admission: scripts/benchmark_phase_adapt_runtime.py compared36 raw queries spanning all32 source recordings against the original runtime. Every full fusion state, width logits and three TTC predictions matched exactly. Receipt: output/phase_adapt_final/runtime_parity.json. A preliminary isolated seven-query run also matched exactly:4.556s cold capture, then0.213–0.745s per query. The broader parity run overlapped CPU tests for part of its duration; its86.68s original/41.13s cached totals are not a clean throughput benchmark. Actual extraction throughput will be reported separately.
- Compression probe: a representative1738-byte NPZ takes approximately0.119ms to compress in memory over100 repetitions; compression is retained. Training already preloads all feature/target tensors onto the GPU once per seed. Explicit speculative disk prefetch is not required to obtain the measured gain; cache reuse and graph warmup address the observed dominant costs.
- Verification:81 focused tests passed and1 skipped; Ruff clean; targeted Pyright with the actual .venv-transfer environment reports0 errors. The initial Pyright invocation used the repository's broken .venv configuration and reported missing torch attributes; rerunning with a task-local environment configuration resolved it.
- data_v3 explicitly migrates verified v2 shards, binds the new runtime and parity receipt, and records per-run graph/cache diagnostics. MIGRATION.json's inherited reason describes unchanged sensor forward semantics; RUNTIME_ADMISSION.json records the changed execution mechanism. The extraction command continues directly into seeds7/13/21 training and report generation only after COMPLETE.json is written successfully.
