# Source-to-queue integration (pre-freeze)

`campaign_sources.CampaignSources` connects the real compiled-cache loader to
the registered arm bindings. `train` is the `source_loader` callback consumed by
`queue.run_phase`; `source(spec, "outer_dev")` supplies the separate endpoint
inference population. Both apply the same canonical arm transformation. No raw
events or expert forward passes occur in this adapter.

`scripts/prepare_simplex_t_head_sources.py` verifies the acknowledged role and
ancestry interfaces and loads all three complete compiled D0 folds. It produces
TRAIN and OLD_DEV identities for every possible registered D0 arm, including
TRAIN-only normalizer and control identities. It retains one fold/feature family
at a time, then releases it. This preparation reads no scores, fits no model,
and consumes zero optimizer updates.

The preparation output is explicitly **not scientific freeze**. Preparing T3,
latent and replicate source identities does not pass their practical/technical
gates. D1 is not silently replaced by D0; graphs containing an unintegrated pool
are rejected. Future authoritative D1 availability still needs a separate pool
adapter before the pre-score availability decision.

Expected compiled input layout:

```text
compiled_root/
  outer0/COMPILED.json
  outer1/COMPILED.json
  outer2/COMPILED.json
```

The existing `compiled_context.compile_fold` generates each directory only from
a complete fold. No partial 8192-query source is accepted. Call the preparation
script with explicit `--local-paths`, `--coordination`, `--compiled-root`,
`--index`, `--dedup`, and a new `--output` receipt path. CPU settings are four
threads and two interop threads. The resource admission retains the absolute
RAM/disk limits; no percentage resource threshold is introduced.

Verification in this change: four new adapter tests plus six existing source
loader tests pass, without optimizer updates; Ruff and Pyright pass for the new
production adapter and preparation entry point. These are synthetic wiring
tests, not production replay parity or real-history exact-resume evidence.

Still pending: run compilation and preparation on complete real folds; connect
freeze/QA/practical-gate validation and source identities into the scientific
CLI; connect sealed endpoints to analysis and final packaging; complete actual
history resume QA and the complete freeze before the first scientific fit.

## Sealed endpoint inference connection

`phase_inference.iter_phase_predictions` now connects the complete sealed phase
to the fixed-endpoint loader and cached-head inference. It checks the caller's
prerequisite callback first, then the pinned phase manifest, exact registered fit
set, canonical model configurations, path scope and non-aliasing, checkpoint
hashes, and all completed-update training identities before invoking any OLD_DEV
source loader. The caller supplies pre-frozen OLD_DEV identities; each source is
checked before inference. Only complete per-fit outputs and the actually consumed
history slots are yielded. The caller still owns metadata alignment, per-query
export, phase-wide aggregation, practical gate reporting and publication.

Ten new inference-wiring cases and the four existing sealer tests use explicitly
mocked checkpoints: they are not scientific fits or proof of production endpoint
execution. They cover partial/duplicate seals, altered model/bytes, escaping
paths, gate rejection, source mismatch and resource interruption without partial
fit output. No optimizer work is performed.

The first type check exposed `QuerySource.population` as a mutable protocol
attribute although `CachedQueries.population` is read-only. The protocol now
declares a read-only property, accepting both existing stored counts and computed
properties. This is an interface typing correction, not a model, optimizer,
sampler, loss or checkpoint numerical change. Full actual-history resume QA
remains pending as before.

## Paired factorial analysis

`factorial_analysis.paired_factor_effects` aligns seed7 T2 cells against an
explicit authoritative 8192-query OLD_DEV identity table, including sequence,
track and outer fold. Input row order is irrelevant; missing, duplicate,
reassigned or nonfinite rows are errors rather than complete-case filtering.
The eight-cell graph reports D/H/C and their interactions using the registered
algebra. If D1 is technically unavailable, the four D0 cells report H, C and HxC
only. No D estimate is manufactured. Controls do not become factorial cells.

Seven new algebra/alignment tests plus six existing evaluation tests pass with
synthetic losses and zero optimizer work. Initial pandas-stub type errors were
resolved using explicit DataFrame selection; Ruff and Pyright then pass. The
returned per-query contrasts retain grouping identifiers for subsequent paired
sequence/track uncertainty estimates. This does not yet implement their
confidence intervals, authorize D1 availability, or establish real results.

## Development export connection

`development_export.development_frame` connects complete per-fit inference
outputs with the role-validated metadata order and the current observation's
original expert TTC from the coherent FP32 cache. It checks the frozen source
identity, exact consumed history, historical metadata hash and query order, and
compiled manifest/expert array hashes. Only identity and target CSV columns are
decoded, not historical score columns. It exports query/track/fold identities,
current experts, prediction diagnostics and consumed history indices through
the existing `prediction_frame` implementation, adding source identity, explicit
retrospective-current-ROI semantics, anchor/ROI availability and history span.

Six synthetic integration cases pass, covering unchanged infinite expert
selection and rejection of source, history, metadata order/bytes and expert
cache changes. Ruff and Pyright pass (the initial pandas `usecols` stub mismatch
was resolved with explicit column filtering and a required-column check).
These checks perform no optimizer updates and are not real model results.

## Paired uncertainty connection

`uncertainty_analysis.paired_uncertainty` verifies identical complete nine-sequence
cohorts (query, track, fold and target), then passes all named loss columns together
to the existing historical `hierarchical_losses` implementation. It preserves
the strict OLD metric and shared sequence/whole-track draws, writes the original
draw log and arrays, and reports every named comparison against an explicit
reference. Exact sequence-only intervals, sequence wins and omit-one-score
sensitivity use the existing `exact_sequence_diagnostic`. No TTC values are
averaged and bootstrap fractions are not called posterior probabilities.

The output directory must be new. A resource interruption preserves partial
draw evidence without publishing a complete summary; a retry requires a new
directory. This analysis has no optimizer state and performs zero updates.

Seven tests pass: six plumbing/failure tests plus one execution of the actual
historical 8192-draw engine on synthetic complete-track losses. The latter
verifies output draw files and a known constant paired difference. It is not
scientific uncertainty for SIMPLEX-T. Ruff and Pyright pass. Real endpoint
prediction assembly, practical gates, freeze and final package integration
remain required before scientific completion.

## Combined verification receipt

At implementation commit `021d449`, all 50 tests across campaign sources,
context sources, phase inference/sealing, factorial alignment/evaluation,
development export and uncertainty passed together (zero skipped or failed).
Evidence: `artifacts/simplex_t/T0/qa_source_analysis_integration_021d449.xml`,
SHA256 `c935c60107f57240b04426a6a6724f8327fc5aca4dabb6a446f7879e2299fa4f`.
This combined run performs zero optimizer updates. Historical failure receipts
remain unchanged; this is not baseline-versus-new full-repository QA or the
pending actual-history 10-versus5+5 resume proof.

## Executable fold compilation

The main Python entry point now accepts `prepare --compile-fold 0|1|2` with a
new `--output` directory and the existing `--local-paths` argument. The PowerShell
wrapper forwards `-CompileFold`. It uses the companion's canonical T1 cache,
index and deduplication paths, then calls the existing complete-fold compiler.
Output directories should be named `outer0`, `outer1`, `outer2` under the
`--compiled-root` subsequently supplied to `prepare_simplex_t_head_sources.py`.
Their parent must already exist.

The command rejects `--resume`, missing output and incompatible commands before
reading local paths. It does not pretend an incomplete compilation can resume;
existing partial outputs remain explicit evidence and are not overwritten.
Successful compilation means a complete fold cache, not scientific freeze or
permission to fit. Five CLI wiring tests, Ruff, Pyright and PowerShell syntax
parsing pass. The tests mock the compiler and consume no optimizer updates;
actual complete-fold execution still depends on finishing its cache.
