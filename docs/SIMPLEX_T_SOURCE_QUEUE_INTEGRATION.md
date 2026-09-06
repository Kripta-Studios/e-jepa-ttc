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
