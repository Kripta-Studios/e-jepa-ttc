# Scientific Recovery V8 status

Branch: `scientific-recovery-v8-temporal-mechanisms`.

V8 has no completed result. No V8 metric, comparison or winner appears in this document.

## Current state

P0 integrity work and the V8 implementation are in progress. The frozen protocol fixes the 8,192-row outer-OOF contract, three folds and the screen seed. The implementation must emit signed local artifacts before the project can evaluate a phase.

The execution order remains P0, A, R, B1/B2, conditional B3/C1, D0-D4 and one confirmation candidate. C1 remains closed until its signed mechanism gate passes. B3 requires the B1 screen gate.

Public validation, private test, EvTTC test and CodaBench remain sealed. No V8 script may use them for selection or reporting.

## Evidence policy

Each run must record its experiment identity, config and dataset hashes, row identity hash, host and library versions, timestamps, status, checkpoint path and SHA-256, metrics path and artifact SHA-256. The report builder accepts signed JSON and signed CSV sources. Missing evidence leaves a phase pending or blocked.

The project does not claim SOTA. A signed aggregate can support only the conclusion its frozen protocol permits.
