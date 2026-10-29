# ADR-0044: Preserve published and runtime-bound orchestration source

Date: 2026-10-29

Status: accepted

## Context

Six RunPod orchestration wrappers from 27 October 2026 remain part of the
published research history. Their authorization windows have expired. Nineteen
additional queue, watchdog, RGB-context, migration, and upload sources are bound
by runtime receipts and caller hashes in the integrated research record. These
files contain compact operational syntax that predates the repository's current
Ruff format and annotation rules.

## Decision

Preserve the six expired wrappers byte-for-byte against published commit
`2ad7a05797799caf1bcdc9202c307d244f9fe31d`. Preserve the nineteen additional
runtime-bound sources against integration commit
`2a047bf1afc5c584e0cf6e3546dfe7bbb5883541`. Parametrized regression tests bind
every path to its SHA-256 digest. Ruff continues to lint all files, with per-file
ignores limited to diagnostics present in the bound bytes. Ruff formatting
excludes exactly these twenty-five paths.

The exception does not apply to successor scripts or other historical source.
The expired wrappers retain their existing deadline, receipt, lineage, and
expiry tests. The additional files are runtime-bound sources; they are not all
expired authorization wrappers.

## Consequences

CI detects any byte change while continuing to report diagnostic classes not
covered by the narrow exceptions, including all F, B, UP, and N findings. The
preserved source records published or runtime-bound program identity. Its
presence, byte identity, and receipt relationships are not proof that a pod
executed any particular file.
