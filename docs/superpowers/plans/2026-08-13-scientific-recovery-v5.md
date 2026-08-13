# Scientific Recovery V5 execution plan

> Status: historical/completed through the A8.0 gate. The fold-local parent
> correction and observed results are in
> [Scientific Recovery V5](../../SCIENTIFIC_RECOVERY_V5_STATUS.md).

## Scope and hard walls

- Keep `private/test`, CodaBench, and EvTTC test closed.
- Use only the nine public training sequences for A8 model selection.
- Keep the neural forward event-only. Oracle boxes remain preprocessing/training supervision, never transport inputs.
- Require `causal_left`; do not claim strict end-to-end streaming causality while oracle/common-ROI preprocessing remains outside the model-prefix audit.
- Run one CUDA job at a time with `num_workers=0`; monitor JSON/logs, never checkpoints.

## Requirement map

| Requirement | Code or artifact | Verification | Scientific gate |
|---|---|---|---|
| Namespaced optional `track_id` | `scripts/paired_cluster_bootstrap.py` | symmetric track/no-track unit cases plus real rerun | exact-sample paired comparison |
| Cryptographic paired provenance | paired bootstrap source contract | SHA unit tests and real input hashes | stale-result rejection |
| Fail-closed claim readiness | `scripts/build_scientific_claim_readiness.py` | matching/mismatching candidate, Garl, type, status, scope tests | no false claim readiness |
| Fresh paired outputs only | `scripts/run_scientific_recovery_master_v3.ps1` | failure-path regression plus PowerShell parse | master fail-closed behavior |
| Fixed A4 seed-7 parent for current A6 mechanism replication | causal freezer, runner, replication summary | frozen config/replication tests and checkpoint hashes | valid geometry attribution |
| Train-only grouped development | new V5 protocol builder and dataset view | determinism, exhaustiveness, uniqueness, sequence disjointness, hashes | A8 selection isolation |
| Frozen A8 geometry | A8 runner/checkpoint audit | parameter membership, tensor SHA/equality, fixed-probe output equality | geometry preservation |
| Prefix causality | preprocessing/model/branch/fusion tests and signed audit | mutate appended future inputs | causal promotion |
| A8.0 dual transport | A7-derived causal config and grouped folds | fold summaries, provenance, finite metrics | MiD <= 175 first-stage promotion |
| Conditional A8.x | one preregistered change at a time | same folds/budget, causality, geometry, cost | promote only after prior evidence |
| Documentation consistency | repository Markdown/docs sweep | global stale-reference search | V5 handoff completeness |

## Execution order

1. Audit source snapshot, Git/remotes, documentation, data/split/target/metric/ROI/causality/config/checkpoint/artifact paths.
2. Reproduce P0 failures and establish the targeted test baseline.
3. Add failing P0 regression tests, implement minimal fixes, and run the required validation commands.
4. Recompute A6-vs-Garl and A5 diagnostic-only paired artifacts without retraining; rebuild claim readiness.
5. Commit and push the coherent P0 integrity change after explicit-path staging.
6. Build, test, sign, and freeze the 3-fold train-only grouped-development protocol before observing A8 results.
7. Preregister and run A8.0 sequentially on the frozen folds; verify geometry, causality, provenance, and gates.
8. Research and execute only the next ablation justified by A8.0. Stop the chain on a failed gate unless a separate diagnostic is explicitly scoped.
9. Complete the audit/status reports and documentation sweep; run targeted and full feasible verification, then commit/push and inspect CI.
