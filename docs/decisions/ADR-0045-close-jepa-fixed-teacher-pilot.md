# ADR-0045: Close the fixed-teacher JEPA pilot after the failed benefit gate

Date: 2026-10-31

Status: accepted by the user

## Context

The bounded three-seed pilot completed all nine fixed-length training arms and
its terminal evaluation. Future teacher endpoint prediction improved, but the
preregistered frozen-encoder expansion benefit gate failed against the static
and shuffled-future controls. A bounded post hoc analysis of all eight held-out
sequences found heterogeneous effects without a validated reopening mechanism.

## Decision

Close this campaign and preserve its negative evidence. Do not start automatic
rollout training, scaling, or integration into the published PHASE-TTC model.
Preserve final checkpoints, source identities, protocol, logs and predictions
in a verified archive; retain original data and intermediate artifacts.

Reopening requires a concrete new hypothesis with supporting evidence, fixed
controls and endpoints, an explicit compute budget, and a distinction between
explored data and independent confirmation. An untested variant alone is not
sufficient. This decision does not claim that JEPA is universally ineffective
or that RoboJEPA's autoregressive planning recipe has been tested here.

## Evidence

See the [closure report](../jepa_closure_20261031/README.md),
[generated results](../jepa_closure_20261031/closure_analysis.json), and
[archive receipt](../jepa_closure_20261031/evidence/CLOSURE.json).

The current model and published results are unchanged. No additional training
was launched during closure; unrelated pod workloads were left intact.
