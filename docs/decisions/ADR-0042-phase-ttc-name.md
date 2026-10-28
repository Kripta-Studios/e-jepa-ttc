# PHASE-TTC project naming

Date: 2026-10-28

Status: accepted following the research review and owner-authorized rename.

## Decision

Use **PHASE-TTC** as the public project name and `phase-ttc` as the GitHub
repository and Python distribution name. The descriptive subtitle is
**Causal Event–RGB Fusion for Time-to-Contact Estimation**.

“Phase” names the signed motion coordinate used by the temporal context and
residual fusion models. It is a descriptive word, not a newly invented acronym.

## Why the name changes

E-JEPA-TTC accurately records the project's original predictive-representation
hypothesis. The final TRAIN40 system instead combines event and RGB producers,
PAIR readouts, temporal context refiners, and phase-residual fusion. Supervised
geometry and privileged teacher training are also part of its history. The
experiments do not establish that JEPA causes the final improvements. Keeping
JEPA in the primary name would overemphasize a hypothesis rather than describe
the deployed computation.

This is a naming migration, not a new architecture, training run, benchmark
score, or release. The original implementation remains accessible in history.
The final system is recorded on `perf/train40-persistent-50hz`; the earlier
`main` lineage is not replaced by the rename.

## Compatibility and provenance

- Keep the `e_jepa_ttc` Python import namespace, model classes and serialized keys.
- Provide `phase-ttc` and retain `e-jepa-ttc` as console entrypoints to the same CLI.
- Prefer new distribution metadata while supporting old installed metadata.
- Preserve original experiment names, artifacts, signed manifests, source
  snapshots, branch names and local worktree directories.
- Keep archived E-JEPA-TTC manuscript files as historical outputs; publish the
  revised manuscript under PHASE-TTC filenames with a naming note.
- Update the shared repository remote after the GitHub rename. Independent
  clones can update their origin without modifying their scientific branches.

The current scientific limitations remain unchanged: no official SOTA claim,
no established universal JEPA benefit, no guaranteed event contribution across
all cohorts, and no demonstrated sustained 50 Hz deployment.

## Commands

```sh
git clone https://github.com/Kripta-Studios/phase-ttc.git
cd phase-ttc
uv sync --dev
uv run phase-ttc --help
# Historical scripts remain supported:
uv run e-jepa-ttc --help
```

For an existing HTTPS clone:

```sh
git remote set-url origin https://github.com/Kripta-Studios/phase-ttc.git
```

SSH users should retain their existing GitHub host alias while changing only
the repository component to `Kripta-Studios/phase-ttc.git`.
