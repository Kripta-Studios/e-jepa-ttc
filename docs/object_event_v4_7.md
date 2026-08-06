# Object Event TTC v4.7 — high-resolution foreground extent

## Decision from v4.6

V4.6 learned foreground localisation on held-out sequences (`soft IoU = 0.725`)
but its learned per-frame scale correction did not generalise (`height-ratio
Pearson = 0.286`). The learned blend collapsed to approximately zero, preserving
the frozen v4.2 prediction instead of improving it.

The supervision itself is valid: in the recorded v4.6 validation predictions the
official visible-height ratio has Pearson above 0.98 with the TTC-derived
`log(eta)` target. The failure is therefore representation/estimation, not target
misalignment.

## Controlled representation change

V4.7 removes the content-dependent scale head and the TTC fusion head. It keeps
the v4.6 geometry encoder initialisation, decodes a 64x64 foreground mask, and
computes height only from a differentiable top/bottom extent of that mask. A
constant scale cancels in the temporal ratio, so no sequence-dependent absolute
height calibration is needed.

Inference remains event-only. Boxes and visible heights are training targets only.
The experiment is a representation screen: it must first establish a robust
height-ratio signal before any fusion with v4.2 is reconsidered.

## v4.7.1 checkpoint-selection hotfix

The original v4.7 runner ranked overfit checkpoints with sequence-level terms that
are unstable on a balanced 64-sample subset. A later epoch could pass every overfit
gate while `best_observed.pt` still pointed to an earlier checkpoint that failed the
foreground gate. The hotfix now records both `best_observed.pt` and
`best_gate_passing.pt`; whenever at least one epoch passes every mode-specific gate,
the latter is used for final evaluation and eligibility. Overfit checkpoint ranking
also excludes sparse per-sequence terms that are not part of the overfit contract.
