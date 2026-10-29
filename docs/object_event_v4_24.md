# Object Event TTC v4.24 — train-only schedule orchestrator

V4.23 improved the transferable geometry learned in v4.22 but made the direct
TTC readout strongly positive-biased on development validation. The geometry
therefore remains useful; the unresolved question is which trainable block and
loss schedule causes the sign-generalisation regression.

V4.24 replaces one-short-run-per-version with a successive-halving experiment.
Five scientifically distinct schedules are tested:

1. the exact v4.23 schedule as a control;
2. geometry-tail-only fine-tuning with the existing motion/TTC readout frozen;
3. motion/TTC-head-only fine-tuning with geometry frozen;
4. conservative joint fine-tuning;
5. geometry-heavy joint fine-tuning.

Arm selection never uses the development validation split. Stage 1 evaluates
all five arms with seed 13 in three sequence-grouped train folds. The best three
continue to stage 2 and are confirmed with independent seeds 7 and 23 on the
same grouped-train protocol. A single champion is selected from multiseed OOF
predictions. Only then is the champion trained on the full train split from the
three independent v4.22 checkpoints and evaluated once on development
validation.

The objective rewards Pearson, balanced sign accuracy, worst-sequence Pearson,
worst-sequence negative accuracy and retained dense geometry, while penalising
negative-class prior drift. Hard train-only eligibility criteria are reported
but do not cause an operational failure; if no arm satisfies every criterion,
the best available OOF candidate is still run as an explicitly labelled
fallback.

Official eAP test and EvTTC remain sealed. Boxes and visible heights remain
train-only targets and never become forward features.
