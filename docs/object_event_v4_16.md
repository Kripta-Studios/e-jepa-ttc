# Object Event TTC v4.16 — causal temporal dual head

## Why v4.16 exists

V4.15.2 repaired the systematic receding-motion failure without destabilising
TTC magnitude: validation negative accuracy rose from 0.639 to 0.716 and the
worst sequence from 0.107 to 0.429, while RTE and saturation remained close to
the frozen v4.10 baseline.  It still missed the minimum-sequence Pearson gate by
0.0033.  Inspection showed that the true repaired negatives form a persistent
run on one track while many false flips are isolated across positive tracks.

V4.16 therefore ends the post-hoc threshold/override line.  It learns a causal
temporal model that emits sign and magnitude jointly from event features.

## Architecture

1. Freeze the three true-seed v4.8 backbones (7, 13, 23).
2. Reuse the exact-odd v4.12 descriptor construction and the robust mean/median
   multibackbone consensus from v4.15.
3. Sort samples chronologically inside `(sequence_id, track_id)` and build a
   right-aligned causal history of 12 samples. IDs are grouping metadata only.
4. The sign branch applies a bias-free odd MLP to every history element and a
   learned global causal recency kernel. Because the kernel is input-independent,
   negating every odd descriptor negates the final sign logit exactly.
5. The magnitude branch consumes only absolute descriptors plus a positive
   anchor derived from the frozen v4.8 backbones. It predicts a bounded
   multiplicative residual, so magnitude is exactly even under descriptor sign
   reversal and strictly positive.
6. Final expansion is `sign * magnitude`; there is no opposite-sign blending,
   probability threshold sweep, or validation-tuned override rule.

The v4.10 ensemble magnitude is used only as a train-split distillation target.
It is never a forward input and validation inference is event-only.

## Scientific protocol

The first v4.16 experiment is intentionally a short screen. Three grouped folds
on the nine train sequences measure train-side generalisation under a fixed 24-epoch
head schedule. The final head uses the same preregistered 24 epochs. Validation is
evaluated once after training. The official eAP test and EvTTC
remain closed. If this screen succeeds, the next experiment is a true-seed
replication followed by selective unfreezing of the last v4.8 temporal blocks;
only then is a long training justified.

## v4.16.1 overfit checkpoint-selection hotfix

The initial v4.16 run reached 0.96875 balanced sign accuracy on the balanced
96-window memorisation check but evaluated only the final epoch. v4.16.1 keeps
the architecture, losses, learning rate, temporal window, fold schedule and
screen gates unchanged. It evaluates the same train-only overfit subset after
every epoch, stops at the first gate-passing checkpoint, and otherwise records
the best observed checkpoint under the unchanged gates. The overfit-only budget
is increased from 100 to 200 epochs; grouped OOF and final screen schedules are
unchanged.


## v4.16.2 single-importance-weight correction

The v4.16.1 screen exposed a training-contract bug: sequence/sign weights were
used both to sample rows and again inside the loss, effectively squaring rare
cell leverage. v4.16.2 traverses every selected row once per epoch in a random
order and applies the sequence/sign importance weight only in the loss. Model
architecture, causal window length, losses, learning rate and scientific gates
remain unchanged. The screen also reports two diagnostic hybrid metrics
(temporal sign with frozen baseline magnitude, and frozen baseline sign with
temporal magnitude) to separate sign from magnitude generalisation failure;
these diagnostics are never used for selection or gates.
