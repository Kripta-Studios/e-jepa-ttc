# Object Event TTC v4.12 — reversal-balanced directional sign probe

## Why this experiment exists

V4.10.5 is stable in magnitude and rank across seeds, but the three experts make
the same sign error on a short negative track in `DGqicHUGWb`. V4.11 trained a
router from final expert outputs and improved the global negative accuracy only
slightly. Its grouped-CV training predictions already had almost perfect sign,
so the router had no representative false-positive examples from which to learn.

V4.12 moves the sign decision upstream. It freezes the validated v4.8 motion and
foreground networks, extracts signed temporal feature maps, and trains a small
binary direction head. The magnitude remains the fixed v4.10 ensemble magnitude.

## Reversal-balanced supervision

For every event tensor `(t0,t1,t2)` the probe also sees `(t2,t1,t0)`. The reversed
sample receives the opposite direction label. The loss combines:

- class-balanced BCE on the original sample;
- class-balanced BCE on the reversed sample;
- an antisymmetry penalty requiring opposite logits;
- a small signed margin.

This creates hard directional examples without touching validation labels.

## Inputs and scientific contract

At inference the sign probe receives only the event tensor. It does not receive
boxes, visible heights, sequence IDs, track IDs, RGB, official eAP test labels,
or EvTTC labels. Boxes and heights remain part of earlier training stages only.
The screen is development evidence, not an official eAP result.

## Decision rule

The probe supplies the sign and v4.10 supplies the absolute magnitude. V4.12 is
accepted only if it improves negative direction while preserving Pearson and
MAE, passes temporal-reversal accuracy, and remains event-dependent.


## v4.12.1 exact odd-symmetry hotfix

The first v4.12 overfit reached 0.984 balanced sign and 1.0 negative accuracy on
the original windows, but only 0.609 reverse-sign accuracy.  The directional
signal was therefore present; the failure came from asking a generic GELU/Dropout
MLP to learn time-reversal antisymmetry as a soft regularizer.

v4.12.1 antisymmetrises the frozen descriptor directly,
`d_odd(x)=0.5*(d_raw(x)-d_raw(Rx))`, and replaces the head with a bias-free odd
MLP (non-affine LayerNorm, Linear without bias, tanh).  Consequently
`logit(Rx)=-logit(x)` by construction.  The implementation deliberately makes
no assumption about polarity or temporal-bin ordering within the 12 event
channels.  All screen gates remain unchanged.


## v4.12.2 screen checkpoint-gate hotfix

The v4.12.1 overfit passed exact reversal symmetry, but the first screen epoch
raised `KeyError: zero_event_pearson_drop`. Epoch-level checkpoint selection had
called the final screen gate set before the selected-checkpoint zero-event and
shuffled-event evaluations existed. V4.12.2 separates the contracts:

- checkpoint selection uses the core predictive/sign gates available each epoch;
- the selected checkpoint is still evaluated with the full event-dependence gates;
- a passed overfit is reused when resuming an incomplete screen.

No scientific threshold is relaxed and no validation label is used to fit the
event-dependence checks.
