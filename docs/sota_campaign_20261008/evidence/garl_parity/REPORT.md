# Garl native parity

Status: **PASSED**. Samples: 3 across 3 eAP sequences. Maximum preprocessing error: `0.0`.

The event-only and full adapters use the published height order `(t0,t1)`, `dT=0.1 s`, and the unclipped signed equation `TTC=dT/(1-height_t0/height_t1)`. The published timevolume ignores polarity.

EvTTC uses exact microsecond windows and explicit cross-camera calibration. These are declared transfer adaptations; the audit does not select eAP's `x+5` or millisecond flooring based on EvTTC errors.
