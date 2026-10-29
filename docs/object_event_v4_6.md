# Object Event TTC v4.6 — learned foreground height ratio

## Decision from v4.5

V4.5 improved ensemble Pearson, expansion MAE and the fragile-sequence negative
recall, but it reduced seed agreement and improved weighted MiD by only 0.38%.
Seeds 13 and 23 selected epoch zero.  The result rejects further loss-only tuning
as the main research direction.

## Controlled representation change

V4.6 keeps the validated v4.2 event-only predictor frozen and adds a separate
trainable event encoder with:

1. a learned foreground probability map at each causal endpoint;
2. a differentiable vertical-extent estimate;
3. a per-frame log-height correction predicted from foreground-pooled features;
4. a height-ratio estimate `log_eta_h = log(h_t1) - log(h_t2)`; and
5. a bounded learned interpolation with the frozen v4.2 `log_eta` prediction.

Official boxes and visible heights are supervision-only.  They are never accepted
by `forward`; inference remains strictly event-only.  The crop and split are
unchanged.

## Execution contract

The wrapper first runs a balanced 64-sample overfit.  Full held-out training is
launched only when the model can fit foreground, height ratio, expansion and sign.
The full screen starts again from the original seed-7 v4.2 checkpoint rather than
from the overfit model.

V4.6 advances to multiseed only if it improves weighted MiD by at least 5% over
the matching frozen baseline, preserves Pearson/sign, and materially repairs the
eligible worst-sequence negative accuracy.
