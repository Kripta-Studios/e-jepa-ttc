# Object Event TTC v4.8 — dense foreground temporal field

## Motivation

V4.7 learned a useful high-resolution foreground mask, but a mask with soft IoU
around 0.72 was not accurate enough to recover the very small frame-to-frame
height changes required by TTC. The overfit passed, while the held-out screen
stayed near Pearson 0.34. Therefore v4.8 does not derive TTC from mask edges.

## Hypothesis

The foreground network is frozen. A new head receives only encoded temporal
differences and event-activity differences and predicts a dense `log(eta)` field.
The final scalar is a confidence-weighted average inside the predicted
foreground. During training, boxes define only the foreground region and visible
heights define only the target log-scale. Neither is accepted by `forward`.

## Falsification order

1. Overfit 64 balanced examples.
2. If all overfit gates pass, run one held-out seed-7 screen.
3. Do not fuse with v4.2 unless the dense field reaches held-out Pearson >= 0.50,
   balanced sign >= 0.68 and preserves event dependence.
4. Keep the official eAP test and EvTTC sealed.
