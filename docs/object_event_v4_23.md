# Object Event TTC v4.23 — joint geometry + TTC/LHR fine-tuning

V4.21 established that the object-level box geometry is a valid target: oracle
box-affine divergence reaches Pearson 0.670 on development validation and the
visible-height log ratio reaches 0.760. V4.22 then moved this supervision into
the final geometry-encoder layers and improved frozen dense divergence from
0.250 (v4.19) to 0.309, while vertical-scale reaches 0.299 and remains positive
on all three development sequences. Representation drift is only about 0.5%.

V4.23 is the first integrated downstream fine-tuning stage. It starts from each
v4.22 adapted seed independently and jointly trains:

* the same final 8 geometry-encoder parameter tensors;
* the existing v4.8 `temporal_projection` motion block;
* the existing v4.8 `field_head`.

Everything else remains frozen, including the foreground decoder/refiner. There
is no new sign router, calibration MLP, or post-hoc override.

The objective combines the original v4.8 TTC/LHR objective with the v4.22 dense
geometry objective:

```
L = L_v4.8_TTC/LHR + 0.25 L_dense_geometry + 0.005 L_anchor
```

`L_v4.8_TTC/LHR` contains the existing pooled/dense visible-height log-ratio,
expansion, correlation, sign, confidence, background and TV terms. The geometry
auxiliary contains train-only box pseudoflow, divergence and vertical-scale
supervision. The anchor penalizes drift of every trainable tensor from its v4.22
initial value.

Boxes and visible heights are training targets only and are never forward
features. TTC labels are used only on the train split. Development validation is
evaluated once after a fixed six-epoch schedule and never selects epochs, seeds
or hyperparameters. Official eAP test and EvTTC remain sealed.

V4.23 reports direct event-only TTC predictions as the primary output and keeps
divergence/vertical-scale probes as representation diagnostics. This is meant
to test whether the transferable geometry discovered in v4.22 can regularize a
TTC model, not whether geometry alone can replace TTC prediction.
