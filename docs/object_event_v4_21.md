# Object Event TTC v4.21 — oracle audit of box-affine pseudoflow supervision

V4.19 found a transferable signal in frozen dense correspondence: divergence
correlated 0.305 on train and 0.250 on development validation. V4.20 trained a
small refiner against affine box pseudoflow. The refiner improved OOF slightly
(0.318) but degraded development-validation divergence to 0.183.

Before allowing this loss to update the encoder, v4.21 audits the supervision
target itself. This is necessary because a bounding-box resize is only a proxy
for physical image flow: visibility changes, truncation, lateral motion,
rotation and annotation jitter can change box size without matching TTC.

No model is trained. For train and development validation independently, the
script constructs the exact forward/reverse affine pseudoflow target used by
v4.20 and measures its endpoint-antisymmetrised divergence against the TTC
expansion target. It also audits simple width/height/geometric box-scale proxies,
anisotropy and normalized centre translation, globally and per sequence.

Validation boxes are used only as an oracle diagnostic to test whether the
training target itself transfers; they are never a model input or optimisation
target. Orientation is fitted on train only and then applied unchanged to
validation.

The output makes a representation decision, not a gate:

- `box_pseudoflow_supervision_supported_proceed_partial_unfreeze`:
  target geometry transfers, so v4.22 may update encoder layers with it;
- `box_target_sequence_shift_stop_box_supervision_use_event_native_flow`:
  target works on train but changes relationship on new sequences;
- `box_target_insufficient_stop_box_supervision_use_event_native_flow`:
  box geometry is not a sufficiently faithful flow target even on train.

Official eAP test and EvTTC remain unopened.
