# Object Event TTC v4.22 — object-centric geometry auxiliary partial-unfreeze

V4.21 audited the train-only box pseudoflow target before any encoder update. The
oracle affine-box divergence reaches Pearson 0.751 on train and 0.670 on
 development validation, with minimum validation-sequence Pearson 0.587. The
strongest individual object-scale observable is the log visible-height ratio
(Pearson 0.793 train / 0.760 development validation). Therefore the failure of
v4.20 is not a target mismatch: the frozen v4.8 representation/refiner does not
recover enough of the object-specific scale geometry already present in the
annotations.

The eAP release is object-level: a source frame can contain multiple annotation
rows with different `instance_id` values and different TTC values. V4.22 makes
that contract explicit before training. `sequence_id`, `sample_token`, and
`track_id` remain metadata only; one cache row represents one tracked object
through the temporal crop. The preflight refuses adaptation if duplicate object
rows are found or if no multi-object source-frame keys survive in the screen
cache.

V4.22 then adapts the representation, not a post-hoc TTC classifier:

1. load each original v4.8 seed (7, 13, 23) independently;
2. freeze the whole network, then unfreeze only the final 8 parameter tensors of
   `foreground_model.geometry_encoder`;
3. preserve the object-specific common temporal ROI used by Object Event v4;
4. obtain differentiable local-correlation flow from the v4.8 dense maps;
5. supervise forward/reverse flow with train-only affine box pseudoflow;
6. supervise endpoint-antisymmetric divergence;
7. add an explicit vertical log-scale auxiliary loss, because v4.21 and the
   Garl-TTC geometry both identify visible-height change as the most reliable
   TTC-related object geometry;
8. anchor the updated encoder tensors to the original checkpoint to limit drift.

The optimisation objective contains **no TTC label**. Boxes/heights are
train-only supervision and are never forward inputs. Validation boxes are not
used by v4.22 optimisation. Validation is evaluated once after the fixed 8-epoch
schedule. Official eAP test and EvTTC remain sealed.

The experiment reports divergence and vertical-scale probes separately; they
are not naively averaged. A useful outcome is improvement of either transferable
geometry score across all development sequences. A negative result means that
post-hoc geometry has reached its limit and the next model should predict object
height/foreground or motion geometry more directly from the event encoder.
