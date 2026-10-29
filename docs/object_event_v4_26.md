# Object Event v4.26 — leak-free OOF geometry residual stack

V4.25 cross-fitted the geometry features but selected them against the v4.10 `ensemble_train_predictions.csv` anchor, whose train predictions were produced by models trained on those same train sequences. That made the meta-CV asymmetric: geometry was OOF while the anchor was in-sample. Because the in-sample v4.10 anchor is nearly perfect, the meta-readout was biased toward `baseline_control`.

V4.26 repairs that protocol without opening official eAP test or EvTTC. It rebuilds the v4.24 `geometry_only_regularized` family in 3 seeds x 3 grouped folds and uses the **same held-sequence-out models** to obtain all three first-level features: TTC anchor, divergence and vertical scale. A second grouped meta-CV then learns only a non-negative geometry **residual** while fixing the anchor coefficient to exactly 1.0.

The final stack is standard cross-fitted stacking: residual calibration and coefficients are fit from first-level OOF train predictions, then applied once to full-train v4.24 champions on development validation. V4.10 remains an external validation benchmark only; its in-sample train predictions cannot participate in selection.

V4.26 additionally reports metrics per object `track_id`, including macro track Pearson and the worst negative-track sign accuracy. This is required because v4.25 exposed a failure mode concentrated in a single negative object trajectory.

This is the last post-hoc readout experiment. If the leak-free OOF stack is not selected or does not beat the v4.10 development anchor convincingly, the next architecture is an explicit event-native LHR head trained inside the model rather than another post-hoc coefficient sweep.
