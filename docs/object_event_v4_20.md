# Object Event TTC v4.20 — train-only box pseudoflow decoder

V4.19 found the first direction representation in this recovery line that
transfers with the same orientation across all three development-validation
sequences: local feature-map divergence. Its score correlation is 0.305 on
train and 0.250 on validation, while the radial-slope statistic is unstable
across sequences.

V4.20 therefore does not add another TTC/sign classifier. It asks a narrower
question: can a tiny shared flow refiner learn, from **training-only boxes**,
to denoise the frozen v4.8 local-correlation flow into an affine object-motion
field whose divergence transfers better than the raw v4.19 divergence?

Training supervision is geometric only. For a point at normalized coordinates
inside the endpoint-1 box, the pseudo correspondence is the same normalized
point inside endpoint-2:

    x2 = cx2 + (w2 / w1) * (x1 - cx1)
    y2 = cy2 + (h2 / h1) * (y1 - cy1)

The reverse direction is trained with the same decoder and the inverse box
mapping. Boxes are never accepted by forward inference. TTC labels are not used
by the decoder loss; they are used only to score OOF/development-validation
correlation after training.

The refiner receives only:
- frozen v4.8 local-correlation flow `(u,v)`;
- local matching confidence;
- event-only foreground overlap;
- three-seed flow disagreement.

It predicts a bounded residual around the raw correspondence. The final scalar
probe is endpoint-antisymmetrised divergence. Radial slope is deliberately
excluded because v4.19 showed a sign reversal on `DGqicHUGWb`.

There is no scientific pass/fail exit code. The result chooses between:
1. decoder improves transferable divergence -> partial-unfreeze v4.8 with
   pseudoflow/divergence auxiliary loss;
2. decoder is neutral -> add the auxiliary loss directly during partial
   unfreeze;
3. decoder degrades transfer -> train the encoder itself with dense flow
   supervision rather than stacking more heads.

Official eAP test and EvTTC remain unopened.
