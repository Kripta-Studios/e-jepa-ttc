# Object Event TTC v4.13 — conservative dual-head fusion

V4.12.2 proved that the frozen event representation contains direction: it
recovered most receding examples, including 25/28 negatives in DGqicHUGWb.
Using the sign head as a full replacement, however, flipped too many positive
examples and reduced Pearson/MAE.

V4.13 therefore keeps the v4.10 ensemble prediction as the default magnitude and
sign. The odd v4.12 head supplies a continuous residual everywhere. A
sign-changing correction is allowed only for baseline-positive samples with
`p(receding) >= 0.985`. The locked blend is 0.20 normally and 0.51 for those
high-confidence overrides.

These constants were informed by the already-open seed-7 development screen.
Therefore v4.13 is a development ablation, not independent validation. A pass
only authorises a locked multiseed replication; it does not authorise opening
eAP test or EvTTC.
