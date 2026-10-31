# Causal event-expansion expert research plan

User authorizes continued autonomous research toward comparable EvTTC SOTA.
This follows two rejected feature-fusion campaigns, not evidence of a model ceiling.

## Scientific motivation and scope

The first Slider query has one box but a causal event stream. Bbox expansion
cannot estimate motion from one box. A separate physical event expert may add
information absent from the four cached estimates. Do not silently replace bbox
geometry or use its reliability features for this different estimator.

Primary references: Gallego et al. contrast maximization arXiv1804.01306;
STRTTC arXiv2407.07324 and NAIL-HNU/event_aided_ttc commit
79ff0842955304ec4f6164ec09baddc71386d225, utils/str_warping.m. Referenced mathematics
only; write original Python code, no vendoring code with uncertain license.
The official exact EvTTC CMax solver is unavailable in inspected public trees.
This experiment is not a claimed reproduction of its leaderboard score.

## Fixed prototype contract before real-data scoring

Input causal event x,y,time_us,polarity, current ROI, query timestamp, optional
intrinsics supplied by the audited adapter. Never target, depth or future boxes.
Endpoint affine translational warp p_ref = p + (q*p - b)*(t_ref-t), q=1/TTC_ref.
Use centered ROI-normalized image coordinates to condition optimization; b remains
free, so the ROI center is not imposed as the focus of expansion. Separate
polarities for image accumulation. Only a currently supplied ROI is required.

First milestone: synthetic fixture-controlled prototype and support diagnostics,
not production submission. Original NumPy/SciPy implementation with vectorized
bilinear splatting, fixed bounded multi-start optimization over q,bx,by. Profile
actual time before considering CUDA; cache causal event selections outside solver.
Explicit validity flags for temporal/spatial support, nonpositive warp scale,
lost events, flat objective, endpoint search optimum and optimizer failure.
Avoid accepting apparent contrast improvement from discarding events; preserve a
fixed padded canvas and penalize event loss. Record objective and ambiguity.

Tests: endpoint TTC and time shift; off-center expansion with translation;
negative/receding motion; pure translation/zero motion no finite confident TTC;
empty/noise/edge support; future rejection; deterministic seeded event sampling;
timestamp-unit invariance; no history boxes required. GPU-specific speedups and
real-event quality remain unproven at this stage.

Next milestone only after synthetic gates: use audited SensorReader on eligible
Slider500 and road records. Preserve packet/timestamp/camera provenance. Select
search/acceptance on synthetic and eligible development only. Freeze before any
Slider750/1000 or original eight road sequence scoring. Evaluate conditionally
with abstention AND combined full-coverage fallback; never drop difficult queries.

Third milestone if useful: new five-expert/versioned quality contract, grouped
training, three seeds, source-preservation gate and native runtime replay. Do not
promote merely from synthetic success or diagnostic benchmark improvement.
