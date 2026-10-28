# ADR-0043: Frozen PHASE-TTC on the EvTTC competition sensors

Date: 2026-10-28

Status: accepted for prediction generation; official server acceptance and scoring pending.

## Context

The ten-sequence EvTTC competition download differs from the public development
data used by the previous transfer scripts. Eight road sequences provide right-
camera-aligned six-column CSV annotations and HDF5 sensors. The two Slider
sequences provide ROS1 bags, a different camera pair, and approximately 25 Hz RGB.
The earlier development loader selected left cameras, JSON annotations, and a
historical query subset. Reusing it unchanged would select incompatible inputs.

## Decision

Add a separate `operational/evttc_competition` adapter and
`scripts/run_evttc_competition.py`. Preserve the sealed nine-component TRAIN40
model, normalizers, canonical representations, and final 100 ms phase mapping.
Do not edit the frozen runtime source files or retrain any component.

Use every supplied bbox CSV row in its original output order. Preserve its index
and timestamp text; use exact sensor timestamps internally. There are 365 road
rows, 51 Slider-750 rows, and 18 Slider-1000 rows. Keep Slider indices 1--51 and
18--35. Do not infer time offsets from published result plots.

Choose the right camera pair for road data. Timestamp correspondence and sensor
overlays support interpreting the supplied boxes in the event raster. Use
zero-based half-open bounds and record explicit clipping. This is a documented
interpretation of the headerless annotation files, not an organizer-confirmed
annotation specification.

Project event-camera rays to RGB through calibrated intrinsics, distortion, and
relative rotation. Translation is omitted because inference does not use object
depth. The Slider rotation is derived from a consistent cluster of calibration-
target poses in the supplied MATLAB file and checked on sensor overlays. These
are approximate corresponding ROIs, not exact depth-dependent stereo crops.
Use the last two causal boxes to form one common square per modality, with the
existing 0.25 margin; when only one box exists, reuse that box for ROI construction
without duplicating any sensor frames.

Road event timestamps remain in their supplied relative microseconds. Slider
uses event payload timestamps and RGB header timestamps, with the first RGB
header as a common origin. Keep negative event times. ROS playback timestamps do
not define the sensor clock. Road millisecond indices and nonzero Slider packet
headers establish conservative recording bounds; unknown silent edges are not
invented.

Adapter v2 considers the same four event anchors at every query: 300, 200, 100,
and 0 ms before the query. At each anchor with at least three microseconds of
recorded history, form three native 100 ms windows if possible; otherwise
partition its actual recorded prefix into three positive-duration windows.
Use measured endpoint deltas. All these observations become available at the
current query because they share its causal ROI. The anchor schedule is fixed
independently of model outputs. This is a declared startup preprocessing
adaptation. A physically
recorded empty window is valid input and receives explicit zero-support
diagnostics; missing recording history is unavailable input. Do not substitute
constant TTCs or fabricated pre-recording events.

Select actual causal RGB predecessors nearest to 100 ms spacing. Pass measured
frame intervals to the unchanged producers. This yields approximately 80.164 ms
spacing on Slider without interpolation. Use genuine T2/T3 observations and at
most eight valid history entries. The complete information span, including the
first event-window start, stays within 650 ms. Short-window and changed-cadence
inputs are distribution shifts; preserving model weights does not establish
preserved accuracy.

Record every row's observation windows, event counts, ROI bounds, frame indices,
measured deltas, startup mode, producer support diagnostics, and output. Preserve
signed model predictions; do not apply an absolute value or a new positive clamp.
Do not reject predictions based on test errors or change thresholds after viewing
outputs. Keep unavailable rows and reasons in the full ledger.

The frozen expert-to-phase conversion cannot represent NaN, zero, or finite
TTC in (0, 0.1] seconds. If that exact conversion error occurs and the captured
expert output confirms mathematical unrepresentability, reject the entire
observation and record its producer values, support, known flags, and compute
cost. Keep the remaining observations in chronological order. The known flags
do not determine admission. No valid RGB history invokes the existing native
event fallback; no valid event history yields an explicit unavailable query.
Other exceptions still stop the run. In particular, infinite expert TTC has a
phase limit but violates the native producer-feature finite-value contract;
that distinct error remains fail-fast and is not observation-filtered.
No output clipping, expert substitution,
alternative crop search, or model adjustment is introduced.

Freeze model, source, input, query, calibration, and policy identities before the
first benchmark forward pass. Recheck them before packaging. Per-estimate cost
includes sensor reads and representation preparation, all model stages, and a
completed host output. One-time model loading, indexing, calibration, and hashing
are recorded separately and excluded from that cost. This offline implementation
does not establish sustained real-time performance.

## Validation and release boundary

Required checks cover half-open timestamps, payload-versus-playback clocks,
recorded silence, future-data poisoning, chronological ROI history, actual RGB
cadence, calibration direction, exact frozen-model parity, signed output
preservation, query identity, and incomplete-package rejection.

The first v1 attempt completed all 365 road queries and two Slider-750 queries,
then stopped at Slider-750 index 3. Its E_C2F expert emitted
0.06156697869300842 seconds on a short-prefix observation whose latest ROI was
empty. The failed run and its 367 successful records are preserved unchanged.
V2 records the earlier attempt's hashes and reruns the whole population in a
new directory. This change was motivated by a runtime-domain failure, without
test labels or an accuracy comparison; it is not a predeclared original policy.

A complete local package has ten four-column `results.txt` files and 434 finite
model outputs. A missing output produces an explicitly partial artifact. Neither
local validation nor finite coverage is official accuracy or a benchmark-defined
failure rate. The public instructions do not document signed-value acceptance,
the exact required query population, or startup exclusions. Those server details
remain unverified until organizer evaluation.

No application email, upload, or leaderboard submission is performed by this
adapter. Published comparison curves are not digitized or used to tune this run.

## References

- https://nail-hnu.github.io/EvTTC/competition/
- https://nail-hnu.github.io/EvTTC/download/data_format/
- https://nail-hnu.github.io/EvTTC/calibration/
