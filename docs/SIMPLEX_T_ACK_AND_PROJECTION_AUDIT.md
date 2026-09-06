# Acknowledged interfaces and bounded projection audit

This continuation supersedes the earlier missing-ACK status, not historical evidence.
The user-supplied ACK byte SHA256 is
3e55ab3c6e9a57eecd862ad05e999627ea90957e58e329b2eb3652120e953318.
The request, role map, time charter, historical ancestry and resource amendment
referenced by it were rehashed successfully. Roles are adopted read-only.
Original-8192 temporal scope and all protected/confirmation exclusions remain.
Exclusive replay is not granted. Expansion worker PID38464 and launcher36040
were confirmed live in this continuation; neither was modified.

The user's subsequent instruction replaces 60 GiB and ACK120 GB disk floors with
40,000,000,000 bytes free after outstanding output reservations. This is decimal
GB, not GiB. RAM remains minimum8 GiB available/max4 GiB process-tree RSS,
four CPU threads/two interop. Stage70 code and its own configurations are unchanged.

## Real local geometry evidence

The user authorized investigating reconstruction using existing 3D annotations.
Only original sequence2cyv0Oedzg was used. No TTC, velocity, raw events, RGB,
confirmation or protected payloads were opened. Depth is used solely inside the
existing geometric projection, not as a proposed head input or eligibility target.

The first explicit join on Garl track_id versus eAP track_id found zero matches;
that failed audit is retained in T0/PROJECTION_FEASIBILITY.json. Inspection of
input identity fields established that eAP also provides instance_id. A second
audit used exact frame member paths and Garl track_id versus eAP instance_id,
without constructing IDs by guessed suffixes.

T0/PROJECTION_FEASIBILITY_INSTANCE_ID.json records2995 FRAME rows,17611 OBJECT
rows and128 TTC-PAIR audit rows. Of256 selected observations,108 had an exact
frame/instance-key match and a projection;148 had no matched frame/object key.
For the108, median maximum-corner error versus published boxes is580 pixels;
maximum1034 pixels. These are geometric diagnostic errors, not TTC scores.
They establish that the existing event-camera projection and this proposed join
do NOT currently reproduce Garl boxes. They do not establish that reconstruction
is impossible: camera convention, clipping and cross-release identity still need
verification. No numerical search over TTC/model scores was performed.

Neither audit establishes complete history or authorizes replacing frozen expert
preprocessing. No new scientific fit or optimizer step ran. Totals remain585
technical updates and0 scientific updates. Ten ACK/resource/preflight tests passed;
the initial projection type errors were fixed (optional-frame narrowing), with
the final targeted type check reporting zero errors. No scientific freeze exists.

Next: audit cross-release identity and projection provenance, validate real cached
current-only inputs independently, finish production orchestration, and obtain a
coordinated inference slot when required. Do not infer a slot from process exit.

## Same-frame candidate diagnostic

PROJECTION_FRAME_CANDIDATES.json extends the same128-pair audit to every eAP
3D object in each referenced frame. For each of256 observations it records the
nearest two projected boxes, solely as a diagnostic, not as an identity assignment.
Even the nearest candidate has maximum-corner error at least39.1258877 pixels;
mean125.6778264 and maximum428.1859298 pixels. Thus selecting another same-frame
ID alone does not establish parity with this projector. No guessed mapping was
adopted. The projection source and source3D file hashes are retained in the audit.

## Original release acquisition after user authorization

The official catalog at
https://nail-hnu.github.io/eAP_dataset/assets/data/release_catalog.json links
per-sequence Dropbox ZIPs, distinct from the Hugging Face Parquet release.
Selective byte ranges successfully retrieved annotations.pkl and frames.pkl for
all nine original-role sequences. Total transferred range bytes19,095,691;
extracted bytes89,957,465. All18 member SHA256 values were reverified locally;
ZIP member CRCs were checked during extraction. Receipts reside beside the
files under artifacts/simplex_t/original_annotations/<sequence>/.
No RGB/event media, confirmation or protected sequence archive was downloaded.
The files were not unpickled/executed. An opcode-only inspection of the first
annotation file confirms documented bbox, instance_id and exposure timestamp
keys. Full schema, identity correspondence and history eligibility remain to audit.
This supersedes the missing-loose-original-files prerequisite; it does not yet
establish scientific cache readiness or authorize an exclusive replay slot.

## Primitive-only decoding and nine-sequence input audit

All nine annotation/frame pairs were subsequently decoded using a primitive-only
pickle opcode allow-boundary: globals, reducers, object construction and persistent
references are refused before unpickling. Three decoder/field-projection tests pass.
There are182086 object observations and26957 frames, with unique frame/instance
keys and no object referencing an absent frame in each source. These counts describe
the published originals, not proof of every acquisition observation being retained.

Replacing TTC, velocity and bbox_3d with nonnumeric placeholders leaves the complete
whitelisted observation digest unchanged for every sequence. No target value enters
this projection. This is a real-source input membership test, not yet a full temporal
history/producer-cache integration test.

The first128 Garl pairs per sequence were compared by exact frame member and ID.
Missing ID matches persist. The documented xywh interpretation yields no exact-box
match in these audits. A diagnostic raw-xyxy interpretation yields only2 unique exact
matches across the9 bounded samples. Example original bbox[236,499,348,601] versus
Garl[234,497,343,601] suggests close but nonidentical raw bounds in one case.
Neither convention nor cross-release persistent identity is promoted from this
diagnostic. The earlier statement about coordinates exceeding720 was conditional
on the documented xywh interpretation, not proof of a different camera resolution.
Source format/provenance and causal history attachment remain under investigation.

## Current-only historical table audit

CURRENT_TABLE_AUDIT.json verifies all6 compact historical tables (three outer
folds, inner_oof and outer_dev separately). CSV and NPZ byte hashes match the
frozen index; ancestry hashes match the acknowledged historical audit. Selected
identity columns contain only original-role sequences. Features17, expert TTC,
expert phase and target phase arrays are finite and match declared shape/dtype.
Each outer fold has disjoint training/development token sets totaling8192; the
three development sets jointly cover8192 unique queries. No performance score
was computed. No pooled outer-OOF table was used to synthesize temporal histories.

The first audit attempt correctly stopped on an assumed inner_fold column in
outer_dev; the actual outer_dev schema has no inner_fold. The adapter now requests
that column only for inner_oof. The audit then completed on all six tables.
This establishes compact H1 input availability, not producer replay parity or
scientific readiness. Row-level transitive lineage, production loader, full
pre-fit QA and freeze remain required. No optimizer updates were executed.
