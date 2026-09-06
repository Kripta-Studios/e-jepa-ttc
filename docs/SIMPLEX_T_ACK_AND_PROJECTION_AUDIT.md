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

## Current-input lineage and normalization integration

CURRENT_INPUT_INTEGRATION.json records successful loading of all6 tables through
the new current_inputs adapter. Each row's outer/inner family is checked against
the historical ancestry: A5/C2F exclude that sequence, PAIR references the same
A5 parent, and the row PAIR checkpoint hash matches. This consumes the previously
verified ancestor audit; it does not independently replay teachers or checkpoints.
Normalizer statistics and population weights are fit only on each inner_oof TRAIN
table; outer_dev is explicitly refused by that operation. Three tests pass.

The initial float64 anchor equality check failed by at most2.7755575615628914e-17;
the feature phase and expert phase columns are bit-identical in registered FP32.
The adapter checks exact FP32 equality, not an adjustable tolerance. Historical
bytes, head architecture and loss were not changed. Initial typing errors from
pandas named tuples were corrected with explicit record dictionaries. Ruff/types
pass. No fit is authorized by this loader alone; replay and scientific freeze
remain outstanding. Technical updates585, scientific updates0.

## Real-array H1 head exact-resume probe

CurrentQueries now connects normalized current inputs to the registered engine,
binding the exact model inputs, expert phases, targets, masses, timing and
normalizer tensors into its source identity. Timing is mandatory, not inferred.
The bounded probe explicitly uses zero timing as a technical fixture: it does
not certify real timing lineage or authorize a production fit.

The global technical ledger reserved20 updates before execution. Outer0 TRAIN
ran10 continuous updates and5+save+5 at batch128, registered FP32 recipe, seed7.
The complete checkpoint state digests match exactly, including model, optimizer,
sampler, RNG and logs. Evidence: T0/current_array_resume/RESUME_QA.json. No score
was computed. Raw experts were not rerun. Total executed technical updates605;
scientific updates0. No scientific freeze exists. CPU head/source integration
has advanced; the temporal-source and replay gates remain independent.

## Complete D0 exposure timing audit

CURRENT_EXPOSURE_TIMING.json binds all 8192 original queries to the selected
frame exposure metadata, using only input fields and the verified charter.
The final selected exposure ends 1003–19992 microseconds after the event anchor;
none has zero exposure age. This is not proof that the event-only historical
expert consumes RGB, or that annotation generation was available online.
The recorded conditional H1 timing must not be adopted as a final producer
cutoff without establishing the historical ROI dependency contract. The earlier
zero-timing resume probe remains a technical fixture, not production parity.

A live read-only CIM check on this continuation still found Stage70 worker
38464 and launcher36040 running the expansion-cache command. No exclusive
inference slot was inferred or taken. An owner handoff request was presented
to the user for after the entire command finishes. No optimizer updates were
performed by either metadata audit; the technical total remains605.

## Complete original/HF/Garl identity contrast

ORIGINAL_RELEASE_IDENTITY_FULL.json covers all21471 Garl input pairs in the nine
original roles (22716 unique frame/track/box observations). It is not a sample
of128 pairs and is not limited to the8192 target-stratified D0 queries.
All182086 original (frame, instance_id) keys exactly equal the HF eAP keys, with
zero missing keys on either side in every sequence. HF's track_id is a shortened
ID (for example000005 versus instance_id2cyv0Oedzg_000005); this spelling
difference is not evidence of independent tracks.

Only938 Garl observations have an exact original frame/instance key. None of
those938 has an exact matching box under either raw-xyxy or documented-xywh
interpretation. This rules out the tested direct-key and box-format-only adapter;
it does not prove that a source-authoritative crosswalk cannot exist. No nearest
box match or TTC/3D-based association was adopted. The missing prerequisite is
a verified Garl-to-eAP persistent-object crosswalk or the original unfiltered
Garl per-track annotations, not another copy of these eAP files.

The audit projects only identities and input boxes. The HF geometry/velocity
columns are not loaded. Original primitive pickle decoding is unavoidable, but
numeric target/velocity/3D values are never inspected or used in membership.
Ruff and Pyright pass for both new metadata scripts and the delivery module.

## Canonical arm-to-engine binding

The arms module now resolves every registered D0/D1, density, temporal control,
Transformer, latent and replication FitSpec into its exact TemporalConfig and
cached-source transformation. It rejects unknown widths, seeds, folds, stages,
endpoints and registered fits absent from the caller's frozen graph. It does
not authorize the supplied graph or certify its stage-gate evidence.

Eight tests pass without optimizer updates (qa_arm_binding.xml). They cover all
registered configurations, rejection cases, and REPEAT_CURRENT gather parity:
current experts, timings, masks, targets and masses remain unchanged, the shared
normalizer is retained, and the original cache is not mutated. Ruff and Pyright
pass. Scientific freeze, source validation and the production runner are still
pending; this module is an integration component, not a completed campaign.

## Exact outstanding source request

For the Stage70 owner, ask only whether a verified source mapping already exists:
"Do you have an authoritative crosswalk from Garl public_track_id/track_id to
eAP instance_id for the nine original groups, or the original Garl train/*.pkl
per-track annotations consumed by build_garlttc_dataset.py? Please provide paths,
SHA256 and provenance only, without scores or protected data. The eAP ZIP/HF
frame/instance keys agree exactly; direct Garl keys/boxes do not. If you do not
have this mapping, no reconstruction or nearest-box association is requested."

The same source question can be passed to the dataset publisher if the owner
does not possess it. This is a request draft, not an externally sent message.

## Fixed-endpoint inference integration

The endpoint module checks the phase-manifest checkpoint byte hash, complete
checkpoint integrity, completed2500 status and exact TRAIN-source/freeze/config/
seed/device/Torch identity before restoring the head in evaluation mode. Cached
inference visits queries in order with FP32 batches of128 including the final
short batch. Resource interruption raises without returning partial outputs.
It performs no raw expert replay. The caller still must validate the phase
endpoint manifest, role permissions and scientific freeze; this component does
not create any of those missing authorizations.

QA covers complete query order, tail batches, partial-checkpoint refusal,
incorrect checkpoint bytes, training-mode refusal, resource interruption, exact
model restoration and mismatched seed/source/freeze rejection. The positive
loading test mocks the verified decoder boundary using initialized weights; it
is explicitly not a trained checkpoint or a2500-update experiment. No fabricated
optimizer history is published. Existing corruption/publication tests are rerun
alongside these tests. No optimizer updates occur; the campaign remains at605
technical updates and0 scientific updates. Ruff and module Pyright pass.

The owner ACK/request files remain unchanged. A current read-only process check
again found Stage70 worker38464/launcher36040 live; no slot was taken or job
interrupted. The previous goal turn made implementation progress, not scientific
progress or completion. Production data/replay/freeze and execution remain open.

## Physical work accounting after interruption

WorkBudget separates saved scientific progress from a conservative upper bound
on work lost after a checkpoint. It reserves a chunk ending at the next100-update
checkpoint boundary. Synchronous resource-pause settlement at update37 counts37
saved updates, not100. Restart recovery at the old checkpoint retains the entire
unsaved chunk as uncertain work, not as an execution claim. The full graph plus
technical reservations and possible lost work cannot exceed250000. A failed
recovery cannot erase its pending chunk or permit a blind restart.

Five state-transition tests pass (qa_work_budget_cap.xml), with zero optimizer
updates. Ruff and Pyright pass. This journal is not yet connected to engine
callbacks or the production queue; that integration is required before fits.
No real crash/replay counts were invented and the actual technical total stays605.

The relayed publisher email resolves HF access, not unfiltered-history lineage.
A live API check found the same local revision and TRAIN LFS hashes, no new
Garl PKL/crosswalk. SOURCE_ESCALATION.json now distinguishes those facts.
The latest CIM query no longer found the two known Stage70 PIDs38464/36040.
This is terminal evidence for those handles only, not proof of campaign success,
absence of other jobs, or an exclusive slot. No new shared owner release exists.
