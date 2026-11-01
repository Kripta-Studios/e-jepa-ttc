# EvTTC final submission implementation plan

> Execute inline with executing-plans and TDD; one fresh final review.
> User requests autonomous research and a final solution ready for organizers.

Goal: deliver one reproducible adapted candidate and all434 predictions in the
official format, together with an accurate response about the unchanged
2026-10-28 submission and its newly supplied organizer email.

Spec: user request, organizer email supplied in this conversation, previous
startup/robust research reports and official EvTTC format. Existing isolated
research/evttc-transfer-v2 branch retained. No email/upload is authorized by
"para enviarles"; prepare complete reviewable files and draft, do not send.

User-confirmed listing: sole author Álvaro Schwiedop Souto; independent researcher,
no institutional affiliation. Preserve historical PHASE-TTC name. New adapted
candidate must have a distinct version identifier.

Design: retain original sealed TRAIN40 producers; refit the already selected
inverse QA router with the fixed recipe on24eligible road recordings and Slider500
only. Explicitly exclude benchmark Slider750/1000 and the eight matched road
recordings. Do not claim historical exposure has been erased. Three seeds7/13/21,
300updates, unchanged augmentation/normalization/loss. No new hyperparameter
search against official predictions. Replay all10 sequences through the exact
competition right-camera/Slider sensor contract, target-blind. Use bounded raw
cache, measured complete query cost and synchronized GPU outputs. Freeze all
sources/weights/data before replay and reject changes at packaging.

Review focus: accidentally fitted benchmark rows; signed-output rewriting;
timestamp/index precision; native-left versus competition-right contract;
reported runtimes excluding meaningful work; bundle/export identity changes.

- [x] 1. Audit original submitted ZIP against historical run; record hardware,
  every camera and focal-length evidence. Save user email as provided evidence.
  Research public evaluator/source. Implement tested interior negative-output
  interpolation for evaluation ONLY; never rewrite submitted predictions.
- [x] 2. Train three final inverse QA heads on2608eligible rows; hard-fail overlap,
  preserve seeds and exact recipe. Save deterministic input/weight/source hashes,
  export/reload proof and compact final model bundle.
- [x] 3. Implement tested complete ten-sequence inference/export/verification,
  reject missing/nonfinite rows, preserve signed outputs and original timestamps.
  Run raw-sensor inference on all434 queries with actual runtime profiling.
- [x] 4. After prediction seal, score only locally supported targets; organizer-
  confirmed interpolation shown separately. No extrapolated missing targets or
  unverifiable official-score claim. Compare frozen original/geometry references
  without new test-driven selection. Preserve geometry/exposure limitations.
- [x] 5. Prepare organizer metadata response and separate new-candidate cover
  note, provenance/model card, code/weights/results/verification package. Reuse
  existing application, do not invent another registration requirement.
- [x] 6. One final independent review, required targeted/full regression, static
  checks, local source commit and ZIP with member hashes independently checked.

Research rulings: the organizer email confirms all supplied bbox timestamps and
permits retaining negatives; their evaluator linearly interpolates the two
negative outputs between surrounding valid predictions. The old public docs'
uncertainty about those rules is superseded for this submission. The PDF target
alignment remains unproven unless the declared processing reproduces its values.
The model prepared in this iteration is a disclosed adapted research result;
official listing and acceptance remain the organizers' decision.

Validation: full pytest 7329passed,46skipped,17subtests; zero failures/errors,896.729s. Focused7passed; Ruff/format/Pyright clean. All three fresh-install checkpoints byte-identical. Source/replay identities verified. Local commit and archive receipts record their final identities.
