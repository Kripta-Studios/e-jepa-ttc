# Final conformance finding — no retrospective rescue

The runner reached RISK_ROUTER_DEV_CANDIDATE after all twelve Stage 64 endpoints
and six Stage 65 ridge fits. These observed results are retained, not rewritten.
The final audited acceptance is INTEGRITY_BLOCKED, a distinct outcome.

Document05 requires CE17 equivalence before fitting RISK. The frozen execution
code invokes FullRegretRidge.fit in the train-all loop and _ce17_replay only in
the later outer-evaluation loop. Replay did match R2, but the required ordering
was not implemented. There is no claim that a hash proves this ordering.

The same frozen code lacks the mandatory eval-only PAIR-phase permutation,
oracle/headroom/regret/sign diagnostics and the document06 sequence-only
bootstrap and leave-one-sequence-out sensitivity. Targeted QA did not establish
coverage of these requirements. They were discovered after outer scores, so no
post-hoc model inference, retraining, gate change or candidate promotion is used
to conceal the omission. They remain explicitly not executed.

The reporting audit only reloads verified checkpoints on CPU (no forwards),
verifies fit/output identities and independently aggregates the frozen CSV
predictions by the specified sequence/bucket weights. It supplies bucket scores
and fold-separated CSVs with phase, target/observation anchors, failure/fallback.
This is reporting, not a corrected scientific execution of the missing controls.

The twelve high/medium review findings were addressed in commit4c9240e with
temporal cache v2, coherent frozen A5 replay, nested ancestry, complete component
hashes, real CUDA resume QA, checkpoint recovery, central authorization, live
resource guards, typed failure closure, durable access journal, canonical metric
validation and verified packaging. This does not erase the distinct omissions
above. Access coverage is explicitly non-exhaustive; unknown does not mean false.

User-authorized time-cap removal is separately documented by execution commit
362177c and WALL_TIME_AMENDMENT.json. It preceded new outer evaluation and changed
neither numerical training code nor gates. Original budget and lock bytes remain.

No seeds13/23 were authorized after RAW failed its gates. Stage65 used only the
verified nested A5/C2F/PAIR sources, not the failed RAW expert. No new data, public
validation, private test, CodaBench, downloads, push or PR were used for this run.

Next action is none. Any future repair, additional diagnosis, training or
confirmation needs an explicit prospective amendment; these nine adaptively
reused groups and the positive numerical gates do not establish SOTA.
