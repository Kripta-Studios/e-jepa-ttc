# SIMPLEX-T query-conditioned context amendment — pre-fit

Authority: the user explicitly states that original histories are unavailable,
requests an executable alternative, and confirms effective replay handoff.
No scientific fit has run and no scientific model/config freeze exists.

The candidate alternative is retrospective sensor context conditioned on the
current query's supplied detection ROI. It does not reconstruct Garl/eAP IDs,
use target-filtered prior query membership, or assert verified same-object
trajectories. Prior windows must come from raw event timestamps on a fixed
input-only schedule; their crop depends on the current supplied ROI. Consequently
their availability is at least the current ROI availability, not their historical
sensor endpoint. They are NOT predictions that an online tracker emitted then.

This preserves the comparison of additional past sensor information versus
current-only correction, unchanged expert selection and smoothing, but changes
the object-history estimand. Results must use a separate query-context namespace
and cannot satisfy a claim of original SIMPLEX-T same-track history equivalence.
The original source-absence diagnosis is retained, not overwritten.

Before this alternative can be frozen: audit the existing producer crop and
clock implementations, demonstrate current-query replay, establish strictly
past event access and current-ROI dependency timestamps, implement controls and
input-only membership tests, and bind the resulting cache/schema. Preserve the
registered head recipe, finite fit budget, group exclusions and all closed roles.
No expert refit or teacher change is authorized. Do not train from a speculative
adapter or claim that this design note is a validated executable alternative.

First execution step: the already selected64 TRAIN queries are replayed through
all12 matching A5/C2F/PAIR families using historical frozen input tensors. Export
actual differences and latent values; a failed replay is evidence, not permission
to tune tolerances or silently alter preprocessing. This diagnostic has zero
optimizer updates and is independent of the missing trajectory source.

Operational note: user-confirmed replay handoff supersedes the old absent-slot
status. Stage70 cache worker29120 remained visible at confirmation. Do not kill,
modify or resume it; keep a single inference process and bounded host memory.
