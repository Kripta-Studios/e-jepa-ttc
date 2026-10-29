# Independent TRAIN24 causal raw history

The admitted metadata-only artifact is
`output/independent_phase/raw_history_index_v4`. V1/V2/V3 remain immutable historical
attempts; V4 closes their verifier and initial-bin gaps. It contains exactly 53,047 queries from
24 TRAIN24 sequences, plus exactly the producer and refiner view for each of six
rotations. It contains no TTC value, learned feature, cache, normalizer, checkpoint,
or historical TRAIN40 learned parent.

Each row describes 31 endpoints spanning 600 ms at 20 ms increments, 30 endpoint
transitions, and 31 distinct half-open observation packets. The first observation
is `[anchor-620 ms,anchor-600 ms)` and the last is `[anchor-20 ms,anchor)`; their
union is the exact 620 ms source span required by the runtime's 600 ms initialization
plus final 20 ms packet. The sensor anchor is the end of the
last Garl event window and equals the **start** of the current eAP RGB exposure.
Availability is the **end** of that exposure. The Garl frame timestamp is the label
clock and is deliberately recorded separately; it is not substituted for a sensor
clock.

The V4 verifier physically rehashes and verifies Fold V2 (`ef67d270...d81a4`) and
its source receipt, rederives every identity and clock from raw Garl/eAP Parquet,
checks both RGB members and both exposures, requires the canonical per-sequence
`data/train/<sequence>/events.h5` path without opening it, and compares every column
of all 12 role views with the Fold V2 roles. A closed file inventory prevents hidden
side inputs.

`stream_bounds_checked=false` and `training_admitted=false` are intentional. The
training loader fails closed. A later, separate admission contract must physically
hash every event H5 and prove its timestamp bounds before any training consumer may
load this index.

Reproduce on CPU from the repository root:

```powershell
python scripts/prepare_independent_raw_history.py
```

The CLI uses a fresh staging directory directly below `output/independent_phase`,
self-verifies it, and publishes with an atomic rename. It refuses any existing
destination. Failed staging trees are preserved with `FAILED.json`; no V1/V2 or
failed output is overwritten.
