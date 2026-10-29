# Independent teacher crosswalk V3

V3 maps TRAIN24 to the frozen TRAIN40 teacher strictly by unique `sample_token`.
It never equates raw `public_source_ordinal` with `teacher_ordinal`. The observed
regression case is raw ordinal 2102/token `pzJcg1OCfc_000002_12321000000` to
teacher ordinal 71589, shard 2237, local row 5.

The audit covers 2,774 manifest-ordered shards, all 88,744 unique TRAIN40 tokens
and ordinals 0..88743, exact raw Garl token-universe equality, scientific shard
receipts, two RGB digests per row, and all twelve TRAIN24 producer/refiner token
sets. No feature tensor is loaded by the builder, and extraction/training remain
false.

Artifacts under `output/independent_phase/teacher_crosswalk_v3`:

- `RECEIPT.json`: `267e5d5ceb5589f1a251e237e5e7fd8c552b319b28d4e1b6d3ca4859d560233e`
- `teacher_source_inventory.json`: `e23946fe0d10accb28e28eb3799432a73edf571efc96d95df9ca4b54b1524ba7`
- `teacher_crosswalk.parquet`: `9545f48fdfe5f0cf9222972cfc3d042a4b0cf23daf274fdae0949e98f8569655`

Future builds use a private staging directory, run the full physical verifier,
and atomically publish only after PASS; failed staging directories are preserved.
