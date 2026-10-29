# Independent PHASE teacher-cache admission

The independent refit may use the existing frozen DINOv3 RGB teacher only as a
fixed external parent. It may not read historical PHASE/H8 learned caches or use
the old TRAIN40 teacher directory during fitting.

`operational/independent_refit/teacher_cache.py` implements a CPU-only,
write-once conversion. It verifies the original manifest, freeze, completion
admission, every one of the 2,774 NPZ shards and receipts, the model file SHA,
and the ordered state-tensor SHA. It never loads Torch/Transformers, RGB, events,
annotations, or TTC values.

The population gate is the v2 fold contract only, SHA-256
`ef67d27078a978cdfb94f0f99f660a44d64079fe5e9e2c1887841e751b6d81a4`.
The converter rejects v1 or a regenerated contract even if its population rows
look equivalent.

Weights and converted cache remain distinct parents. The v2 physical-weight
receipt `TEACHER_PARENT.json` is pinned at
`f9cdfb13ba99299de717423cc8d0334ded8e9628895e7a7f4921fcdb13584b75`;
it binds repository, revision, resolved 784,957,976-byte file/hash, ordered
state-tensor hash and TRAIN24 token set. The converter verifies and copies that
receipt rather than treating the converted shards as the teacher weights.

The exact identity join is:

`unique sample_token -> teacher ordinal/shard/local row`

`public_source_ordinal` remains the independent raw-Garl identity and is never
interpreted as a frozen-teacher ordinal. The converter retains both coordinate
systems explicitly. It uses a bounded eight-shard decoded LRU (only a few MiB),
not a whole-population in-memory feature cache.

Both the ordinal and token must match. The output is 53,047 rows in canonical
`fit_ordinal` order, canonical shards of 32 rows, and `index.parquet` containing
only the ten declared identity/location columns. Each new shard records the
source shard hashes used to build it. The manifest declares `TRAIN24_ONLY`, no
fitting, zero optimizer updates, and no TTC or sensor reads.

The loader is role-scoped. Only `producer` and `refiner` are accepted, and the
allowed token set is recomputed from the fully verified rotation contract.
Generation, selection, calibration, unknown tokens, and arbitrary cache paths
fail closed. Each open writes a role receipt binding cache, fold, rotation,
ordered/set token hashes, fit groups and the disjoint generation groups.

## Historical row-locality audit

Filtering the frozen cache is scientifically admissible only if a target does
not depend on the other rows that happened to share its original batch. The
historical materializer source is pinned at
`c03174474ddde846266b85356f7032f5ce18158e97af6cdf8dbf1634e7b77370` and
the exact relation implementation at
`e778eb51d46514beef2a1700d22d26b08ad3453a3fda628df4fb811afedd57ed`.

The chain is row-local: crop/resize and fixed ImageNet normalization operate on
one endpoint image; the frozen ConvNeXt is in eval mode with zero drop-path and
LayerNorm parameters (the safetensors header has no batch-running statistics);
`functional.normalize(..., dim=-3)` normalizes channels within each spatial
location; offset slicing and the cosine sum retain all leading batch axes.

The CPU fixture compared one anchor feature map alone and in two batches with
different neighbouring rows. Values had maximum absolute difference `0.0` and
validity masks were exactly equal. This audit reads only source, config and the
safetensors header; it does not deserialize model tensors. Therefore filtering
the already frozen targets is equivalent to re-materializing only the TRAIN24
rows with respect to cross-row dependence.

That evidence is an execution gate, not supporting prose: the exact receipt SHA
`9ed1ff47286f200407c18ea894cf760644238ae9977409ec1ad6b4c4fdbaac7e`
must appear in the root authorization, is checked before output creation, and is
copied into and rebound by the final cache manifest.

## Executed v2 conversion

The first authorized v1 attempt failed closed before writing a shard: it had
incorrectly treated raw `public_source_ordinal=2102` as the frozen-teacher
ordinal. The expected token actually resides at teacher ordinal 71,589,
`shard_02237.npz`, local row 5. That failed staging and its diagnosis remain
preserved; no v1 cache was published.

After introducing and independently reviewing the token-exact V3 crosswalk,
the root-authorized v2 conversion completed and was published at
`output/independent_phase/train24_teacher_cache_v2`. Its measured evidence is:

- 53,047 rows in 1,658 shards; 934,564,681 bytes across 3,326 files.
- Manifest SHA-256
  `adadd943e54d0526b982f715ddf589da1d2539070519c8fd9bc9246cbb878408`.
- Conversion wall time 1,390.001 s and peak RSS 391,647,232 bytes, with all
  numeric thread controls fixed to one.
- `gpu_used=false`, `teacher_model_deserialized=false`, zero optimizer updates.
- Execution receipt SHA-256
  `1606796f9fa50268b16647efee836418f0168248f9b5eb5d5c70b0036f1f4ab2`.
- A fresh-process, full post-publication verification passed all 53,047 rows in
  183.368 s; its receipt SHA-256 is
  `a5d61b394a48c88230c4a91af9b4d9087807d605e27f02d454fd44b6bc7ce00b`.

The source is authenticated immediately before and after each selected shard
load. A bounded eight-shard decoded LRU kept memory low, although the mismatch
between fit order and teacher-shard order made those reads the dominant runtime
cost. This is a future I/O optimization target, not a reason to change or
re-materialize the verified cache.

The executed command form was:

```powershell
python scripts/prepare_independent_phase_teacher.py `
  --source-root <frozen-train40-root> `
  --weight-path <model.safetensors> `
  --authorization <root-reviewed-authorization-v2.json> `
  --teacher-crosswalk-receipt <verified-crosswalk-v3/RECEIPT.json> `
  --output-dir output/independent_phase/train24_teacher_cache_v2
```

The resulting `MANIFEST.json` SHA is pinned by every fit caller. Use
`--verify-only --manifest-sha256 <sha>` to rehash the complete standalone cache.
