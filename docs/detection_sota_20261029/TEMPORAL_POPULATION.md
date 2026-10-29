# Temporal RGB population admission

## Result

The S3 temporal population is admitted for the frozen D4096-v5 train cohort and
the complete selection cohort. The receipt contains 4,096 train currents and
23,958 selection currents. It adds the two canonical causal RGB history slots
without changing either current-frame cohort.

The rule is fixed before model training: take the latest two frames from the
same sequence whose exposure ended no later than the current exposure start,
then apply a 500,000 microsecond age bound and the shared media-availability
mask. A failed slot is never replaced with an older frame. This establishes
sensor-timestamp causality only because the release has no arrival clock.

The real CPU audit produced:

| Quantity | Result |
|---|---:|
| Current frames | 28,054 |
| Unique canonical history frames | 32,080 |
| History frames already covered by the v5 current receipt | 23,971 |
| New history members strictly decoded and SHA-256 hashed | 8,109 |
| TARs containing new members | 78 |
| New TARs requiring another whole-file hash | 0 |
| Maximum-age rejections | 0 |
| Media failures in the canonical population | 0 |

All 78 TAR bindings reuse the exhaustive `rgb_full_preflight_v1` whole-TAR
receipts and remain bound to the pinned public-file inventory. Only the new
member bytes were read, decoded strictly with truncated-image loading disabled,
and hashed. A sampled working set during the run was about 310 MiB, below the
1 GiB contract. No labels, event HDF5, test data, or GPU were used.

History support is:

| Role | No history | One frame | Two frames |
|---|---:|---:|---:|
| Train | 24 | 0 | 4,072 |
| Selection | 8 | 8 | 23,942 |

The known unreadable frame
`OYgB6RGWcq:000000661_000000160` does not occur in these canonical histories.
Its frozen failure receipt is nevertheless bound in the admission receipt, and
the implementation fails closed: if it is ever selected, that exact slot is
masked for every temporal variant without backfilling an alternative frame.
The unit fixture exercises this behavior directly.

## Evidence

The reviewed outputs are under
`output/campaign_data/temporal_d4096_v5_admission_v2`. Revision v1 is preserved
as superseded evidence. Review found that v1's runner rejected any negative
history before its documented masking path could execute. Revision v2 removes
that contradiction while keeping an unreadable current frame fatal.

The v2 runner now binds an unreadable history member to its member SHA-256 and
size, observed exception, source failure receipt, and pinned whole-TAR SHA-256
and size. It then masks that exact canonical slot without backfilling. A
run-level regression exercises the complete receipt-building path with a
truncated PNG. The real population has no such history member, so all population
and support counts remain unchanged.

| File | SHA-256 |
|---|---|
| `temporal_media_admission.json` | `1e00d240adba04011063341a3d8d6cea6c0afd3bce310a18d4a8710701e43d38` |
| `history_media_receipt.json` | `0804d352451fbf5296dc95aa4f61c4c7335bf3b53a0d71181196ec7a9f4fb9c5` |
| `negative_audit.json` | `072b6a6c8ac4c2269c0e74258953e813a880b925692483db3995f7ded8a2110a` |
| `provenance.json` | `ee7a0a0ef9e9b04e24bf97a02225480fa280257b792da03c2a761dcc0d4e8ff2` |
| `summary.json` | `8f1daa5a974040fba53812f0fd28d2e9731704b29610d90a73ba4fc4885bae33` |

Each current-frame entry records both canonical pre-media history and the final
fixed history, masks, exposure ends, ages, media receipt data, and mask reasons.
The script also verifies the exact audited population counts, role ownership,
input hashes, split disjointness, and the absence of silent substitutions.

## Reproduction

From the repository root, with `PYTHONPATH=src`:

```powershell
python scripts/prepare_detection_temporal_population.py `
  --data-root E:/eAP_dataset `
  --split data/splits/eap_detection_candidate_v1.json `
  --v5-manifest output/campaign_data/d4096_v5_readable/d4096_manifest.json `
  --v5-receipt output/campaign_data/d4096_v5_readable/receipt.json `
  --v5-media-admission output/campaign_data/d4096_v5_readable/media_admission.json `
  --full-selection output/campaign_data/full_selection_v1/full_selection_tokens.json `
  --preflight-summary output/campaign_data/rgb_full_preflight_v1/summary.json `
  --preflight-shard-ledger output/campaign_data/rgb_full_preflight_v1/shard_status.jsonl `
  --pinned-manifest data/manifests/eap_detection_public_files_v1.json `
  --output-dir output/campaign_data/temporal_d4096_v5_admission_v2
```

The output directory is write-once: the command refuses to overwrite an
existing receipt.
