# Independent raw-history H5 admission

## Current result

Physical H5 admission V1 was executed on 2026-10-29. The write-once inventory binds the
exact 24 TRAIN24 event streams and their 12 producer/refiner roles to Raw History
V4. Their total size is 255,166,331,276 bytes (237.64 GiB). No full-file hash was
started only after an explicit I/O-window approval.

The sequential single pass hashed all 255,166,331,276 bytes in 321.074 seconds
(757.91 MiB/s), with 184 MiB peak worker RSS. All 24 per-file receipts were
published and retained under `raw_history_h5_admission_v1/parts`. The ordinary
post-write verifier passed without a second full-file read. Admission applies
only to the 52,950 `history_valid=true` rows; the 97 unverified-leading-context
rows remain excluded.

Inventory V2 binds every local path and byte size to the pinned public eAP
manifest, including the expected release SHA-256. Metadata preflight V4 keeps
the complete 53,047-query population and teacher
ledger unchanged, while conservatively marking 97 queries as
`history_valid=false`. The remaining 52,950 queries are admissible by observed
event extent. The invalid reason is
`NO_AUTHORITATIVE_RECORDING_START_BEFORE_FIRST_EVENT`: it means the leading
context is unverified, not that it is known to lie outside the recording.

All 24 files have empty root, `events`, `events/t`, and `ms_to_idx` attributes.
Their absolute `ms_to_idx` arrays contain zero-valued entries at all 19 affected
required positions. Those entries prove an empty event-reader result, but do not
prove acquisition coverage before the first stored event. V4 therefore keeps
`recording_acquisition_bounds_verified=false`, `stream_bounds_checked=false`,
and `training_admitted=false`; it does not invent zero-filled observations.

V4 emits one immutable history-admissibility ledger, exact masks for all 12
producer/refiner roles, and six generation masks. The verifier re-derives every
identity from Raw History V4 and requires each admissible query to occur in
exactly one generation mask.

A one-file probe on `DGqicHUGWb` measured H5 open at 3.6 ms and header/boundary
reads at 20 ms. All 24 header/index-bound inspections took 1.43 seconds. These
metadata reads registered the eAP HDF5 compression plugin before opening files.

## Artifacts

| Artifact | File SHA-256 | Canonical receipt SHA-256 | Status |
|---|---|---|---|
| `raw_history_h5_inventory_v1/INVENTORY.json` | `7ad0fed2afcb79d1dde4834add15aa156944d97bea073c4e3901e3142ddbdfa0` | `e90baee57e9cd8bac468c6a947d6a643a3f7553bbd08e77324350cc70f26cd52` | Inventory only; not hashed/admitted |
| `raw_history_h5_metadata_preflight_v1/PREFLIGHT.json` | `8a423a38b4bab31c892ce514f939cb65fe37c867a93596363e0d9e54d6baa5b1` | `815a710f43694643d0794d3a6c6a762207f65def85fbb1869f3d116eb7ab3486` | Superseded: first event was treated as recording start |
| `raw_history_h5_metadata_preflight_v2/PREFLIGHT.json` | `e99af26c1698c86931bce186693defd0fc603c8a1044042a72eaf3463be7146f` | `6d9a2d37486c4406d01882bd81ae641c427974b49a54478b48cdf200e14d8e96` | Superseded: leading zero index was treated as acquisition coverage |
| `raw_history_h5_metadata_preflight_v3/PREFLIGHT.json` | `f300f0e4ef47cb6eae68ba671ad005599b4bf3c81a47482b3dd609877bdfb7ec` | `66e6b670ba0e850e8c5d537e35214cbc4db7eeaebbf45a7ab065a27a29eb4795` | 52,950 valid; 97 leading contexts unverified; not hashed/admitted |
| `raw_history_h5_inventory_v2/INVENTORY.json` | `99a7394ad44678fd9c8def1f8bc7e1df83d8c38746046ff65385b8249590614e` | `dcf63110837a3322a7eb185648835537289487799f5eee9acaee09b7ae0b21f7` | Public release SHA/size pinned; not hashed/admitted |
| `raw_history_h5_metadata_preflight_v4/PREFLIGHT.json` | `c9141674d95af89ff67352a09ad5a9c25b7a84942c88c08546f66dbdb0ee4b92` | `e94527e338831f1b5b68cc68a7e81ee8f66bec6fa2e880108f2e5ef4a28bf2ba` | Current: 52,950 valid; 97 leading contexts unverified; not hashed/admitted |
| `raw_history_h5_admission_v1/ADMISSION.json` | `17ca9176d8a4dadd06fa79b0e4ac2d3c055228fc22c534639fd18c6a8b23fb1e` | `31ee610e9c90d7d73b2cdfe2a216baf310c210eabd81f58537fb958b46013dc5` | 24/24 public-release hashes and physical bounds verified |

The V4 admissibility ledger SHA-256 is
`851387c00b9d7fee374717fbe06bc6c04cf3e100037b5a39be6e77fa1438d6ca`.

The full sequential hash would read 237.64 GiB. Pure throughput estimates are
20.3 minutes at 200 MiB/s or 8.1 minutes at 500 MiB/s; these are theoretical,
not measured. The actual approved run achieved 757.91 MiB/s; this measurement
must not be used as an uncontended training-throughput estimate because R0 shared
the drive during part of the window.

## Commands

```powershell
python scripts/prepare_independent_h5_admission.py inventory
python scripts/prepare_independent_h5_admission.py metadata
python scripts/prepare_independent_h5_admission.py verify --receipt output/independent_phase/raw_history_h5_metadata_preflight_v4/PREFLIGHT.json
python scripts/prepare_independent_h5_admission.py verify --receipt output/independent_phase/raw_history_h5_admission_v1/ADMISSION.json
```

`admit` is implemented as a sequential write-once operation with full SHA-256,
per-file schema, timestamp, `ms_to_idx`, causal-bound and role checks, atomic
publication, resumable per-file receipts, and failed-attempt preservation. It has
completed successfully. Completed parts are checked before pending files are hashed, and
the ordinary verifier uses immutable path/stat/public-digest bindings rather than
silently performing a second 237.64 GiB read. A full rehash requires explicit
`verify --full-hash`. All earlier outputs remain preserved.
