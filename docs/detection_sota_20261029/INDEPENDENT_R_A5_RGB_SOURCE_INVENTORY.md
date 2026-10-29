# Independent R_A5 RGB source inventory

## Decision

R_A5 can reuse the exact raw RGB row semantics from source commit
`2ad7a05797799caf1bcdc9202c307d244f9fe31d`, but it cannot reuse TRAIN40
weights, features, normalizers, role membership, or native ordinals. The only
safe identity join from independent TRAIN24 to the historical RGB metadata is
the unique `sample_token`.

The current source tree is byte-identical to that commit for the relevant
files:

| File | SHA-256 |
|---|---|
| `operational/final_refit/producers.py` | `45717d9ca859461d403f938f80388dfcf3bb1ecf292b5f9beb439982397d8c14` |
| `operational/final_refit/rgb.py` | `77dd54104d04095948d3ed06964978028277e25b3290342629c7c636fc2c60ab` |
| `src/e_jepa_ttc/rgb_port/data.py` | `a33a0ddeb62b7a668d25e9d2e8f880261cc7e325d8bcfe953bcee18e4b0410ec` |
| `operational/rgb_port/cache_prepare.py` | `352a20124fabc91f57670965c0c4eebde405fc3d473b01f85a13e1288bf3109b` |
| `configs/rgb_port/producers.json` | `9ffb221ebe540ee5fe0013d69be07012954f008ffdd988a15acc2b2d2106f0aa` |

## Physical inputs

The historical P/H/V manifests are present under
`e-jepa-ttc-v13-rgb-port/artifacts/rgb_port_20261008` and bind the row files on
`E:/E_JEPA_TTC_RGB_PORT/metadata`:

| Role | Manifest SHA-256 | Rows | Rows SHA-256 |
|---|---|---:|---|
| P | `6f22b070c53c96aa851d90ae2c6836b604e8280965f4887ba6ef3a0cd6d3727c` | 53,890 | `d0341f7aedede35e47ba2e8afa1452478fd39ba43615a85627804ab1fa183b94` |
| H | `ac26288c11583d859166737feb53491167f2f1d908b309fe555220cb9ee27a92` | 18,422 | `137420177757b802abd66b9077cb50ef0a37d8d55951dbcb8582d316d1ee60a1` |
| V | `f73aaab45d868fcab406f8d90aea65dc97c7bd702647677e7cc77b0781f9d720` | 16,432 | `c0344a33739e845a5e5f7ddd5d7098c1fbc08a2fb5b6ad426400ee07c8423402` |

All 52,950 `history_valid=true` identities occur exactly once in the P/H/V
union: P=35,780, H=9,866, V=7,304. They comprise 2,261 T2 and 50,689 T3
queries, 156,589 frame references, and 78 distinct public RGB tar shards. All
78 tar files exist below `E:/eAP_dataset`, and every relative path is pinned by
`data/manifests/eap_detection_public_files_v1.json`. Their aggregate file size
is 72,373,954,560 bytes; no tar payload was opened for this inventory.

For rotation-0 producer R_A5, the exact independent mask contains 35,048 valid
queries and excludes 68 leading-context queries. The valid rows map to P=26,228,
H=1,516, V=7,304, with 1,547 T2 and 33,501 T3 inputs. They reference 103,597
frames in 53 distinct tar shards (49,355,161,600 aggregate tar bytes).

The token join also proves equality of sequence ID, track ID, public track ID,
and `timestamp_us == query_time_us` for all 52,950 rows. Historical
`source_ordinal` and independent `public_source_ordinal` differ for every row;
neither may substitute for the token join.

## Decode and cache semantics

The exact loader uses `(producer_shard, producer_member)` as the raw-frame LRU
key. It decodes original RGB uint8, applies one shared clipped square ROI to all
two or three chronological frames, then converts once with
`float32 / 255`. It does not load a learned pixel normalizer. Boxes, visible
heights, delta times, foreground mask, and ROI are derived row-locally from the
same frozen row.

The prepared-cache key is historical role-local dataset index plus
`sample_token`. Shards are separated by T2/T3 and contain at most 32 rows. The
cache stores lossless `rgb_uint8` and derived geometry/time arrays; float
conversion remains on read. No learned features or weights are stored.

`E:/E_JEPA_TTC_RGB_PORT/prepared/P` is present and complete for all 53,890
historical P rows. Its manifest SHA-256 is
`5b4484d62906b8839b361837eae2fb370f090bd35c37bdc774f017409bdcf9cb`,
its parity receipt SHA-256 is
`ee3d9b91d138e8a01ce35d4d06c67c8f2d84c304539a081876edc711f2f9d38f`,
and its recorded compressed size is 5,434,729,896 bytes. The parity receipt is
bit-exact for every stored field at the first and last T2/T3 probes.

The following required admission artifacts are absent:

- a cache binding for the existing P cache;
- prepared H and V caches;
- an independent TRAIN24 RGB source descriptor and row-freeze receipt;
- an independent rotation-0 RGB role manifest/binding over exactly 35,048 rows;
- a first-batch receipt proving raw/cache parity for P/H/V-derived independent
  rows.

The historical `PREPARE_FREEZE.json` is also absent from the RGB artifact root.
Historical P/H/V manifests and cache parity are useful source evidence, but are
not by themselves an independent freeze or training admission.

## Minimal reusable implementation

Add an `IndependentRGBProducerSource` beside the event source with this narrow
contract:

1. Verify the five exact source/config hashes above, the three role manifests,
   their row hashes, Raw History V4, H5 admission V1, and the independent role
   mask.
2. Join P/H/V rows to the independent mask by unique `sample_token`; preserve
   `historical_role` and role-local index solely for physical decoding/cache
   lookup, and preserve independent `fit_ordinal` for sampling.
3. Re-derive the exact 35,048 rotation-0 identities and 68 exclusions. Reject
   missing, duplicate, or extra rows and any clock/track mismatch.
4. Decode through the exact `RGBPortDataset`/`decode_query` path. Permit the P
   prepared cache only after a new hash-bound independent binding; use raw tar
   fallback for H/V until independently prepared caches exist.
5. Attach the already independent role-scoped teacher by `sample_token`. Never
   expose TRAIN40 DINO features, learned normalizers, checkpoints, or native
   ordinal lookup.
6. Publish a write-once row manifest, source descriptor, cache bindings, and a
   small first/last T2/T3 parity receipt before allowing the first optimizer
   update.

This adapter is reusable across later rotations because the same verified P/H/V
union covers all 52,950 admissible identities; only the independent role mask
and fit-order projection change.
