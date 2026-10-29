# Independent PHASE TRAIN24 population

## Result

The CPU-only, label-free foundation for an independent PHASE-TTC retraining is
materialized at `output/independent_phase/train24_population_v2`. The earlier
v1 bytes remain preserved but are superseded because their consumer verifier
did not rederive every identity and their parent gate trusted declarations. V2 admits
exactly 53,047 raw GarlTTC queries from the 24 groups assigned to `train` by the
pinned detection split. The 18,594 selection queries and 17,103 calibration
queries have zero entries in every new population and fold artifact.

This step did not train a model, load a checkpoint, materialize sensor data,
extract teacher features, or read TTC values. The annotation Parquet was read
only through its five identity columns to prove exact row alignment with the
raw query metadata. A later supervised fit must call `join_supervised_targets`;
that function accepts only a verified rotation's own `producer` or `refiner`
role and rejects `generation` rows.

Each population row contains:

- `public_source_ordinal`: the original ordinal in the 88,744-query raw GarlTTC
  TRAIN40 metadata;
- `fit_ordinal`: a contiguous TRAIN24 ordinal from 0 through 53,046;
- exact query token, sequence/group, track, public track, and timestamp keys.

The raw metadata audit checked all 177,488 declared RGB history slots. It found
zero future timestamps and zero shard paths crossing a sequence boundary. No
image, event, or other sensor payload was opened.

## Sources and hashes

| Source | SHA-256 |
|---|---|
| Detection split | `91a13c5ae7f6e40ee5fdb4a62da481eba5ffff2b8209ea1110befe3ad7cdc903` |
| GarlTTC query metadata | `03dd3022db4b5f43bb10244fc8778476d74351e764f73a90c8566af949c17fd6` |
| GarlTTC annotation metadata | `64e9a81232643ecce1d2f66e906c7cff6ea1e9e6958f81588b95e112693b11ad` |
| eAP public train metadata | `69ffedab74ea28b295fe3ec2c2c83c08716bc31c216bc2ec9971f8ca9644a314` |

The two GarlTTC hashes were observed from the local raw files; they were not
invented or treated as prior pins. `SOURCE_RECEIPT.json` records them and every
subsequent verifier rehashes those files before admitting a fit contract.

Principal output hashes:

| Artifact | SHA-256 |
|---|---|
| `SOURCE_RECEIPT.json` | `0b83cbb94408d10c40daf30fba24a22d6b84c90caf13272c6148eb8577ad0019` |
| `population.parquet` | `1af6a55143a623232345b5ea52f13c9186444f6115f5dfa90b6b123f93926784` |
| `POPULATION.json` | `cface67d204e9a4f38630683a8648f733f6bd587a49869302589aed64c736348` |
| `BLOCKS.json` | `d2f3501ef037cf04d7918166f9bb6c61612072e378beadc3821ba0166530fd62` |
| `FOLD_CONTRACT.json` | `ef67d27078a978cdfb94f0f99f660a44d64079fe5e9e2c1887841e751b6d81a4` |
| `TEACHER_PARENT.json` | `f9cdfb13ba99299de717423cc8d0334ded8e9628895e7a7f4921fcdb13584b75` |
| `LINEAGE.json` | `eb896d889fa9089110512337aa5984404fdee43c9ce1563828c36f449b3d8bf9` |

The exact TRAIN24 sample-token set hash is
`3dec20a6b65ad1b092c443f4c4d8bb7f18f31986fe4ad8f5df171d27de71f0ce`.

## Six-fold rotation

The partition is deterministic with seed 7 and does not accept a target vector.
Groups are ordered by descending query count; ties use
`sha256("7:<group_id>")` and then group ID. Each group is assigned to the
non-full block with the lowest current query total, then lowest cardinality,
then block number. Every block has exactly four groups.

| Block | Queries | Groups |
|---|---:|---|
| B0 | 9,104 | A1Nhv0KtmL, DGqicHUGWb, L36QTEoVkB, mHGFBekt7X |
| B1 | 8,827 | 9ITOs4twfk, eX8t2nGd6O, qGsgzl4Q8B, qoohcdtLDH |
| B2 | 8,882 | 34K84TaN11, OYgB6RGWcq, nRcwUlHyAd, wE9MZUmiJP |
| B3 | 8,817 | 2cyv0Oedzg, 6WCDpro5rE, 8SdieOgbva, t79dBxj1WS |
| B4 | 8,813 | GjMYZjmSQb, LMHPShQkaU, Po6ePjWYAz, pzJcg1OCfc |
| B5 | 8,604 | 5ilM1PX2vz, ClXqVoZOFu, q2k6O0X6JY, xkiVthsGwf |

For rotation `i`, generation is `B[i]`, refiner is `B[(i+1) % 6]`, and the
other 16 groups are producer. Thus every group is generation once, refiner
once, and producer four times. Within every rotation, all three roles are
disjoint and cover TRAIN24. The six generation roles cover every query exactly
once. A generated query's group is absent from both learned roles in its
rotation.

## Lineage gates

`verify_independent_contract` rehashes all raw sources, receipts, population,
role Parquets, and manifests; validates exact tokens, ordinals, group counts,
16/4/4 role coverage, and six-fold OOF coverage; and fails closed on any change.

`validate_fit_parents` resolves and rehashes every parent receipt and artifact.
It enforces the explicit graph bases→teacher, PAIR→producer base and
context→producer bases/PAIR, while context supervision remains refiner-only.
It permits only:

1. in the future, a parent produced by an admitted independent fitter for the
   exact component edge and rotation, with producer16/refiner4 roles and
   generation groups excluded; or
2. the locally present frozen DINOv3 teacher at repository
   `facebook/dinov3-convnext-large-pretrain-lvd1689m`, revision
   `e959efa74c867491dcfe3ec3e4f97382e39025b3`, file SHA-256 `8bab82...33d`
   and distinct ordered-state-tensor SHA-256 `96ad081d...628`, with a
   `TRAIN24_ONLY` query whitelist.

Paths indicating historical `operational/final_refit` checkpoints,
normalizers, learned H8/history caches, or the old RGB-port learned artifacts
are rejected as fit parents. Historical TRAIN40 identities may be compared,
but historical learned artifacts cannot become live inputs.

No independent fitter/checkpoint identity schema exists yet. Consequently all
`independent_phase_fit` parents currently fail closed even when their external
JSON fields match the graph. This prevents a copied historical checkpoint or
arbitrary bytes plus a self-authored receipt from becoming a parent. Only the
physical frozen teacher is admitted at this stage.

## Reproduction

From the repository root:

```powershell
python scripts/prepare_independent_phase_population.py `
  --garl-data "<GARL_ROOT>/data/train.parquet" `
  --garl-annotations "<GARL_ROOT>/annotations/train.parquet" `
  --eap-data "<EAP_ROOT>/data/train.parquet"
```

The destination is write-once. Existing output is never replaced. A partially
built failed attempt is renamed to a timestamped `.failed.*` directory with a
`FAILED.json` receipt for diagnosis. One such preserved pre-fix attempt records
the initial NumPy-list parsing failure; it is not an admitted output.

## Teacher cache now materialized; fitting remains absent

The standalone TRAIN24 teacher cache has now been converted and self-verified
under the gates described in the
[teacher-cache admission document](INDEPENDENT_PHASE_TEACHER_CACHE.md). Its
write-once `MANIFEST.json` has SHA-256
`adadd943e54d0526b982f715ddf589da1d2539070519c8fd9bc9246cbb878408`;
the post-publish verification has SHA-256
`a5d61b394a48c88230c4a91af9b4d9087807d605e27f02d454fd44b6bc7ce00b`,
and the CPU execution receipt has SHA-256
`1606796f9fa50268b16647efee836418f0168248f9b5eb5d5c70b0036f1f4ab2`.
The relative evidence root is
`output/independent_phase/train24_teacher_cache_v2/`.

This establishes a frozen feature population only. It does **not** establish an
independent producer, refiner, normalizer, learned cache, fitted checkpoint, or
any fit result. Every future fit artifact must still cite the V2 fold contract
and the admitted teacher-cache manifest through the lineage gates above.
