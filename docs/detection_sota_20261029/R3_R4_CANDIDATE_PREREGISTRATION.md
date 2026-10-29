# RGB-R3/R4 candidate preregistration

Status: engineering-ready design only. The active immutable RGB-R0 producer does
not import `campaign_candidate_controls.py`; no scientific run is authorized by
this document.

## Controlled questions

- RGB-R3 changes only moderate photometric augmentation relative to the chosen
  RGB-R1/R2 parent.
- RGB-R4 changes only the train sampler relative to RGB-R3.
- Both retain D4096, R50, eight classes, 50 queries, 11 groups, FP32, batch four,
  8,192 updates, 32,768 image exposures, optimizer, scheduler, losses, resolution,
  evaluation populations and checkpoint-selection rule from their parent.

The selection population never generates augmentation parameters, rare classes,
identity membership, weights or caps.

## RGB-R3 fixed candidate

The preregistered screening magnitudes are brightness `0.15`, contrast `0.15`
and saturation `0.10`. They operate on RGB after undoing ImageNet
normalization, clamp to `[0,1]`, then restore the same normalization. Evaluation
is an exact no-op. Geometry, K, velocity and targets are untouched.

The candidate uses an isolated CPU Torch generator. Its full state, seed, policy
and policy hash must be included in every durable checkpoint. Resume must match
the future augmented tensors exactly in the CPU fixture. The effective recipe
must record these magnitudes before the first update; changing one creates a new
candidate ID rather than a resume.

## D4096 train-only identity inspection

The inspection used the 4,096 train tokens in final manifest
`4af615689bab28016d28881a9339be6c58fe0f35f5d234c0db090eafeff9fb45`, the
pinned label files, `(sequence_id, track_id)` identity, and the exact current
640×384 trainer-admission geometry. It did not read selection labels to design
the sampler.

| Class | Trainable annotations | Train identities | Frames per identity min/median/max |
|---|---:|---:|---:|
| bicycle | 162 | 69 | 1 / 2 / 8 |
| bus | 110 | 24 | 1 / 2.5 / 16 |
| car | 10,452 | 1,548 | 1 / 2 / 168 |
| motorcycle | 693 | 362 | 1 / 1 / 25 |
| other_vehicle | 7 | 6 | 1 / 1 / 2 |
| pedestrian | 245 | 130 | 1 / 1 / 19 |
| tricycle | 97 | 51 | 1 / 1 / 24 |
| truck | 1,994 | 388 | 1 / 2 / 63 |

These are identity-frame memberships, not independent acquisitions. In
particular, repeating six `other_vehicle` IDs cannot establish class robustness.

## RGB-R4 fixed candidate

The preregistered rare classes are `bicycle`, `bus`, `motorcycle`,
`other_vehicle`, `pedestrian` and `tricycle`, selected solely from the train
support above. The proposed train-only mixture reserves 25% of each fixed-size
epoch for identity-balanced rare draws. Hard caps are:

- at most four exposures of one frame per epoch;
- at most four times the number of supporting frames for one identity per epoch;
- exactly 4,096 total draws per epoch, so the update and exposure budgets do not
  increase.

Each rare draw cycles identities before revisiting one, then cycles that
identity's train frames. A frame containing multiple rare identities increments
and must satisfy every affected identity cap. If the requested rare quota cannot
be filled under the caps, the sampler records the shortfall and uses eligible
uniform train draws; it never relaxes a cap.

Every epoch receipt must expose the requested/realized rare draws, uniform
fallback, token exposure counts (the effective per-frame weights), support and
exposure count for every rare identity, population/policy hashes and exact order
hash. Checkpoints retain the generator, epoch order, cursor and receipt.

The membership artifact required for integration must itself bind the D4096
manifest hash, train role, label-file aggregate hash, admission-protocol hash and
the canonical token→`(class, sequence:track)` mapping. Any token outside train,
unknown identity, changed source hash or selection-derived mapping is rejected.

## Acceptance before integration

1. Freeze the winning R1/R2 parent and its checkpoint/hash before R3.
2. Freeze the R3 policy/hash before its first update; compare with the same seed,
   exposure budget and validation cadence.
3. Run R4 only from the frozen R3 recipe and initialization, changing only the
   sampler.
4. Prove exact sampler and augmentation continuation from checkpoint on CPU;
   report all effective exposure ledgers.
5. Evaluate natural-prevalence selection unchanged. Report rare-class recall/AP
   and degradation for car/truck separately; an enriched train diagnostic cannot
   replace selection.
6. Preserve negative results. A gain on repeated rare identities is evidence for
   this bounded intervention, not evidence of eight-class robustness.
