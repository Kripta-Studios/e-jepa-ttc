# Post-Dfull event screening contract

This is a short development screen, not a replacement for the full S4 acceptance
criteria in the development plan. No event quality result is implied by admission.

The three new arms start from the same selected Dfull checkpoint and use the
temporal screening recipe: seed 7, 1,024 updates, batch four, FP32-highest, LR
`2e-5`, backbone multiplier 0.1, 5% warmup and cosine scheduling. Evaluation uses
the fixed 1,024-frame selection panel at updates 512 and 1,024. Final-update
comparisons are fixed in advance; selected checkpoints are also retained.

- F01 events: current RGB plus the current causal event window.
- F01 RGB-capacity control: the same extra encoder and residual mixer, receiving
  resized current RGB instead of events. Its ten channels are RGB repeated three
  times plus the channel mean. Availability observables are constants, not event
  measurements. It reads no event tensor values.
- F11 events: two causal past RGB frames plus the current event window. Its RGB
  reference is the completed two-frame temporal arm, including its negative result.

The corresponding F00 and F10 controls were already executed with matched update
budgets. New event parameters are initialized in an isolated RNG context. All
heads, dimensions, class count and velocity semantics remain bound to Dfull.
The extra branch is zero-initialized and its parameters receive the common
detector/fusion LR. This limited budget does not establish convergence of a new
encoder, and a negative result cannot exclude better event recipes.

The screen gate is an absolute AP gain of at least 0.005 over the relevant RGB
controls, including the RGB-capacity control. Any passing candidate still needs
full-selection confirmation and explicit zero/permuted/delayed event perturbations
before adoption. Inference perturbations and trained ablations must be identified
separately. No independent holdout or official-score claim follows from selection.

## Input preparation and integrity

`prepare_detection_event_panel.py` reads public metadata and event HDF5 only;
object labels are never loaded. Its fixed panel is 4,096 TRAIN and 1,024 selection
frames. Each sequence's HDF5 is verified against its public full SHA-256 before
encoding. A resumed sequence must retain its admitted size and modification time.

Every window is `[RGB exposure end - 50 ms, RGB exposure end)`, with five bins per
polarity. Pixel-center mapping produces FP32 tensors of shape `[10,96,160]`.
Normalization is local to each window. Compression is lossless and the exact
tensor is checked after the compression roundtrip. CPU workers each retain at
most one HDF5 handle. Completed frames and sequences survive interruption.

`seal_event_panel()` requires the complete producer index and binds both each
tensor file and its causal receipt. The training loader requires the externally
pinned contract and sealed-index hashes. Every consumed frame is checked against
its role, current RGB identity, exposure cutoff, tensor hash and receipt hash.
Additional event arrays, future-window changes and modified receipts fail closed.

The cache is prepared locally and transferred by completed sequence with SCP.
Remote byte hashes are verified before publication; raw HDF5 files are not sent.
Root manifests and the final admission are published only after the transfer
completes. These caches already exist locally and are excluded from reverse
checkpoint backups.

## Model and training entry points

`TemporalDetectorAdapter.project_temporal()` exposes the existing causal projected
features before decoding without accepting targets. `EventTemporalDetectorAdapter`
adds an explicit event tensor to those features and calls the original decoder.
It stores no per-request sensor state. RGB-only calls cannot inherit an earlier
event input. History remains detached and sequence/generation checks are retained.

The GPU engineering gate used real label-free RGB and nonzero synthetic event
tensors. F01 and F11 matched all 29 output tensors exactly; 386 gradient tensors
were finite and the complete model state was unchanged. No optimizer update was
performed. The source hashes and parent checkpoint are recorded in
`evidence/post_dfull_20261031/event_gpu_smoke.json`.

Train a configured arm with:

```text
python scripts/train_detection_temporal_campaign.py --config <temporal-wrapper.yaml> --event-config <sensor.yaml> --output <new-run-directory>
```

The sensor YAML has exactly these keys:

```yaml
schema: eap_detection_event_screen_v1
control: F01  # F11 requires a temporal wrapper with past RGB history
input_mode: events  # or rgb_capacity
cache_root: <completed-event-cache>
cache_contract_sha256: <CONTRACT.json SHA-256>
cache_index_sha256: <SEALED_INDEX.json SHA-256>
```

Runtime provenance binds this file, all event implementation sources, parent
weights, normalization and population. Checkpoints preserve optimizer, scheduler,
sampler and RNG state. `--resume` rejects a changed sensor or producer contract.
The existing frozen temporal runs are not modified by these source additions.

The cloud queue reserves a slot while full-ranking GPU replay is pending or active,
then permits two event-training arms. CPU evaluation, local preparation and
content-addressed recovery backups run independently. Additional GPU capacity is
not rented by this continuation.
