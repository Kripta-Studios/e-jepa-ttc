# Object-centric JEPA-LHR training

## Why this route exists

The historical raw-event trainer assigns one object-level TTC label to a
full-frame event tensor. The audited local release contains 20,097 full-frame
input signatures with multiple tracks and conflicting TTC labels (46,710 rows,
up to seven tracks for one input). A deterministic regressor therefore has no
well-defined target and can minimize its robust loss with a near-constant
prediction.

This route preserves object identity before learning:

1. select the official `boxes_xyxy` endpoint pair for one `track_id`;
2. create the official square ROI for each endpoint;
3. materialize both the official 20-plane time-volume representation and a
   21-channel ROI representation compatible with the existing JEPA backbone;
4. predict positive visible heights `h1` and `h2`;
5. derive signed TTC with `delta_t / (1 - h1/h2)`;
6. select checkpoints only after the full geometric curriculum and reject
   constant TTC or constant log-ratio predictions.

Bounding boxes, 3-D geometry, TTC, category, and masks are never model inputs.
The box and 3-D fields are cache-time/supervision-only data. Masks are optional;
missing masks produce no segmentation loss and are never replaced by rectangle
masks.

## Curriculum

- epochs 1-5: log-visible-height Smooth L1;
- epochs 6-10: height + visible log-ratio + approaching/receding direction;
- epoch 11 onward: previous terms + TTC-aligned MiD log-ratio term;
- optional focal mask loss is applied only to endpoints whose real mask file was
  successfully loaded.

The direct signed-TTC auxiliary is disabled by default. It exists only as an
explicit ablation and is not the primary objective.

## Screen run

```powershell
uv run --no-sync python scripts/run_e_jepa_object_lhr.py `
  --profile screen `
  --stages cache train `
  --eap-root "E:\eAP_dataset" `
  --garlttc-root "E:\GarlTTC_dataset" `
  --include-masks `
  --device cuda
```

The cache stage is resumable:

```powershell
uv run --no-sync python scripts/run_e_jepa_object_lhr.py `
  --profile screen `
  --stages cache `
  --eap-root "E:\eAP_dataset" `
  --garlttc-root "E:\GarlTTC_dataset" `
  --include-masks `
  --resume
```

Then train from the completed cache:

```powershell
uv run --no-sync python scripts/run_e_jepa_object_lhr.py `
  --profile screen `
  --stages train `
  --device cuda
```

## Exact JEPA transfer screen

The object cache stores `jepa_event_roi` with the same 21-channel representation
contract as the label-free encoder. Transfer remains exact and fails closed on
any key, shape, or structural-config mismatch.

```powershell
uv run --no-sync python scripts/run_e_jepa_object_lhr.py `
  --profile screen `
  --stages train `
  --pretrained "$ROOT\pretrain\level\seed-7\checkpoint.pt" `
  --device cuda
```

The runner separates outputs automatically under `scratch/` and
`level-transfer/`. During the first five epochs of a transferred run, the
backbone LR is zero while pooling and object readouts learn the task. Scratch
does not freeze its random backbone.

## Full candidate

Only run this after the screen demonstrates non-collapsed validation log-ratio
and TTC predictions.

```powershell
uv run --no-sync python scripts/run_e_jepa_object_lhr.py `
  --profile full `
  --stages cache train `
  --eap-root "E:\eAP_dataset" `
  --garlttc-root "E:\GarlTTC_dataset" `
  --include-masks `
  --device cuda
```

By default the full profile runs seeds 7, 13, and 23 and requires a clean
worktree. A seed-matched transfer run must pass exactly one `--seeds` value and
its matching SSL checkpoint. It remains training-only until external
eAP/CodaBench and EvTTC evaluation are frozen.

## Tests

```powershell
uv run --no-sync pytest `
  tests/unit/test_object_lhr.py `
  tests/unit/test_object_lhr_data.py `
  tests/unit/test_object_lhr_training.py `
  tests/unit/test_garlttc_object_cache_extension.py `
  tests/integration/test_object_lhr_step.py
```

## Expected artifacts

- cache manifest: `artifacts/cache/garl_object_lhr_screen_v1/manifest.json`;
- scratch summary: `artifacts/runs/e_jepa_garl_object_lhr_screen_v1/scratch/seed-7/summary.json`;
- transfer summary: `artifacts/runs/e_jepa_garl_object_lhr_screen_v1/level-transfer/seed-7/summary.json`;
- validation predictions: `best_validation_predictions.csv`;
- best checkpoint: `best.pt`;
- failure details: `FAILURE.json`.

Do not compare this event-only route directly with the RGB+event SOTA. The fair
first comparison is the official event-only height-ratio model under the same
split and metric protocol.
