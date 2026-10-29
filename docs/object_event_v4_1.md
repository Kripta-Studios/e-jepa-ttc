# Object Event TTC v4.1 — event-only learnability gate

## Why this diagnostic exists

The v4 cache correction succeeded, but the first bounded screen showed that the
fused model improved through its observable-motion branch while the event-only
branch stayed near chance and saturated when converted to TTC. Continuing that
screen would only strengthen the shortcut.

V4.1 asks one bounded question before any more full training:

> Can a small model using only the corrected `t0/t1/t2` event tensor memorize a
> balanced 64-sample subset and retain non-trivial signal on held-out sequences?

## Deliberate removals

V4.1 does not receive observable motion, boxes, visible heights, TTC-derived
motion features or a fusion gate. It does not optimize loss in TTC seconds. The
supervised target is the bounded signed expansion `delta_t / TTC`.

Temporal reversal is no longer hard-coded as `forward - reverse`. The model
predicts directly from causal temporal and spatial differences. Reversal enters
only as a low-weight auxiliary after step 240.

## Architecture

The cached 128×128 events are downsampled by area averaging to 64×64 for a fast
diagnostic. A shallow shared spatial encoder produces feature maps for `t0`,
`t1` and `t2`. The direct head receives spatially pooled levels, first
differences, second differences and their local interactions. A separate
activity head uses per-channel means, standard deviations and temporal
differences. Both branches are reported independently.

## Fail-closed gates

The 64-sample overfit gate requires:

- train Pearson >= 0.95;
- train balanced sign accuracy >= 0.95;
- train TTC-conversion saturation <= 5%;
- train expansion MAE <= 0.005.

The held-out screen additionally requires:

- validation event-only Pearson >= 0.20;
- validation balanced sign accuracy >= 0.55;
- Pearson degradation under shuffled events >= 0.05.

A failed gate is a scientific result. Do not relax thresholds after observing
results and do not resume full v4 training.

## Outputs

`artifacts/debug/object_event_v4_1_overfit/` contains:

- `summary.json`;
- `history.jsonl`;
- `best.pt` and `last.pt`;
- train and validation predictions;
- `FAILURE.json` only for operational failures.

## Execution

Run the focused tests first:

```powershell
uv run --no-sync pytest -q `
  tests\unit\test_object_event_v4_1.py `
  tests\integration\test_object_event_v4_1_step.py `
  tests\integration\test_object_event_v4_1_synthetic_learning.py
```

Then execute the bounded diagnostic against the already materialized v4 cache:

```powershell
& .\scripts\run_object_event_v4_1_overfit.ps1 -Device cuda -Force
```

This diagnostic must finish before any new fused, Level-transfer or full-data
training is authorised.
