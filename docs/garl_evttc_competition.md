# Garl-TTC on the frozen EvTTC competition queries

## Status and scope

The run completed on 28 October 2026 in `output/garl_evttc_competition_20261028`.
All 1,302 model-query combinations were processed. The independent audit passed;
the preserved PHASE-TTC artifacts remained unchanged.

| Model | Finite predictions / queries | Unavailable | Median absolute disagreement with PHASE |
|---|---:|---:|---:|
| PHASE-TTC reference | 434 / 434 | 0 | — |
| Garl full RGB + event | 424 / 434 | 10 | 0.261 s |
| Garl event-only | 434 / 434 | 0 | 0.681 s |
| Garl visual-only | 426 / 434 | 8 | 0.236 s |

Disagreement is computed only on shared finite predictions and is not prediction
error. Eight road startup queries lack two real causal RGB frames. The full
variant also lacks event coverage for its earlier endpoint at Slider-750 indices
1 and 2. No model invocation produced a numerical singularity or nonfinite TTC.
Negative predictions are preserved: PHASE 2, Garl full 0, event-only 10, visual-only 5.

The event-only archive contains all 434 rows. Full and visual archives remain
explicitly marked partial; their missing inputs have not been replaced with
another model's output. The aggregate run status is therefore `PARTIAL`, despite
complete processing of the entire planned population.

The generated [comparison report](../output/garl_evttc_competition_20261028/comparison/REPORT.md),
[per-query records](../output/garl_evttc_competition_20261028/comparison/per_query_prediction_disagreement.csv),
and [independent audit](../output/garl_evttc_competition_20261028/independent_audit.json)
contain the detailed evidence. No official EvTTC accuracy or leaderboard result
was obtained by this local run.

The experiment applies three frozen public Garl-TTC checkpoints to the same 434 query identities used by the preserved PHASE-TTC run. It then compares model predictions at identical sequence, row, frame-index, and timestamp identities. The comparison has no verified ground truth for those exact queries. It measures prediction disagreement and coverage. It cannot measure error, RTE, accuracy, or a winner.

The runner performs inference only:

- zero training updates;
- no target values;
- no test-driven checkpoint selection;
- float32 inference with gradients disabled;
- strict loading of one checkpoint per released variant.

## Published evidence

Garl-TTC appears in *Toward Deep Representation Learning for Event-Enhanced Visual Autonomous Perception: the eAP Dataset*, [arXiv:2603.16303v1](https://arxiv.org/abs/2603.16303). The authors define Garl-TTC as Geometry-Aware Representation Learning for object TTC estimation. The [official project page](https://nail-hnu.github.io/eAP_dataset/) and [official repository](https://github.com/NAIL-HNU/Garl-TTC) provide the paper, code, and released checkpoints.

Table VI on page 13 reports an eAP-trained model evaluated on three EvTTC road sequences without fine-tuning:

| Sequence | Garl-TTC RTE | Runtime |
|---|---:|---:|
| CCRs2-medium | 8.31% | 13 ms |
| CCRs2-high | 10.56% | 13 ms |
| CCRm-medium | 12.93% | 12 ms |
| Three-sequence average | 10.60% | 13 ms |

The paper defines relative TTC error as

\[
\mathrm{RTE}=\frac{|\tau-\hat{\tau}|}{\tau}\times 100\%.
\]

The 10.60% result covers three named sequences. It does not cover the ten sequences on the [public EvTTC competition page](https://nail-hnu.github.io/EvTTC/competition/). The original EvTTC paper, [arXiv:2412.05053](https://arxiv.org/abs/2412.05053), reports older methods on all ten sequences in Table V, page 7; it contains no Garl-TTC row.

A later paper, [REACT, arXiv:2609.19204v1](https://arxiv.org/abs/2609.19204), reports 9.44% for Garl-TTC on an in-domain validation split. It does not provide the sequence list, sample count, checkpoint command, or derivation for that value. The eAP paper does not report 9.44%. This guide therefore does not use 9.44% as a ten-sequence competition result.

The source-backed literature receipt lives at `tmp/garl_evttc_literature/literature_receipt.json`.

## Frozen checkpoints

The run binds three public checkpoints and their matching release configurations:

| Variant | Checkpoint | Input channels | Released mode |
|---|---|---:|---|
| `full` | `paper_ours_full.pth` | 46 | RGB + event, late fusion, height ratio |
| `event` | `paper_event_only_lhr.pth` | 40 | Event-only, height ratio |
| `visual` | `paper_visual_only_lhr.pth` | 6 | RGB-only, height ratio |

The inventory records file sizes and SHA-256 hashes for each checkpoint, configuration, and imported native source file. The runtime rehashes them before inference, loads each state dictionary with `strict=True`, and rejects source imports outside the pinned code root. It removes branch-initializer checkpoint paths before construction so one released checkpoint cannot import weights from another variant.

These controls prove byte identity and strict state loading. The public model card and download receipts do not establish the full training-sequence ancestry of the checkpoints. In particular, they do not rule out prior exposure to EvTTC sequences. Treat this run as an adapted frozen-checkpoint comparison. Do not describe it as clean zero-shot transfer unless the checkpoint authors provide training-lineage evidence.

The execution freeze records seed 7, two CPU threads, disabled TF32, disabled cuDNN benchmarking, and float32 tensors. These settings reduce avoidable variation. They do not establish bitwise reproducibility across different GPU, CUDA, PyTorch, and driver combinations.

## Frozen query population

The runner reads the query manifest from the completed PHASE-TTC reference and requires exact equality for:

- all 434 query identities in their original CSV order;
- raw sensor file identities;
- camera calibration records.

The PHASE reference must contain 434 finite predictions and a passing independent audit. The runner rehashes `RUN_FREEZE.json`, `all_rows.jsonl`, `summary.json`, `independent_audit.json`, and `results.zip`. A changed or incomplete reference stops the run.

Each Garl variant receives the same query identity. Input construction differs because the public checkpoints expect two endpoint representations rather than PHASE-TTC's causal history. The comparison script joins rows by the frozen identity key and leaves unavailable Garl predictions as gaps.

## Input construction

### Temporal endpoints

The released configurations use two endpoints, `window_interval: 1`, and the native model field `dT=0.1`. The runtime leaves that field unchanged. The inventory proves configuration and checkpoint byte identity, but it does not prove the checkpoints' training cadence or full training ancestry. The adapter records the measured endpoint spacing and uses that value when converting the two predicted heights to TTC.

For `full` and `visual`, the adapter selects the latest real RGB frame at or before the query, then selects the preceding real frame closest to 100 ms earlier. It does not duplicate or interpolate frames. For the event part of `full`, each endpoint receives a causal event window ending at that RGB timestamp. Window support is capped at `min(measured endpoint spacing, 100 ms)` for each endpoint.

For `event`, a mature query uses two consecutive 100 ms causal event windows. A startup query uses the observed causal prefix split into two real intervals. Queries without two real intervals remain unavailable.

The native event representation remains fixed:

- 20 polarity-agnostic time-volume planes per endpoint;
- 100 ms representation range;
- 5 ms bin width;
- released first-event-relative boundary handling;
- no short-window relabeling.

The model receives the measured endpoint spacing only in the final height-ratio conversion:

\[
\hat{\tau}=\frac{\Delta t_{\mathrm{measured}}}
{1-\hat h_0/\hat h_1}.
\]

Available preparations record `conversion_delta_t_s` and `native_training_delta_t_s`; model outputs repeat both the conversion and native deltas. Input-unavailable rows can terminate before those fields exist and instead retain a null prediction plus its reason.

### Object boxes and cameras

The adapter treats supplied boxes as zero-based, half-open event-plane coordinates and clamps them to the event raster. For each endpoint it chooses the latest supplied box available at or before that endpoint. If the endpoint precedes all query-causal boxes, it uses the earliest query-causal box and records that fallback as becoming available at the query timestamp.

Road sequences use the Blackfly-right RGB stream and Prophesee-right event stream. The CSV timestamps match the right RGB frame clock, and input-only overlays support the right event-plane box interpretation.

Slider sequences use the supplied event and RGB streams. Their calibration file has no named authoritative event-to-RGB extrinsic. The adapter derives a candidate rotation from the supplied camera-pose sets. Both road and Slider RGB crops use calibrated intrinsics, distortion, and rotation-only ray projection from the event box. The projection omits translation because the competition inputs provide no object depth. It provides an approximate common physical region, not exact stereo registration.

The adapter derives a common square edge length from the two endpoint boxes. Each event endpoint crop retains its own endpoint-box center, while the RGB path applies the current endpoint's square center to both RGB frames. All crops are resized to 128 by 128. The adapter preserves the official preprocessing order and uses ImageNet normalization for RGB. The eAP-specific `x+5` alignment offset is disabled because the EvTTC camera geometry differs.

## Terminal prediction contract

Each query and variant produces one terminal ledger row:

- `ok` with a finite TTC value whose sign remains unchanged; or
- `unavailable` with `prediction_ttc_seconds: null` and a named reason.

The runner does not clip negative values and does not replace them with absolute values. Empty raw event windows, missing RGB pairs, nonpositive endpoint intervals, and singular or nonfinite height-ratio outputs produce nulls. An empty event ROI inside a nonempty raw window remains a valid all-zero time volume.

A variant receives a complete `results.zip` only when all 434 rows contain finite predictions. Otherwise the runner writes a `partial_results.zip` that retains each null. In this context, "signed" means that numeric sign is preserved; it does not mean a cryptographic signature.

## Accuracy boundary

The required `--ground-truth-audit` file records a negative finding:

```text
schema: evttc_competition_gt_identity_inventory_v1
status: NO_VERIFIED_EXACT_QUERY_GROUND_TRUTH
query_count: 434
```

The runner reads the audit identity and hash. It does not read TTC labels. It refuses an audit with another schema, status, or query count. Do not create this file by hand to bypass the gate; preserve the source audit that established the missing exact-query mapping.

The comparison script reports:

- signed prediction ranges and explicit null counts;
- per-query Garl-minus-PHASE disagreement;
- median and P95 absolute disagreement on shared-valid rows;
- preprocessing and model-wall telemetry under the local execution policy.

It does not report MAE, RTE, accuracy, SOTA status, or an official leaderboard rank.

## Timing boundary

The runner executes variants in fixed order: `full`, `event`, then `visual`. It prepares each tensor independently and synchronizes CUDA around the model call. The recorded full cost adds preparation time and synchronized model-to-host wall time.

Timing excludes model loading, file indexing, calibration loading, source hashing, and result packaging. The operating-system page cache, HDF5 page cache, and GPU warm state can carry across variants despite independent tensor preparation. One fixed-order pass cannot support cross-variant latency ranking. PHASE-TTC also uses a different history and adapter, so these values cannot support an efficiency ranking between PHASE-TTC and Garl-TTC.

## Reproduction

Run commands from the repository root. Use a fresh output directory for the first attempt. Set `EVTTC_BENCHMARK_ROOT` to the directory that contains the ten competition sequence folders.

### 1. Check the frozen prerequisites

```powershell
$python = "..\e-jepa-ttc\.venv\Scripts\python.exe"
$inventory = "tmp\garl_evttc_readiness\inventory.json"
$gtAudit = "tmp\garl_evttc_readiness\ground_truth_identity_audit.json"
$phaseRun = "output\evttc_competition_20261028_v2"

Test-Path $inventory
Test-Path $gtAudit
Test-Path "$phaseRun\independent_audit.json"
```

The three checks must print `True`. The runner verifies the inventory schema, checkpoint set, source hashes, ground-truth audit, and PHASE reference again. The readiness inventory contains absolute paths to the pinned checkpoint and native-code copies on the machine that produced it. Build a new audited inventory when those assets move; editing path strings would break the byte-bound provenance record.

### 2. Run frozen inference

```powershell
$run = "output\garl_evttc_competition_20261028"

& $python scripts/run_garl_evttc_competition.py `
  --data-root $env:EVTTC_BENCHMARK_ROOT `
  --inventory $inventory `
  --ground-truth-audit $gtAudit `
  --phase-run $phaseRun `
  --output $run `
  --device cuda
```

Use `--device cpu` for a CPU run. Use `--resume` only after an external interruption leaves a valid freeze, an intact row ledger, and no nonempty error ledger. A nonempty `errors.jsonl` blocks every resume attempt: preserve that failed output directory as evidence, diagnose and correct the cause, then start a new attempt under a new output path. To resume an otherwise intact interrupted run:

```powershell
& $python scripts/run_garl_evttc_competition.py `
  --data-root $env:EVTTC_BENCHMARK_ROOT `
  --inventory $inventory `
  --ground-truth-audit $gtAudit `
  --phase-run $phaseRun `
  --output $run `
  --device cuda `
  --resume
```

The runner also rejects a resume when the freeze, source, inventory, raw inputs, calibration, audit, or PHASE reference changed.

### 3. Independently verify the run

```powershell
& $python scripts/verify_garl_evttc_competition_run.py `
  --run $run `
  --data-root $env:EVTTC_BENCHMARK_ROOT `
  --output "$run\independent_audit_recheck.json"
```

The verifier independently reconstructs temporal plans from sensor clocks,
checks all identities and packages, replays FP32 height-to-TTC arithmetic, and
rehashes raw inputs, checkpoints, executed sources and the PHASE reference.
It does not rerun the models or reconstruct tensor pixels. The original audit
and the promoted verifier's recheck both passed for the completed run.

### 4. Build the label-blind comparison

```powershell
& $python scripts/compare_garl_evttc_competition.py `
  --run $run `
  --phase $phaseRun `
  --output "$run\comparison"
```

The comparison writes `comparison.json`, `REPORT.md`, a per-query disagreement CSV, common-scale figures, a runtime figure, and `comparison_manifest.json`. The manifest binds each generated file to the Garl run, PHASE reference, negative ground-truth audit, literature receipt, and comparison source.

## Result acceptance checklist

Accept a completed run for descriptive reporting only after these checks pass:

- `RUN_FREEZE.json` binds the 434-query PHASE manifest and the three checkpoint variants.
- `summary.json` accounts for every expected terminal row and declares `official_accuracy_evaluated: false`.
- Each `ok` row contains a finite sign-preserving prediction.
- Each unavailable row contains a null and a reason.
- The run contains no nonempty `errors.jsonl`.
- `comparison_manifest.json` rehashes all inputs and generated outputs.
- The report labels every Garl-versus-PHASE quantity as disagreement or coverage.
- The report keeps the published 10.60% three-sequence RTE separate from the unscored 434-query run.

The completed run passed these checks. Retain the frozen inference ledger when
regenerating presentation artifacts or evaluating the predictions with future
organizer-provided ground truth.
