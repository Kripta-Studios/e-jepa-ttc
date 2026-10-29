# Deferred B1/B2/B4 evaluation profile

Status: the isolated GPU profile was executed after exclusive admission. Execution completed, but
the preregistered numerical gate rejected B2 and B4. The ordinary evaluator and every
frozen/preregistered runtime remain unchanged; batched evaluation is **not admitted**.

## Fixed inputs and scope

The profiler uses the existing immutable 16-token selection panel at
`output/campaign_smoke_admission_v1/inference_tokens.json` (SHA-256
`eab440fa0ddd4de5e7a13bfb2408e7b5ba4c2c8ddceb7f8b34905532182c8829`). It also accepts an explicit
32-token panel, but rejects any size other than 16/32, duplicates, and tokens outside the frozen
selection role.

It restores the campaign checkpoint through `verify_checkpoint_runtime()`, which verifies checkpoint
schema/producer/inference contract, runtime provenance, frozen code hashes and the complete portable
source tree before strict model-state loading. It does not read or trust a checkpoint batch-size
field. All inputs are prepared once from annotation-empty samples and the model always receives
`targets=None`.

Only H2D, synchronized forward wall time, decode host time and peak CUDA allocation/reservation are
reported. Reader/preparation/scoring are deliberately outside forward timing. The script does not run
the detection protocol or produce AP.

## Preregistered comparison

- FP32 and `torch.set_float32_matmul_precision("highest")` only.
- Seed 7 and the configured four CPU threads are applied before model construction. The receipt
  records effective threads, Torch/CUDA/cuDNN versions, GPU identity/capability and all relevant
  TF32, cuDNN benchmark/deterministic and deterministic-algorithm flags without changing them.
- Batch sizes 1, 2 and 4; the 16/32 panel divides exactly, so no padding or dropped samples.
- One untimed warm-up per batch size.
- Three retained repeats, ordered B1/B2/B4, B4/B2/B1, B1/B2/B4.
- Every tensor leaf in every output head is sliced back to a single-sample structure and compared to
  the first retained B1 output. Velocity is mandatory.
- Tensor acceptance fixed before results: `atol=1e-6`, `rtol=1e-5`, identical structure/dtype/shape.
- Decoded structure and categorical fields must be exact; decoded finite numbers use `atol=1e-6`.
- Any failed repeat yields `REJECT_BATCHED_EVALUATION`; passing yields only
  `ELIGIBLE_FOR_NUMERICAL_REVIEW`, never automatic integration.

Evaluation mode uses one query group because the frozen model slices its query embedding when
`self.training` is false. Batch dimension can still select different CUDA/cuDNN kernels, so exact
equality is measured rather than assumed. B4 can also exceed available VRAM; the failure must be
retained, not silently retried with a smaller batch.

## Executed command

The admitted run used this command contract in a fresh process (with the absolute output path shown
below):

```powershell
python scripts/profile_detection_batched_evaluation.py `
  --checkpoint output/detection_campaign/rgb_screen_v5_seed7_v1/automatic_runs_v1/r0_8192/checkpoints/selected.pt `
  --expected-checkpoint-sha256 5e42f17f4eaac802e6c83ee4265896054167a8b26f0b6e7432501355bd115996 `
  --tokens output/campaign_smoke_admission_v1/inference_tokens.json `
  --output D:/PHASE_TTC_DETECTION_CAMPAIGN_20261029/batched_evaluation_profile_v1/receipt.json
```

The output is write-once: an existing receipt path is refused before checkpoint loading or CUDA
admission. No runner change is authorized from this result.

The required checkpoint SHA is verified before the strict loader calls `torch.load`. Any failure
writes one immutable FAILURE receipt with the exception, completed modes, current CUDA resources and
source/checkpoint hashes; it never retries or silently lowers the batch size. Each mode synchronizes,
empties unused allocator cache, resets peaks and records baseline plus incremental peak allocation and
reservation. Cleanup cost is recorded separately and excluded from forward timing.

The current strict loader and the frozen executed copy both hash to
`1ca8799517464a14c306378b40aa3e4fbb05f1f7f90a48e0d7535b2e9842edf5`; the loader's own historical
identity gate remains active. The executed selected checkpoint hashes to
`5e42f17f4eaac802e6c83ee4265896054167a8b26f0b6e7432501355bd115996`.

## Result and interpretation

The immutable receipt is
`D:/PHASE_TTC_DETECTION_CAMPAIGN_20261029/batched_evaluation_profile_v1/receipt.json`, SHA-256
`77e1e19b13115231fdef2f575b7002c63df5aa070b7afd5be59203daa9766a5c`. Its two status fields have
different meanings:

- `status: PASS` means the profiling procedure completed and produced valid evidence.
- `recommendation: REJECT_BATCHED_EVALUATION` is the numerical admission decision.

Median synchronized forward time per image was 0.038843 s for B1, 0.023659 s for B2 and 0.017715 s
for B4. These are forward-only diagnostic timings, not end-to-end evaluator or campaign speedups.
Peak allocated memory was 311,370,240 bytes for B1, 448,323,072 for B2 and 713,352,192 for B4.

B1 reproduced the first B1 reference exactly in all three retained repeats. B2 and B4 were stable
within their respective modes across all three repeats, but neither matched B1: every one of the 464
compared tensor leaves differed. Maximum raw absolute differences were 0.05023575 for B2 and
0.01781082 for B4; maximum decoded absolute differences were 16.29508 and 16.29510. Therefore the
measured throughput opportunity cannot be used without changing predictions, and no AP equivalence
was established.

### Read-only cause analysis

The evidence establishes a batch-shape-dependent numerical path, but does not localize its first
divergent operation. The strongest bounded hypotheses are:

1. The recorded runtime had `cudnn_allow_tf32=true`, `cuda_matmul_allow_tf32=false`,
   `cudnn_deterministic=false` and deterministic algorithms disabled. Convolutions and reductions in
   the frozen ResNet and region head may therefore select batch-shape-dependent CUDA/cuDNN paths.
   The non-monotonic B2/B4 maxima and perfect repeatability are consistent with different stable
   kernels, but do not prove which operation diverges first.
2. The compared output tree includes `pred_region_prob`, which is produced before both detector
   transformers and portable deformable attention. Because all 464 tensor leaves differed, the first
   exposed divergence is already present no later than the region-head output. The portable
   `grid_sample` path may amplify later differences but cannot explain that earlier exposed change.
3. Small upstream changes pass through the depth predictor, two transformer stages and all prediction
   heads. The decoder
   then ranks queries by class score. A small score change can alter positional list comparison and
   produce a much larger decoded-coordinate difference. The receipt did not retain query-index
   correspondences, so query reordering remains a hypothesis rather than a finding.

Frozen BatchNorm uses stored statistics, and the inspected inference source contains no intended
cross-sample aggregation in the active region-head path. Thus there is no current source evidence
for semantic coupling between samples; numerical kernel selection is the leading explanation.

A scientifically useful follow-up, if batched evaluation remains a priority, is one small admitted
localization run on four fixed tokens. It should compare B1/B4 at named boundaries (backbone,
input projections, region head, depth predictor, 2D transformer, 3D transformer and final heads),
then separately test disabling cuDNN TF32 and enabling deterministic forward algorithms. Any changed
flag defines a new inference producer: its B1 output must first be compared with the historical B1
producer, and passing a localization run would still not establish AP equivalence. No rerun is
authorized automatically.
