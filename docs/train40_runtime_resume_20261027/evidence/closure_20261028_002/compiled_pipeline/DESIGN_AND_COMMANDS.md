# Compiled transport integration: design and current admission

This is an experimental private adapter. It does not modify frozen scientific sources, weights, normalizers or preserved predictions.

`operational/train40_runtime/compiled_pipeline.py` uses one `torch.library.custom_op` (`train40_compiled_pipeline::transport_pair`). The operator implementation calls the compiled complete-transport block. TorchScript records that opaque node instead of tracing through Dynamo. CUDA Graph capture should then record the compiled kernels; actual GPU execution and final TTC parity still require validation.

The existing `attach_transport_pair_core` performs its audited two AST replacements. The new adapter substitutes only the attached fine/coarse tensor cores. The original encoder and tail remain unchanged. `CompiledTransportProducerRuntime` inherits the existing input/output guards, private capture streams, inter-stream last-use events and owned result clones. Event reuse defaults off, RGB reuse on. Callers must use RGB object chunk size one to retain the original admitted B7 endpoint encoder batch.

Operator handles use weak registry ownership: model wrappers retain their executor; stale standalone traces fail explicitly after the owning wrapper is gone. A runtime must outlive all outstanding calls and graphs. No process-global model mutation or hidden eager fallback occurs.

CPU proofs use `torch.compile(backend="eager", fullgraph=True)` to establish tracing compatibility and preserve numerical reference arithmetic, not to claim Inductor numerical equivalence or CUDA speed. Tests cover the opaque IR node, real compiled-call dispatch after tracing, changed and empty inputs, model states/parameter identity, finite outputs, invalid-clock/sensor/RGB false guards, explicit async guard rejection and expired ownership.

GPU benchmark `scripts/benchmark_train40_compiled_pipeline.py` compares prepared-input original CUDA Graphs against compiled-transport CUDA Graphs using the same encoder reuse/chunking policy. It uses distinct translated real-sensor ROIs, preserves object zero, compares all three final TTC outputs against original eager producer batching outside the latency clock, and records source bytes/hashes and unchanged artifacts. Timing excludes raw preparation, H2D and final context: no complete-update or 50 Hz claim is permitted from this driver.

Initial command (run only after the root grants the exclusive GPU slot):

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' scripts/benchmark_train40_compiled_pipeline.py `
  --candidate 'D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE' `
  --manifest artifacts/final_train40_transfer_20261027/QUERY_MANIFEST.json `
  --reference artifacts/final_train40_transfer_20261027/final_train40/own `
  --compiler-root artifacts/train40_runtime_20261027/compiler_probe `
  --output artifacts/train40_runtime_resume_20261027/compiled_pipeline/gpu_single_v1 `
  --indices 0 137 488 505 514 533 553 --objects 1 --warmup 3 --repeats 20
```

Then use a fresh evidence directory with `--indices 553 --objects 20`, keeping the default per-object event chunk and RGB chunk1 first. A grouped event batch is a separate variant requiring the same fixed TTC gate. Defaults are FP32, TF32 off, `use_fast_math=False`, `force_same_precision=True`, `triton.cudagraphs=False`.

The precision council found that these compiler defaults still permit FMA. If final TTC parity fails, independently measure `emulate_precision_casts=True` (disables automatic fp fusion) and `eager_numerics.division_rounding=True` (FP32 div_rn) under new result directories; do not loosen the fixed 1e-5 absolute/relative gate. Neither option guarantees identical reductions.
