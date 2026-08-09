# Object Event TTC v4.30 — stable multiscale similarity

## Authoritative correction (v4.31)

The completed 2,048-row result is authoritative and negative: SHA256
`9722202A4D33F6B5D1B933EEDA1F9143E13E4E2FD64B21356E93783AFAA1C689`, status
`completed_oof_gate_failed`. Stabilization passed (JS median `.0010116798`, JS
p95 `.0423071422`, displacement p95 `.1308624286`). The rank-only winner was
`stable_multiscale_similarity`; champion is null. Its best-arm Pearson was
`.4791568608`, negative accuracy `0`, balanced accuracy `.5`, std ratio
`.3731916487`, slope `.1788173388`, high-bucket Pearson `-.1972577670`, and
magnitude ratios `.92439/.58893/.48926/.30467`. Both arms failed; no sealed data
was opened. The next action after Sol's rethink is a TTC-label-free but
train-box-conditioned common-object-ROI v4.31 redesign: selection independent
of TTC/sign/buckets, immutable sequence/time-disjoint train-only stabilization
and audit pools, sanitized event/ROI-only artifacts, exact physical reversal
controls, and no development/test/EvTTC. The direct full-frame v4.31 draft was
rejected before execution and is not evidence.

The target-free saved-NPZ post-hoc audit is not preregistered: forward-vs-swap
`log_eta` correlation `+.53338`, zero sign flips, and 95.8% coverage at
`|log_eta| >= .005`.

**Superseded status:** this preregistration-era statement preceded the completed
authoritative v4.30 negative result above. Development validation, eAP official
test, EvTTC, RGB and sealed inputs remain closed.

ELI5: three frozen teachers look locally for where an event pattern moved. Their probability maps are multiplied into a consensus, then an event-only student learns it. A small physics fit extracts zoom, rotation and sideways motion. A totally empty event window is `UNKNOWN`, never a made-up answer.

The only forward input is `[B,3,C,H,W]`. TTC, boxes, visible heights, translation, IDs and metadata never reach `forward`. TTC and t1/t2 annotation invariants are explicitly annotation-conditioned losses; t0 boxes are forbidden.

All locked geometry seeds `{7,13,23}` retain EMA epochs 8–10. Their dense maps are centered and shrinkage-whitened. Correlation posteriors at `{1,2,4}` are geometrically averaged and distilled using `KL(P_consensus || P_student)` for the same three student seeds. There is no teacher, seed or best-checkpoint selection. Python, NumPy, Torch CPU/CUDA, sampler and workers are isolated.

Before any student epoch, the analyzer now builds one deterministic, CPU-float32 consensus table for the selected train rows. It runs each of the three frozen teachers once per cache batch, averages all three event-derived foreground maps, releases the teachers, and gathers the aligned cached rows during every fold/seed/arm epoch. The cache hash, configuration hash, checkpoint hashes, row count, timing, build count and teacher-forward-batch count are recorded in the output summary. It contains only event-derived posteriors—never TTC, boxes, IDs or metadata—and is never reused across different hashes.

Before arm fitting, held-out posterior JS median/p95 must be at most `.02/.08` and expected-displacement p95 at most `.5` feature pixel. Failure is `stabilization_gate_failed`; it stops without opening sealed data.

Raw channels `0:10` define activity as `clip(sum(abs(E))/sqrt(mean_{a>0}(a^2)),0,4)`. Support is exactly `(0.05+foreground)*activity*confidence`, then fixed 4x4 tile mass balancing. There is no activity threshold or invalid row fallback. Any nonzero input has a finite estimate/covariance from fixed `.01` Cholesky ridge and three Huber IRLS passes. Zero mass is `UNKNOWN`.

The fit is `d(x)=[[kappa,-omega],[omega,kappa]](x-c)+[tx,ty]`. `log_eta=.5*log((1+kappa)^2+omega^2)` and `expansion=1-exp(log_eta)`. Adjacent fits have composition/cycle loss. Arm B adds only voxel-native normal-flow residuals; it adds no TTC MLP, labels readout, calibration or sweep.

Controls are promotion gates: zero event stays `UNKNOWN`, temporal shuffle must halve Pearson, and the signed endpoint-swap Pearson must be finite and at most `.15` (so the physical reversal is accepted when anti-correlated). The reported absolute endpoint-swap Pearson is diagnostic only and cannot satisfy the gate. Every frozen coverage, uncertainty, magnitude, track-tail, sequence, calibration and seed gate in YAML must pass. `rank_winner` may exist; `promoted_champion` is null unless all gates pass. Arm B also needs its locked paired gain.

```powershell
uv run --no-sync python scripts/preflight_object_event_v4_30.py --help
powershell -ExecutionPolicy Bypass -File scripts/run_object_event_v4_30_stable_similarity.ps1 -Device cuda
uv run --no-sync pytest -q tests/unit/test_object_event_v4_30.py
```

The wrapper description below is superseded operational history. v4.30 has
completed with the authoritative OOF-gate failure above; no further v4.30 run is
the next action.

The automatic decision tree is fixed. Stabilization failure, diagnostic mode, or an OOF gate failure writes a nonselectable result and returns without statting/reading either development reference or materializing validation. A genuine OOF promotion re-runs ten stabilization epochs over all train rows for every seed, averages epochs 8–10, trains a fresh champion head for twelve fixed epochs, then reads the v4.10 comparator and materializes validation exactly once. It reports `development_validation_completed_passed` or `development_validation_completed_failed`; neither outcome opens official eAP or EvTTC.

This is a long GPU procedure, not a smoke command: it performs grouped OOF for both arms and then, only if promoted, three full-train stabilization passes plus three twelve-epoch final heads. Expect hours rather than minutes on a consumer GPU. `--diagnostic-samples` is nonselectable and never reaches development validation.

Operational history and diagnostics, not selectable results:

- Before the wrapper hotfix, an invocation supplied `-DiagnosticSamples 12`, but a PowerShell nullable-parameter boxing bug omitted `--diagnostic-samples`. It therefore began the full 2,048-row OOF and was safely terminated after about 50 minutes, before stage-1 artifacts, summary, validation access, or scientific metrics. GPU use was about 7 GiB. No sealed data was opened.
- After the hotfix, the 12-row diagnostic completed in 24.7 s. Its recorded JS median was `.013007` and JS p95 `.100864` (already a failure against `.08`). Its recorded displacement p95 `.306892` **must not be treated as a pass**: the analyzer then used under-scaled coarse offsets. It stopped before either arm; it selected no arm and opened no sealed data.
- The earlier 96-row record (51.7 s; SHA256 `1A607311C140D7E8A063F139C1FFDCCF826A19D99CBD2BDFF3E6B74815F73C10`) remains **invalid/superseded** for displacement because offsets were under-scaled.
- The post-fix 96-row train-only diagnostic was historical `diagnostic_only`, not the current record: `artifacts/debug/object_event_v4_30_diagnostic/summary.json`, SHA256 `CF9EC7D67EB421AA86304ABD4AB4582F6865CCEABD8D29F5CD7EC4EADBA06BD3`. JS median `.010237284936010838` passed; JS p95 `.19495552778244019` failed; base-feature-pixel displacement p95 `.5500071191315064` failed. All 9/9 KL histories decreased. Its one teacher-cache build recorded 96 rows, 36 batches, build count 1, and `4.1370828000363` s. Rank and champion were null; all sealed flags were false. It stopped before arms. The prior SHA `D9DE07D69CCEB9EAE1C087071A5F46F125655B6C0D0DEE65A86F712D63D3F2D8` is superseded pre-fix history, not current evidence.

The four corrected blockers were the real t1/t2 visible-height schema, a signed endpoint-swap anti-correlation gate, one support-weighted joint multiscale centre in base feature pixels, and truthful `effective_seed = fold_seed + seed` provenance. Earlier emptiness/interruption statements are superseded by the authoritative completed summary above.

Verification is targeted v4.30 `30/30`; full Pytest is 100% pass with 7 skipped and inherited UTF-8/PyTorch warnings; focused Ruff is clean; Pyright reports 0. Global Ruff is not clean: it has 872 inherited findings.

These diagnostics are bounded train-only and nonselectable: they cannot relax, replace, or satisfy fixed gates. The fixed full v4.30 result has completed and failed its gates. The next decision is the box-conditioned TTC-label-free v4.31 redesign above, not relaxed gates or a direct full-frame audit.

Research-transfer decision: [SPAE](https://arxiv.org/abs/2608.01306) identifies high-frequency spectral mismatch and channel entanglement in pretrained visual latents. It supports a diagnostic for radial-spectrum and cross-seed disagreement, and a compact/channel-structured bottleneck only if that audit localizes this JS/displacement tail; its ImageNet/DiT generation results do not transfer to event TTC. [INTACT-JEPA](https://arxiv.org/abs/2607.26056) and its [reference implementation](https://github.com/zju3dv/INTACT-JEPA) support only a later controlled ablation: one shared operator/input grammar over adjacent physical temporal changes, a proper likelihood, and asymmetric gradient routing. They do not directly solve current cross-seed matcher instability and must not be grafted before stabilization.
