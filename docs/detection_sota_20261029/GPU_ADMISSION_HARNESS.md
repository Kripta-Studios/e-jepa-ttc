# Serial GPU admission for the indexed campaign runner

This is an engineering admission, not a scientific training result. Its three
outputs are permanently labelled `ENGINEERING_NON_CANDIDATE_DISCARD_OUTPUTS`.
The scientific campaign may resume only from the pinned original or reviewed
migrated parent, never from an admission-arm checkpoint.

## Fixed design

The serial order is `old_A`, `old_B`, `optimized`. Every arm executes exactly
16 updates from continuation-state-identical CPU checkpoints. Sixteen updates
are enough to observe first-forward equality, repeated backward variability,
both halves of preparation timing, and the mandatory stop diagnostic. They do
not support an extrapolation to 8,192 updates.

The draft JSON must contain:

- schema `eap_detection_gpu_admission_plan_v1`;
- fixed arm order and `updates: 16`;
- bindings for config, protocol, producer trees and optimized preregistration;
- a resume and selected checkpoint for every arm;
- exact argv, new output, metrics, summary and final-evaluation paths;
- `execution_receipt_path` and physical RAM/free VRAM gates;
- the default tolerance object emitted by `Tolerances` unless a stricter object
  was reviewed before any result existed.

`prepare` hashes every input and checkpoint, validates the exact budget and
writes the plan once. The launcher additionally proves identical model,
optimizer, scheduler, sampler, RNG, progress, update, best metric and RGB ledger
state across the three resumes and selected checkpoints.

Checkpoint config/protocol/runtime-preregistration hashes are checked against
those bindings, and each runtime preregistration must bind the exact producer
bytes invoked by argv. `selected.pt` must be the sibling of its `last.pt`.
Metrics, summary and evaluation paths must all remain below the new arm output.

## Numerical gates fixed before results

The first loss has absolute floor `1e-7` and cap `1e-6`. Later losses use floor
`5e-6` and cap `1e-2`. Gradient norm uses floor `1e-5` and cap `1e-3`. For each
update the optimized allowance is:

```text
min(cap, max(floor, 4 * abs(old_A - old_B)))
```

The nonzero floors cover FP32 scalar serialization and a zero repeat delta; the
caps prevent an unexpectedly noisy repeat from making arbitrary optimized drift
acceptable. These values must not be widened after observing results.

The exact sample-token sequence, learning-rate arrays and exposure count must
match at every update. Preparation speedup must be at least 1.05× independently
in both eight-update halves. Failure of any gate means “do not adopt”; it does
not authorize tuning the gate.

## Resource and timing evidence

Execution requires the explicit `--execute-gpu` flag and the reviewed plan hash
as `--approval-token`. Before the suite and after each arm, the launcher rejects
other NVIDIA compute processes, insufficient physical RAM or insufficient free
VRAM. Desktop graphics processes are not treated as compute competitors.

On Windows, `prepare` seals the existing `C+G` WDDM population by PID and
creation timestamp. A later `C+G` process is rejected rather than silently
whitelisted. A type `C` process is always rejected unless it is a creation-time
checked descendant of the currently launched arm. This handles a venv launcher
whose CUDA-owning Python process has a different PID without allowing a generic
`python.exe` exception.

The launcher records wall time through the first metric record and time from the
last metric record through evaluation/finalization. Runner summaries provide
preparation, compute, peak allocated/reserved memory and optimized phase metrics.
Every arm must produce its required final evaluation. No process is killed,
paused or retried automatically.

## Commands

```powershell
python scripts/detection_gpu_admission.py prepare `
  --draft path/to/reviewed_draft.json `
  --output path/to/admission_plan.json

python scripts/detection_gpu_admission.py execute `
  --plan path/to/admission_plan.json `
  --receipt path/fixed/in/the/plan/execution.json `
  --execute-gpu `
  --approval-token <plan_sha256>

python scripts/detection_gpu_admission.py verify `
  --plan path/to/admission_plan.json `
  --output path/to/verification.json
```

The verifier is evidence-only: it never selects a continuation, deletes an arm,
or claims a full-campaign speedup.
