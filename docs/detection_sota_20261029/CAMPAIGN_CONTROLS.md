# Campaign controls v5

This campaign control is preregistered before inspecting any R0 predictions. It fits the
existing constant-category prior only on the 4,096 v5 train frames admitted by the legacy
trainer profile, then evaluates it independently on the 1,024-frame monitor and the complete
23,958-frame selection population under the frozen local v2 protocol.

The runner reads metadata, calibration and pinned raw labels. It does not open RGB members or
use a GPU. Every one of the 40 public-train label files is checked against the SHA-256 and byte
size recorded by the v5 population audit. Image dimensions come from the 28,054-entry media
admission ledger, whose entries already contain per-member hashes and strict decode receipts.

## Execution contract

- The prior is fitted before either evaluation population is read.
- Selection labels never affect class centers, dimensions, yaw or scores.
- AP101 is accumulated once over each complete population. Chunk AP values are never averaged.
- Ground truth is built directly from raw `bbox_3d_ego`, calibration and admitted image size.
- Only one frame's GT is resident in the scoring loop. Compact output omits PR point arrays and
  per-match records, while retaining global flags, denominators and GT hashes.
- The process RSS gate is 1 GiB. A measured breach fails the run rather than being extrapolated.
- Results are local development metrics, not official eAP scores.

Run after source review:

```powershell
python scripts/evaluate_detection_campaign_controls.py
```

The output directory is `output/campaign_controls/v5_train_constant_v1`. It contains a
pre-execution receipt, machine-readable global results and a generated summary.

## Deferred comparisons

Observed R0, prediction derangement and paired sequence bootstrap are intentionally deferred.
The comparison CLI is unimplemented until the actual completed R0 summary, evaluation hash,
token map and GT source contract are reviewed and bound. No provisional completion schema is
accepted.
