# V8 experiment ledger

This ledger is a planning and evidence index. Its `signed evidence` column must point to a signed artifact before a row moves to `completed` or `confirmed`. It records no performance claims.

Allowed states: `planned`, `frozen`, `running`, `completed`, `failed_integrity`, `failed_gate`, `confirmed`.

| arm | phase | state | seed | folds | config/manifest | signed evidence | gate decision | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V8 branch creation | P0 | completed | n/a | n/a | V8 master plan | n/a | branch exists | Only repository branch creation is represented as completed; no experiment has run. |
| P0 | prerequisite freeze and smoke | planned | 7 | n/a | V8 protocol/config freeze | pending | pending | Establish the signed protocol and quality gate before experimental execution. |
| A | no-training autopsy | planned | n/a | evidence-defined | A5/C2F/Garl signed evidence inventory | pending | pending | Audit prior A5, C2F, and Garl evidence; this arm does not train a model. |
| R | prospective nested A5/C2F router | planned | 7 | nested grouped folds | V8 R router manifest/config | pending | pending | Prospective routing arm; retain all failed and negative runs. |
| B1 | TIMEVOL20-3 | planned | 7 | nested grouped folds | V8 TIMEVOL20-3 config | pending | pending | Sequential screen; do not select against test. |
| B2 | EXP6-3 | planned | 7 | nested grouped folds | V8 EXP6-3 config | pending | pending | Sequential screen; do not select against test. |
| B3 | PAIR20-2 | planned | 7 | nested grouped folds | V8 PAIR20-2 config | pending | pending | Conditional on the preceding diagnostic gate. |
| C1 | GATED-EXP6-3 | planned | 7 | nested grouped folds | V8 GATED-EXP6-3 config | pending | pending | Conditional on prerequisite routing and diagnostic evidence. |
| D0 | scratch diagnostic | planned | 7 | nested grouped folds | V8 D0 scratch config | pending | pending | Diagnostic evidence only; not a result claim. |
| D1 | random frozen diagnostic | planned | 7 | nested grouped folds | V8 D1 random-frozen config | pending | pending | Diagnostic evidence only; not a result claim. |
| D2 | JEPA frozen diagnostic | planned | 7 | nested grouped folds | V8 D2 JEPA-frozen config | pending | pending | Diagnostic evidence only; not a result claim. |
| D3 | JEPA partial fine-tune diagnostic | planned | 7 | nested grouped folds | V8 D3 JEPA-partial-FT config | pending | pending | Diagnostic evidence only; not a result claim. |
| D4 | shuffled future diagnostic | planned | 7 | nested grouped folds | V8 D4 shuffled-future config | pending | pending | Diagnostic evidence only; not a result claim. |
| confirmation | locked confirmation | planned | 7, 13, 23 | held-out grouped folds | V8 confirmation config | pending | pending | Select one seed-7 winner under the frozen rule, then confirm it with seeds 13 and 23. |
| calibration | post-selection calibration | planned | n/a | locked validation/calibration split | V8 calibration manifest | pending | pending | Follow §15 using only the selected, frozen confirmation candidate. |
| robustness | corruption evaluation | planned | n/a | held-out grouped folds | V8 robustness config | pending | pending | Follow §15; record corruption seed, intensity, and uncertainty response. |
| efficiency | latency and resource evaluation | planned | n/a | deployment benchmark set | V8 efficiency manifest | pending | pending | Follow §15; record preprocessing, inference, memory, and environment. |
| export | deployment parity | planned | n/a | n/a | V8 export manifest | pending | pending | Require signed PyTorch/ONNX parity evidence. |
| package | reproducibility package | planned | n/a | n/a | release checklist | pending | pending | Package only evidence that has passed required gates. |
