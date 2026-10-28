# Compiler precision variants: measured rejection

Every row below is generated from its saved summary.json. One real primary ROI was tested at seven sensitive queries; no two-object compiled run was promoted. The fixed gate is all three TTC outputs with atol=rtol=1e-5. Intermediate features are diagnostics only. Every final PHASE output passed.

| Variant | All-three-TTC queries passed | Failed outputs | Files preserved |
|---|---:|---|---:|
| default | 5/7 | 488 R_CTX_TRAIN40: abs 0.00165939331 > allowed 0.000373525505; 505 E_CTX_TRAIN40: abs 0.000602722168 > allowed 0.000140657902 | 602 |
| no-fma | 4/7 | 488 R_CTX_TRAIN40: abs 0.00149917603 > allowed 0.000373525505; 505 E_CTX_TRAIN40: abs 0.000459671021 > allowed 0.000140657902; 533 E_CTX_TRAIN40: abs 0.000400543213 > allowed 0.000353280449 | 602 |
| no-fma-exact-div | 5/7 | 488 R_CTX_TRAIN40: abs 0.000427246094 > allowed 0.000373525505; 505 E_CTX_TRAIN40: abs 0.000293731689 > allowed 0.000140657902 | 602 |

Every variant preserved loaded weight values and passed its file snapshot. Their prepared GPU producer latencies were approximately 10 ms, but raw preparation, H2D and final context were outside that timing. These rejected variants do not establish complete-update 50 Hz.

The next native-ATen entropy-log boundary is implemented behind --native-log. It has targeted CPU validation only; no Inductor/CUDA/final-TTC admission has been run. The next driver also records feature errors separately for E_A5_MATCHED, E_C2F_MATCHED, R_A5 and R_C2F, under matched encoder reuse and batch shapes.
