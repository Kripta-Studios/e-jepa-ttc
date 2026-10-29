# V18 RGB P2: fixed endpoints and development result

Regenerate: `python -B -m operational.joint_measurement.rgb_report`.

Frozen decision: **NO_CANDIDATE_RGB_PAIR**. No system promotion or official claim.
Two seed-7 fits completed 12,000 updates each before H evaluation. P is training;
H contains eight held-out groups. Both routes use two RGB observations with their
own sensor clocks. The native P2 reference has a different objective and architecture.

| Role | Model | Rows | Aggregate | MiDc | Negative MiD | MAE (s) | Sign errors |
|---|---|---:|---:|---:|---:|---:|---:|
| P | R_A5-P2-FROZEN | 53890 | 89.994828 | 131.282987 | 42.885214 | 2.846627 | 2.0653% |
| P | RGB-PAIR-CORR-FT-S7 | 53890 | 50.526252 | 46.537213 | 100.205463 | 5.246288 | 7.4003% |
| P | RGB-PAIR-CORR-FROZEN-S7 | 53890 | 50.982027 | 46.414485 | 97.193858 | 5.224566 | 7.0403% |
| H | R_A5-P2-FROZEN | 18422 | 113.910110 | 167.601681 | 52.966741 | 3.819361 | 3.9952% |
| H | RGB-PAIR-CORR-FT-S7 | 18422 | 90.909602 | 115.950679 | 105.633376 | 5.437178 | 7.6647% |
| H | RGB-PAIR-CORR-FROZEN-S7 | 18422 | 97.852413 | 126.815853 | 102.215201 | 5.331355 | 6.8396% |

The aggregate and crucial-band gains coexist with worse negative TTC and sign
errors, including on P. This is not solely a held-out transfer failure. Both
variants fail the registered band and sign checks. Retain the positive signal
without selecting either endpoint for V, replication or refit under these gates.

Paired cost covers prepared tensors resident on the GPU through TTC emission;
it excludes sensor loading and preprocessing. It is not full-pipeline latency.
Six prediction tables, NumPy metric verification, group bootstraps and three
portable exports are recorded in `rgb/evidence/COMPLETE.json`. Exact endpoint
verification separately confirms encoder changes for FT and unchanged encoder
weights for FROZEN. The historical R_CTX/F_TRUE systems remain comparators.

C1 is an independently registered event geometry experiment enabled by the
earlier real-temporal correspondence diagnostic. Its execution does not promote
these rejected RGB endpoints. RGB-anchor fusion remains a separate prospective
experiment; using historical H-trained heads must be labelled sequential supervision.
