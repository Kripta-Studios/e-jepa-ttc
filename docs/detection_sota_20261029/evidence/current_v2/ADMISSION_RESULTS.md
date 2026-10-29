# Data and RAM admission evidence v2

All values below are derived from the packaged receipts and ledgers.

## Data admission v5

| Item | Value |
|---|---:|
| Status | PASS |
| Train frames | 4096 |
| Selection frames | 1024 |
| Full selection frames | 23958 |
| Per-member media hashes | 28054 |
| Exact substitutions | 1 |

Replacement at index `2063`: `OYgB6RGWcq:000000661_000000160` -> `OYgB6RGWcq:000000660_000900159`.

| Class | GT evaluable v2 (train) | Trainable legacy |
|---|---:|---:|
| bicycle | 173 | 162 |
| bus | 130 | 110 |
| car | 11194 | 10455 |
| motorcycle | 749 | 693 |
| other_vehicle | 8 | 7 |
| pedestrian | 252 | 245 |
| tricycle | 105 | 97 |
| truck | 2219 | 1994 |

Rare trainable identities: bicycle=69, other_vehicle=6.

## Original strict preflight

| Records | PASS | FAIL |
|---|---:|---:|
| Tokens (28054) | 28053 | 1 |
| TAR overall member-read status (107) | 106 | 1 |

TAR file hashes matching their pins: `107/107`. Native unreadable members: `1`.

Strict decode: `True`. Native TAR matches its pinned SHA-256: `True`.

## RAM/serialization admission

| Item | Value |
|---|---:|
| Status | PASS |
| Observed frames | 4096 |
| Full population frames | 23958 |
| Full population observed | False |
| Measured evaluator peak RSS (bytes) | 487043072 |
| Projected full evaluator RSS (bytes) | 4273160859 |
| Projected detector + evaluator RAM (bytes) | 7494386331 |
| Required RAM before loading trainer, historical admission (bytes) | 9104999067 |
| Hash-bound reference receipts | 7 |
| Serializer byte parity | True |
| Metric parity | True |

The 23,958-frame figures are projections, not observations or scientific metrics.
