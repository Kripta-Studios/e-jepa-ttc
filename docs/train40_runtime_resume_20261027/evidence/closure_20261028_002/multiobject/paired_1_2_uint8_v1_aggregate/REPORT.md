# TRAIN40: paired PIL / uint8, one and two objects

All reported times are complete updates in milliseconds, including current ROI work, transfers, producers, context and all host TTC outputs. Sample counts appear in n; quantiles use linear interpolation. The second ROI is artificial, not an annotated track.

| Mode | Objects | Index | n | Mean | Median | p95 | p99 | Max | Parity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PIL | 1 | 0 | 10 | 26.197 | 25.620 | 31.588 | 32.216 | 32.373 | PASS |
| PIL | 1 | 137 | 10 | 26.082 | 24.704 | 35.371 | 40.694 | 42.025 | PASS |
| PIL | 1 | 553 | 10 | 72.785 | 79.312 | 100.756 | 101.153 | 101.252 | PASS |
| PIL | 2 | 0 | 10 | 62.001 | 55.210 | 92.390 | 94.150 | 94.590 | PASS |
| PIL | 2 | 137 | 10 | 49.298 | 48.955 | 59.357 | 60.703 | 61.039 | PASS |
| PIL | 2 | 553 | 10 | 127.634 | 133.765 | 144.535 | 146.729 | 147.278 | PASS |
| uint8 | 1 | 0 | 10 | 35.712 | 29.563 | 62.669 | 67.260 | 68.407 | PASS |
| uint8 | 1 | 137 | 10 | 25.052 | 24.784 | 28.810 | 28.902 | 28.926 | PASS |
| uint8 | 1 | 553 | 10 | 77.668 | 80.765 | 85.784 | 85.808 | 85.814 | PASS |
| uint8 | 2 | 0 | 10 | 56.759 | 53.856 | 76.224 | 79.483 | 80.298 | PASS |
| uint8 | 2 | 137 | 10 | 49.139 | 47.830 | 63.541 | 64.825 | 65.147 | PASS |
| uint8 | 2 | 553 | 10 | 157.257 | 135.838 | 251.100 | 287.012 | 295.990 | PASS |

The JSON retains every sample, first-repeat and later-repeat distributions, actual first updates per object count, startup receipts, and hashes of all input evidence.

PIL ROI-first remains the conservative default. Uint8 is numerically admitted for this pilot but does not show a consistent latency benefit. Neither route demonstrates 50 Hz or under 20 ms. ABBA timing drift and host commit pressure prevent a clean causal speedup claim; the independent parity checks remain valid.
