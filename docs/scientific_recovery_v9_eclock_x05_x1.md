# E-Clock X0.5 → X1 preregistered continuation

This continuation is bound to source commit
`57865bea943f7c1518341003170cec1c221aa093` and to X0 bundle SHA-256
`b8b228a6f0ea039238cf96d046228759d1a64f65fdda02b16353d63450beb7f9`.

X0.5 replays the signed A5, BASE-U and DYN-U outer-fold evidence, requires exact
phase equality for BASE-U/DYN-U, and exports the nine `m12` transport slots before
any control zeroing. A float64 weighted ridge is selected by leave-one-sequence-out
inside each meta-train partition. Its outer meta-test is never available to fitting,
normalization or selection. DYN9 must beat both CAL and the target-blind,
within-sequence SHUFFLE9 control with the frozen confidence gates. It must also beat
raw A5 by at least 1 MiD before X1 can be authorized.

X1 is fully defined even if it is never authorized. It uses a zero-initialized
9→32→32→1 residual adapter over frozen A5 phase and frozen transport slots. ZERO-U,
DYN-U and SHUFFLE-U share topology, initialization, normalization, batch schedule,
optimizer and the fixed 1,000-update budget. The update-1,000 checkpoint is the only
scientific checkpoint, and outer-dev is evaluated once after freezing. Seeds 13 and
23 are conditional on seed 7 passing the primary, shuffle and practical-utility
gates simultaneously.

Public validation, private test, EvTTC test and CodaBench remain sealed. X0-DYN-W,
X2, X3, Track M, recurrent-depth variants and post-hoc sweeps are outside this
protocol. All scientific metrics are regenerated from signed CSV/JSON artifacts;
Markdown is descriptive only.
