# Object Event TTC v4.9 — fixed two-expert fusion

V4.8 proved that a foreground-conditioned dense temporal field generalises, and
it repaired the severe sign failure of `DGqicHUGWb`.  Its global ranking and MiD
were still weaker than the strongest encoded event expert.  V4.9 therefore tests
whether the two event-only experts are complementary before any learned gate is
introduced.

The primary prediction is predeclared:

```text
g_v49 = 0.5 * g_v42 + 0.5 * g_v48
```

The coefficient is fixed before the v4.9 execution and is not fitted by the v4.9
script.  It is nevertheless informed by the earlier development screens, so this
remains a development-validation experiment and cannot support a final benchmark
claim.  An alpha sweep is exported only as a diagnostic and may not replace the
primary result.  Both experts receive the
same event tensor; boxes, visible heights, RGB and observable motion are absent
from the fusion input.

If the fixed fusion passes, the next justified step is to reproduce v4.8 and the
same fixed fusion for seeds 13 and 23.  A learned reliability gate is not allowed
until the fixed multiseed result is stable.
