# Object Event TTC v4.3 — multiseed robustness gate

V4.2 established a real event-only signal on held-out sequences for seed 7:
validation Pearson 0.5607, balanced sign 0.7370, and shuffled-event Pearson near
zero.  It also exposed a sequence-specific sign failure: `DGqicHUGWb` retained
positive Pearson but classified only 1/28 negative examples correctly.

V4.3 therefore does **not** add motion, boxes, Level transfer, TTC-domain losses,
or a new architecture.  It repeats the exact v4.2 protocol with seeds 7/13/23
and aggregates:

- per-seed mean, standard deviation, and worst seed;
- equal-weight prediction ensemble;
- pairwise seed agreement and per-sample prediction dispersion;
- track-cluster bootstrap confidence intervals;
- per-sequence Pearson, balanced sign, and negative accuracy;
- fail-closed gates that prevent a global mean from hiding a fragile sequence.

Only a robust pass permits the next experiment: grouped sequence CV.  Motion or
Level transfer remain prohibited until event-only robustness survives that CV.
