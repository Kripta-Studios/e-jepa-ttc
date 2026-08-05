# Object Event TTC v4.2 — full event-only screen

V4.2 is the full-data successor to the passed v4.1 learnability diagnostic. It
uses all screen-cache rows but retains the strict event-only contract:

- input: common-coordinate real-event `t0/t1/t2`, 12 active channels;
- no observable motion, boxes, heights, geometry prior, Level transfer or fusion;
- direct signed expansion `g = delta_t / TTC` training;
- encoded spatial-temporal branch only; the failed global activity branch is frozen;
- checkpoint selection only on held-out validation sequences;
- repeated global event shuffles rather than one batch-local roll;
- sequence-level, bootstrap, sign, saturation and event-dependence gates.

Passing v4.2 justifies seed repeats. It does not yet justify motion fusion or a
claim against Garl-TTC.
