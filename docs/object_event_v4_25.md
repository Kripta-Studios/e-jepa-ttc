# Object Event v4.25 — anchored geometry-conditioned TTC readout

v4.24 exhausted the tested fine-tuning schedules: grouped train OOF looked excellent, but every schedule still lost TTC sign generalization on the development sequences. At the same time the learned dense geometry continued to improve and remained transferable.

v4.25 therefore freezes the v4.24 champion representations and asks one narrow question: **can the stable v4.10 expansion anchor and the learned divergence/vertical-scale proxies be combined by a low-capacity, physically monotone readout?**

The readout has no hidden layers, no bias and no sign router. Coefficients are constrained non-negative. Geometry scores are calibrated to expansion units using train only. The geometry side is first rebuilt as 3-seed x 3-fold grouped OOF features, so each train row is scored by a representation that did not train on its sequence. Candidate feature sets/ridge strengths are then selected by a second grouped leave-sequence-out meta-CV on train; development validation is evaluated once after selection.

This is a terminal diagnostic for post-hoc readout design. If it fails, the next model must train an explicit LHR/geometry-conditioned TTC head rather than add another router or threshold.
