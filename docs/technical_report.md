# Technical Report

Current thesis-style technical report:

- [E-JEPA-TTC paper](e_jepa_ttc_paper.md)

The old scaffold has been replaced by the paper because reproducible
experiments now exist. The paper records:

1. dataset and split protocol;
2. model architecture;
3. JEPA/self-supervised objective;
4. anti-leakage controls;
5. full-starter all-window results;
6. bbox/ROI diagnostic results;
7. SkyJEPA-style latent and rollout prober ablations;
8. final verification commands;
9. official SOTA claim limits;
10. next work required for a real EvTTC benchmark claim.

Final local claim:

> Tubelet-masked event-token JEPA with a dense transformer predictor and causal
> integrated-navigation conditioning is the strongest local all-window result in
> this repository: `0.231 +/- 0.018 s` validation MAE and `0.312 +/- 0.044 s`
> diagnostic CPLA-high MAE over seeds 7/13/21.

This is not an official EvTTC SOTA claim.
