# Frozen EvTTC comparison with full RGB+event Garl

Previous event-only results are preserved; all three H8 heads are reused exactly. No training or tuning.

## Accuracy on identical common finite support

| Model | Labeled | Finite | MAE s | Median AE s | RMSE s |
|---|---:|---:|---:|---:|---:|
| H8_seed7 | 3640 | 3640 | 1.361283924691238 | 0.7218486548063361 | 3.4753928964291 |
| H8_seed13 | 3640 | 3640 | 1.394969588613207 | 0.6992462247615228 | 3.6177486367623883 |
| H8_seed23 | 3640 | 3640 | 1.456491555929561 | 0.7020513066353757 | 4.003137581802338 |
| public_Garl_event_lhr | 3640 | 3640 | 4.55971165260342 | 0.9578366008360603 | 83.89608032982709 |
| public_Garl_rgb_event_full | 3640 | 3640 | 94.509601884153 | 0.6614158970779118 | 5561.621766218054 |

Common finite support across all models: 3640 queries.

## Paired macro-sequence differences versus full Garl

- H8_seed7: -89.863998 s, 95% sequence-bootstrap CI [-268.08810846541684, -0.3014353738901018].
- H8_seed13: -89.833573 s, 95% sequence-bootstrap CI [-267.99693202769936, -0.2989799089350109].
- H8_seed23: -89.757229 s, 95% sequence-bootstrap CI [-267.89237581847965, -0.26496075810714026].
- public_Garl_event_lhr: -86.455690 s, 95% sequence-bootstrap CI [-265.80933660180125, 5.024164616225024].

## Coverage on the original population

- H8_seed7: 3640/3640 labeled queries; coverage 1.0.
- H8_seed13: 3640/3640 labeled queries; coverage 1.0.
- H8_seed23: 3640/3640 labeled queries; coverage 1.0.
- public_Garl_event_lhr: 3640/3640 labeled queries; coverage 1.0.
- public_Garl_rgb_event_full: 3640/3640 labeled queries; coverage 1.0.

## Limitations

- Historical development cohort and follow-up after seeing event-only results; not blind model selection.
- H8 uses events plus oracle ROI metadata; full Garl additionally receives RGB pixels.
- Native frozen temporal contexts differ across models.
- RGB/event timing and geometry approximations are declared in the frozen input contract.
- H8 native TTC support +/-60 seconds; Garl native conversion unbounded, no clipping.
- Public Garl training/selection ancestry not independently certified.
- Sampled cross-dataset transfer comparison, not a reproduction of the official paper benchmark.
