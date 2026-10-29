# Diagnósticos

## Gradientes iniciales H

| mass | sampling | batches | seed | batch_size | trace_sample_covariance | mean_squared_error_to_exact_H_gradient | exact_H_gradient_l2 | mean_batch_gradient_l2 | clip_fraction_at_initialization | observed_zero_rare_batches | probability_zero_rare_batch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | uniform | 32 | 12345 | 128 | 0.0334359 | 0.0329578 | 0.156638 | 0.215908 | 0 | 23 | 0.696409 |
| M0 | importance | 32 | 12345 | 128 | 0.0359779 | 0.0372335 | 0.156638 | 0.244437 | 0 | 22 | 0.66858 |
| M1 | uniform | 32 | 12345 | 128 | 0.131452 | 0.128389 | 0.543471 | 0.588732 | 0.15625 | 23 | 0.696409 |
| M1 | importance | 32 | 12345 | 128 | 0.0229633 | 0.0226371 | 0.543471 | 0.561316 | 0 | 7 | 0.173972 |

## Clipping durante S

| fit | updates | fraction_clipped | last250_fraction_clipped |
| --- | --- | --- | --- |
| S-M0-L0-S7 | 2500 | 0.1732 | 0.128 |
| RT-M0-L0-S7 | 2500 | 0.1632 | 0.112 |
| S-M1-L0-S7 | 2500 | 0.1884 | 0.2 |
| RT-M1-L0-S7 | 2500 | 0.3716 | 0.424 |

## Clipping de todos los fits V15

| fit | updates | fraction_clipped | last250_fraction_clipped | gradient_norm_p95 | gradient_norm_max |
| --- | --- | --- | --- | --- | --- |
| R-C05-CTX-S7 | 2500 | 0.2072 | 0.128 | 1.45698 | 2.87775 |
| R-C05-S7 | 6000 | 0.9995 | 1 | 4.79835 | 73.3637 |
| R-C20-CTX-S7 | 2500 | 0.1872 | 0.152 | 1.42903 | 2.43651 |
| R-C20-S7 | 6000 | 1 | 1 | 8.33335 | 75.567 |
| S-M0-L0-S7 | 2500 | 0.1732 | 0.128 | 1.36979 | 2.44948 |
| S-M1-L0-S7 | 2500 | 0.1884 | 0.2 | 1.33723 | 2.4059 |

## Alcance condicional P/H

| role | producer | band | queries | mean_residual_over_cap | overestimate_count | unavailable_bounds | mean_prediction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H | R-C05-S7 | 0.1-1 | 52 | 0.787647 | 52 | 0 | 1.91435 |
| H | R-C05-S7 | 1-2 | 524 | 0.722075 | 332 | 0 | 2.49628 |
| H | R-C05-S7 | 2-3 | 1251 | 0.616832 | 2 | 0 | 2.91507 |
| H | R-C05-S7 | negative | 7626 | -0.289356 | 0 | 0 | -7.32728 |
| H | R-C05-S7 | other_positive | 8969 | 0.337935 | 0 | 0 | 6.26179 |
| H | R-C20-S7 | 0.1-1 | 52 | 0.213314 | 0 | 0 | 3.02849 |
| H | R-C20-S7 | 1-2 | 524 | 0.187969 | 0 | 0 | 2.51327 |
| H | R-C20-S7 | 2-3 | 1251 | 0.15463 | 0 | 0 | 3.13788 |
| H | R-C20-S7 | negative | 7626 | -0.0709286 | 0 | 0 | -7.60367 |
| H | R-C20-S7 | other_positive | 8969 | 0.0823132 | 0 | 0 | 6.51336 |
| H | historical_A5 | 0.1-1 | 52 | 0.753065 | 52 | 0 | 2.25729 |
| H | historical_A5 | 1-2 | 524 | 0.701256 | 332 | 0 | 2.59355 |
| H | historical_A5 | 2-3 | 1251 | 0.605547 | 2 | 0 | 2.95869 |
| H | historical_A5 | negative | 7626 | -0.276212 | 0 | 0 | -7.60842 |
| H | historical_A5 | other_positive | 8969 | 0.324488 | 0 | 0 | 6.5841 |
| P | R-C05-S7 | 0.1-1 | 208 | 0.902466 | 208 | 0 | 2.00519 |
| P | R-C05-S7 | 1-2 | 2038 | 0.818036 | 1409 | 0 | 2.41799 |
| P | R-C05-S7 | 2-3 | 3823 | 0.639355 | 12 | 0 | 3.30106 |
| P | R-C05-S7 | negative | 22397 | -0.315845 | 0 | 0 | -7.38783 |
| P | R-C05-S7 | other_positive | 25424 | 0.338789 | 0 | 0 | 6.52408 |
| P | R-C20-S7 | 0.1-1 | 208 | 0.301066 | 0 | 0 | 1.97814 |
| P | R-C20-S7 | 1-2 | 2038 | 0.221831 | 0 | 0 | 2.36618 |
| P | R-C20-S7 | 2-3 | 3823 | 0.158145 | 0 | 0 | 3.32108 |
| P | R-C20-S7 | negative | 22397 | -0.0765355 | 0 | 0 | -7.53278 |
| P | R-C20-S7 | other_positive | 25424 | 0.0817577 | 0 | 0 | 6.76137 |
| P | historical_A5 | 0.1-1 | 208 | 0.884566 | 208 | 0 | 2.41996 |
| P | historical_A5 | 1-2 | 2038 | 0.792553 | 1409 | 0 | 2.42751 |
| P | historical_A5 | 2-3 | 3823 | 0.619801 | 12 | 0 | 3.25562 |
| P | historical_A5 | negative | 22397 | -0.298473 | 0 | 0 | -7.85013 |
| P | historical_A5 | other_positive | 25424 | 0.322855 | 0 | 0 | 6.9182 |

## Predicciones del productor en TTC ≤1 segundo

| role | producer | queries | target_mean_s | prediction_mean_s | prediction_median_s | prediction_p95_s | mae_s | absolute_error_p95_s | late_warning_rate | sign_error_rate | known_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H | R-C05-S7 | 52 | 0.884656 | 1.91435 | 1.90437 | 4.27016 | 2.08662 | 7.47273 | 1 | 0.0576923 | 1 |
| H | R-C20-S7 | 52 | 0.884656 | 3.02849 | 1.9451 | 7.50753 | 3.70329 | 14.8987 | 0.942308 | 0.0576923 | 1 |
| H | historical_A5 | 52 | 0.884656 | 2.25729 | 1.90386 | 5.23068 | 2.3501 | 9.7309 | 1 | 0.0576923 | 1 |
| P | R-C05-S7 | 208 | 0.846636 | 2.00519 | 1.70639 | 3.73767 | 1.86352 | 3.65797 | 1 | 0.0192308 | 0.990385 |
| P | R-C20-S7 | 208 | 0.846636 | 1.97814 | 1.29962 | 4.88913 | 1.56894 | 5.51053 | 0.759615 | 0.0192308 | 0.995192 |
| P | historical_A5 | 208 | 0.846636 | 2.41996 | 1.72918 | 4.27011 | 1.9101 | 4.90638 | 1 | 0.0192308 | 0.995192 |

La varianza se estima con 32 batches de diagnóstico y se contrasta con el gradiente exacto de todo H en la inicialización seed7. No garantiza menor varianza para todos los parámetros o durante todo el entrenamiento. La igualdad de riesgo esperado no implica igualdad tras clipping y AdamW.

La masa no se normaliza dentro del batch. El sampler usa probabilidades globales float64 y la corrección w/p. El cambio no crea escenas independientes. SAMPLER_EXPOSURE.csv conserva los draws por grupo y rango.

El alcance del productor mantiene la geometría fija y permite residuales máximos independientes en ambas parejas. Se marca como no disponible cuando se cruza un polo del transporte. No es un límite absoluto de PAIR/H8 ni de un encoder distinto. Se conservan predicciones, known, soporte, edades y correcciones por consulta en PRODUCER_REACH_P_H.csv.

EVENT_CROP_ROWS_P_H.csv y EVENT_CROP_SUMMARY.csv usan las coordenadas reales del índice de eventos congelado. La fuente identifica secuencias y tracks, pero no episodios independientes; estos últimos permanecen sin inventar.

PRODUCER_METRICS_P_H.json compara A5 original y ambos readouts sobre P/H. P es entrenamiento de readout; H entrena las cabezas. Estas cifras describen el mecanismo y no son selección ni validación independiente. Las normas de gradiente R proceden de un objetivo distinto del de S; sus valores absolutos no son comparables como una única pérdida.

![Muestreo S frente a sus referencias V14](../../artifacts/range_readout_20261012/figures/S_SAMPLING_CURVES.png)

![Readouts A5 sobre P](../../artifacts/range_readout_20261012/figures/R_READOUT_CURVES.png)

![Refinadores sobre H](../../artifacts/range_readout_20261012/figures/R_CONTEXT_CURVES.png)
