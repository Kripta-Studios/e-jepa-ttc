# Localización de deriva en el original nativo

Comparación sin targets y sin ejecutar el candidato; tolerancias intactas.

| Modalidad | Campo | Máximo Δ crudo | Máximo Δ normalizado |
|---|---|---:|---:|
| event | event_count_log1p_mean | 5.7220459e-06 | 4.29153442e-06 |
| event | event_rate_log1p_mean | 4.76837158e-06 | 3.57627869e-06 |
| event | event_a5_flow | 3.61353159e-07 | 4.8995018e-05 |
| event | event_a5_margin | 0.000919342041 | 0.000173807144 |
| event | event_a5_log_variance | 0.000215530396 | 6.66379929e-05 |
| event | event_c2f_flow | 3.96743417e-07 | 5.30779362e-05 |
| event | event_c2f_margin | 0.00154876709 | 0.000294923782 |
| event | event_c2f_log_variance | 0.000282287598 | 8.86917114e-05 |
| event | event_a5_phase | 1.66893005e-06 | 8.10623169e-05 |
| event | event_c2f_phase | 2.8014183e-06 | 0.000136375427 |
| event | event_pair_phase | 1.55717134e-06 | 7.22408295e-05 |
| event | event_pair_minus_a5 | 3.2261014e-06 | 0.000564038754 |
| event | event_pair_minus_c2f | 4.35858965e-06 | 0.000642120838 |
| event | event_c2f_minus_a5 | 1.13248825e-06 | 0.000250771642 |
| event | event_abs_pair_minus_a5 | 3.2261014e-06 | 0.000715851784 |
| event | event_abs_pair_minus_c2f | 4.35858965e-06 | 0.00082463026 |
| event | event_abs_c2f_minus_a5 | 1.13248825e-06 | 0.00032389164 |
| rgb | rgb_luminance_mean | 0 | 0 |
| rgb | rgb_spatial_gradient_log1p | 0 | 0 |
| rgb | rgb_a5_flow | 1.80676579e-07 | 1.99079514e-05 |
| rgb | rgb_a5_margin | 0.00242042542 | 0.000408172607 |
| rgb | rgb_a5_log_variance | 0.000425338745 | 0.00014424324 |
| rgb | rgb_c2f_flow | 1.46776438e-06 | 0.000120460987 |
| rgb | rgb_c2f_margin | 0.00137710571 | 0.000233411789 |
| rgb | rgb_c2f_log_variance | 0.000348091125 | 0.000122666359 |
| rgb | rgb_a5_phase | 4.39211726e-06 | 0.000199317932 |
| rgb | rgb_c2f_phase | 2.48476863e-06 | 0.000111579895 |
| rgb | rgb_pair_phase | 4.82425094e-06 | 0.000202417374 |
| rgb | rgb_pair_minus_a5 | 1.83656812e-06 | 0.000214606524 |
| rgb | rgb_pair_minus_c2f | 2.33948231e-06 | 0.000236168504 |
| rgb | rgb_c2f_minus_a5 | 3.94508243e-06 | 0.000595092773 |
| rgb | rgb_abs_pair_minus_a5 | 1.83656812e-06 | 0.00024241209 |
| rgb | rgb_abs_pair_minus_c2f | 2.33948231e-06 | 0.000280529261 |
| rgb | rgb_abs_c2f_minus_a5 | 3.94508243e-06 | 0.000725448132 |

La normalización amplifica diferencias según su escala congelada.
Este diagnóstico localiza el primer bloque observado que difiere,
pero no identifica por sí solo el kernel, dispositivo o política numérica causante.
No prueba un fallo del candidato ni autoriza relajar la paridad histórica.
