# Resultados y selección


| fit | stage | MiDc | MiDs | MiDl | MiDn | overall_MiD | mean_MiD | MAE_s | late_warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R-C05-CTX-S7 | screen | 155.846 | 53.8675 | 53.1181 | 61.2272 | 105.518 | 70.234 | 2.58797 | 0.960526 |
| R-C20-CTX-S7 | screen | 155.904 | 54.3092 | 53.0565 | 60.9944 | 105.65 | 69.9487 | 2.54712 | 0.973684 |
| S-M0-L0-S7 | screen | 153.788 | 55.5223 | 52.4788 | 61.6959 | 104.968 | 70.4329 | 2.53593 | 0.947368 |
| S-M1-L0-S7 | screen | 136.782 | 65.041 | 71.3357 | 81.0557 | 103.142 | 83.7817 | 2.93385 | 0.828947 |
| RT-M0-L0-S7 | historical_V14 | 156.832 | 54.7756 | 52.2174 | 61.3816 | 106.208 | 70.3671 | 2.51236 | 0.947368 |
| RT-M1-L0-S7 | historical_V14 | 134.47 | 63.1119 | 76.397 | 84.208 | 102.229 | 85.2914 | 2.97851 | 0.894737 |

## Gates registrados


| family | contrast | eligible | failed |
| --- | --- | --- | --- |
| R | paired | False | aggregate, crucial_10pct, crucial_groups_60pct, late_warning |
| R | historical_V14_M0 | False | crucial_10pct, late_warning |
| S | paired | False | large, mae, negative, sign, small |
| S | historical_V14_M0 | False | large, mae, negative, sign, small |


## Diferencias de los contrastes previstos


| candidate | control | delta_MiDc_pct | delta_MiDs_pct | delta_MiDl_pct | delta_MiDn_pct | delta_overall_pct | delta_MAE_pct | groups_crucial_improved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S-M0-L0-S7 | RT-M0-L0-S7 | -1.94053 | 1.36323 | 0.500517 | 0.511931 | -1.16762 | 0.937784 | 8 |
| S-M1-L0-S7 | RT-M1-L0-S7 | 1.71899 | 3.05666 | -6.62511 | -3.74337 | 0.89323 | -1.49936 | 3 |
| S-M1-L0-S7 | S-M0-L0-S7 | -11.0583 | 17.1438 | 35.9324 | 31.3795 | -1.73949 | 15.6914 | 6 |
| S-M1-L0-S7 | RT-M0-L0-S7 | -12.7842 | 18.7407 | 36.6128 | 32.0521 | -2.8868 | 16.7764 | 6 |
| R-C20-CTX-S7 | R-C05-CTX-S7 | 0.0366873 | 0.819918 | -0.115981 | -0.380171 | 0.124767 | -1.57867 | 2 |
| R-C20-CTX-S7 | RT-M0-L0-S7 | -0.591813 | -0.851498 | 1.60681 | -0.630858 | -0.526153 | 1.38326 | 6 |
| R-C05-CTX-S7 | RT-M0-L0-S7 | -0.62827 | -1.65782 | 1.72479 | -0.251644 | -0.650109 | 3.00944 | 5 |


## Bootstrap por grupos


| contrast | delta_MiDc | CI95_low | CI95_high | groups |
| --- | --- | --- | --- | --- |
| R-C05-CTX-S7_vs_RT-M0-L0-S7 | -0.985327 | -3.44056 | 1.65585 | 8 |
| R-C20-CTX-S7_vs_R-C05-CTX-S7 | 0.0571764 | -3.10499 | 2.57045 | 8 |
| R-C20-CTX-S7_vs_RT-M0-L0-S7 | -0.928151 | -3.11927 | 1.78978 | 8 |
| S-M0-L0-S7_vs_RT-M0-L0-S7 | -3.04337 | -4.03048 | -1.91609 | 8 |
| S-M1-L0-S7_vs_RT-M0-L0-S7 | -20.0497 | -33.5341 | -5.93987 | 8 |
| S-M1-L0-S7_vs_RT-M1-L0-S7 | 2.31154 | -1.01644 | 5.3416 | 8 |
| S-M1-L0-S7_vs_S-M0-L0-S7 | -17.0064 | -30.142 | -3.44893 | 8 |

Los candidatos prospectivos son S-M1 y R-C20-CTX. Se aplican los gates V14 frente a su control pareado y al M0/L0 histórico. S-M0 y R-C05 no se promocionan a posteriori. Un intervalo favorable no es un gate adicional.

Umbrales: MiDc ≤90 % del control y mejora en ≥60 % de sus grupos; overall sin aumento; MiDs/MiDl/MiDn como máximo control + max(2 puntos, 5 %); MAE ≤102 %; error de signo como máximo +0,5 puntos porcentuales; p95 de sobreestimación crucial ≤105 %; aviso tardío y falsa alerta como máximo +1 punto porcentual, con denominadores ≥30. Se exige cobertura completa. Cada candidato debe pasar frente a su pareja y al control histórico M0.

MiD por banda promedia primero dentro de cada grupo y después entre grupos. Overall = 0,5 MiDc + 0,3 MiDs + 0,1 MiDl + 0,1 MiDn; permanece ausente si falta una banda. mean_MiD y MAE son medias por consulta. El bootstrap agrupa por escenario: 1.000 draws, seed12345, descriptivo ante múltiples contrastes.

El bootstrap conserva la lectura decimal CSV de la ruta V14. La reproducción exacta de métricas recupera primero los bits FP32 de la salida nativa; ambos procedimientos quedan identificados, sin cambiar predicciones ni gates.

Las tablas completas por consulta y grupo están en evaluation/. Los siete contrastes iniciales se conservan en bootstrap/. Si hay réplica, se promedian pérdidas por consulta sobre seeds7/13/23; no se construye un ensemble TTC. El p95 agregado usa el error positivo medio por consulta antes de su cuantil.

V es desarrollo interno expuesto; no es test oficial ni confirmación externa. La comparación histórica tiene seed7 y no se inventan sus seeds13/23.
