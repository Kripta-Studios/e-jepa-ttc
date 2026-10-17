# V19: descomposición del error negativo L0/LW

Análisis descriptivo de predicciones congeladas completas P/H. Cero fits, updates o inferencias nuevas. H es desarrollo expuesto y no selecciona ningún modelo aquí.

Los IDs, targets, roles y orden coinciden exactamente con las filas fuente. Los relojes se admiten mediante los mismos hashes de manifest, filas y sensor en ambos recibos; no están duplicados en los parquet de predicciones ni se reproduce el sensor crudo.

Las cuatro celdas particionan todos los targets negativos. MAE, sesgo y MiD de celda son promedios pooled. La contribución MiDn usa 1/(número de grupos × negativos del grupo), por lo que las cuatro contribuciones suman la métrica negativa nativa. Saturación se calcula con la conversión FP32 original, no con igualdad decimal a 60.

## P: 53,890 consultas completas

Negativos: 22,397; grupos: 24.

| Celda | n | MAE L0/LW s | MiD pooled L0/LW | Sesgo L0/LW s | Saturados L0/LW | Δ suma error absoluto s | Δ contribución MiDn |
|---|---:|---:|---:|---:|---:|---:|---:|
| both_correct | 19102 | 7.640/3.654 | 70.660/53.136 | -7.117/-2.889 | 844/117 | -76147.292 | -15.158047 |
| L0_only_correct | 35 | 45.063/59.384 | 137.343/182.626 | -45.063/59.384 | 24/24 | 501.217 | 0.068990 |
| LW_only_correct | 1296 | 50.080/31.913 | 200.264/129.830 | 50.080/-31.906 | 556/428 | -23544.295 | -3.596481 |
| both_wrong | 1964 | 22.735/34.583 | 276.358/239.254 | 22.735/34.583 | 65/378 | 23268.060 | -3.079497 |

| Total negativo | L0 | LW |
|---|---:|---:|
| mae_s | 11.478004 | 8.088161 |
| bias_s | -1.248681 | -1.185128 |
| absolute_error_sum_s | 257072.856815 | 181150.546836 |
| mean_MiD | 96.301772 | 74.097228 |
| group_macro_negative_MiD_contribution | 93.546748 | 71.781713 |
| sign_error_count | 3260.000000 | 1999.000000 |
| saturated_count | 1489.000000 | 947.000000 |

La contribución de ambos signos correctos es -15.158 MiDn (69.6% del cambio neto): la mejora incluye cambios de magnitud sin recuperar signo. Hay 1296 correcciones de signo y 35 regresiones.

En los 1964 casos de signo incorrecto compartido, el MAE pasa de 22.735 a 34.583 s y el MiD pooled de 276.358 a 239.254. No son correcciones de signo. La transformación de fase comprime diferencias cuando TTC se aleja hacia infinito; por eso MiD puede mejorar mientras el error en segundos empeora. Las saturaciones correspondientes figuran en la tabla.

## H: 18,422 consultas completas

Negativos: 7,626; grupos: 8.

| Celda | n | MAE L0/LW s | MiD pooled L0/LW | Sesgo L0/LW s | Saturados L0/LW | Δ suma error absoluto s | Δ contribución MiDn |
|---|---:|---:|---:|---:|---:|---:|---:|
| both_correct | 6686 | 6.603/3.589 | 60.703/49.956 | -5.918/-2.653 | 260/44 | -20149.351 | -10.465183 |
| L0_only_correct | 15 | 50.816/64.185 | 124.104/161.060 | -50.816/64.185 | 14/12 | 200.547 | 0.071545 |
| LW_only_correct | 355 | 50.311/33.697 | 180.394/115.089 | 50.311/-33.697 | 148/129 | -5897.760 | -3.444053 |
| both_wrong | 570 | 23.744/33.763 | 268.827/238.517 | 23.744/33.763 | 34/102 | 5710.378 | -2.474052 |

| Total negativo | L0 | LW |
|---|---:|---:|
| mae_s | 10.005777 | 7.365311 |
| bias_s | -1.171484 | -1.244940 |
| absolute_error_sum_s | 76304.052022 | 56167.865094 |
| mean_MiD | 81.955821 | 67.300263 |
| group_macro_negative_MiD_contribution | 87.087086 | 70.775344 |
| sign_error_count | 925.000000 | 585.000000 |
| saturated_count | 456.000000 | 287.000000 |

La contribución de ambos signos correctos es -10.465 MiDn (64.2% del cambio neto): la mejora incluye cambios de magnitud sin recuperar signo. Hay 355 correcciones de signo y 15 regresiones.

En los 570 casos de signo incorrecto compartido, el MAE pasa de 23.744 a 33.763 s y el MiD pooled de 268.827 a 238.517. No son correcciones de signo. La transformación de fase comprime diferencias cuando TTC se aleja hacia infinito; por eso MiD puede mejorar mientras el error en segundos empeora. Las saturaciones correspondientes figuran en la tabla.

## Interpretación y límites

La celda both_correct compara magnitudes conservando signo correcto. LW_only_correct recoge correcciones de signo; L0_only_correct recoge regresiones. both_wrong conserva errores de signo. Estas categorías describen dónde cambia el error: una celda que cambia de signo también puede cambiar de magnitud, por lo que no estima un efecto causal puro del signo. No se crea ningún vector contrafactual ni selector por target.

La mejora negativa no revoca el empeoramiento crítico/agregado ni habilita por sí sola replicación o promoción. La partición se aplica idénticamente a P y H, sin buscar umbrales o coeficientes. La variación entre entrenamientos no se estima con una semilla.

Evidencia: `artifacts/decision_recovery_20261017/loss_negative_analysis/REPORT.json`, CSV de todos los negativos P/H y `RECEIPT.json` con hashes de entradas, código y salidas.

Reproducción CPU: `python -B -m operational.decision_recovery.loss_negative_analysis`. Coordinar la ventana con el propietario GPU antes de iniciarla.
