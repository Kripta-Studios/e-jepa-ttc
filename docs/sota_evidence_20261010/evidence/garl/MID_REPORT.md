# MiD con el evaluador oficial de GarlTTC

Menor es mejor. Se ejecuta el scorer CodaBench conservado y verificado por SHA-256 sobre predicciones y etiquetas locales. No son puntuaciones oficiales de test12.

| Datos | Método | MiD medio | MiDc | MiDs | MiDl | MiDn | overall_MiD | FR (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DEV32 | H8_median3 | 138.377 | 213.223 | 55.043 | 32.382 | N/D | N/D | 0.000 |
| DEV32 | Direct_median3 | 137.323 | 209.002 | 57.124 | 31.872 | N/D | N/D | 0.027 |
| DEV32 | public_Garl_event_lhr | 165.984 | 249.314 | 74.536 | 49.071 | N/D | N/D | 0.000 |
| DEV32 | public_Garl_rgb_event_full | 136.336 | 197.576 | 65.891 | 61.405 | N/D | N/D | 0.000 |
| FCWD | H8_median3 | 79.952 | 177.938 | 25.835 | 17.638 | N/D | N/D | 0.000 |
| FCWD | Direct_median3 | 84.458 | 183.276 | 24.634 | 19.280 | N/D | N/D | 0.000 |
| FCWD | public_Garl_event_lhr | 61.817 | 95.057 | 36.698 | 34.212 | N/D | N/D | 0.000 |
| FCWD | public_Garl_rgb_event_full | N/D | N/D | N/D | N/D | N/D | N/D | N/D |

MiD = 10⁴ · |log(1 − 0,1/GT) − log(1 − 0,1/predicción)|.

`overall_MiD = 0,5 MiDc + 0,3 MiDs + 0,1 MiDl + 0,1 MiDn`. Dev32 y FCWD no contienen GT negativos: MiDn y overall_MiD quedan N/D. No se redistribuye el peso de esa banda.

El scorer calcula medias de MiD finitos. El CSV añade el número de MiD inválidos por banda y una media estricta que queda N/D ante cualquier MiD inválido. Las medias condicionales no eliminan esos fallos.

MiD medio pondera muestras y puede incluir GT fuera de las cuatro bandas; no equivale a overall_MiD. FR oficial cuenta TTC no finitos o de magnitud <0,1 s, y no equivale a la tasa de avisos urgentes fallidos.

En TTC largos, diferencias grandes en segundos pueden producir diferencias pequeñas en la razón de alturas. Por eso ordenar modelos por MAE y por MiD puede dar resultados distintos. No se debe concluir superioridad en MiD a partir del MAE.

Diferencia de MiD medio: H8 menos comparador; un valor negativo favorece a H8. IC percentil del 95 % con 10.000 remuestreos pareados de secuencias completas. La media conserva la ponderación por muestras; no se remuestrean ventanas independientes. FCWD solo tiene tres secuencias y su intervalo es especialmente frágil.

| Datos | Comparador | Δ MiD | IC 95 % | Secuencias |
|---|---|---:|---|---:|
| DEV32 | public_Garl_event_lhr | -27.607 | [-43.689, -11.634] | 32 |
| DEV32 | public_Garl_rgb_event_full | 2.041 | [-14.335, 18.800] | 32 |
| FCWD | public_Garl_event_lhr | 18.135 | [15.166, 23.171] | 3 |

Casos con mayor MiD: `MID_LARGEST_ERRORS.csv`. Los intervalos exploratorios no corrigen la exposición previa de estas cohortes.
