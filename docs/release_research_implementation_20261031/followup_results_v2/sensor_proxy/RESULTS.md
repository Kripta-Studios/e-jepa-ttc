# S1 exploratorio con referencia RGB aproximada

72/149 consultas con celdas admitidas; 24 adquisiciones; 96 ajustes.
Sin targets TTC. R0 tiene exposición histórica; la calibración se ajusta excluyendo la adquisición evaluada.

| Control | MSE macro de tasa firmada | MAE macro de tasa firmada |
|---|---:|---:|
| activity | 137.794593 | 5.934052 |
| texture | 146.001947 | 6.286781 |
| physical | 146.918881 | 6.296040 |
| permuted | 147.320948 | 6.274677 |

| Físico menos control | Delta MSE | CI95 por adquisición |
|---|---:|---|
| activity | 9.124289 | [-12.248288782116068, 30.07257101908819] |
| texture | 0.916934 | [-0.06636549985347871, 2.1608020101775463] |
| permuted | -0.402066 | [-1.0255128396547473, 0.23385978563967794] |

La escala ajustada no equivale al umbral físico del sensor.
RGB/ISP, umbrales asimétricos, paralaje y textura de referencia pueden dominar el residuo.
Slider se conserva como no admitido por alineación provisional; ninguna consulta desaparece del inventario.
Las celdas silenciosas no imponen q=0 ni una pérdida de silencio.
Este resultado no mide mejora TTC ni abre S2 automáticamente.
