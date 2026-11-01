# S1 exploratorio con referencia RGB aproximada

72/149 consultas con celdas admitidas; 24 adquisiciones; 120 ajustes.
Sin targets TTC. R0 tiene exposición histórica; la calibración se ajusta excluyendo la adquisición evaluada.

| Control | MSE macro de tasa firmada | MAE macro de tasa firmada |
|---|---:|---:|
| activity | 136.458609 | 5.895193 |
| texture | 144.744301 | 6.249378 |
| physical | 145.648563 | 6.260611 |
| permuted | 146.016965 | 6.240431 |
| rgb_observed | 96.657290 | 4.458287 |

| Físico menos control | Delta MSE | CI95 por adquisición |
|---|---:|---|
| activity | 9.189954 | [-12.946395862905911, 30.584788524453508] |
| texture | 0.904262 | [-0.07322695087571626, 2.190336105325985] |
| permuted | -0.368402 | [-0.997070469355619, 0.24884885372061757] |
| rgb_observed | 48.991273 | [0.25612292825419086, 83.93149591787844] |

La escala ajustada no equivale al umbral físico del sensor.
RGB/ISP, umbrales asimétricos, paralaje y textura de referencia pueden dominar el residuo.
Slider se conserva como no admitido por alineación provisional; ninguna consulta desaparece del inventario.
Las celdas silenciosas no imponen q=0 ni una pérdida de silencio.
Este resultado no mide mejora TTC ni abre S2 automáticamente.
