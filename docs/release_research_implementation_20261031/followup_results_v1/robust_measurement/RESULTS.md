# Control de lectura robusta sobre los mismos planos

Desarrollo expuesto; parámetros históricos intactos, sin entrenar gate.

| Lectura | q disponible | TTC disponible | Signos q erróneos | MAE con fallback R0 |
|---|---:|---:|---:|---:|
| ols | 63 | 53 | 26 | 1.593770789 |
| historical_robust | 8 | 8 | 4 | 0.779543461 |

Modos del solver robusto: {'invalid': 140, 'affine': 8, 'no_planes': 1}.
El modo isotrópico introduce una hipótesis de movimiento explícita.
Se conserva la población y ausencia original; no se cambia la admisión O0.
