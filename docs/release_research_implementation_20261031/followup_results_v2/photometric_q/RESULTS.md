# Recuperación de q desde el proxy fotométrico

Desarrollo expuesto. Escala ajustada excluyendo la adquisición y lecturas selladas antes de unir TTC.

| Lectura | q disponible | TTC disponible | Mediana error q | Signos erróneos | MAE con fallback R0 |
|---|---:|---:|---:|---:|---:|
| radial | 72 | 72 | 0.316184 | 13 | 13.241014767 |
| translation | 72 | 72 | 0.257795 | 14 | 12.050232373 |
| affine | 72 | 72 | 0.241709 | 13 | 9.168523738 |
| spatial_permutation | 72 | 72 | 0.438956 | 37 | 45.629594312 |

La lectura afín permite traslación, deformación y luz global; no una textura libre por píxel.
La escala RGB/eventos sigue siendo un proxy: desfase, ISP, paralaje y respuesta pueden sesgar q.
Se conservan todos los fallos, la ausencia original y el control espacial permutado.
No se modifica R0 ni se abre automáticamente entrenamiento de productores o JEPA.
