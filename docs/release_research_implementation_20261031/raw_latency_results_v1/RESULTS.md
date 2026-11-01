# Latencia desde eventos RAW y cierre de la evidencia disponible

Este suplemento completa la frontera de latencia del informe de campaña.
El ensayo anterior lee representaciones de eventos precalculadas en NPZ;
no incluye su construcción. El ensayo RAW sí incluye lectura HDF5, recorte,
voxelización, transferencia y todos los productores, contextos y fusión,
junto con la lectura y decodificación RGB. Las cajas vienen suministradas.
No mide detección, adquisición física ni transporte del sensor.

Paridad: 120 observaciones reconstruidas exactamente;
120 consultas físicas seleccionadas por prefijo temporal.
Todas las predicciones cronometradas coinciden exactamente con el replay
local completo. Se reinician lectores y cachés por secuencia y brazo; el
caché del sistema operativo puede estar caliente. El orden de brazos alterna.

| Frontera | Brazo | Consultas | p50 ms | p95 ms | p99 ms |
|---|---|---:|---:|---:|---:|
| NPZ preparado + RGB | canonical | 120 | 346.744 | 1658.066 | 1791.578 |
| NPZ preparado + RGB | original | 120 | 329.344 | 1603.654 | 1737.175 |
| RAW + RGB | canonical | 120 | 534.662 | 1008.124 | 1208.804 |
| RAW + RGB | original | 120 | 532.024 | 963.353 | 1014.155 |

Cociente p95 canónico/original desde RAW: **1.046474**.
Cumple el presupuesto descriptivo de +10%: **True**.
Son mediciones separadas: no se suman percentiles ni se atribuye su
diferencia al preprocesado aislado. No hay garantía de servicio a 50 Hz.

GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU. Máximo CUDA asignado: 69235712 bytes;
reservado: 92274688 bytes; RAM máxima del proceso:
2433994752 bytes. Las colas por secuencia están en RAW_LATENCY.json.

## Qué permite concluir

La cadencia tiene una señal favorable de desarrollo en Slider; el cambio
de carretera es pequeño y su intervalo incluye cero. La comparación
nativa completa mide sensibilidad local; no demuestra conservación del
score oficial porque la paridad histórica intermedia falló y no se
leyeron etiquetas privadas. El release oficial sigue intacto.

Los controles de observabilidad, reloj, sensor y superficies están
ejecutados con los datos disponibles. Sus resultados no admiten los
entrenamientos condicionales O2/D4/S2/V2 ni JEPA definidos en el dossier.
Eso no prueba que cualquier variante futura sea incapaz de mejorar.

El usuario confirma que no existen adquisiciones reservadas ni calibración
fotométrica adicional. Repartir de nuevo los mismos datos no crea una
confirmación independiente. Quedan sin demostrar la generalización de
la ganancia y la no inferioridad del score nativo; no se promociona el candidato.
