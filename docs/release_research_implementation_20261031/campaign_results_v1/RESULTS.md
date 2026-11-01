# Resultado de la investigación con los datos disponibles

La corrección de cadencia supera los criterios fijados de desarrollo en 2608
consultas. La comparación nativa local y la latencia están ejecutadas. Esto no
demuestra conservación del score oficial: la réplica histórica del original
falla su criterio intermedio y no se han abierto targets privados.

El usuario confirma que no existen adquisiciones reservadas ni calibración
fotométrica adicional. Las conclusiones mantienen su exposición de desarrollo.

## Cadencia: ampliación completa

| Dominio | Pares/población | MAE original (s) | MAE canónico (s) | Cambio relativo MAE |
|---|---:|---:|---:|---:|
| road | 2531/2531 | 0.810620018 | 0.810489612 | -0.0161% |
| slider | 76/77 | 1.178155898 | 1.120808950 | -4.8675% |

Admisión F0: `True`. CI95 del delta MAE de carretera: `[-0.0005047002743243974, 9.314967648063326e-05]` s.
Slider es una sola adquisición, reutilizada del screen; no constituye una
confirmación nueva independiente. La ampliación añade 2459 consultas de carretera.
La mejora media de carretera es pequeña y su intervalo incluye cero.
Cobertura, error relativo, signos y todas las grabaciones: [informe completo](POPULATION.md).

## Conservación histórica y comparación local

La ruta histórica original se conserva cerrada por fallo de paridad. Las
cuatro políticas numéricas se probaron exclusivamente en el original:

| Política | Consultas históricas que pasan |
|---|---:|
| matmul_0_cudnn_0 | 0/2 |
| matmul_0_cudnn_1 | 0/2 |
| matmul_1_cudnn_0 | 0/2 |
| matmul_1_cudnn_1 | 0/2 |

El ensayo posterior responde a una pregunta distinta: cuánto cambia el
candidato respecto al original ejecutado en este mismo equipo. Primero exigió
repetibilidad exacta de todos los PAIR/PHASE17, contextos y cabeza originales.

- Consultas físicas: 5805; emparejadas: 5805.
- Predicciones idénticas: 86.
- Nuevas ausencias: 0; cambios de signo: 0.
- Máximo cambio absoluto: 0.00649642944 s.
- Percentiles 50/95/99 del cambio absoluto: `[1.430511474609375e-05, 7.677078247070312e-05, 0.00012104034423828132]` s.
- Contexto de eventos idéntico: `True`.
- Original local frente a histórico: 5805 consultas fallan algún campo; 504 fallan la tolerancia TTC.
- Aliases lógicos resueltos: 6762; no son observaciones independientes.

Una diferencia pequeña no acredita conservación de Overall/Crucial/Negative.
El release oficial permanece intacto. [Datos de sensibilidad](NATIVE_SENSITIVITY.json).

## Latencia en el equipo local

Primera pasada sin hooks de medida, desde preparación/lectura hasta salida host,
con sincronización CUDA. Pesos calientes y cachés independientes por brazo.
La selección utiliza los primeros diez ejemplos físicos por secuencia; no
representa toda la distribución de despliegue. No equivale al servidor histórico.

| Brazo | Consultas | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|---:|
| canonical | 120 | 346.744 | 1658.066 | 1791.578 |
| original | 120 | 329.344 | 1603.654 | 1737.175 |

Cociente p95 candidato/original: 1.03393. Dentro del presupuesto descriptivo +10%: `True`.
Memoria pico CUDA asignada: 66.03 MiB; pico RAM del proceso: 2064.27 MiB.
La carga de pesos, importación, admisión de medios y adquisición física previa
no forman parte del tiempo predict. Las colas son simulaciones de servidor único;
no prueban servicio sostenido a 50 Hz. [Mediciones completas](LATENCY_MEASUREMENTS.json).

La segunda pasada instrumentada separa costes y añade sincronizaciones.
Sus medianas no deben sumarse ni confundirse con la primera pasada:

| Brazo | Preparación/caché (ms) | Productores (ms) | Contextos/fusión/host (ms) |
|---|---:|---:|---:|
| original | 119.562 | 300.216 | 24.055 |
| canonical | 117.872 | 302.643 | 23.698 |

## Qué queda de las demás propuestas

Los diez contrastes previos están en [el informe de mecanismos](../followup_results_v2/RESULTS.md).
O0 no supera los controles; la lectura robusta mantiene poca cobertura. Los
filtros del reloj tampoco superan al contexto tras corregir la cadencia.
S1 encuentra información de movimiento en RGB observado, pero su recuperación
de q empeora TTC; no basta mejorar el residuo sensorial para entrenar ese auxiliar.
La medición de superficies disponible tampoco mejora TTC.

Por ello no se han entrenado O2, D4, S2/S3, V2/V3 ni JEPA condicional. Sus
bloques y controles implementados no se presentan como modelos entrenados.
Los resultados negativos delimitan estas implementaciones y esta población;
no demuestran que toda variante posible sea inútil.

No quedan extracciones de esta ejecución declaradas completas sin recibo. La
confirmación independiente y la conservación de precisión oficial siguen sin
demostrarse. La variante de cadencia se conserva como candidato de desarrollo.
