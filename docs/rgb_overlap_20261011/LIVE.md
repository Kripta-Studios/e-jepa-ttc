# R_A5 reanudado: observación del entrenamiento completo

Regenerado desde `LIVE_RESULT.json` por `publish_live.py`, sin argumentos. Esta es una ventana registrada, no un monitor actualizado automáticamente.

Implementación: `2b38c31`. Restauración desde **2124 updates durables**, con el hash exacto del checkpoint conservado.

| Ventana | Inicio | Fin | Duración (s) | Avance | Updates/min |
|---|---|---|---:|---:|---:|
| Anterior | 2026-10-11T02:18:23.330276+02:00 | 2026-10-11T02:20:39.900071+02:00 | 136.57 | 71 | 31.19 |
| Overlap | 2026-10-11T02:32:54.574672+02:00 | 2026-10-11T02:37:59.136829+02:00 | 304.56 | 183 | 36.05 |

Última muestra: **2321/30330**; durables **2300**, recuperación acumulada **0**.

Extrapolación para terminar sólo R_A5 al ritmo de toda la ventana: **12.95 horas**.

La referencia excluye el plateau de la pausa voluntaria en 2.124; la ventana nueva comienza en la primera muestra con avance tras restaurar. Las ventanas tienen distintos batches, duraciones y carga del host. No constituyen una comparación controlada ni incluyen las fases posteriores de V13.

## Ventanas recientes

| Solicitada | Duración realmente disponible (s) | Updates/min | Cobertura completa |
|---|---:|---:|---|
| 5 min | 304.56 | 36.05 | Sí |
| 10 min | 304.56 | 36.05 | No |
| 15 min | 304.56 | 36.05 | No |

Si falta historial, la tabla muestra la ventana disponible y lo indica. No se fabrican muestras ni ritmos de 10/15 minutos.

## Checkpoint y preparación

Recibo durable observado: **2300 updates**; **276.85 s** desde la restauración hasta el recibo.

Espera acumulada del consumidor por entradas: **0.220 s** (**0.08%** de ese intervalo). Esto mide espera en los futuros, no toda la preparación ni una descomposición del cómputo GPU. Preparación especulativa y trabajo solapado siguen incluidos en sus contadores separados.

El recibo confirma **cuatro decodificadores, dos batches anticipados y optimización geométrica desactivada**. BF16, pérdida, sampler y estados del entrenamiento se mantienen.

Verificación independiente de la cola: 2026-10-11T02:38:10.970086+02:00; **2327 updates**, pausa **False**, STOP_REQUEST **False**. Los procesos observados incluyen launcher y ejecutable Python para cada módulo; no representan entrenamientos duplicados.

## Foco restante según pilas

Hilo principal: **919 muestras**. Porcentajes inclusivos dentro de sus muestras registradas:

| Función presente en la pila | Muestras ponderadas (%) |
|---|---:|
| `_producer_loss` | 82.3 |
| `forward` | 64.7 |
| `transport_physical_features` | 22.1 |
| `_summarise_one` | 21.7 |

Los porcentajes inclusivos se solapan: **no deben sumarse ni interpretarse como tiempo CUDA o tiempo de pared**. Los lectores siguen descomprimiendo datos; esa actividad se solapa con el entrenamiento y la espera medida del consumidor es pequeña en este recibo. El foco siguiente es el trabajo del modelo, sus operaciones geométricas y el lanzamiento de operaciones, que requieren perfiles adicionales antes de atribuir tiempos GPU concretos.

## Memoria observada

Se selecciona por muestra el proceso de entrenamiento con más memoria privada; se conservan también las mediciones del launcher en el JSON.

| Medida | Mínimo (GiB) | Mediana (GiB) | Máximo muestreado (GiB) |
|---|---:|---:|---:|
| RSS | 4.026 | 4.419 | 4.484 |
| Privada | 6.666 | 7.416 | 7.515 |
| RAM disponible | 12.153 | 12.323 | 12.813 |

Son muestras periódicas, no picos exactos ni una garantía de estabilidad prolongada. La memoria privada no equivale a RSS ni al margen global de memoria comprometida de Windows.
