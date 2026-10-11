# Evidencia de componentes: overlap RGB

Generado por `regenerate.py` desde `EVIDENCE.json`. No es un monitor en vivo.

## Regresiones CPU

| Casos | Nuevos overlap | Fallos | Errores | Omitidos |
|---:|---:|---:|---:|---:|
| 119 | 51 | 0 | 0 | 0 |

## Preparación de entradas CPU

Estado del archivo completo: **COMPLETE**.

Muestras por caso: 512. Comparación desde update 1500.

| Caso | Hilos decode | Prefetch | Espera acumulada (s) | Variación frente a media serial |
|---|---:|---|---:|---:|
| baseline_A | 0 | False | 6.334 | +3.1% |
| decoder_2 | 2 | False | 4.862 | -20.8% |
| decoder_4 | 4 | False | 4.255 | -30.7% |
| baseline_B | 0 | False | 5.948 | -3.1% |

Paridad exacta de batch: **True**. Cursor/RNG preservados: **True**. Origen sin modificar: **True**.

### Pruebas de paridad con anticipación

| Caso | Espera acumulada (s) |
|---|---:|
| decoder_2_prefetch_2 | 3.290 |
| decoder_4_prefetch_2 (prueba posterior) | 0.911 |

Paridad adicional con cuatro lectores: **True**. Cursor/RNG: **True**.
Reserva pendiente máxima observada: **109239808 bytes**, límite **134217728 bytes**.

Dos referencias seriales; una ejecución por variante. Caché del sistema sin vaciar. Con prefetch, el cálculo de hashes puede solaparse con lecturas. Estos tiempos de espera no prueban aceleración del entrenamiento completo.

## Cómputo GPU con entradas preparadas

### Primer ABBA completo: coincidió con el benchmark CPU

Estado: **PASSED**.

Paridad exacta GPU: **True**. Archivos protegidos sin modificar: **True**.

| Implementación | Mediana por batch (s) |
|---|---:|
| Referencia A | 1.009 |
| Máscaras cacheadas B | 1.553 |

Cociente de medianas A/B: **0.650×**.

**Resultado de rendimiento negativo:** la variante cacheada tardó más en esta comparación. La paridad no demuestra una mejora de velocidad.

### Repetición sin entrenamiento ni benchmark CPU concurrente

Estado: **PASSED**.

Paridad exacta GPU: **True**.

| Implementación | Mediana por batch (s) |
|---|---:|
| Referencia A | 0.574 |
| Máscaras cacheadas B | 0.923 |

Cociente de medianas A/B: **0.621×**.

La repetición tampoco demuestra aceleración. La máscara cacheada queda **desactivada** en el runtime admitido.

Tiempos individuales (Primero); se conserva la variación entre A1 y A2:

| Ronda | Segundos por batch |
|---|---|
| A1 | 0.185, 0.363, 0.466, 1.224 |
| B1 | 0.748, 1.555, 1.539, 1.601 |
| B2 | 0.740, 1.570, 1.551, 1.584 |
| A2 | 0.794, 1.657, 1.539, 1.523 |

Tiempos individuales (Aislado); se conserva la variación entre A1 y A2:

| Ronda | Segundos por batch |
|---|---|
| A1 | 0.165, 0.317, 0.304, 0.302 |
| B1 | 0.222, 0.583, 0.477, 1.134 |
| B2 | 0.713, 1.384, 1.440, 1.397 |
| A2 | 0.831, 1.483, 1.518, 1.471 |

La variabilidad temporal impide atribuir causalmente todo el delta a la caché. La decisión conservadora es mantener el cómputo original.

Una ejecución anterior quedó incompleta por un error de instrumentación al serializar un tensor con stride singleton. Se conservó su evidencia; no se admite como prueba de paridad. El ajuste afectó al capturador del benchmark, y la comparación se repitió sin updates de optimizador.

Orden ABBA tras calentamiento; forward, pérdida, backward y clipping. Excluye preparación, paso del optimizador y copias de QA. Las pruebas no consumen updates del optimizador de la campaña.

## Alcance

La admisión requiere todos los gates, aunque un componente sea más rápido. Faltan aquí observaciones de rendimiento sostenido de la campaña y métricas TTC nuevas. No se extrapola una ETA ni se afirma SOTA.
La observación breve posterior del entrenamiento real está en [LIVE.md](LIVE.md).
