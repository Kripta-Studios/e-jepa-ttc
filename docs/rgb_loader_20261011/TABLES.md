# Tablas regeneradas: optimización RGB, 11 de octubre de 2026

Generadas desde los JSON públicos por `regenerate_tables.py`. Unidades de tiempo: segundos.

## Fuente CPU: comparación ABBA

| Ensayo | Ruta | Ejemplos | Tiempo fuente | Lecturas DINO | Lecturas RGB |
|---|---|---:|---:|---:|---:|
| A1 | Original | 512 | 22.2716 | 355 | 22 |
| B1 | Optimizada | 512 | 6.8875 | 31 | 22 |
| B2 | Optimizada | 512 | 7.3466 | 31 | 22 |
| A2 | Original | 512 | 22.2275 | 355 | 22 |

Media original: **22.2496 s**. Media optimizada: **7.1170 s**. Reducción: **68.01 %**.

Paridad exacta de todos los campos en los cuatro ensayos; estados aleatorios conservados.

## Coste del control completo de recursos

| Ronda | Variante | Tiempo | Decisión permite continuar | Procesos del proyecto |
|---|---|---:|---|---:|
| 1 | original | 4.7734 | True | 2 |
| 1 | filtered | 0.2146 | True | 2 |
| 2 | filtered | 0.1901 | True | 2 |
| 2 | original | 5.4671 | True | 2 |
| 3 | original | 5.9409 | True | 5 |
| 3 | filtered | 0.2146 | True | 5 |

Medianas: **5.4671 → 0.2146 s** por llamada; cociente **25.48×**.

Las parejas de cada ronda identificaron los mismos procesos; hubo cambios de procesos entre rondas. Las seis decisiones observadas coincidieron.

## QA

| Comprobación | Resultado |
|---|---|
| Pytest | 68 aprobadas |
| Ruff | PASSED |
| Pyright | 0 errores, 0 advertencias |
| Updates de optimizador de la intervención CPU | 0 |

Estos tiempos son de componentes CPU. No sustituyen una medición del entrenamiento completo tras reanudar.
