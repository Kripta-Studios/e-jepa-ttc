# Mediciones regeneradas — TRAIN40, 27 de octubre de 2026

Generado desde los JSON del directorio `evidence/`. Los relojes de estas tablas tienen alcances diferentes y no deben dividirse entre sí para afirmar un speedup.

## Piloto completo, un objeto, tres consultas

Desde paquete RAW CPU y frames RGB disponibles: incluye ROI, representación, transferencias, productores, contexto y TTC. Excluye reproducción de disco, inicialización del historial y priming sintético inicial. No es un stream sostenido.

| Índice original | Actualización completa (ms) | Entrada exacta |
|---:|---:|---|
| 0 | 44.1235 | True |
| 200 | 20.3230 | True |
| 553 | 27.9516 | True |

Estado: **COMPLETE**. Media 30.7994 ms; p95 42.5063 ms; p99 43.8001 ms. Solo tres observaciones: estos percentiles no estiman una cola estable.

| Salida | Máxima diferencia absoluta respecto al original (s) |
|---|---:|
| PHASE_ONLY_TRAIN40_S7 | 4.291534424e-06 |
| E_CTX_TRAIN40 | 1.239776611e-05 |
| R_CTX_TRAIN40 | 9.536743164e-06 |

Tolerancia fijada: `atol=rtol=1e-5`. Preservación: 601 archivos, 0 cambiados.

## Productores preparados con reutilización por objeto

Tensores ya residentes en GPU. Incluye productores, PAIR, PHASE17 y guards; excluye sensores, H2D, contexto y TTC final. Carga multiobjeto artificial declarada.

| Objetos | Paralelo mínimo (ms) | Paralelo máximo (ms) | Serial/paralelo exactos |
|---:|---:|---:|---|
| 1 | 10.142 | 11.429 | True |
| 2 | 16.247 | 17.871 | False |
| 5 | 40.428 | 43.637 | True |
| 10 | 74.818 | 82.727 | True |
| 20 | 179.488 | 181.531 | True |

El caso de dos objetos no es exacto y su deriva no se cuantificó al pausar. Ninguna fila demuestra paridad del TTC final multiobjeto contra el oracle.

## Representación de eventos para veinte ROI

Nuevo paquete de 510,778 eventos más veinte ROI: **75.212 ms**. ROI distintas de carga, sin veinte objetos anotados. Modelos excluidos.

## Compilación aislada de correlación

B8, C64, 32×32, radio 1; features sintéticas. No es un productor completo.

| Ejecución | Mediana (ms) | p95 (ms) |
|---|---:|---:|
| eager | 1.9454 | 2.2577 |
| eager_graph | 0.2876 | 0.3285 |
| compiled | 0.2307 | 0.2505 |

Compilación inicial: 10.193 s. Paridad del helper a 1e-5 aprobada, sin igualdad bit a bit. Falta paridad de TTC y compilación del core completo.

## Resultado rechazado

`pilot_v1/summary.json` es FAILED: un módulo aditivo cambió durante el ensayo. Se conserva para auditoría; no se admite su tabla de tiempos. El candidato congelado no cambió.
