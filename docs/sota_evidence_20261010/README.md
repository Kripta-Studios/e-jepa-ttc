# Evidencia frente a Garl-TTC y REACT — 10 de octubre de 2026

Actualización posterior: la campaña activa se centra en GarlTTC; REACT quedó
fuera del alcance por decisión del usuario. Véanse el [estado final local](STATUS_20261010.md),
el [informe de streaming](STREAMING_OPTIMIZATION.md) y la [evidencia publicada](evidence/README.md).
El texto siguiente conserva la auditoría inicial y su contabilidad de aquella ejecución.

**Estado: superioridad SOTA no establecida.** Esta ejecución recalcula métricas
sobre predicciones congeladas; no entrena, selecciona checkpoints ni crea un test ciego.
V13 conserva su ejecución. GPU adicional consumida: 0 segundos.

## Resultados recalculados

RTE micro pondera consultas; RTE macro pondera secuencias por igual.
Los resultados son diagnósticos locales; no son las poblaciones de los artículos.

| Cohorte | Método | n | RTE micro (%) | RTE macro (%) |
|---|---|---:|---:|---:|
| DEV32 | H8_median3 | 3640 | 54.489 | 65.168 |
| DEV32 | Direct_median3 | 3640 | 37.321 | 43.329 |
| DEV32 | public_Garl_event_lhr | 3640 | 161.121 | 188.996 |
| DEV32 | public_Garl_rgb_event_full | 3640 | 5180.182 | 5015.576 |
| FCWD | H8_median3 | 597 | 27.856 | 27.890 |
| FCWD | Direct_median3 | 597 | 28.986 | 29.020 |
| FCWD | public_Garl_event_lhr | 597 | 26.887 | 26.841 |
| FCWD | public_Garl_rgb_event_full | 597 | N/D | N/D |

Las tablas también contienen cada semilla, cada secuencia, TTC positivos, bandas
de ambos artículos y una vista secundaria con límite ±60 s igual para todos.
El bootstrap pareado remuestrea secuencias enteras (10.000 réplicas); no elimina
la correlación entre familias ni la exposición previa.
FCWD solo aporta tres secuencias.
No se selecciona el modelo con mejor resultado después de observar estas tablas.

## Contratos de los artículos

- Garl: MiD = 10^4 |log(1−0,1/pred) − log(1−0,1/GT)|; bandas
  (0,3], (3,6], (6,10], [−10,0), pesos 0,5/0,3/0,1/0,1.
  El código público excluye −10 exactamente, a diferencia del texto. Se registran
  ambas convenciones. La tabla VI publica RTE promedio 10,60 %; es referencia
  bibliográfica, no una reproducción nuestra. [Artículo](https://arxiv.org/abs/2603.16303).
- REACT: su MiDw usa |log(pred/GT)| para TTC positivo, con bandas
  [0,3), [3,6), [6,10). Es otra magnitud. Sus pesos suman 0,9; registramos
  suma literal y normalizada como sensibilidad, sin elegir la más favorable.
  La tabla II declara 9,59±0,74 % sobre validación en dominio con cinco semillas
  y 4,6 ms; la tabla III declara 25,7 % en FCWD. Sin población/ejecutable
  idénticos no se restan esas cifras a las nuestras para proclamar victoria.
  [Artículo](https://arxiv.org/abs/2609.19204).

El artículo de REACT anuncia código tras publicación. La web del autor y la lista
de repositorios públicos revisadas no proporcionaron una reproducción verificable.
Esto describe las fuentes examinadas; no demuestra ausencia absoluta de código.
El 9,44 % de Garl citado en REACT tampoco equivale al promedio 10,60 % de la
tabla VI de Garl: hace falta reconstruir la población antes de comparar rankings.

## Integridad de las métricas

`REFERENCE_PARITY.json` contrasta cada fila local con el evaluador Garl archivado.
Su media MiD omite NaN, y una predicción infinita puede generar MiD finito pese
a marcarse como fallo. Conservamos esa salida solo como referencia de compatibilidad.
La salida estricta declara N/D si alguna fila requerida es inválida; registra
fallos y tamaños de banda. No cambia umbrales para mejorar resultados.
Una banda vacía impide el agregado ponderado: no se redistribuyen sus pesos.

## Barreras de la evaluación oficial

Las 40 secuencias del índice real de entrenamiento coinciden con train40 oficial
y no intersectan test12 (`TRAIN_TEST_DISJOINT.json`). Esto cierra la comprobación
de IDs de secuencia; no sustituye auditoría de escenarios ni puntuación de test.
- Test oficial: 6762 consultas, 12 secuencias, sin etiquetas TTC públicas.
- Eventos ausentes: 125856313274 bytes
  según el inventario remoto archivado. Descarga completa pendiente.
- Preparar inferencia causal para esos IDs, comprobar relojes y ROI y obtener
  predicciones completas antes de exportar. `submission.py` exige todos los IDs
  y valores finitos; la plantilla descargada no se usa como predicción.
- Obtener puntuación externa de CodaBench con el candidato y protocolo fijados
  previamente. Las etiquetas privadas no se pueden reconstruir del conjunto público.
- Para REACT: disponer de pesos/código, split y timestamps exactos, configuración
  de entrenamiento y medición, y reevaluar ambos métodos en la misma población.
- Las tres cabezas comparten productores: faltan repeticiones independientes del
  sistema completo para estimar variabilidad de entrenamiento.

## Alcance de una afirmación

H8 usa eventos con ROI oráculo e historia 650 ms; Garl usa sus entradas nativas
200 ms, y la rama full añade RGB. REACT no recibe ROI. El entrenamiento previo
de nuestros productores incluyó teacher RGB: event-only describe inferencia, no
toda la supervisión. Los protocolos no son equivalentes.
La latencia propia incluye lectura HDF5; los costes publicados tienen otros
límites y hardware. La historia de 650 ms no es automáticamente 650 ms de
espera adicional en cada consulta de un sistema causal ya inicializado.
Se necesitan warm-start, cadencia, antigüedad de observaciones y cola separados.

Podemos afirmar mejoras medidas en los protocolos locales descritos. No podemos
afirmar mejor rendimiento global, tiempo real, autonomía sin ROI ni SOTA.

## Reproducir

```powershell
python -m operational.sota_evidence.run `
  --source-root artifacts/ttc_revision_20261009 `
  --train-index artifacts/train40_system_20261005/TRAIN40_ROWS.parquet `
  --output artifacts/sota_evidence_20261010 --media-root $env:EAP_ROOT `
  --garl-evaluator $env:GARL_EVALUATOR --report docs/sota_evidence_20261010/README.md
python -m pytest tests/unit/test_sota_evidence.py `
  tests/unit/test_ttc_revision_score.py
```

Fuentes y entradas: `SOURCES.json`, `INPUTS.json`, `PROTOCOL.json`.
Resultados: `METRICS.csv`, `PAIRED_MACRO_RTE.csv`, `REFERENCE_PARITY.json`,
`INVENTORY.json`, `RESULT.json`. Las fuentes bibliográficas nunca se mezclan
con resultados reproducidos en una clasificación conjunta.
