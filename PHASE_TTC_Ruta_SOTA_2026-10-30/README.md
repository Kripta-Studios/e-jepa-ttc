# PHASE-TTC — Ruta de investigación hacia un resultado SOTA defendible

**Fecha de corte:** 30 de octubre de 2026. **Autor del proyecto:** Álvaro Schwiedop Souto. **Base revisada:** `Kripta-Studios/phase-ttc`, commit `30b28b5c14eb9ed60067ed2f1dc262d7b60f2654`.

## Decisión principal

Continuar con PHASE-TTC, conservar su candidato original y resolver primero el contrato de evaluación, la transferencia de sensores y la disponibilidad de datos independientes. Si esos contratos pasan, el primer experimento aprendido debe ser una adaptación pequeña de la fusión, con productores congelados y controles de capacidad. Una rama JEPA centrada en movimiento es una investigación posterior, condicionada a demostrar que las representaciones actuales pierden la señal temporal relevante.

Esta recomendación maximiza la información obtenida por experimento. No garantiza un puesto en un leaderboard: el SOTA se demuestra sobre una tarea, un protocolo, una población y un conjunto de competidores concretos.

## Qué contiene este paquete

Este es un plan de ejecución y una revisión crítica, no una implementación ni un nuevo resultado experimental. Se revisaron el plan y su anexo, código del commit indicado, documentación de los runs y fuentes primarias hasta la fecha de corte. Se recalcularon los agregados del anexo. No se han releído aquí los 184 GiB de HDF5 del equipo del autor, verificado sus pesos locales, entrenado modelos ni enviado predicciones. Las observaciones históricas sobre esos recursos conservan su procedencia.

Las rutas marcadas **existente** se comprobaron en el repositorio. Las rutas y comandos marcados **propuesto** son contratos para implementar; no deben confundirse con herramientas disponibles. Todas las cifras de presupuesto o criterios de promoción propuestos son decisiones operativas, no resultados ni requisitos oficiales del benchmark.

## Orden de lectura

| Documento | Para qué sirve |
|---|---|
| [01 — Estado y diagnóstico](01_ESTADO_Y_DIAGNOSTICO.md) | Qué sabemos, qué cambia respecto al plan inicial y dónde está la mayor oportunidad. |
| [02 — SOTA y lecciones](02_SOTA_Y_LECCIONES.md) | Fuentes verificadas, comparabilidad y técnicas con utilidad concreta. |
| [03 — Protocolo y datos](03_PROTOCOLO_Y_DATOS.md) | Genealogía, exclusiones, grupos, exposición histórica y datos Slider nuevos. |
| [04 — Evaluador y baselines](04_EVALUADOR_Y_BASELINES.md) | Targets, relojes, referencia temporal, negativos y comparaciones válidas. |
| [05 — Diagnóstico de sensores y modelo](05_DIAGNOSTICO_DE_SENSORES_Y_MODELO.md) | Pruebas físicas y trazas que distinguen causas. |
| [06 — Intervenciones mínimas](06_INTERVENCIONES_MINIMAS.md) | Adaptador, arranque y consistencia temporal causal. |
| [07 — Fusión y fine-tuning](07_FUSION_Y_FINE_TUNING.md) | Diseño exacto del primer piloto aprendido y sus controles. |
| [08 — JEPA y arquitecturas futuras](08_JEPA_Y_ARQUITECTURAS_FUTURAS.md) | Cuándo investigar nuevas representaciones y cómo probar su contribución. |
| [09 — Matriz experimental](09_MATRIZ_EXPERIMENTAL_Y_DECISIONES.md) | Dependencias, promoción, rechazo y presupuesto de búsqueda. |
| [10 — Confirmación y SOTA](10_CONFIRMACION_ESTADISTICA_Y_SOTA.md) | Estadística agrupada, evaluación reservada y alcance de las afirmaciones. |
| [11 — Runtime y recursos](11_RUNTIME_RECURSOS_Y_REPRODUCIBILIDAD.md) | Ejecución en 12 GiB, cachés, latencia y reproducibilidad. |
| [12 — Backlog y primera iteración](12_BACKLOG_Y_PRIMERA_ITERACION.md) | Tareas pequeñas, entregables y criterios de cierre. |
| [13 — Referencias y alcance](13_REFERENCIAS_Y_ALCANCE.md) | Fuentes, versiones y límites de la investigación. |
| [14 — Contratos y comandos](14_CONTRATOS_Y_COMANDOS_PROPUESTOS.md) | Interfaces, esquemas y ejemplos que puede implementar un agente. |
| [15 — Revisión independiente](15_REVISION_INDEPENDIENTE.md) | Preguntas críticas para evitar falsos éxitos antes de cerrar una fase. |

Hay cuatro plantillas: [experimento](plantillas/EXPERIMENTO.md), [auditoría de datos](plantillas/AUDITORIA_DATOS.md), [reporte](plantillas/REPORTE_RESULTADOS.md) y [handoff](plantillas/HANDOFF.md). El [manifiesto](MANIFIESTO.md) permite comprobar el contenido y la identidad de todos los documentos.

## La primera iteración concreta

1. Congelar la identidad del candidato y levantar la genealogía de sus nueve componentes, sus normalizadores y sus decisiones de selección.
2. Convertir la auditoría de solapamiento en un CLI que reproduzca las ocho correspondencias y los 365 frames, incluyendo casos negativos y ambiguos.
3. Inventariar relaciones de adquisición y exposición histórica de las 24 candidatas. Establecer qué permiten medir realmente.
4. Construir el evaluador diagnóstico, manteniendo las 11 consultas fuera de cobertura como ausentes y verificando el instante y la geometría del TTC.
5. Añadir trazas de todos los productores aceptados y rechazados, y ejecutar fixtures físicos de relojes, expansión, paralaje y causalidad.
6. Emitir una decisión basada en evidencia: corregir un defecto, probar consistencia causal, ajustar fusión o recopilar datos antes de entrenar.

**Resultado esperado de esta iteración:** una causa localizada o un conjunto de hipótesis descartadas y un experimento bien definido. No se exige que ya reduzca el error.

## Tres líneas que deben conservar identidades distintas

| Línea | Entrenamiento o calibración en EvTTC | Qué puede demostrar |
|---|---|---|
| `PHASE-FROZEN` | Ninguno en el nuevo experimento; genealogía previa verificada aparte | Transferencia del candidato original bajo el adaptador declarado. |
| `PHASE-ADAPT` | Solo grupos de desarrollo elegibles | Mejora tras adaptación al dominio; no transferencia sin ajuste. |
| `PHASE-MOTION` | Preentrenamiento y/o ajuste de una representación nueva, con datos declarados | Contribución específica de aprender dinámica, si supera controles equivalentes. |

Son etiquetas propuestas para organizar la campaña, no nombres de modelos ya entrenados. Una corrección de adaptador también tiene versión propia, aunque los pesos no cambien.

## Condición para hablar de SOTA

Publicar un resultado mejor bajo un protocolo comparable y permitido; declarar modalidades, cajas, historia causal, entrenamiento, cobertura y tratamiento de fallos; confirmar que la mejora no se limita al test ya inspeccionado. Si solo existe una mejora interna, se denomina mejora interna. Si solo existe una mejora de clasificación oficial, se acota al leaderboard y a su fecha.

El plan original sigue disponible en [GitHub, versión exacta revisada](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/superpowers/plans/2026-10-30-evttc-transfer-diagnosis.md). Este paquete lo amplía; no cambia los recibos históricos ni da por ejecutadas sus casillas pendientes.
