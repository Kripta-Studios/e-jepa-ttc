# Historia científica y de ingeniería: junio–octubre de 2026

Corte documental: **27 de octubre de 2026**. Esta entrega reúne la evolución del proyecto desde el primer commit de junio hasta el candidato multimodal TRAIN40, las rotaciones fuera de muestra, la ejecución local y en Runpod, y la comparación final con Garl-TTC sobre EvTTC. Se apoya en commits, código, recibos y resultados conservados; no convierte propuestas ni mediciones parciales en experimentos completados.

El resultado final de esta etapa es `PHASE_ONLY_EVENT_S7_TRAIN40_SCIENTIFIC_OOF`: nueve componentes entrenables identificados, dos normalizadores y un ZIP de 6.762 predicciones oficiales Test12. El archivo está **preparado, no enviado**. V18 mantiene su cierre histórico `NO_CANDIDATE`. La evaluación EvTTC posterior es transferencia descriptiva sobre un dataset ya observado, no una puntuación oficial ni una demostración de SOTA.

## Lectura por tema

| Capítulo | Contenido |
|---|---|
| [01. Cronología científica](01_SCIENTIFIC_CHRONOLOGY.md) | JEPA, recuperaciones, A5/C2F, RGB, V16/V18/V19 y decisiones que condujeron al refit |
| [02. TRAIN40, OOF y entrenamiento final](02_TRAIN40_OOF_AND_FINAL_TRAINING.md) | Poblaciones, genealogía, rotaciones, exposición, dependencias y componentes finales |
| [03. CUDA, precisión y transporte](03_CUDA_PRECISION_AND_TRANSPORT.md) | BF16/FP32, sincronizaciones, CUDA Graphs, paridad y optimizaciones rechazadas |
| [04. Caché, prefetch y entrada/salida](04_CACHE_PREFETCH_AND_IO.md) | NPZ/ZIP, staging, memoria, disco local, volumen de red y procesamiento RAW |
| [05. Concurrencia y recuperación](05_CONCURRENCY_RESOURCES_AND_RECOVERY.md) | Reservas Windows/Linux, checkpoints, journals, pausas, reanudación y fallos |
| [06. Migración y colas Runpod](06_RUNPOD_MIGRATION_AND_QUEUES.md) | Pods RGB/eventos, automatización de dependencias, recursos, presupuesto y descarga |
| [07. Evaluación y latencia](07_EVALUATION_METRICS_AND_LATENCY.md) | Métricas, cohortes, Garl oficial, precisión frente a coste y límites de comparación |
| [08. Exportación y preservación](08_EXPORT_PRESERVATION_AND_REPRODUCTION.md) | Identidad del candidato, export de cabeza, Test12, archivos locales y reproducción |
| [09. Limitaciones y siguientes pasos](09_LIMITATIONS_AND_NEXT_STEPS.md) | Qué se puede afirmar, qué falta demostrar y qué no debe repetirse |
| [10. Publicación y validación](10_PUBLICATION_AND_VALIDATION.md) | Alcance del commit, comprobaciones actuales y conservación de bytes científicos |
| [Catálogo de evidencia](EVIDENCE_INDEX.md) | Recibos públicos compactos, hashes y procedencia |

## Cómo interpretar las fuentes

Los informes de cada campaña conservan su fecha y sus gates. Una decisión posterior no reescribe el resultado anterior. Las tablas de velocidad requieren leer también dispositivo, batch, precisión y fronteras del cronómetro: un forward sobre tensores preparados no mide la ruta desde sensores.

Los enlaces `../../artifacts/...` de los capítulos remiten al archivo local de trabajo. Esas carpetas están excluidas de Git por tamaño o naturaleza operativa: **no todos esos enlaces funcionarán en un clon público**. Los informes bajo `docs/` y el catálogo de evidencia proporcionan la parte publicable; los bundles completos de recuperación se conservan separadamente. Esta distinción evita confundir un recibo disponible localmente con un archivo incluido en GitHub.

Los modelos, datos crudos, cachés, entornos y credenciales no se añaden al repositorio. El objetivo de esta publicación es conservar código, metodología, resultados y trazabilidad; los checkpoints y predicciones completas tienen además su preservación local verificable.

## Referencias de identidad

- Base Git antes de esta publicación: `771c96a23784e1f3955cec21ba1f4d9a063a9145`.
- Código de la entrega histórica V18: `c3f14408babda11f9f9d50ab1e4158992e6f9c19`.
- Recibo documental histórico V18: `a811cc3a39334532af457034bab5457127335bbd`.
- Sello del candidato TRAIN40: `cf5f4eac22e572679c77cd0f6a16425f978600597d4be77f0b1e576c226165a4`.
- ZIP Test12 TRAIN40: `5eb69749de7aff682eea640d843ee635379702337d929c86f50472264f1d88bc`.

El hash del commit que contiene este índice se consulta con `git log -1`; no se escribe anticipadamente un identificador inexistente dentro del propio commit.
