# Índice público de evidencia

Este catálogo conserva identidades y recibos compactos del candidato final y de su evaluación de transferencia. No contiene datasets, pesos, archivos de despliegue privados ni credenciales. Publicar estos documentos no ejecuta entrenamiento, inferencia, validación científica adicional ni un envío a CodaBench.

El [manifiesto del catálogo](evidence/EVIDENCE_MANIFEST.json) registra tamaño y SHA-256 de cada documento, ubicación lógica del original y SHA-256 de su fuente. Los originales se copiaron byte por byte. Los resúmenes derivados son documentos nuevos: su hash no sustituye al del recibo original.

## Recibos originales

| Archivo | Evidencia que conserva | Alcance |
|---|---|---|
| [MODEL_SEAL.original.json](evidence/MODEL_SEAL.original.json) | Identidad del candidato, dependencias de despliegue, normalizadores y ascendencia de entrenamiento | El estado `SEALED_NOT_EVALUATED` corresponde al momento del sellado; no se reescribe retrospectivamente tras predecir. |
| [TEST12_PREDICTION_RECEIPT.original.json](evidence/TEST12_PREDICTION_RECEIPT.original.json) | 5.805 predicciones físicas y 6.762 tokens oficiales, bajo el sello fijado | Sin targets privados, sin fallback y sin envío externo. |
| [TEST12_PACKAGE.original.json](evidence/TEST12_PACKAGE.original.json) | ZIP de submission y su identidad | `PACKAGED_NOT_SUBMITTED`; no acredita puntuación oficial. |
| [PHASE_ONLY_HEAD_EXPORT.original.json](evidence/PHASE_ONLY_HEAD_EXPORT.original.json) | Exportación TorchScript y paridad de la cabeza para batches 1 y 7 | Es la cabeza sobre estado físico de 340 coordenadas; no exporta la cadena completa desde sensores. |
| [COHORT_REGISTRATION.original.json](evidence/COHORT_REGISTRATION.original.json) | Registro de 554 consultas de transferencia en 32 secuencias | Cohorte fijada sin leer GT; contiene las 46 consultas históricas y la extensión. |
| [COHORT_VALIDATION.original.json](evidence/COHORT_VALIDATION.original.json) | Unicidad, categorías, intersección y exclusiones por soporte de sensores | 512 consultas extendidas, cuatro compartidas con las históricas: 508 nuevas. |
| [TIMING_SUMMARY.original.json](evidence/TIMING_SUMMARY.original.json) | Resumen final de tiempo instrumentado de 5.805 consultas | No es un benchmark controlado de latencia. |
| [TIMING_METHODOLOGY.original.json](evidence/TIMING_METHODOLOGY.original.json) | Definiciones de tiempos, cachés, alias, exportación y telemetría | Distingue tiempo de pared, kernels GPU y preparación de datos. |

## Resúmenes derivados

| Archivo | Transformación y límites |
|---|---|
| [PRESERVATION_SUMMARY.derived.json](evidence/PRESERVATION_SUMMARY.derived.json) | Conserva los contadores y hashes de los grupos archivados; sustituye ubicaciones privadas por nombres de archivo. El recibo documenta 263.923 miembros únicos y nueve checkpoints de despliegue. Los archivos grandes no se han rehasheado de nuevo para crear este catálogo. |
| [FINAL_PACKAGE_REVIEW.derived.json](evidence/FINAL_PACKAGE_REVIEW.derived.json) | Conserva los hallazgos del examen técnico del ZIP; omite la ruta local del archivo. Registra identidad exacta de tokens, finitud y equivalencia a precisión float32. |
| [TRANSFER_REVIEW.derived.json](evidence/TRANSFER_REVIEW.derived.json) | Conserva el examen independiente de sellos, IDs, métricas e intervalos; sustituye rutas absolutas por nombres. Remite a los resultados completos ya publicados. |
| [LATENCY_SCOPE.derived.json](evidence/LATENCY_SCOPE.derived.json) | Convierte los estadísticos guardados de segundos a milisegundos. Mediana 112,04 ms y p95 140,94 ms en el recorrido instrumentado con cachés; excluye inicialización, preparación cruda y escritura de diagnósticos. No demuestra tiempo real ni latencia desde sensores crudos. |
| [PUBLICATION_AUDIT.derived.json](evidence/PUBLICATION_AUDIT.derived.json) | Fotografía anterior a la publicación: tamaños, búsqueda de secretos y problemas de CI/identidad de bytes detectados entonces. No describe automáticamente el estado del commit final ni sustituye comprobaciones posteriores. |

## Identidades principales

- Sello del modelo: `cf5f4eac22e572679c77cd0f6a16425f978600597d4be77f0b1e576c226165a4`.
- ZIP Test12: `5eb69749de7aff682eea640d843ee635379702337d929c86f50472264f1d88bc`.
- Manifiesto de transferencia: `9a87a31a435d7922c22c479fe3a8e425e6227b8fab55109c9a2f85a91bb34d31`.

Los [resultados completos de transferencia](../final_train40_transfer_20261027/RESULTS.json) y sus [filas por consulta](../final_train40_transfer_20261027/QUERIES.csv) ya están en `docs/`; no se duplican aquí. La evaluación es descriptiva sobre EvTTC previamente observado: no equivale al ranking Test12, a una reproducción exacta de Table VI ni a una confirmación ciega. Los targets válidos puntuados son positivos; estos resultados no validan TTC negativo.

## Material conservado fuera de Git

`LOCAL_PRESERVATION/` en el manifiesto identifica el archivo local de preservación `E_JEPA_POD2_RETURN_20261027`, no una URL pública ni un directorio incluido en este repositorio. Allí se conservan el candidato, checkpoints, normalizadores, predicciones, diagnósticos, entradas derivadas, configuraciones y paquetes de código congelado. Los originales bajo `artifacts/` tampoco se incluyen automáticamente en Git.

Las rutas `/workspace/...` que aparecen en recibos originales son ubicaciones históricas del entorno de ejecución; no son endpoints de acceso. Se mantienen porque forman parte de los bytes firmados. La ejecución en otro equipo necesita las dependencias, datos y artefactos de modelo identificados, además del código. Los hashes permiten verificar esos archivos; no hacen que los archivos estén disponibles por sí solos.

Para reproducir las tablas desde evidencia guardada no hace falta volver a entrenar. Regenerar predicciones desde cero sí requiere los medios públicos y el sistema completo. Las referencias de release, hashes de inputs y diagnósticos preservados permiten distinguir esas dos operaciones.

## Verificación posterior a la auditoría de publicación

El [recibo de QA](evidence/FOCUSED_QA_RECEIPT.original.json) y el [informe de QA](evidence/FOCUSED_QA_REPORT.original.md) documentan el trabajo posterior al snapshot de auditoría: 864 pruebas pasaron inicialmente, ocho fallaron y una se omitió; las cinco fixtures corregidas pasaron las 22 pruebas de comprobación dirigida y resolvieron los ocho fallos. La evidencia combinada cubre 872 pruebas aprobadas y una omitida, sin afirmar una repetición íntegra posterior. Los tests V8/V9 ya generan fixtures portátiles; este resultado posterior prevalece sobre el hallazgo inicial de dependencia de `artifacts/`.

La comprobación de [PowerShell](evidence/POWERSHELL_PARSE.original.json) cubre 80 scripts sin errores sintácticos. [Pyright con el entorno instalado](evidence/PYRIGHT_ACTUAL_ENV.original.json) no detectó errores en los cinco archivos de su configuración; no implica cobertura de todos los módulos nuevos. Ruff global seguía registrando 495 problemas y 144 archivos por formatear en ese recibo. No se ocultaron esos resultados ni se reformatearon masivamente fuentes congeladas.
