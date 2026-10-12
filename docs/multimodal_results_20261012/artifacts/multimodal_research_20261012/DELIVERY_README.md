# Continuación multimodal: evidencia y reproducción

El candidato es `PHASE_ONLY_EVENT_S7`: una corrección de 1.955 parámetros sobre R_CTX, con productores, PAIR, refinadores y normalizadores V13 congelados. Ese número cuenta sólo la cabeza nueva, no todo el sistema. Las semillas 7/13/23 pasan los criterios internos originales del 2 %. La selección es la primera semilla registrada, no la mejor semilla externa.

Esta investigación ejecutó 32 fits completos de 2.500 updates: 80.000 físicos/científicos, sin recuperaciones. La autorización ampliada del usuario reemplazó el presupuesto inicial de cero updates; el informe histórico de la etapa sin entrenamiento se conserva aparte.

## Archivos principales

- `MULTIMODAL_REVIEW_FINAL.md`: informe final generado cuando existe el paquete test12 verificado. Mientras la inferencia está activa, consultar `CURRENT_RESEARCH_REVIEW.md` y `RECOVERY_STATUS.json`.
- `NEXT_DECISION.json`, `ACCOUNTING.json`, `AUTHORIZATION.json`: decisión, consumo real y autorización.
- `test12_phase_only/MODEL_INPUT_FREEZE.json`: identidad inmutable de la receta, padres, fuentes e inputs.
- `test12_phase_only/INPUT_AUDIT.json`: separación de grupos P/H/V/test, causalidad, identidades físicas, fuentes científicas originales congeladas y recuperación declarada.
- `test12_phase_only/CACHE_RESUME_PARITY.json`: paridad de todos los arrays en 16 consultas guardadas seleccionadas por hash. `PREDICTION_CACHE_INTEGRITY_INCIDENT.json` conserva la cuarentena de ocho archivos corruptos previos.
- `test12_phase_only/TEST_PREDICTIONS.csv`: predicciones de los cinco sistemas comparados, una fila por token oficial; disponible al completar la inferencia.
- `test12_phase_only/PHASE_ONLY_S7_GarlTTC_submission.zip` y `PACKAGE_VERIFICATION.json`: candidato en formato oficial, verificado localmente y no enviado.
- `studies/`: todos los ensayos, incluyendo resultados negativos, controles, cobertura y contrastes.
- `POSITIVE_BUCKET_COMPONENTS.csv`: contribución observada de los buckets positivos con pesos oficiales; el componente negativo sigue desconocido y no se publica como score completo.
- `PHASE_ONLY_GARL_POSTHOC.csv`: contrastes descriptivos por secuencias contra Garl y F_TRUE.
- `PHASE_ONLY_ATTRIBUTION.*`, `PHASE_ONLY_STRATA*`, `PHASE_ONLY_WRONG_SIGN_CASES.csv`: trazas y localización de fallos, sin modificar predicciones.
- `PHASE_ONLY_ROUTE_COST.*`: 128 mediciones CPU con sensores y transformaciones precalculadas. `phase_only_cost_v1/` conserva la primera versión.
- `FCWD_GPU_ROUTE_COST.*`: medición adicional, si está completa, incluyendo registro fresco y padres GPU; su propio recibo define el perímetro y la paridad.
- `METRICS_REPLAY_VERIFICATION.json`: recálculo de 13 tablas internas y 34 externas, con 46.778 comprobaciones escalares y 32 endpoints verificados.

## Verificar el bundle

Desde la raíz de una extracción nueva del ZIP de investigación, ejecutar `python VERIFY.py`. Usa sólo la biblioteca estándar y verifica tamaño y SHA-256 de cada payload del manifiesto. El archivo `.verification.json` junto al ZIP verifica también su CRC y hash global. El ZIP antiguo V16 se conserva.

Para recalcular las métricas desde los CSV, con las dependencias Python del proyecto instaladas, ejecutar desde esa misma raíz:

```powershell
$env:PYTHONPATH = 'src;.'
python -B -m operational.multimodal_research.research_metrics_replay
```

Este comando no realiza inferencia sensorial, no entrena y no utiliza etiquetas privadas. La comprobación de una extracción independiente se registra por separado cuando termina el bundle.

## Dependencias sensoriales

El bundle incluye fuentes, predicciones, cachés de estados, pesos nuevos y los doce endpoints V13 con normalizadores y recibos. `DEPENDENCY_INDEX.json` relaciona las dependencias externas archivadas con sus rutas originales. Incluye el parquet público de inputs test12 y el freeze histórico que identifica los tensores event actuales. Se preservan las rutas históricas: no se afirma que el runtime sensorial funcione tras mover la extracción sin preparar esas dependencias.

Los HDF5, los archivos RGB y la caché sensorial histórica H8 permanecen externos, identificados por los freezes. De esa caché se utilizó únicamente la observación event actual verificada contra el adaptador nativo, no la historia ni la predicción H8. Los checkpoints públicos Garl full/event permanecen externos; el visual descargado sí está incluido. No se descarga nada al verificar hashes o recalcular CSV.

## Alcance científico

El 4 % es una sensibilidad posterior del margen de degradación permitido en TTC largo. No cambia los scores ni sustituye el gate original. V, Dev32 y FCWD son desarrollo expuesto; los intervalos externos son descriptivos y no corrigen la búsqueda adaptativa. Las tres semillas replican cabezas, no productores completos. Las métricas externas carecen de etiquetas negativas y no constituyen un `overall_MiD` oficial.

Test12 conserva los 6.762 tokens, aunque 957 sean aliases de observaciones físicas repetidas. Las historias son causales, de la misma identidad y con soporte de hasta 650 ms. El sistema usa RGB y eventos; FCWD emplea registro estimado asistido por eventos, no calibración extrínseca oficial. No se presenta esta receta como un nuevo refit de TRAIN40.

No hay puntuación oficial ni SOTA demostrado. El release requiere scoring oficial con etiquetas privadas. El ZIP sólo acredita formato y contenido de las predicciones. No se ha hecho ningún envío, y la instrucción original del usuario excluye submissions. `OFFICIAL_ACCESS_REVIEW.json` conserva además la falta actual de navegador conectado y de una tabla de clasificación verificable.

La recuperación no cambia el modelo: carga caches verificadas y usa el mismo bloque de cálculo. El control completo se ejecuta por bloques acotados, manteniendo límites de memoria, presupuesto y caducidad. `WAITING_OWNER_CONTRACT.json` distingue un supervisor V18 en espera de un trabajador GPU activo; sus fuentes de coordinación observadas se conservan en `resource_sources/V18/`. Ningún proceso o fichero de V18 se modifica.
