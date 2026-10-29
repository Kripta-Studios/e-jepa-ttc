# Reparación de la materialización RGB OOF — 23 de octubre de 2026

La rotación 0 se detuvo a las 13:50 (Madrid), después de completar E_CTX,
con 137.500 updates científicos durables y ninguno pendiente. La preparación
de R_CTX falló en `materialize_rgb_observations`: `old != enriched` intenta
convertir arrays NumPy a un booleano al comparar observaciones compartidas.

La auditoría de las 88.744 consultas TRAIN40 encontró 89.875 observaciones
únicas y 294.897 reutilizaciones. Los únicos campos distintos entre referencias
a una misma observación fueron `sample_token`, `query_time_us` y `query_age_us`.
Son propiedades de la consulta, no de los frames/ROI de la observación.

La corrección compara recursivamente los metadatos del sensor, incluidos arrays
de objetos anidados, sin tolerancia numérica ni broadcasting. Excluye de esa
comparación únicamente los tres campos de referencia de consulta. Conserva el
rechazo de diferencias en grupos, frames, timestamps del sensor, ROI y cualquier
otro campo. El materializador original conserva cada fila de consulta y calcula
sus relojes usando su propio tiempo y el anclaje/disponibilidad de la observación.
`decode_query` no consume los tres campos excluidos.

## Implementación y verificación

- `operational/oof_rgb_metadata_fix.py` sustituye exactamente una comparación
  del AST de la función original, exigiendo su SHA-256 conocido.
- `operational/scientific_oof_rgb_fixed.py` comprueba los hashes de la reparación,
  las pruebas y la auditoría antes de llamar al ejecutor existente. Registra una
  sesión con la identidad de esta reparación.
- `scripts/run_rgb_fixed_oof_stage.ps1` conserva el protocolo de bloqueo,
  resultado durable y salida del lanzador anterior.
- 32 pruebas pasan: reproducción del fallo, arrays anidados, rechazo de conflictos,
  relojes distintos para consultas que comparten observación, recuperación de
  caché, materialización y verificaciones OOF existentes. Ruff también pasa.
- `scripts/audit_oof_rgb_metadata.py` reproduce la auditoría real sin inferencia
  ni optimizer updates.

Los 246 archivos científicos y los siete archivos del runtime de prefetch
anterior permanecen ligados a sus hashes originales. No se cambian pesos,
datasets, semillas, pérdidas, precisión ni scheduler. Los entrenamientos ya
completados se reutilizan. Las pruebas y la auditoría no ejecutan optimizer updates.

## Trazabilidad operativa

En `artifacts/final_refit_20261019_scientific_oof/`:

- `RGB_METADATA_REAL_AUDIT.json`: población real y diferencias observadas.
- `RGB_METADATA_REPAIR_TESTS_V2.xml`: resultado final de las 32 pruebas.
- `RGB_METADATA_REPAIR_ADMISSION.json`: hashes de código/evidencias y alcance.
- `RGB_METADATA_REPAIR_SESSION_<pid>.json`: aplicación efectiva por proceso.
- Plan, estado y recibo del error anteriores archivados antes de reanudar.

El plan operativo incorpora el lanzador corregido y sus dependencias; su nuevo
hash es `6d724a7cb8018ffcfb02c5a58b2c7c0a677fa5badd6e3e35c38cba7de0507fef`.
La autoridad científica `SCIENTIFIC_CONFIG.json` conserva el SHA-256
`3268e2998391b40bb72b07b6fe2bfc873f3bf61f159ca581b28575ba068d767e`.
La reparación es un registro adicional: no se reescribe retrospectivamente la
genealogía histórica ni se presenta como una mejora científica del modelo.

## Segundo fallo detectado durante la recuperación

El primer arranque de verificación encontró una discrepancia exclusiva en
`receipt_payload_sha256` al reutilizar E_CTX. El escritor y el validador original
usan JSON con escape ASCII; `_expected_role_receipt` usaba UTF-8 sin escape. La
ruta real contiene «Álvaro», de modo que las representaciones diferían aunque
arrays, campos de recibo, padres y normalizador fueran idénticos.

La revisión 2 utiliza la función de hash del escritor solamente al reconstruir
el recibo esperado. No cambia los recibos existentes ni la función global de
hash científica. La prueba de regresión reproduce el fallo en una ruta Unicode,
comprueba la reutilización sin modificar el recibo y mantiene el rechazo de
padres distintos. También se verificó la reutilización de la caché real E_CTX.
Se archivaron el plan, estado, admisión y recibo de error de la revisión 1.

## Avance observado tras la corrección

El intento 12 superó ambos puntos de fallo. Entre las 16:06:50 y las 16:07:25
(Madrid), R_CTX pasó de 640 a 1.056 observaciones durables (20 a 33 fragmentos),
con la cola RUNNING. Son aproximadamente 711 observaciones/minuto en una ventana
de 35 segundos, no una medición de 15 minutos ni una ETA de toda la campaña.
El recibo `RGB_METADATA_REPAIR_PROGRESS.json` conserva ambas lecturas. Se
mantienen los 137.500 updates científicos previos; esta generación no hace
optimizer updates. El monitor habitual de la cola continúa activo.
