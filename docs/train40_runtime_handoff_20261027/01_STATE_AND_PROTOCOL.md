# Estado, identidad y protocolo

## Identidades que no se deben cambiar

- Candidato: `PHASE_ONLY_EVENT_S7_TRAIN40_SCIENTIFIC_OOF`.
- Nueve checkpoints: E_A5, E_C2F, R_A5, R_C2F, PAIR_E, PAIR_R, E_CTX_MATCHED, R_CTX y PHASE_ONLY_EVENT_S7_TRAIN40.
- Dos normalizadores de 17 coordenadas; modelo multimodal RGB+eventos.
- MODEL_SEAL SHA-256: `cf5f4eac22e572679c77cd0f6a16425f978600597d4be77f0b1e576c226165a4`.
- ZIP Test12 SHA-256: `5eb69749de7aff682eea640d843ee635379702337d929c86f50472264f1d88bc`.
- Manifiesto EvTTC: `artifacts/final_train40_transfer_20261027/QUERY_MANIFEST.json`, SHA `9a87a31a435d7922c22c479fe3a8e425e6227b8fab55109c9a2f85a91bb34d31`.

El contrato preservado incluye 554 consultas físicas de 32 secuencias, con cuatro observaciones event y cinco RGB por consulta. Las cuarenta consultas sin un target TTC asociable siguen siendo casos de inferencia y deben incluirse en paridad y latencia. Hay 46 consultas históricas y 508 extendidas sin duplicar anclas. No leer TTC para optimizar ni seleccionar casos favorables.

Los 32 HDF5 registrados existen localmente y suman 197.781.014.225 bytes. No hace falta descargarlos. El conjunto original contiene un objeto primario anotado por consulta; no prueba capacidad sobre veinte objetos reales.

## Criterio numérico fijado antes de los probes

`prediction_parity` exige cobertura exacta de IDs, ausencia de duplicados, mismas historias y motivos de disponibilidad RGB, y coincidencia de las tres salidas con `atol=1e-5`, `rtol=1e-5`. Registra por separado igualdad exacta, máscara de finitud, error absoluto y relativo. No aumentar tolerancias después de ver un fallo.

Las representaciones CPU optimizadas conservan las fórmulas originales. El ring de momentos enteros reordena sumas y por ello necesita su propia admisión, aunque en los tres casos medidos la salida float32 coincida exactamente. Reutilizar encoders o cambiar batch también puede cambiar reducciones de convolución. La igualdad serial/paralelo de una variante no demuestra su igualdad contra el oracle original.

El rango de TTC original incluye valores negativos y el clamp cercano a −60 s. No filtrar negativos, saturados o consultas sin GT. Las comprobaciones de guards no se pueden omitir para reducir latencia.

## Fronteras de reloj

1. **Offline con disco:** `benchmark_train40_runtime.py` mide lectura/cache, ROI, representaciones, transferencias, modelos y salidas host. Alterna orden de rutas y conserva repeticiones por separado. La caché del sistema operativo no está controlada; se declara.
2. **Último paquete RAW disponible:** `benchmark_train40_integral.py` reconstruye antes el historial causal y carga solo frames RAW; después cronometra el paquete nuevo de 20 ms, ROI actual, representaciones, transferencias, modelos y TTC host. La primación de formas usa datos sintéticos una sola vez. La fase temporal irregular de cada consulta original requiere reinicialización explícita. Este runner valida consultas registradas, no continuidad.
3. **Tensores GPU preparados:** probes de productores y compilación. No incluyen la entrada. No anunciar sus milisegundos como sensor→TTC.
4. **Servicio sostenido pendiente:** reloj de liberación independiente `t0+k*20 ms`, veinte IDs distintos por actualización, sin reiniciar deadlines cuando el servicio se retrasa. Registrar tiempo hasta el último TTC, espera de cola, descartes y trabajo pendiente. Repetir una consulta o distribuir su tiempo entre veinte no sirve.

`scheduled_stream_report` exige veinte objetos distintos completados en cada actualización por defecto, 50 Hz, 60 segundos de observación prospectiva, cobertura, p95/p99, y cero trabajo anterior pendiente en cada llegada. Además registra el gate estricto `max response <20 ms`. Una declaración de scope no reemplaza evidencia de llegadas físicas. Los runners actuales marcan explícitamente la meta como no demostrada.

## Preservación y fallos

Los runners verifican sellos, fragmentos y archivos antes/después. Incluyen fuentes aditivas en el snapshot: modificar código durante un ensayo invalida la evidencia. Eso ocurrió en `pilot_v1`; se conserva como FAILED aunque sus predicciones pasaran. `integral_pilot_v1` sí concluyó COMPLETE con 601 archivos sin cambios.

No se han vuelto a escribir ni inferir dentro de las carpetas de predicciones originales. Todas las salidas nuevas usan directorios distintos. Las fuentes científicas originales ligadas al archivo cloud deben seguir pasando `verify_frozen_runtime`.
