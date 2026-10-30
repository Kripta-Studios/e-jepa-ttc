# PHASE-ADAPT: refit final con datos locales compatibles

Estado: **modelo entrenado, exportado, verificado y empaquetado el 30 de octubre de 2026**.
Se entrega PHASE-ADAPT F1, semilla 7, tras dos épocas completas y 1.444 actualizaciones
por réplica sobre 92.315 ejemplos. Las semillas 13 y 21 también han terminado.

La instrucción del usuario del 30 de octubre autoriza entrenar un candidato final
usando todos los datos locales compatibles. Este experimento conserva la arquitectura
F1: ocho productores/contextos congelados y ajuste de los 1.955 parámetros de
`PhaseOnlyFinal`, inicializados desde el candidato TRAIN40 preservado.

El inventario contiene 88.744 estados OOF de TRAIN40 y 3.828 consultas nativas con
sensores/ROI compatibles de las 32 grabaciones EvTTC. De estas últimas, 3.571 tienen
target dentro del soporte de interpolación y salida; 257 se conservan como exclusiones.
Las 3.571 consultas con target admitido produjeron un estado de fusión válido.

Los datos de competición derivados de esas grabaciones no se añaden por duplicado.
Slider no tiene todavía un contrato supervisado compatible verificado en esta ruta.
Los targets privados y las cachés TEST12 no forman parte del entrenamiento. La
exposición de EvTTC queda declarada; el modelo adaptado no es transferencia sin ajuste.

La receta se fija antes de entrenar: AdamW1e-4, decay1e-4, batch128, clip1, hasta
30épocas completas y2000updates; peso total igual por dominio y por grabación dentro
del dominio. Las semillas7,13,21 producen réplicas. Se entrega la última iteración
de la semilla7 por elección previa, sin seleccionar por score o por semilla.

La ponderación por grabación no certifica independencia de adquisiciones. Los
resultados de ajuste se calculan sobre entrenamiento y no prueban generalización,
calibración fuera de dominio ni superioridad oficial. El protocolo anterior de
auditoría sigue intacto y conserva sus límites.

## Reproducción

Desde el repositorio con `.venv-transfer` y los recursos locales preservados:

```powershell
$env:PYTHONPATH = 'src;.'
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
.venv-transfer/Scripts/python.exe scripts/prepare_phase_adapt.py --candidate "$env:EVTTC_CANDIDATE_ROOT" --manifest data/manifests/evttc_all32_local.yaml --data-root "$env:EVTTC_DATA_ROOT" --output output/reproduction_phase_adapt/data --device cuda
.venv-transfer/Scripts/python.exe scripts/train_phase_adapt.py --candidate "$env:EVTTC_CANDIDATE_ROOT" --data output/reproduction_phase_adapt/data --output output/reproduction_phase_adapt/model --device cuda
.venv-transfer/Scripts/python.exe scripts/report_phase_adapt.py --data output/reproduction_phase_adapt/data --model output/reproduction_phase_adapt/model
```

El primer comando valida la preservación de TRAIN40 desde el directorio padre del
candidato. Las rutas de datos se pasan como argumentos y no están codificadas en
los módulos. Las salidas distintas deben usar directorios nuevos; los recibos son
inmutables. `--migrate-from` permite una migración explícita y verificada de los
fragmentos de la primera implementación de esta campaña, conservando su procedencia.

Para reproducir la extracción acelerada, generar primero un inventario separado con
`prepare_phase_adapt.py --inventory-only`, manteniendo los demás argumentos. Ejecutar
`scripts/benchmark_phase_adapt_runtime.py --population <inventario>/POPULATION.json
--candidate <candidato> --output <recibo_paridad>.json` y pasar después
`--runtime-evidence <recibo_paridad>.json` al extractor en un directorio nuevo.
El recibo vincula el candidato, el inventario y el código del adaptador acelerado.

La ruta acelerada usa caché acotada de eventos y frames, cuatro trabajadores para
recortar RGB y CUDA Graphs de los productores con calentamiento y admisión de paridad.
Conserva los batches nativos y la ruta original de contexto/fusión. La comprobación
de esta campaña comparó 36 consultas de las 32 grabaciones y obtuvo igualdad exacta
en los 340 valores de entrada, los logits de anchura y las tres predicciones TTC.
Se conserva compresión NPZ sin pérdida; el entrenamiento precarga las features y
targets en GPU una vez por semilla. La implementación no cambia precisión, resolución,
historia, ROI ni población para ganar velocidad.

El exportado TorchScript corresponde a la cabeza de fusión. Requiere entradas
FP32 finitas `[B,340]` y `[B,2]` producidas por la cadena incluida; no recibe eventos
crudos directamente. La ruta de sensores utiliza `AdaptedPredictor`, que verifica
el grafo padre, el checkpoint nuevo, su semilla y su población de entrenamiento.


## Entrega y evidencia medida

- Paquete local: `output/phase_adapt_final/PHASE_TTC_ADAPT_F1_S7_2026-10-30.zip`.
- Modelo extraído: `output/phase_adapt_final/model/FINAL_MODEL/`.
- Ficha y tabla generadas: `MODEL_CARD.md` y `training_metrics.csv` dentro del modelo.
- Recibos versionados: [evidence/phase_adapt_final](evttc_transfer_v2/evidence/phase_adapt_final/RESULTS.json).
- La población final contiene 88.744 estados TRAIN40 y 3.571 EvTTC, de las 32 grabaciones;
  257 consultas quedan excluidas por target no compatible. No hubo exclusiones adicionales
  por falta de estado de fusión en las consultas con target admitido.
- Cada réplica visitó todos los ejemplos en ambas épocas. Se verificó que hubo actualizaciones
  reales, que los pesos finales son finitos y que sus métricas sobreviven a guardar/cargar.
- Los 22 archivos del paquete pasaron verificación SHA-256; exportación CPU exacta y replay
  desde sensores crudos pasaron. El ZIP contiene 24 entradas y pasó la comprobación CRC.
- La mejora de ajuste EvTTC coexiste con un pequeño aumento del MAE en TRAIN40. Las cifras
  completas, incluidas las tres semillas, proceden de RESULTS.json; no se sustituye la
  semilla elegida previamente ni se presenta el modelo como una mejora universal.

La extracción acelerada calculó 2.228 estados nuevos: media 0,5299 s, mediana 0,3967 s,
p95 1,3020 s por consulta sensorial. Estos tiempos incluyen preparación e inferencia y
excluyen la escritura posterior del shard. Se conservaron además 1.377 shards migrados,
34 de ellos ajenos a los targets admitidos; 1.343 contribuyeron a la población final.
La preparación de sensores consumió el 69,66 % del tiempo medido. Los contadores registran
205,75 GB acumulados de eventos servidos desde caché y 42,45 GB leídos. Véase PERFORMANCE.json.

En Windows, cargar TorchScript desde un archivo binario abierto evita el problema de
`fopen` con rutas Unicode absolutas. La ficha generada incluye esta indicación; la paridad
se volvió a verificar usando esa ruta de carga.

Este cierre entrega el refit F1 de la campaña. Los experimentos condicionales JEPA, la
confirmación externa de Slider y la certificación de SOTA siguen requiriendo sus propios
criterios y datos; no quedan declarados como ejecutados por entrenar esta cabeza.
