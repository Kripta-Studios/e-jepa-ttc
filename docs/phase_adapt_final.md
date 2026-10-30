# PHASE-ADAPT: refit final con datos locales compatibles

Estado: extracción completa en ejecución; esta ficha se completará con los recibos
del modelo entrenado antes de la entrega.

La instrucción del usuario del 30 de octubre autoriza entrenar un candidato final
usando todos los datos locales compatibles. Este experimento conserva la arquitectura
F1: ocho productores/contextos congelados y ajuste de los 1.955 parámetros de
`PhaseOnlyFinal`, inicializados desde el candidato TRAIN40 preservado.

El inventario contiene 88.744 estados OOF de TRAIN40 y 3.828 consultas nativas con
sensores/ROI compatibles de las 32 grabaciones EvTTC. De estas últimas, 3.571 tienen
target dentro del soporte de interpolación y salida; 257 se conservan como exclusiones.
La cantidad final utilizada depende además de que exista estado de fusión válido.

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
```

El primer comando valida la preservación de TRAIN40 desde el directorio padre del
candidato. Las rutas de datos se pasan como argumentos y no están codificadas en
los módulos. Las salidas distintas deben usar directorios nuevos; los recibos son
inmutables. `--migrate-from` permite una migración explícita y verificada de los
fragmentos de la primera implementación de esta campaña, conservando su procedencia.

El exportado TorchScript corresponde a la cabeza de fusión. Requiere entradas
FP32 finitas `[B,340]` y `[B,2]` producidas por la cadena incluida; no recibe eventos
crudos directamente. La ruta de sensores utiliza `AdaptedPredictor`, que verifica
el grafo padre, el checkpoint nuevo, su semilla y su población de entrenamiento.
