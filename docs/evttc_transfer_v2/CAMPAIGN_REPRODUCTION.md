# Reproducción de la campaña adicional

Esta campaña continúa la entrega F1, sin sustituir sus pesos ni cambiar el
protocolo oficial. El plan y sus decisiones están en
`docs/superpowers/plans/2026-10-30-phase-campaign-completion.md`.

## Entradas

- `PHASE_DATA`: directorio de la caché verificada que contiene `training.npz`,
  `POPULATION.json`, `TARGET_LEDGER.json`, `BINDING.json` y `COMPLETE.json`.
- `PHASE_CANDIDATE`: candidato TRAIN40 original, con los nueve componentes y sellos.
- `PHASE_MODELS`: salida del refit final que contiene `seed_7`, `seed_13`, `seed_21`.
- `PHASE_SLIDER`: fuente Slider-500 pública: `slider_500.bag`, `bounding_box.csv`,
  `gt_ttc.csv`, `calibParams.mat`. No se incluyen sensores crudos en Git.
- `PHASE_RUN`: directorio de salida nuevo. Los recibos son inmutables: no mezclar
  una reproducción con cambios de código y los recibos de una ejecución anterior.

En el host local se utiliza `.venv-transfer`; el entorno original `.venv` no es
la instalación validada para este trabajo. Dependencias del repositorio y CUDA
deben estar disponibles. La fase de representación y productores necesita GPU;
los probes y la reescala de intervalos se ejecutan en CPU.

```powershell
$env:PYTHONPATH='src;.'
$env:PYTHONUTF8='1'
$env:UV_PROJECT_ENVIRONMENT=(Resolve-Path .venv-transfer).Path
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'

.venv-transfer/Scripts/python.exe -X utf8 scripts/analyze_phase_campaign.py --data "$env:PHASE_DATA" --candidate "$env:PHASE_CANDIDATE" --adapted "$env:PHASE_MODELS/seed_7/checkpoint.pt" --protocol data/splits/evttc_transfer_v2.yaml --output "$env:PHASE_RUN/diagnosis"
.venv-transfer/Scripts/python.exe -X utf8 scripts/train_phase_campaign_cv.py --data "$env:PHASE_DATA" --candidate "$env:PHASE_CANDIDATE" --protocol data/splits/evttc_transfer_v2.yaml --output "$env:PHASE_RUN/F1_grouped"
.venv-transfer/Scripts/python.exe -X utf8 scripts/calibrate_phase_campaign.py --data "$env:PHASE_DATA" --campaign "$env:PHASE_RUN/F1_grouped" --output "$env:PHASE_RUN/calibration"
.venv-transfer/Scripts/python.exe -X utf8 scripts/diagnose_phase_temporal_probes.py --data "$env:PHASE_DATA" --diagnosis "$env:PHASE_RUN/diagnosis" --protocol data/splits/evttc_transfer_v2.yaml --output "$env:PHASE_RUN/temporal_probes"
.venv-transfer/Scripts/python.exe -X utf8 scripts/evaluate_phase_external.py --source "$env:PHASE_SLIDER" --candidate "$env:PHASE_CANDIDATE" --models "$env:PHASE_MODELS" --output "$env:PHASE_RUN/Slider500" --prepare-only
.venv-transfer/Scripts/python.exe -X utf8 scripts/evaluate_phase_external.py --source "$env:PHASE_SLIDER" --candidate "$env:PHASE_CANDIDATE" --models "$env:PHASE_MODELS" --output "$env:PHASE_RUN/Slider500"
.venv-transfer/Scripts/python.exe -X utf8 scripts/diagnose_phase_external_geometry.py --external "$env:PHASE_RUN/Slider500" --output "$env:PHASE_RUN/geometry"
.venv-transfer/Scripts/python.exe -X utf8 scripts/diagnose_phase_restarts.py --external "$env:PHASE_RUN/Slider500" --candidate "$env:PHASE_CANDIDATE" --output "$env:PHASE_RUN/restarts"
.venv-transfer/Scripts/python.exe -X utf8 scripts/robustness_phase_campaign.py --data "$env:PHASE_DATA" --candidate "$env:PHASE_CANDIDATE" --adapted "$env:PHASE_MODELS/seed_7/checkpoint.pt" --protocol data/splits/evttc_transfer_v2.yaml --output "$env:PHASE_RUN/robustness"
.venv-transfer/Scripts/python.exe -X utf8 scripts/report_phase_campaign.py --campaign "$env:PHASE_RUN" --output "$env:PHASE_RUN/report"
.venv-transfer/Scripts/python.exe -X utf8 scripts/supplement_phase_review.py --campaign "$env:PHASE_RUN"
.venv-transfer/Scripts/python.exe -X utf8 scripts/verify_phase_campaign.py --campaign "$env:PHASE_RUN"
```

No ejecutar dos entrenamientos/inferencias GPU simultáneamente para interpretar
tiempos. La medición de robustez incluye instrumentación y perturbación; no es una
latencia de despliegue. La entrega anterior conserva el benchmark de paridad y el
perfil de extracción con CUDA Graphs, caché de sensores y carga previa de features.

## Contratos

La validación agrupada reinicia la cabeza original por fold. El checkpoint F1 que
vio las 32 grabaciones no participa como modelo reservado en esa tabla. La tabla
externa sí evalúa ese F1 completo, con semillas predefinidas, antes de abrir los
valores de etiqueta externos. Probes, filtro temporal y reescala son diagnósticos
separados y no se integran silenciosamente en el paquete final.

El perfil `event_aided_ttc` admite exclusivamente `/dvs/events` y `/aps/image_raw`
con los tipos ROS esperados. El perfil original de competición conserva sus
topics y validaciones. Las cajas usan la convención del adaptador; el overlay no
certifica la calibración intersensor, que sigue siendo provisional para Slider.

Las ausencias conservan NaN en NPZ y causa textual en JSON. Una tabla sobre filas
compartidas acompaña la cobertura completa. El error relativo excluye TTC cero
pero MAE lo conserva. No se reemplazan negativos ni ausencias con constantes.

La confirmación oficial, adquisiciones nuevas y genealogía no recuperable desde
los artefactos existentes siguen siendo límites externos; más entrenamiento sobre
la misma población no resuelve esos límites.
