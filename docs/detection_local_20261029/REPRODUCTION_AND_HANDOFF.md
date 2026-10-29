# Reproducción y continuación del trabajo

Fecha: 2026-10-29. Estas instrucciones describen el estado de desarrollo
`5c9fc73062f109f808be732a4a10d6f26a91afee` y la ampliación documental posterior.
Los comandos usan PowerShell y parten de la raíz del repositorio. El entorno
validado es Windows; trasladar el entrenamiento a Linux exige resolver y verificar
sus dependencias de plataforma, además de conservar el protocolo científico.

## Elegir qué se quiere reproducir

| Operación | Recursos necesarios | Evidencia que produce |
|---|---|---|
| Regenerar tablas | Python estándar y recibos en Git | Tablas numéricas idénticas a los agregados archivados |
| Inspeccionar entrenamiento | Logs escalares en Git | Updates, pérdidas, tiempos y exposición por token |
| Inferir con el detector final | GPU, entorno detector, medios eAP, checkpoint local y fuentes congeladas | Cajas 3D y velocidad sin abrir etiquetas |
| Repetir el piloto | Lo anterior más etiquetas train y backbone ImageNet | Un run nuevo con su propio presupuesto y recibos |
| Ejecutar el puente PHASE | Predicciones temporales, RGB/eventos, entorno y candidato PHASE | TTC firmado, fase y límites, o abstención explícita |
| Puntuar en CodaBench | Contrato y plataforma oficiales, medios test completos | No se realizó en esta entrega |

Un nuevo entrenamiento con la misma semilla no garantiza identidad bit a bit:
el backward CUDA de `grid_sample` puede ser no determinista. La paridad exacta
registrada corresponde a inferencia con el mismo checkpoint y sus fuentes.

## Repositorio y archivos de referencia

La rama publicada es `dev/eap-detection-local-20261029` en
`Kripta-Studios/phase-ttc`. Para crear una copia nueva:

```powershell
git clone --branch dev/eap-detection-local-20261029 `
  https://github.com/Kripta-Studios/phase-ttc.git phase-ttc-detection
Set-Location phase-ttc-detection
git rev-parse HEAD
```

El piloto se ejecutó sobre `3ebac039e56e3e5af98e0e3302084f00923235a1`.
El commit `5c9fc730` incorporó resultados y las protecciones finales. No se debe
atribuir al run antiguo la validación del manifest añadida después. El ZIP
[portable_pilot_code_snapshot.zip](evidence/portable_pilot_code_snapshot.zip)
conserva las diez fuentes ejecutadas y el patch de upstream; su manifiesto de
hashes permite verificar los bytes antes de reutilizarlos.

La nueva documentación incluye copias exactas de
[pilot_manifest.json](evidence/pilot_manifest.json),
[pilot_temporal7_tokens.json](evidence/pilot_temporal7_tokens.json) y los tres logs
de entrenamiento bajo `evidence/training_curves/`. El
[recibo de archivo](evidence/documentation_archive.json) conserva rutas, tamaños y
hashes. Git desactiva la normalización de finales de línea en esos recibos.

## Entornos y datos disponibles en este ordenador

El worktree activo está en el directorio hermano `phase-ttc-detection-local`.
La carpeta original `e-jepa-ttc-v18-joint-measurement` conserva `main` en
`f2903f0bfd47d7bf56358a94ad5fac0b8a8906ef`.

| Recurso local | Uso y límite |
|---|---|
| `E:/eAP_dataset` | Medios y metadatos públicos; esta fase usa el split público train |
| `E:/GarlTTC_dataset`, `E:/Garl-TTC` | Inspección de correspondencias y código histórico; no suministran IDs al puente |
| `.venv-detector/Scripts/python.exe` | Python 3.11.15, Torch 2.11.0+cu128, torchvision 0.26.0+cu128; detector RGB |
| `.venv-audit/Scripts/python.exe` | Auditoría CPU, Parquet, pytest y Ruff; no tiene Torch |
| `../phase-ttc-main-integration/.venv/Scripts/python.exe` | Entorno PHASE existente usado en la suite focalizada y el puente |
| `D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE` | Candidato PHASE verificado: nueve checkpoints más normalización y metadatos |

El entorno detector no contiene `h5py`: sirve para el piloto RGB, pero no basta
para las pruebas HDF5 ni para preparar eventos del puente. La primera colección
de tests en ese entorno falló por esta dependencia y el recibo se conservó. La
suite focalizada pasó después en el entorno PHASE existente sin modificarlo.

En una instalación nueva de Windows, el punto de partida es el lock con hashes
del detector. Los siguientes comandos son una receta de instalación; la entrega
comprobó el entorno registrado, no una reinstalación adicional en otro equipo:

```powershell
uv venv --python 3.11.15 .venv-detector
uv pip sync --python .venv-detector/Scripts/python.exe --require-hashes `
  --extra-index-url https://download.pytorch.org/whl/cu128 `
  --index-strategy unsafe-best-match `
  docs/detection_local_20261029/evidence/portable_dependency_lock.txt
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
```

El `uv.lock` principal de PHASE se conserva. Para el entorno PHASE y sus pesos,
consulte también el [README del proyecto](../../README.md); copiar únicamente
el entorno detector no reproduce el ensemble PHASE.

## Fuentes y backbone del detector

El runner requiere el repositorio original fijado y sin cambios tracked. En una
copia nueva del proyecto:

```powershell
git clone https://github.com/PuFanqi23/MonoDGP.git output/detector_sources/MonoDGP
git -C output/detector_sources/MonoDGP checkout aa059a18214aebf644510e7f0793971b403f9d14
```

El runner crea la variante portable en otra carpeta y guarda el patch. No hay
que ejecutar `setup.py` ni compilar la extensión CUDA externa para esta ruta.
El piloto utilizó el archivo oficial `resnet50-0676ba61.pth`, 102.530.333 bytes,
SHA-256 `0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a`.
La URL usada fue `https://download.pytorch.org/models/resnet50-0676ba61.pth`.
El helper exige el hash y verifica los tensores cargados; no descarga pesos de
forma implícita. En este equipo el archivo ya está en `output/external_weights/`.

## Datos exactos y nuevo entrenamiento

Para conservar la selección original, utilice el manifest archivado. El SHA-256
del candidato debe ser `91a13c5ae7f6e40ee5fdb4a62da481eba5ffff2b8209ea1110befe3ad7cdc903`;
el de `data/train.parquet`,
`69ffedab74ea28b295fe3ec2c2c83c08716bc31c216bc2ec9971f8ca9644a314`.
El nuevo guard comprueba esos enlaces y la disjunción antes de construir el modelo.

```powershell
$detectorPython = '.venv-detector/Scripts/python.exe'
$eapRoot = 'E:/eAP_dataset'
$pilotManifest = 'docs/detection_local_20261029/evidence/pilot_manifest.json'
$candidateSplit = 'data/splits/eap_detection_candidate_v1.json'
$newRun = 'output/portable_experiments/pilot256_seed7_reproduction'

& $detectorPython scripts/train_monodgp_portable.py `
  --data-root $eapRoot --manifest $pilotManifest --candidate-split $candidateSplit `
  --output $newRun --role train --steps 512 --max-seconds 1200 `
  --group-num 11 --batch-size 4 --width 640 --height 384 `
  --lr 0.0002 --seed 7 --predict-velocity --velocity-loss-coef 0.1 `
  --backbone-weights output/external_weights/resnet50-0676ba61.pth `
  --precision fp32 --matmul-precision highest
```

El directorio de salida debe ser nuevo. El límite de 1.200 segundos se aplica
al entrenamiento; preparación, evaluaciones y guardado añaden tiempo. La máquina
completó los 512 updates del piloto histórico dentro del límite. Otro run puede
terminar antes y deberá reportar sus exposiciones reales, sin ampliar el presupuesto
después de ver selección.

Para elegir otra población, `scripts/prepare_eap_rgb_pilot.py` acepta `--eap-root`,
`--split`, `--output`, `--receipt`, `--seed` y `--frames-per-sequence`. Fije su
protocolo antes del entrenamiento y conserve el nuevo manifest. Su timestamp
cambia el hash aunque produzca los mismos tokens. La opción `--check-events`
requiere un entorno con HDF5 y los filtros de compresión; es innecesaria para
repetir únicamente el entrenamiento RGB.

## Checkpoint final e inferencia sin etiquetas

El checkpoint registrado está en
`output/portable_experiments/pilot256_seed7_velocity_v1/checkpoint.pt`.
Pesa 498.531.791 bytes y su SHA-256 es
`dc0ce304ad1c8a791ef66395be85b09f22526bc11a834472dcb18ce16384e338`.
Git no incluye sus tensores. Para trasladarlo, copie también `preregistration.json`
y `code/` de ese run; la inferencia verifica el checkpoint y cada módulo congelado.
Los archivos archivados en Git permiten auditar la procedencia, pero no sustituyen
el checkpoint.

```powershell
$originalRun = 'output/portable_experiments/pilot256_seed7_velocity_v1'
Get-FileHash "$originalRun/checkpoint.pt" -Algorithm SHA256

& $detectorPython scripts/infer_monodgp_eap.py --data-root $eapRoot `
  --checkpoint "$originalRun/checkpoint.pt" --manifest $pilotManifest `
  --role selection --output "$originalRun/inference_selection_reproduction.json" `
  --reference-predictions "$originalRun/selection.json"

& $detectorPython scripts/infer_monodgp_eap.py --data-root $eapRoot `
  --checkpoint "$originalRun/checkpoint.pt" `
  --tokens-json docs/detection_local_20261029/evidence/pilot_temporal7_tokens.json `
  --output "$originalRun/inference_temporal7_reproduction.json"
```

El comparador opcional lee predicciones guardadas; la ruta de inferencia no abre
etiquetas. Si `selection.json` no está disponible, omita `--reference-predictions`:
se podrá inferir, pero no registrar paridad frente a aquel archivo. Use nombres
nuevos en cada ejecución para conservar sus recibos.

## Controles y puente PHASE

Los controles son CPU y usan los JSON de evaluación del run:

```powershell
& $detectorPython scripts/check_detection_image_signal.py `
  --train-results "$originalRun/final_train.json" `
  --selection-results "$originalRun/selection.json" `
  --output "$originalRun/image_control_reproduction.json"

& $detectorPython scripts/check_detection_velocity_signal.py `
  --train-results "$originalRun/final_train.json" `
  --selection-results "$originalRun/selection.json" `
  --output "$originalRun/velocity_control_reproduction.json"

$phasePython = '../phase-ttc-main-integration/.venv/Scripts/python.exe'
& $phasePython scripts/infer_detected_roi_phase.py --eap-root $eapRoot `
  --predictions "$originalRun/inference_temporal7_reproduction.json" `
  --sample-token qGsgzl4Q8B:000000929_000731755 `
  --candidate-root D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE `
  --output "$originalRun/phase_bridge_reproduction.json" --device cpu --cpu-threads 1
```

El puente conserva los umbrales predefinidos. Puede abstenerse por falta de historia,
ambigüedad o preparación inválida. Una abstención forma parte del resultado;
no cambie de objeto para conseguir una salida más favorable. q10/q90 permanecen
en fase y el ejemplo de train no mide generalización TTC.

## Auditoría sin GPU y continuación

```powershell
python scripts/summarize_detection_runs.py --docs docs/detection_local_20261029
```

Las [232 pruebas focalizadas](evidence/final_rgb_pipeline_tests.xml) ya se ejecutaron;
no hace falta repetir entrenamiento para auditar la documentación. Los logs
archivados permiten comprobar cuántas veces apareció cada token y dónde terminó
cada presupuesto. Los [resultados detallados](EXPERIMENTS_AND_FINDINGS.md) conservan
el fallo del primer overfit y la baja AP del piloto.

Antes de ampliar entrenamiento, el siguiente agente debe fijar soporte de las
ocho clases, agrupación de sesiones, presupuesto y criterios. Debe medir tiempo
de pared con lectura y preparación, conservar selección y calibración separadas
y tratar la exposición TRAIN40 de PHASE como una limitación independiente.
Una submission oficial requiere resolver el contrato y scorer pendientes y
completar los medios test; el funcionamiento del piloto no los da por aprobados.

El trabajo de esta ampliación documental no ejecuta nuevos modelos ni cambia
sus métricas. Solo añade documentación y archiva metadatos y curvas existentes.
