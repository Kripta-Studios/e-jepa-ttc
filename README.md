# E-JEPA-TTC / OGE-JEPA-TTC

Investigación reproducible de Time-to-Collision con cámaras de eventos. La ruta
activa compara un Event-JEPA global auditado, representación densa causal,
geometría object-centric y una réplica local de Garl-TTC sobre EvTTC-32.

Estado al 30 de julio de 2026:

- `BASE` histórico reproducido con predicciones idénticas byte a byte;
- FlowMimic e inverse-TTC global rechazados por resultados negativos;
- confirmación histórica matched: Dense Patch supera a A0 en ese split;
- grouped CV cerrado (5 folds × 3 seeds): A0 supera a A1 en score y error
  relativo, por lo que A0 es la arquitectura final;
- `R1_MATCHED_BBOX_ROI` completó cinco folds con seed 7 y fue rechazado: usar
  la bbox solo para pooling empeora A0 en score, error relativo y MAE;
- AttnRes y Object-KDA no pasan el gate y no se combinan con el finalista;
- geometría bbox causal y port STRTTC evaluados sin superar todavía el gate;
- screen local Garl G0–G7 ejecutado, aún sin paridad con el protocolo de 50
  épocas y pretraining por ramas del repositorio oficial;
- eAP train-40 completo (536,64 GiB), reservado para un piloto posterior de
  pretraining sin TTC oficial;
- CARLA DVS Looming verificado: 1.406 secuencias, 1.395 utilizables con contexto
  de 100 ms y loader mmap sin duplicar los 71,64 GiB extraídos;
- Benchmark-10 sellado y no abierto.

Documentación:

- [estado verificable](STATUS.md);
- [plan completo v6](PLAN.md);
- [informe técnico](docs/technical_report.md);
- [informe PDF](docs/e_jepa_ttc_paper.pdf);
- [protocolo de datos](docs/dataset_card.md);
- [model card](docs/model_card.md).

## Resultado de referencia

`B0_HISTORICAL_BASE_EXACT`, seed 7, checkpoint downstream de época 26/30:

| Split | MAE | RMSE | Error relativo medio |
|---|---:|---:|---:|
| validation histórica | 0,322892 s | 0,584432 s | 8,1554 % |

La reproducción está en
`artifacts/audit/oge_sota/historical_base_reproduction.json` y demuestra
paridad exacta con el artefacto original. Este resultado es un ancla histórica:
la comparación de arquitectura usa `A0_MATCHED_GLOBAL`, no reutiliza esta fila
como si hubiera sido entrenada con la matriz nueva.

No existe todavía un claim SOTA ni un resultado oficial de Benchmark-10.

## Confirmación matched Core

Los cuatro brazos recibieron el mismo checkpoint BASE, las mismas 1.208/314
ventanas, máximo de 40 épocas, batch efectivo 32, optimizador y regla de early
stopping. El número de épocas completadas puede variar únicamente por esa regla
común.

| Brazo | Mejor época | Épocas | Error rel. macro | MAE macro | ms/ventana |
|---|---:|---:|---:|---:|---:|
| A0 global | 17 | 23 | 16,129 % | 0,701 s | 8,98 |
| **A1 Dense/Patch Policy** | **20** | 26 | **15,210 %** | **0,628 s** | 17,15 |
| A2 + AttnRes | 10 | 16 | 16,136 % | 0,653 s | 16,70 |
| K1 Object-KDA | 7 | 13 | 16,960 % | 0,731 s | 16,99 |

El screen de ocho épocas produjo un falso orden para A1: la representación
densa converge más lentamente. Con entrenamiento suficiente, A1 mejora el
error relativo frente a A0 un 5,70 % y el MAE un 10,45 %, a costa de una
latencia 1,91 veces mayor. AttnRes y KDA no se promocionan en su formulación
actual.

## Grouped CV final y diagnóstico OOD

La selección predeclarada usa cinco folds completos y seeds 7/13/21 desde una
inicialización aleatoria común, sin reutilizar SSL que hubiera visto las
secuencias OOF.

| Brazo | Score (media ± sd seeds) | Error rel. | MAE | ms/ventana |
|---|---:|---:|---:|---:|
| **A0 global** | **0,58452 ± 0,00853** | **30,25 % ± 0,52** | 1,011 ± 0,039 s | 4,54 |
| A1 Dense | 0,59312 ± 0,00349 | 30,55 % ± 0,06 | **1,007 ± 0,013 s** | 9,82 |

A1 mejora el MAE agregado solo un 0,41 %, pero empeora un 0,99 % el error
relativo y un 1,47 % el score; gana 5/15 parejas en score/error relativo y
7/15 en MAE. Usa 1,58× tiempo de entrenamiento y 2,16× latencia. Los tres
bootstrap OOF pareados por secuencia cruzan cero. Conclusión: Dense Patch
conserva señal positiva, pero no pasa el gate de consistencia/coste y no se
promociona.

La ablación posterior `R1_MATCHED_BBOX_ROI` mantiene el backbone, cabeza común,
batch efectivo y optimización de A0/A1, y cambia únicamente el pooling final
para usar la bbox GT. En los cinco folds de seed 7 obtiene score `0,59814`,
error relativo `30,99 %` y MAE `1,0100 s`. Frente a A0 de la misma seed
empeora respectivamente `2,90 %`, `2,74 %` y `4,55 %`. No se repite en otras
seeds: una bbox usada como selector de tokens no sustituye una medición
explícita de expansión/FoE.

El A0 final se ajustó con seeds 7/13/21. El perfil `matched` mejora el score
medio un 10,95 % frente a `throughput`, aunque tarda 4,02× más. Validation
seleccionó seed 13 antes de abrir el diagnóstico familiar.

| Evaluación | Secuencias / ventanas | Score | Error rel. macro | MAE macro |
|---|---:|---:|---:|---:|
| validation | 5 / 314 | 0,28992 | 14,46 % | 0,541 s |
| family-OOD reutilizado | 8 / 481 | 0,53784 | 30,56 % | 0,805 s |

El salto OOD es +85,5 % en score, +111,4 % en error relativo y +48,8 % en
MAE. Este holdout es disjunto del ajuste, pero ya era un diagnóstico reutilizado
en el proyecto; no es Benchmark-10 ni prueba SOTA. Benchmark-10 continúa sin
abrir.

## Instalación

Requisitos principales:

- Windows o Linux;
- Python 3.11;
- PyTorch con CUDA;
- `uv`.

```powershell
uv sync --locked --all-groups --no-editable
uv run --no-sync python -m e_jepa_ttc --help
```

`--no-editable` evita que Python tenga que decodificar un archivo `.pth` con
la ruta absoluta del repo; es el modo robusto en Windows cuando el nombre de
usuario contiene caracteres Unicode. Los scripts del pipeline también añaden
`src` de forma explícita.

En el host auditado:

```text
GPU       NVIDIA GeForce RTX 5070 Ti Laptop, ~12,8 GB VRAM
RAM       32 GB
CPU       Ryzen 9, 32 threads lógicos
PyTorch   2.11.0+cu128
```

El entrenamiento usa BF16, workers persistentes, memoria fijada, prefetch,
microbatch y acumulación. Usar 32 workers no es recomendable en Windows:
multiplica procesos y RAM sin garantizar más throughput.

## Inventario cerrado de datos

```text
datasets/evttc
    EvTTC-32 etiquetado: desarrollo, grouped CV y entrenamiento final

datasets/evttc_official_benchmark_sealed
    Benchmark-10: una inferencia final después del freeze

E:\eAP_dataset\data\train
    eAP Hugging Face train-40: 40 secuencias / 216 archivos / 536,64 GiB,
    sin TTC oficial

datasets/CARLA_DVS_Looming_Dataset/random_spawn
    CARLA DVS Looming: 1.406 secuencias sintéticas / 71,64 GiB extraídos;
    1.395 válidas para contexto de 100 ms; TTC positivo o negativos censurados
```

El inventario queda cerrado a estos cuatro IDs más Benchmark-10 sellado. No se
descarga eAP test y el pseudo-TTC de eAP no se considera ground truth.

## Auditoría exacta de BASE

```powershell
.\.venv\Scripts\python.exe scripts\audit_historical_base.py `
  --checkpoint artifacts\runs\evttc32_article_ablation\base\seed7\ft30\tiny_cnn_best.pt `
  --cache artifacts\features\evttc32_trainval\cache.npz `
  --output artifacts\audit\oge_sota\historical_base_reproduction.json `
  --batch-size 24
```

La auditoría comprueba arquitectura, checkpoint, cache, hashes, métricas y
predicciones.

## Selección rápida de arquitectura

Validar código:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\scripts\run_evttc_architecture_selection.ps1 `
  -Mode Validate
```

Screen Core:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\scripts\run_evttc_architecture_selection.ps1 `
  -Mode Screen `
  -Stage Core `
  -Protocol HistoricalBase `
  -Resume
```

Screen Garl con ResNet-50:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\scripts\run_evttc_architecture_selection.ps1 `
  -Mode Screen `
  -Stage Garl `
  -Protocol HistoricalBase `
  -Resume
```

El perfil Screen utiliza hasta ocho épocas, 304 ventanas train y 80 validation.
Core y Garl escriben resúmenes separados:

```text
artifacts/runs/evttc32_architecture_v4_historical_base_screen/
├── core/fold-0/matrix_summary.json
└── garl/fold-0/matrix_summary.json
```

Un smoke solo comprueba integración. No promueve componentes.

## Grouped CV y semillas

Después del screen se pasan explícitamente solo los candidatos promovidos:

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\scripts\run_evttc_architecture_selection.ps1 `
  -Mode Confirm `
  -Stage Core `
  -Protocol GroupedCV `
  -AllFolds `
  -Seed 7 `
  -RandomControl `
  -Resume `
  -Variants A0_MATCHED_GLOBAL,A1_MATCHED_DENSE_BLOCK
```

`-RandomControl` evita reutilizar en CV un checkpoint SSL histórico que ya vio
eventos de algunas secuencias de validación. Es un control de arquitectura
desde inicialización común; la confirmación anterior conserva por separado el
resultado inicializado desde BASE.

Solo BASE y un máximo de dos finalistas se repiten con:

```powershell
-AllFolds -AllSeeds
```

No se ejecuta el producto cartesiano de módulos, folds y seeds.

## Pipeline final reproducible

La interfaz cross-platform recomendada es `run_evttc_final_pipeline.py`. El
flujo por defecto completa cinco folds, tres seeds y únicamente A0/A1:

```powershell
uv run --no-sync python scripts/run_evttc_final_pipeline.py validate

uv run --no-sync python scripts/run_evttc_final_pipeline.py compare `
  --folds 0 1 2 3 4 `
  --seeds 7 13 21 `
  --variants A0_MATCHED_GLOBAL A1_MATCHED_DENSE_BLOCK `
  --resume
```

El ranking se escribe en
`artifacts/runs/evttc32_architecture_v4_grouped_cv_confirm/core/aggregate.json`.
No se congela ningún candidato hasta que su fila indique
`complete_for_final_selection=true`.

Una vez fijada la arquitectura, se entrena en el protocolo familiar 19/5/8.
El cache contiene los tres roles, pero el trainer solo abre `train` y
`validation`:

```powershell
uv run --no-sync python scripts/run_evttc_final_pipeline.py fit-holdout `
  --variant A0_MATCHED_GLOBAL `
  --seeds 7 13 21 `
  --resume
```

Si se comparan los perfiles `matched` y `throughput`, la selección se regenera
sin leer test:

```powershell
uv run --no-sync python scripts/run_evttc_final_pipeline.py select-final `
  --variant A0_MATCHED_GLOBAL `
  --matched-root artifacts/runs/evttc32_final_family_holdout_matched/core/fold-0/A0_MATCHED_GLOBAL `
  --throughput-root artifacts/runs/evttc32_final_family_holdout/core/fold-0/A0_MATCHED_GLOBAL `
  --output artifacts/metrics/evttc_final_a0_profile_selection.json
```

Evaluación de validation, sin abrir test:

```powershell
uv run --no-sync python scripts/run_evttc_final_pipeline.py evaluate-holdout `
  --checkpoint artifacts/runs/evttc32_final_family_holdout_matched/core/fold-0/A0_MATCHED_GLOBAL/seed-13/best.pt `
  --cache-manifest artifacts/features/evttc32_final_family_holdout_core/manifest.json `
  --splits validation `
  --output-dir artifacts/metrics/evttc_final_a0_seed13_validation
```

El test familiar OOD exige una apertura explícita y se etiqueta siempre como
diagnóstico, nunca como Benchmark-10 oficial:

```powershell
uv run --no-sync python scripts/run_evttc_final_pipeline.py evaluate-holdout `
  --checkpoint artifacts/runs/evttc32_final_family_holdout_matched/core/fold-0/A0_MATCHED_GLOBAL/seed-13/best.pt `
  --cache-manifest artifacts/features/evttc32_final_family_holdout_core/manifest.json `
  --selection-manifest artifacts/metrics/evttc_final_a0_profile_selection.json `
  --splits test `
  --allow-diagnostic-test `
  --output-dir artifacts/metrics/evttc_final_a0_seed13_family_ood
```

El manifest de selección anterior congela el SHA-256 del checkpoint operativo
y es obligatorio para abrir el test familiar. El freeze del ensemble CV se
hace desde un árbol Git limpio:

```powershell
uv run --no-sync python scripts/run_evttc_final_pipeline.py freeze `
  --aggregate artifacts/runs/evttc32_architecture_v4_grouped_cv_confirm/core/aggregate.json `
  --output artifacts/checkpoints/final_freeze_manifest.json
```

Los comandos admiten `--dry-run`. El perfil `matched` es obligatorio para
reproducir las tablas. `--execution-profile throughput` usa microbatch 32 y
acumulación 1, conserva batch efectivo 32 y debe emplearse simétricamente para
todos los brazos de una comparación. Los workers se autodetectan con máximo
12 para evitar presión excesiva de RAM en Windows.

El wrapper PowerShell anterior sigue disponible y acepta
`-ExecutionProfile Throughput`; su default continúa siendo `Matched`.

## Papel de eAP-40

`E:\eAP_dataset\data\train` no contiene TTC oficial. Por tanto, eAP-40 solo
puede alimentar pretraining autosupervisado, probes sin TTC o análisis de
domain shift. No participa en selección A0/A1, fine-tuning TTC, calibración ni
métricas EvTTC. Antes de usar las 40 secuencias debe hacerse un piloto acotado
de 2–4 secuencias y comparar contra el mismo fine-tuning EvTTC.

## CARLA DVS Looming

La distribución de Figshare/University of Sussex contiene 1.406 secuencias
sintéticas a 640×480 con colisiones contra coches o peatones, conducción sin
colisión y negativos difíciles con tráfico/cruces. El ZIP y el TAR locales
coinciden con el MD5 oficial `21a3e72a1c1d9c441a7426393f4e545f`; licencia
CC BY 4.0 y DOI `10.25377/sussex.29114609.v1`.

Preparación reproducible, sin crear un segundo cache de eventos:

```powershell
uv run --no-sync python scripts/prepare_carla_looming.py `
  --root datasets/CARLA_DVS_Looming_Dataset/random_spawn `
  --manifest data/manifests/carla_dvs_looming_v1.json `
  --split data/splits/carla_dvs_looming_blocked_v1.json
```

El manifest está firmado, conserva rutas relativas y fuerza
`numpy_allow_pickle=false`. Se excluyen diez secuencias vacías/con tiempo final
negativo y `example_392`, que dura menos de 100 ms. El split bloqueado contiene
803 secuencias train, 298 validation y 294 test; ningún bloque de 25 IDs
contiguos cruza roles. Este test es out-of-sample dentro del simulador, no OOD
real. La transferencia CARLA→EvTTC es la prueba cross-domain relevante.

CARLA no reemplaza EvTTC: su reloj efectivo está cuantizado a 10 ms, el TTC
positivo llega solo hasta unos 3,85 s y no ofrece bbox temporales. Se usará para
pretraining de percepción/looming y clasificación de riesgo; las secuencias
negativas llevan TTC censurado y nunca una etiqueta de regresión inventada.

## Arquitecturas bajo gate

- `A0_MATCHED_GLOBAL`: control object-cache global.
- `A1_MATCHED_DENSE_BLOCK`: patches espaciales antes de atención temporal
  block-causal.
- `R1_MATCHED_BBOX_ROI`: bbox GT aplicada solo al pooling denso; descartado en
  cinco folds seed 7.
- `A2_MATCHED_DENSE_ATTNRES`: recuperación por tarea a través de profundidad.
- `K1_OBJECT_KDA`: memoria delta temporal posterior a la mezcla espacial.
- `A4_GT_GEOMETRY`: oracle de bbox GT con height/area/affine/event contrast.
- `G0`–`G7`: direct, LHR, early/late fusion y foreground de Garl-TTC.

Estado de promoción:

- A0 global: seleccionado por grouped CV multisemilla;
- A1 Dense/Patch Policy: señal histórica positiva, rechazado por grouped CV;
- R1 bbox-ROI: rechazado; confirma que falta geometría de expansión explícita;
- A2 AttnRes y K1 Object-KDA: rechazados por la confirmación larga;
- TargetQuery, máscara predicha, refiner, router, residual e incertidumbre:
  bloqueados.

La geometría bbox causal produce predicción válida en 311/314 ventanas y baja
el error relativo macro de A1 de 15,210 % a 14,790 % al usar A1 como fallback,
pero empeora el score compuesto de 0,3054 a 0,3114. No alcanza el gate del 5 %.
El port causal, trazable al código público STRTTC, solo resuelve 27/40 muestras
del screen y obtiene 112,96 % de error relativo macro en las exitosas; tampoco
se promociona.

## Ego-motion

El HDF5 EvTTC almacena velocidad norte/este/arriba y heading en grados. El
loader convierte esa señal a la cámara de eventos mediante:

```text
navegación → LiDAR → Blackfly izquierda → Prophesee izquierda
```

La de-rotación por yaw no utiliza datos futuros. El warp traslacional usa
velocidad de cámara, brazo rígido, intrínsecos y profundidad causal. Si la
profundidad procede de la distancia oficial EvTTC, el resultado se etiqueta
obligatoriamente como oracle/teacher y no puede usarse como inferencia final.

## Almacenamiento

- caches EvTTC separados por rol y perfil;
- máximo `best`, `last` y `weights_only` por run;
- sin voxel cache global de eAP;
- CARLA leído por mmap y ventanas; sin segunda copia ni cache voxel global;
- sin extracción masiva de TAR RGB;
- sin logits SAM full-resolution;
- sin hidden states DINO de todas las capas.

Los datos, caches y checkpoints no se versionan en Git.

## Integridad científica

- splits completos por secuencia;
- selección por validación macro de secuencia;
- sin TTC durante SSL;
- sin tuning sobre Benchmark-10;
- resultados negativos conservados;
- latencia medida con el candidato realmente evaluado;
- ningún claim SOTA sin evaluación oficial reproducible.

Este repositorio es investigación. No es un sistema certificado de seguridad ni
debe controlar un vehículo.

## Licencia

Consulta [LICENSE](LICENSE) y las licencias de EvTTC, eAP, CARLA DVS Looming,
Garl-TTC y los teachers antes de redistribuir datos, pesos o derivados.
