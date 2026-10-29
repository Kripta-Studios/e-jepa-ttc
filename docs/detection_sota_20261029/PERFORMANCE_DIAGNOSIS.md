# Diagnóstico de rendimiento del detector

Este informe se regenera exclusivamente desde artefactos locales retenidos. Separa medidas,
cotas aritméticas, prototipos y resultados desconocidos. **No acredita una aceleración de la
campaña completa.**

## Resultado ejecutivo

- El run estable de 1.024 updates midió 0.8807 s de
  preparación mediana y 0.6440 s de step mediano. Como están
  serializados, el solape perfecto tiene una cota aritmética de
  1.731x; no es una predicción.
- El perfil folded de entrenamiento contiene 1171 muestras;
  575 (49.1%)
  incluyen `tarfile.py` en 31.0 s de observación.
  Es proporción de muestras de stack, **no porcentaje de wall time**.
- 147 muestras de entrenamiento
  (12.55%) no contienen stack y se
  clasifican como desconocidas; no se reasignan a TAR, CPU ni GPU.
- AP local-v2 pasó de 0.043137 en update 1.024 a
  0.069496 en 2.048 y
  0.076692 en 3.072. No es score oficial.
- El panel TAR v2 de 128 muestras midió 2.17x
  en lectura raw y 1.776x en preparación, con hashes
  exactos entre productores y trials. El lector v2 cerró revisión `SHIP`; esto no amplía
  el alcance más allá del panel CPU.
- El speedup integral de campaña continúa **desconocido**.

## Evidencia medida

| Evidencia | Población | Resultado | Alcance |
|---|---:|---:|---|
| R0 estable | 1.024 updates / 4.096 exposiciones | prep 0.8807 s; step 0.6440 s | Medianas del runner congelado |
| py-spy entrenamiento | 1171 samples / 102 errors | TAR 49.1% | Sample share; transcripción posterior, no wall time |
| py-spy evaluación | 1451 samples / 22 errors | TAR 4.2% | Sample share; transcripción posterior, no wall time |
| Cache de labels | 128 frames | cache2 22.7555 s; cache24 12.3777 s | Perfil CPU retenido |
| TAR indexado v1 | 32 samples | raw 5.32x; prepared 3.94x | Timing de panel solamente; paridad exacta no admitida |
| TAR indexado v2 | 128 samples / 24 secuencias / 3 trials | raw 2.17x; prepared 1.776x | Panel CPU alternado; review `SHIP` |
| AP101 v2 | 1.024 frames | 1.43x | Timing válido |
| AP101 v4 | 64 frames | 1.41x | Integración medida |

El AP101 v3 conserva un pico de 342,712,320
bytes, pero su corrida de 1.024 frames está contaminada y no se usa para afirmar speedup.
La interpretación firmada de la observación pasiva en update 2.048 es
`SCHEDULED_MONITOR_EVALUATION_IN_PROGRESS` y completó
0 updates. Su CPU y 5,1% medio de GPU
corresponden a la evaluación programada, no al entrenamiento ni a throughput estable.

## Progreso RGB en el corte 3.420

El prefijo inmutable usado por este informe termina en update
`3420` y tiene SHA-256 `a8e64dc260b13bd390352da7098ff0dfa040c01fbb43aae3b7e5e5a0a1735d0f`. La AP local-v2 del
monitor de 1.024 frames en update 3.072 es
`0.07669187`. El run sigue vivo; no existe todavía
un summary final ni un segmento durable apto para `summarize_detection_campaign.py`.

| Ventana | Updates | Prep mediana (s) | Step mediano (s) | Suma componentes mediana (s/update) | Updates/s de componentes |
|---|---:|---:|---:|---:|---:|
| Últimos 100 | 100 | 1.5446 | 1.1958 | 2.7363 | 0.2889 |
| Desde 3.072 | 348 | 1.6518 | 1.2587 | 2.9149 | 0.3336 |

| Clase | GT elegible | Fracción GT | AP101 media |
|---|---:|---:|---:|
| car | 2524 | 54.41% | 0.48391899 |
| bicycle | 26 | 0.56% | 0.00872976 |
| truck | 619 | 13.34% | 0.04618040 |
| pedestrian | 338 | 7.29% | 0.00040344 |
| motorcycle | 963 | 20.76% | 0.05246421 |
| bus | 85 | 1.83% | 0.01679868 |
| tricycle | 80 | 1.72% | 0.00503947 |
| other_vehicle | 4 | 0.09% | 0.00000000 |

## Proyecciones condicionales, no promesas

- Quedan `4772` updates hasta 8.192 en el corte fijado. Aplicar la
  media agregada de los componentes observados en los últimos 100
  (`3.461260` s/update) produce
  `4.59` horas
  de componentes de entrenamiento. La mediana 2,736 s/update de la tabla anterior describe un
  update típico, pero no se usa para totalizar trabajo.
- Quedan monitores programados en `[4096, 5120, 6144, 7168, 8192]`. Escalar el total
  de inferencia observado en update 3.072 a esos monitores añade condicionalmente
  `0.57` horas.
- El subtotal condicional de esos dos componentes es
  `5.15` horas. Excluye scorer,
  checkpoint, serialización, H2D no separado, full-selection y otros overheads.
- Quedan dos pasadas full-selection, en updates
  `[4096, 8192]`; ninguna se ha ejecutado. Multiplicar 23.958
  frames por la mediana actual de
  `0.394161` s/frame da
  `2.62`
  horas por pasada y
  `5.25` horas para
  ambas, solo de inferencia. Ningún subtotal es una ETA integral ni un compromiso de finalización.

## Memoria por formas exactas

| Materialización | Cálculo | Bytes | GiB |
|---|---|---:|---:|
| Imágenes FP32 | 4096×3×384×640×4 | 12,079,595,520 | 11.25 |
| Region masks FP32 BHW | 4096×384×640×4 | 4,026,531,840 | 3.75 |

Estas cifras son cálculos de tamaño, no picos RSS medidos. No se infiere memoria adicional.

## Estado de optimizaciones

| Intervención | Clasificación | Estado |
|---|---|---|
| Prefetch acotado CPU | Prototipo | Tests CPU; no integrado (`9526ec10`) |
| Solape prep/compute | Cota | 1.731x ideal |
| TAR indexado | Medida parcial | Panel v2 exacto: 1.776x prepared; no end-to-end |
| AP101 optimizado | Medida parcial | suffix-max + búsqueda binaria Python; v2 timing válido; v4 integración |
| BF16 / TF32 / compile / CUDA Graphs | Desconocido | Sin experimento admitido |
| Campaña completa | Desconocido | Sin benchmark end-to-end |

No se admite el receipt v1 del lector indexado como prueba de paridad exacta: su hash lógico es
ambiguo. El receipt original fue restaurado byte a byte tras una sobrescritura accidental; la
excepción de restauración está hash-bound y v1 sigue siendo timing-only. TAR v2 es evidencia
separada, write-once, y su 1,776x no se interpreta como ganancia del entrenamiento. Los 102/22
errores de muestreo proceden de transcripciones posteriores del output de
la herramienta, hash-bound al perfil raw; no son logs contemporáneos ni miden tiempo GPU.

## Reproducción

```text
python scripts/build_detection_performance_diagnosis.py
```

El comando escribe `output/performance/performance_diagnosis_v5.json` y este documento.
Las rutas siguientes son relativas al repositorio; el Markdown no conserva rutas de usuario.

| Artefacto | SHA-256 |
|---|---|
| `output/performance/live_training_after2048_v1/pyspy_raw.txt` | `923b8386508ab694c26b2013b6a10dd3d2b17e895b3cc7f06d982b2a9fc0cfc4` |
| `output/performance/live_evaluation_update2048_v1/pyspy_raw.txt` | `51383dce4c6663265360d0f19166e89c00e4f7eff022ecfa8df336a289297128` |
| `output/performance/live_training_after2048_v1/tool_output_transcription.json` | `45fbd509b1c552e70313c4ad23698fd0604280680ce11dbbf1d895e221e2f646` |
| `output/performance/live_evaluation_update2048_v1/tool_output_transcription.json` | `615ee1cfbdf2e523c6e105ed8d12131db13ab250568b2a38dd35748fb661b6e9` |
| `output/performance/live_training_after2048_v1/started_at.txt` | `1a29bc5978ddfdb460a6147a468ad824665c25117191335bb287858bbaca926e` |
| `output/performance/live_training_after2048_v1/ended_at.txt` | `e044701dd9189d69c8cc86218530ef34a48d3379703e8e942849d1b2c5bea417` |
| `output/detection_campaign/rgb_screen_v5_seed7_v1/passive_observation_update2048_v1.json` | `8e3520cd2ce9d92ca903c53c6c3f2ae87b6da1a9971e2f08bc22d34bec40a449` |
| `output/detection_campaign/rgb_screen_v5_seed7_v1/passive_observation_update2048_interpretation_v1.json` | `34e135b441af1596b3b8498bd581c3d9ac10adee8740ea7f614bd8dec2d1c9dd` |
| `output/detection_campaign/rgb_screen_v5_seed7_v1/automatic_runs_v1/r0_1024/summary.json` | `f580bb31ee9e485e2f8810ca055f6933e9c0c2fb54b71ce605a47eedbf65e2f9` |
| `output/detection_campaign/rgb_screen_v5_seed7_v1/automatic_runs_v1/r0_8192/evaluation/selection_update_002048.json` | `a80faf4a2ea65b622ea8436b0db2295c02de348135d7f70db029c269d770d136` |
| `output/detection_campaign/rgb_screen_v5_seed7_v1/automatic_runs_v1/r0_8192/evaluation/selection_update_003072.json` | `4032c65cce016229fc2a2db9a666bb65c40656cef4ac8fcd2cc729a431ff00c1` |
| `output/campaign_data/indexed_rgb_benchmark_v1/receipt.json` | `de9a71a90e96d62fac31779b518e2352ea6040ff09f43807fdbdbf9051776c44` |
| `output/campaign_data/indexed_rgb_benchmark_v1/RESTORATION_EXCEPTION.json` | `065a38bae3dd6a5e9e56ce9d62e1524f8107ae0a2eb4565041e6152316bc0533` |
| `output/campaign_data/indexed_rgb_benchmark_v2/receipt.json` | `198e3d88e5b80ef1c5a2a70935b9868b21a838e87dbfbe2812b4e7a3715d3d31` |
| `output/campaign_data/indexed_rgb_benchmark_v2/code/benchmark_indexed_rgb_data_v2.py` | `0d83c1e60285c4aec574da7c80d2185fd7e59e44fcb3eb8f537272b0bb21da70` |
| `output/campaign_data/read_profile_v3_final/summary.json` | `8a324f9a53090ab3526c1d80e08852488cbaecce0e2a84fa56ff19031b0b96f7` |
| `output/evaluation_ap_fast/ap101_exact_v2/receipt.json` | `7e9f429001b52f75c8203e14e56b7f59b2a644ede9b62515346f8ee7096a3d0f` |
| `output/evaluation_ap_fast/ap101_exact_v3/receipt.json` | `2becd32a9ccadaac069972e5fe9a33303dc9b6462b6195c64e94cfc08945177c` |
| `output/evaluation_ap_fast/ap101_integration_v4/receipt.json` | `aabb6af83b7c110b61372d8b0be76c1bfdcf757fead1d6648aebdd1e0c12f0a8` |
