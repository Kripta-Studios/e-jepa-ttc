# Dataset card

Actualizado: 2026-07-30.

## Inventario cerrado

| ID operativo | Raíz | Uso permitido | TTC oficial |
|---|---|---|---|
| `EVTTC32_LABELLED` | `datasets/evttc` | train, grouped CV, calibración | sí |
| `BENCHMARK10_SEALED` | `datasets/evttc_official_benchmark_sealed` | inferencia final | no consumido |
| `EAP_HF_TRAIN40` | `E:\eAP_dataset\data\train` | SSL/probes no-TTC | no |
| `CARLA_DVS_LOOMING_1406` | `datasets/CARLA_DVS_Looming_Dataset/random_spawn` | SSL/looming/riesgo sintético | TTC sintético positivo; negativos censurados |

Este inventario fue ampliado explícitamente con CARLA DVS Looming. No se
incorporan eAP test, un segundo release eAP ni otros datasets sin revisar el
protocolo.

## EvTTC-32

El manifest local contiene 32 secuencias públicas. Cada secuencia es una unidad
indivisible de split. Las ventanas de una misma secuencia nunca se reparten
entre train y validation.

Datos utilizados según disponibilidad:

- arrays de eventos HDF5;
- timestamps e índices por milisegundo;
- RGB sincronizado para Garl;
- bbox/segmentación;
- tabla oficial de TTC, distancia y velocidad relativa;
- intrínsecos y extrínsecos;
- navegación GNSS/INS.

La tabla TTC se interpola al timestamp de la anotación. Esta es la única
supervisión TTC oficial.

## Navegación EvTTC

Contrato verificado con el formato oficial y los HDF5 locales:

```text
position  = latitud, longitud, altitud
velocity  = norte, este, arriba
attitude  = roll, pitch, heading en grados
```

La trayectoria geodésica y el vector de velocidad coinciden con el heading.
Para obtener velocidad en la cámara de eventos se compone:

```text
GNSS/INS → LiDAR → RGB izquierda → evento izquierda
```

El action vector causal es:

```text
[speed, event_vx, event_vy, event_vz,
 event_ax, event_ay, event_az, yaw_rate_rad_s]
```

Se incluye el movimiento del brazo rígido entre el origen GNSS y la cámara.

## Grouped CV

`data/splits/evttc32_grouped_cv.yaml` define cinco folds. Requisitos:

- 32/32 secuencias cubiertas;
- cada secuencia aparece una vez como validation;
- ningún solapamiento de ventanas;
- balance aproximado de familia, velocidad y overlap;
- selección por métricas macro de secuencia.

El split histórico se conserva únicamente para reproducir `BASE` y ejecutar el
screen inicial comparable.

El protocolo familiar 19/5/8 mantiene CCRs-2, CCRs-3 y CPNAO fuera del ajuste
y del early stopping. Es un diagnóstico OOD por familia, pero su estado local
es `reused_test_diagnostic`; no equivale a un test externo no inspeccionado.

## Benchmark-10

La raíz sellada no se enumera, no se usa para early stopping y no produce
caches de entrenamiento. La inferencia exige autorización explícita, freeze
manifest y checkpoint hash.

## eAP train-40

El release local está completo: 40 secuencias, 216 archivos y 536,64 GiB.
Incluye 40 `events.h5`, 40 tablas Parquet y 136 TAR RGB. Cada secuencia
contiene:

```text
events.h5
labels.parquet
rgb_shards/*.tar
```

Las anotaciones incluyen tracks y cajas 3D, pero no una columna TTC oficial. El
pseudo-TTC track-derived local contiene 804.510 filas, de las que 195.024
(24,24 %) son válidas. Este derivado:

- declara `official_ground_truth=false`;
- usa contexto temporal y no es causal para inferencia;
- no sustituye el fine-tuning EvTTC;
- solo puede evaluarse como ablación posterior.

Sin TTC oficial, eAP sí conserva valor para SSL sobre eventos/objetos, probes
de representación y perturbaciones de dominio. No puede seleccionar A0/A1 ni
producir un MAE TTC comparable. La primera ejecución usa 12 secuencias fijadas
antes de consultar EvTTC: nueve train y tres validation en
`data/splits/eap_pilot12_v1.json`. El loader abre ventanas HDF5 bajo demanda
mediante `ms_to_idx`, no RGB ni un cache derivado masivo. eAP-Geo mejoró A0 en
RTE y MAE en dos folds EvTTC idénticos, por lo que se habilitó
`data/splits/eap_train40_v1.json`: 32 train/8 validation, todas las 40
secuencias y 16.384/4.096 ventanas. El split preserva las tres validation piloto
y añade cinco solo por hash salado de ID; no usa labels ni métricas downstream.

## CARLA DVS Looming

Fuente: University of Sussex Figshare, DOI
`10.25377/sussex.29114609.v1`, licencia CC BY 4.0. El TAR oficial tiene
15.096.280.525 bytes y MD5 `21a3e72a1c1d9c441a7426393f4e545f`;
el archivo local auditado coincide.

Inventario extraído:

```text
secuencias totales                 1.406
eventos totales                    7.692.448.155
resolución                         640 x 480
eventos por secuencia (mediana)    5.027.925
reloj observado                    pasos de 10 ms
```

Con contexto causal de 100 ms son válidas 1.395 secuencias:

| Clase normalizada | Válidas | Rol |
|---|---:|---|
| `car` | 412 | colisión, TTC positivo |
| `pedestrian` | 347 | colisión, TTC positivo |
| `none` | 294 | negativo, TTC censurado |
| `none_with_traffic` | 167 | negativo difícil, TTC censurado |
| `none_with_crossing` | 175 | negativo difícil no documentado en el README original |

Diez secuencias están vacías y tienen `t_end <= 0`; `example_392` contiene
eventos pero no alcanza 100 ms. Ninguna se oculta: sus IDs y causas están en
`data/manifests/carla_dvs_looming_v1.json`.

`data/splits/carla_dvs_looming_blocked_v1.json` usa seed 42 y mantiene bloques
contiguos de 25 IDs completos: 803 train, 298 validation y 294 test. Es un
holdout sintético out-of-sample, no evidencia OOD real. EvTTC constituye el
destino cross-domain.

El adaptador:

- abre NumPy siempre con `allow_pickle=False`;
- interpreta el `diameter_object=None` inseguro como campo ausente;
- usa mmap y búsqueda binaria para ventanas half-open;
- convierte milisegundos a microsegundos y polaridad 0/1 a -1/+1;
- limita el perfil SSL full a 16 ventanas espaciadas por secuencia;
- nunca usa `vel` o `diameter_object` como entrada EvTTC-incompatible.

Con contexto 100 ms, stride 50 ms y horizontes 50/100/250 ms, el perfil full
materializa de forma lazy 12.020/4.457/4.297 pares
train/validation/test. No existe una segunda copia de los eventos ni un cache
voxel global. Los 11 canales auxiliares de BASE son ceros explícitos durante
CARLA para conservar compatibilidad de shape sin inventar navegación o
geometría.

El test CARLA evalúa predicción latente sobre secuencias sintéticas no vistas;
no contiene una cabeza TTC supervisada y no se reporta como error TTC. El
destino cross-domain se mide con grouped CV EvTTC mediante
`scripts/run_carla_evttc_complete.py`.

CARLA es útil para pretraining de eventos, expansión y riesgo, pero no para
entrenar directamente una cabeza bbox-ROI: no contiene cajas temporales. Su
TTC positivo cubre solo el régimen cercano (máximo aproximado 3,85 s).

## Política de derivados

- raw data inmutable;
- caches EvTTC comprimidos y acotados;
- TAR RGB leídos sin extracción masiva;
- sin voxel cache global de eAP;
- sin voxel cache global ni segunda copia de eventos CARLA;
- máscaras SAM RLE/bit-packed;
- tokens DINO compactados, nunca todas las capas;
- presupuesto eAP derived máximo 55 GiB;
- margen libre obligatorio en E: 50 GiB.

## Limitaciones

- la distancia oficial EvTTC no está disponible como entrada legítima del
  Benchmark-10;
- una compensación traslacional que use esa distancia es exclusivamente
  oracle/teacher;
- bbox GT separa calidad de geometría de calidad de localización;
- diferencias de dominio impiden tratar eAP o CARLA como validación TTC real.

No se redistribuyen datos crudos. Deben respetarse las licencias y términos de
EvTTC, eAP y CARLA DVS Looming.
