# Dataset card

Actualizado: 2026-07-30.

## Inventario cerrado

| ID operativo | Raíz | Uso permitido | TTC oficial |
|---|---|---|---|
| `EVTTC32_LABELLED` | `datasets/evttc` | train, grouped CV, calibración | sí |
| `BENCHMARK10_SEALED` | `datasets/evttc_official_benchmark_sealed` | inferencia final | no consumido |
| `EAP_HF_TRAIN40` | `E:\eAP_dataset\data\train` | SSL/probes no-TTC | no |

No se incorporan nuevos datasets, eAP test ni un segundo release eAP.

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
producir un MAE TTC comparable. La primera ejecución debe usar
2–4 secuencias y un cache derivado acotado; las 40 solo se justifican si ese
piloto mejora un fine-tuning EvTTC idéntico.

## Política de derivados

- raw data inmutable;
- caches EvTTC comprimidos y acotados;
- TAR RGB leídos sin extracción masiva;
- sin voxel cache global de eAP;
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
- diferencias de dominio impiden tratar eAP como validación TTC.

No se redistribuyen datos crudos. Deben respetarse las licencias y términos de
EvTTC y eAP.
