# Dataset card

## Exact public validation screen

The signed subset `garl_validation_common_roi_v1` contains exactly 2,048 public
rows from three complete validation sequences, with token and target equality
checked against the Garl public parquets. It contains no private eAP, CodaBench or
EvTTC test rows. The three sequences were present in the official release training
asset list, so release-checkpoint metrics are not a matched-training comparison.

Actualizado: 2026-08-10.

## Inventario activo

| ID operativo | Raíz | Uso permitido | Estado |
|---|---|---|---|
| `EVTTC32_LABELLED` | `datasets/evttc` | desarrollo, grouped CV, calibración | 32 secuencias |
| `BENCHMARK10_SEALED` | `datasets/evttc_official_benchmark_sealed` | inferencia final | sellado |
| `EAP_LOCAL_40_OF_46` | `E:\eAP_dataset` | SSL y Garl supervised raw | faltan 6 secuencias |
| `GARLTTC_PUBLIC_LABELS` | `E:\GarlTTC_dataset` | targets Garl train/validation | solo lectura |
| `GARLTTC_RELEASE` | `E:\Garl-TTC` | paridad con código oficial | solo lectura |

CARLA DVS Looming fue retirado del inventario activo el 2026-08-02 después de
que SSL y TTC sintético empeoraran la transferencia EvTTC. El dataset local,
caches y checkpoints se eliminaron; los resúmenes compactos negativos permanecen
versionados para evitar sesgo de publicación.

## EvTTC-32

Cada secuencia es una unidad indivisible. Nunca se reparten ventanas de la misma
secuencia entre train, validation y test.

Datos usados según disponibilidad:

- eventos HDF5 e índice por milisegundo;
- RGB sincronizado solo en protocolos multimodales declarados;
- bbox/segmentación;
- TTC, distancia y velocidad relativa oficiales;
- intrínsecos, extrínsecos y navegación GNSS/INS.

La tabla TTC se interpola al timestamp de referencia. EvTTC es la única
supervisión TTC oficial disponible localmente para evaluación geométrica. El
manifest canónico es `data/manifests/evttc_all32_local.yaml` y grouped CV usa
`data/splits/evttc32_grouped_cv.yaml`.

## Benchmark EvTTC sellado

No puede usarse para seleccionar arquitectura, seed, época, calibración ni
hiperparámetros. Solo se abre después de un freeze que registre commit, config,
split, hashes y checkpoint.

## eAP y GarlTTC

El inventario local cubre 40 de las 46 secuencias esperadas. El piloto firmado
`data/splits/eap_pilot12_v1.json` usa 9 train y 3 validation. El split full
`data/splits/eap_train40_v1.json` usa 32/8 secuencias sin consultar métricas
EvTTC.

El pipeline activo une por las cinco claves auditadas de GarlTTC y lee los eventos
HDF5 bajo demanda mediante `ms_to_idx`. La entrada event-only contiene cinco pasos
temporales y 21 canales por paso. TTC, profundidad, altura 3D, categoría y máscaras
son targets o metadata de auditoría; no entran al encoder.

eAP público no proporciona un target TTC oficial independiente. Las cifras TTC del
dataset Garl se usan únicamente en entrenamiento/validation supervisados declarados.
No se reconstruye pseudo-TTC para sustituirlas.

### Cache causal-scale screen v1

`artifacts/cache/garl_object_event_common_roi_screen_v4/manifest.json` materializa
4.096 filas balanceadas del split piloto: 2.048 train y 2.048 validation. El cache
ocupa shards locales regenerables, no se sube a Git, y evita recorrer los ~691,5 GiB
de eAP durante cada época. Usa 9 secuencias train y 3 validation sin intersección.

Cada muestra contiene tres endpoints event-only, doce canales y ROI común 128×128.
No contiene RGB ni máscaras. Las cajas están disponibles como metadata/supervisión;
t0 es proxy en este cache y no se usa como target de foreground. Los targets TTC son
los labels públicos oficiales Garl, no pseudo-labels.

### Cache Garl matched oficial

`garl_official_event_only_matched_preprocessing_v1` materializa los mismos 2.048
train/2.048 validation con el preprocessing inmutable del release auditado. Cada
entrada es FP32 `[40,128,128]` (`timevolume20` para dos endpoints). No guarda RGB,
bbox ni TTC como input; sí usa y declara el crop bbox oracle del protocolo Garl.
Su identidad es `92af281030170733411ef9d65b19e88ebc8019c729dd6743e02ae9c40f564b52`.

El split matched firmado `dd08ecc983f30e38a939204f9a2df09e4966bbe73bd764c972f7726e5d4e34d3`
usa estas secuencias completas/disjuntas:

- train: `2cyv0Oedzg`, `5ilM1PX2vz`, `6h5yRW2LGc`, `OBneIVg4Cw`,
  `OYgB6RGWcq`, `WbCh1DRerJ`, `mHGFBekt7X`, `qGsgzl4Q8B`, `t79dBxj1WS`;
- validation: `DGqicHUGWb`, `pBqGOb2vYq`, `qoohcdtLDH`.

Cada rol contiene 2.048 filas. Ninguna fila validation se usa para gradientes;
validation solo selecciona checkpoint. No se abrió test privado, CodaBench ni
EvTTC test.

En A1, `event_v4_boxes_xyxy` se recorta al ROI visible y solo genera cuatro targets
escalares normalizados `h,w,cx,cy`. No se rasteriza como target denso y no forma
parte del esquema de inputs. El modelo sigue recibiendo exclusivamente
`event_v4_common_roi` y `garl_delta_t_s`.

## Contrato de almacenamiento

- datos raw inmutables y fuera de Git;
- ningún full cache high-resolution;
- shards diagnósticos limitados por `--max-samples-per-split`;
- trainer raw/on-demand para screen y full;
- máximo `best.pt`, `last.pt` y predicciones de validation por run;
- resultados cerrados se reducen a JSON/CSV pequeños bajo `artifacts/metrics`;
- `artifacts/runs` y `artifacts/features` son regenerables e ignorados.

Una estimación del full cache Garl dio aproximadamente 455 GiB. Un shard de 256
muestras fue válido; 4.096 muestras alcanzaron cerca de 11 GiB de RAM sin terminar.
Por ello la caché completa no forma parte de ningún protocolo activo.

## Riesgos de datos

- seis secuencias eAP faltan, por lo que no existe cobertura del benchmark oficial;
- EvTTC tiene pocas secuencias y gran correlación temporal;
- bbox/depth GT son oracles salvo que el protocolo declare supervisión únicamente;
- los dominios eAP y EvTTC no son intercambiables;
- selección por ventanas produciría leakage y está prohibida;
- un manifest label-free EvTTC Tabla VI con cobertura/hashes exactos aún no existe.

No se redistribuyen datos crudos. Deben respetarse las licencias y términos de cada
fuente.
