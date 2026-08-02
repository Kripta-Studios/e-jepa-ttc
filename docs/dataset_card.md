# Dataset card

Actualizado: 2026-08-02.

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
