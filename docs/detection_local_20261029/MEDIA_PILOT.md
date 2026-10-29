# Runtime RGB eAP y muestra piloto local

Fecha: 2026-10-29. **512 imágenes seleccionadas leídas correctamente** desde
sus TAR reales; 256 pertenecen al rol train y 256 al rol selection. Se verificaron
también dos intervalos pequeños de eventos reales, uno por secuencia train.
El [recibo](evidence/media_pilot.json) conserva hashes, tokens, conteos, errores y
el tiempo de esta inspección. No representa entrenamiento del detector.

## Selección anterior al entrenamiento

El [preparador](../../scripts/prepare_eap_rgb_pilot.py) usa el candidato 24/8/8
existente y verifica su hash de manifest. Para cada rol ordena las secuencias
por SHA-256 de `pilot:7:role:sequence_id` y elige las dos primeras. Dentro de cada
secuencia selecciona 128 posiciones equiespaciadas después de ordenar por inicio
de exposición y token. La selección no consulta etiquetas, contenido RGB ni
resultados del modelo.

| Rol | Secuencias | Frames |
|---|---|---:|
| Train | `qGsgzl4Q8B`, `nRcwUlHyAd` | 256 |
| Selection | `JaGC4Q9pgU`, `2Y46OkK3ff` | 256 |
| Overfit | Ocho posiciones por cada secuencia train anterior | 16 |

El manifest ejecutable queda en `output/eap_rgb_pilot/manifest.json`. Contiene
listas explícitas de tokens, revisiones, hashes y protocolo; el preparador lo
regenera. No contiene imágenes ni cajas. Los roles usan secuencias disjuntas y
el overfit es subconjunto de train. Calibration permanece sin leer. La relación
entre sesiones de adquisición de distintas secuencias todavía no está probada;
esto limita las afirmaciones de independencia y no cambia la exposición previa
de los pesos PHASE a TRAIN40.

## Soporte de clases observado

Se cuentan anotaciones HF crudas en esos frames; el adaptador geométrico debe
informar por separado cuántas acepta o excluye y por qué.

| Clase | Train 256 | Selection 256 | Overfit 16 |
|---|---:|---:|---:|
| car | 996 | 1169 | 53 |
| bicycle | 3 | 0 | 0 |
| truck | 276 | 729 | 15 |
| pedestrian | 66 | 2 | 2 |
| motorcycle | 171 | 23 | 5 |
| bus | 28 | 13 | 2 |
| tricycle | 22 | 0 | 3 |
| other_vehicle | 3 | 0 | 0 |

Train contiene ocho clases, pero la muestra de overfit contiene seis y selection
cinco. No deben presentarse las clases sin soporte como una validación de ocho
clases. Las métricas deben indicar su población, el número de targets aceptados
y la regla para clases ausentes. Un smoke sintético de ocho clases y este overfit
real responden preguntas diferentes.

## Interfaz y límites del runtime

[`EAPRGBDataset`](../../src/e_jepa_ttc/detection/eap_rgb_data.py) recibe la raíz y
una lista de tokens del manifest público train. Cada `RawRGBSample` entrega:

- `image`: imagen RGB completa, `uint8`, formato HWC y almacenamiento propio;
- `annotations`: registros HF sin reinterpretar dimensiones, yaw o velocidad;
- `intrinsic` y `event_from_ego`: `K_event` y `T_event_ego` del manifest;
- token, secuencia, inicio/final de exposición, metadatos y SHA-256 del miembro RGB.

El modelo recibe la imagen completa; no recibe una ROI derivada de ground truth.
El runtime no convierte targets ni empareja identidades GarlTTC. No aplica una
transformación de calibración o rectificación que no esté explícitamente en el
adaptador posterior.

Se mantienen como máximo dos shards abiertos y dos secuencias de etiquetas en
caché. Los handles se cierran mediante `close()` o el context manager. Se debe
crear una instancia por worker, sin compartir handles TAR entre procesos. Los
paths se validan contra recorrido de directorios, unidades absolutas y escapes
por symlinks; los miembros TAR deben ser archivos regulares con nombres únicos.
Las imágenes se decodifican por completo. Errores de ruta, exposición, lectura,
calibración o etiquetas fallan explícitamente; no se omiten frames silenciosamente.
Una imagen válida sin objetos conserva su lista de etiquetas vacía.

## Eventos y causalidad acotada

La inspección usa `hdf5plugin` para registrar la compresión del HDF5. Lee cinco
milisegundos de eventos que terminan en el final de exposición de una imagen por
secuencia train. `ms_to_idx` delimita una lectura aproximada; después se aplica
la máscara exacta `start_us <= t <= exposure_end_us`. Los recibos registran los
eventos recortados antes y después, los límites retenidos y el hash de sus arrays.
En ambos intervalos, el número de eventos futuros retenidos es cero.

Esto verifica el acceso y recorte de esas muestras. No demuestra monotonicidad
de todo el HDF5, latencia de llegada de sensores, integridad de todos los medios,
sincronización física completa ni rectificación RGB/eventos. El final de
exposición es la referencia temporal declarada; la llegada efectiva no se midió.

## Reproducción y validación

La preparación se ejecutó con el entorno PHASE existente que dispone de pandas,
PyArrow, Pillow, h5py y hdf5plugin, sin modificar sus paquetes:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$auditPython = '../phase-ttc-main-integration/.venv/Scripts/python.exe'
& $auditPython -X utf8 scripts/prepare_eap_rgb_pilot.py `
  --eap-root E:/eAP_dataset --seed 7 --frames-per-sequence 128 --check-events `
  --output output/eap_rgb_pilot/manifest.json `
  --receipt docs/detection_local_20261029/evidence/media_pilot.json
& $auditPython -X utf8 -m pytest -q tests/unit/test_eap_rgb_data.py
.venv-audit/Scripts/python.exe -m ruff check `
  src/e_jepa_ttc/detection/eap_rgb_data.py scripts/prepare_eap_rgb_pilot.py `
  tests/unit/test_eap_rgb_data.py
```

Los **17 casos de prueba de la versión inicial** pasan. Cubren TAR/Parquet reales pequeños, propiedad
de imágenes y arrays de etiquetas, ausencia de tokens, split train, imágenes sin
objetos, nombres TAR ambiguos, paths inseguros, exposición invertida, mezcla de
secuencias, selección determinista, disjunción y recorte causal de eventos.
Ruff y formato pasan en los tres archivos Python nuevos.

El protocolo declara el overfit como prueba funcional de memorización. Exige
valores finitos, actualización de parámetros, menor pérdida y recuperación de
objetos medidas en los mismos 16 frames. El runner debe fijar antes del run su
presupuesto y umbrales numéricos, guardar predicciones iniciales/finales y
checkpoint y separar ese resultado de la evaluación en selection.

## Optimización Arrow posterior a la ejecución v1

Tras completar la inferencia independiente del checkpoint v1 se cambió únicamente
la lectura de Parquet: el runtime pasa filtros `sample_token in requested_tokens`
a Arrow antes de que pandas materialice calibraciones y anotaciones como objetos
Python. Las etiquetas se filtran por los tokens pedidos de su secuencia. Las
comprobaciones del runtime abarcan toda la población solicitada; la estructura
global del release corresponde a la auditoría separada de G0.

La [comparación](evidence/eap_rgb_arrow_parity.json) ejecutó la fuente v1 congelada
y la nueva en procesos CPU separados sobre los mismos 16 frames. Coinciden
exactamente los hashes de RGB, etiquetas crudas, K/T, timestamps, imagen preparada,
calibración, todos los targets, ground truth admitido y reporte. También se
contrastaron fuente y resultados con la preregistración y `final_train.json` de v1.
El [patch](evidence/eap_rgb_arrow.patch) conserva el cambio de lectura.

El pico de working set del proceso pasó de 969.875.456 a 784.588.800 bytes. La
preparación de los 16 frames midió 5,109 s antes y 0,748 s después. Son mediciones
de dos procesos secuenciales con caché de archivos y carga del host distintas:
no equivalen a una aceleración demostrada del entrenamiento. Los recibos separan
el working set inicial, el incremento observado y el pico del proceso completo,
que incluye imports de Torch y pandas.

Los **18 tests del runtime actualizado** pasan, incluido un test que comprueba
que ambas lecturas reciben los filtros Arrow. Los archivos congelados de v1 y su
población permanecen intactos; v2 registra el hash del loader nuevo. El recibo
`media_pilot.json` anterior sigue describiendo la versión original que lo produjo.
