# Base de representación causal de eventos

La base S4 está implementada en
`src/e_jepa_ttc/detection/event_data.py` sin modificar el detector RGB, sus
snapshots o el runner R0. Produce un voxel FP32 `[10,384,640]`: cinco bins
temporales para polaridad positiva y cinco para negativa.

## Contrato

- El cutoff es `rgb_exposure_end_timestamp_us`.
- La ventana es exactamente `[cutoff-window, cutoff)`, con 50 ms por defecto y
  100 ms como único control alternativo.
- Se consumen todos los eventos. `chunk_events=250000` limita asignaciones
  temporales; no limita ni muestrea la población.
- La interpolación temporal es lineal. Sus pesos se acumulan como numeradores
  enteros con denominador común de 50.000 µs y se convierten una sola vez al
  final; la salida es exactamente independiente de la partición en chunks. La
  normalización robusta se aplica después de esa acumulación.
- La polaridad raw eAP 0 va a los canales negativos y 1 a los positivos.
- Los timestamps deben ser enteros, monótonos y permanecer dentro de la ventana.
  Un rollback o un evento futuro produce error.
- El LRU admite como máximo dos handles HDF5 vivos. Desaloja y cierra antes de
  abrir el siguiente, y cierra los restantes al salir del contexto.

La resolución 640×384 coincide con la entrada RGB del detector. La conversión
de coordenadas aplica la convención de centro de píxel de Pillow mediante
aritmética entera:

```text
dst = floor((2 * src + 1) * dst_size / (2 * src_size))
```

Así, los píxeles 0/1279 y 0/719 quedan en 0/639 y 0/383. Una rama posterior a
menor resolución debe conservar el aspect ratio efectivo del detector; 160×96
es una reducción coherente de 640×384. El antiguo 160×90 introduce otra
deformación vertical y no es el default de fusión.

## Probe real

Se procesó en CPU el frame
`qGsgzl4Q8B:000001037_000033536`, sin GPU ni labels. El recibo es
`output/event_feasibility/event_representation_50ms_qGsgzl4Q8B_v3.json`,
SHA-256
`a235d39b8c4ba714be979d043b6cd5f12efd539778958f3479bd23aaf0669d95`.
Está ligado a:

| Campo | Resultado |
|---|---:|
| Implementación ejecutada | `0ee69247ae11ee766c960c3da283eb166ec64f2c99b7c612572820f17df74d55` |
| Entrypoint ejecutado | `afd163cd9091e06ce0e968374d7126c3df5f4dc1a3483953643da18c52ab978f` |
| Metadata train | `69ffedab74ea28b295fe3ec2c2c83c08716bc31c216bc2ec9971f8ca9644a314` |
| Manifiesto público | `2e65cd7f2600eea55f37f59e9a22d55b5ba0aa50445756fd7aaf01332c2b3ff1` |
| SHA esperado de `events.h5` | `b12c384ae5a53d1879d410ba77d22ec47feadb393505b292f3b3fc0fe8127404` |
| Ventana | `[1036986031, 1037036031)` µs |
| Eventos consumidos | 2.014.179 |
| Chunks no vacíos | 9 |
| Máximo por chunk | 250.000 |
| Eventos descartados por cap | 0 |
| Shape | `[10,384,640]` |
| Valores no cero | 680.712 |
| Checksum voxel float32 LE | `8eaf3fac6175cc389ecb11521b403465b840ac4fdf247320f1ef7149c1d1ba96` |
| Lectura y voxelización, wall CPU | 2,489 s |
| RSS child baseline / pico / delta | 606.617.600 / 728.866.816 / 122.249.216 bytes |

El tiempo corresponde a un único probe local y no es un benchmark de
throughput. No incluye dataloader concurrente, detector o transferencia GPU.
El RSS se muestrea dentro del proceso worker real cada 10 ms; no se confunde el
launcher del entorno virtual de Windows con el proceso que carga los datos. El
probe liga el SHA esperado de `events.h5` desde el manifiesto público y verifica
su tamaño local, pero deliberadamente no relee 10,2 GB para certificar su hash
local completo.

Las dos ejecuciones v2 que se sustituyeron durante la corrección no se
ocultan: sus hashes y motivos constan en
`output/event_feasibility/SUPERSEDED_RECEIPT_TOMBSTONES.json`. El v2 que aún
existía al cerrar la corrección también se conserva sin modificar; v3 es la
ruta de admisión y no reutiliza ese nombre.

## Pruebas

`tests/unit/test_detection_event_data.py` cubre:

- evento en el inicio y rechazo del extremo final;
- pesos de interpolación y separación de polaridad;
- identidad exacta entre un chunk y particiones equivalentes, incluyendo
  100.003 eventos repetidos que cruzan bins;
- ventana vacía determinista;
- mapping de píxeles extremos;
- coordenadas y polaridades inválidas;
- rollback dentro de un chunk y entre chunks;
- rechazo al superar el tamaño máximo de chunk;
- cutoff de fin de exposición y pico real de dos handles vivos;
- rechazo de ventanas, bins y handles fuera del contrato S4.

Verificación CPU:

```powershell
$python = '../phase-ttc-main-integration/.venv/Scripts/python.exe'
$env:PYTHONPATH = 'src'
& $python -m pytest -q tests/unit/test_detection_event_data.py
& $python -m ruff check `
  src/e_jepa_ttc/detection/event_data.py `
  tests/unit/test_detection_event_data.py
& $python -m pyright --pythonpath $python `
  src/e_jepa_ttc/detection/event_data.py `
  tests/unit/test_detection_event_data.py
```

La base todavía no demuestra mejora de AP. El siguiente paso S4 es conectar la
rama pequeña al baseline congelado y ejecutar F01/F11 junto con RGB de capacidad
equivalente y los controles de eventos cero, permutados y desfasados.
