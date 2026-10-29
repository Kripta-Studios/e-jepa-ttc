# Base de fusión causal RGB–eventos

`src/e_jepa_ttc/detection/event_fusion.py` implementa la base de ingeniería S4
sin modificar el productor MonoDGP, el adaptador temporal ni la representación
de eventos congelada. Todavía no constituye una integración de entrenamiento ni
una evidencia de mejora de AP.

## Interfaz experimental

La llamada exige declarar una de las cuatro condiciones de la matriz:

| Control | Historia RGB | Eventos | Comportamiento de este módulo |
|---|---|---|---|
| F00 | No | No | Devuelve las cuatro features RGB por identidad |
| F10 | Sí | No | Devuelve las cuatro features temporales RGB por identidad |
| F01 | No | Sí | Añade el residual de eventos |
| F11 | Sí | Sí | Añade el mismo residual a las features RGB temporales |

F00/F10 rechazan entradas de eventos. F01/F11 aceptan un voxel causal ya
normalizado `[B,10,384,640]`, una máscara booleana `[B]` y dos observables sin
labels: número raw de eventos y duración de ventana. Una ausencia completa de
la entrada de eventos, una máscara falsa o una ventana presente pero vacía
devuelven exactamente el baseline RGB para esa muestra.

No hay compuerta aprendida en esta primera variante. Los observables se
conservan para un candidato posterior mediante las transformaciones fijas
`log1p(raw_event_count)` y `duration_us / 100000`; se devuelven detached, no
usan error GT ni estadísticas ajustadas. El CNN tampoco incluye BatchNorm,
GroupNorm o LayerNorm: consume el contrato de normalización del voxel sin
crear otro normalizador aprendido.

## Arquitectura fijada

El encoder usa cuatro bloques convolucionales SiLU. El primero reduce la misma
rejilla de píxeles 640×384 con stride 4 y los siguientes usan stride 2. Cada
nivel se proyecta a C=256 y, si la geometría concreta del backend difiere, se
interpola bilinealmente al `H×W` declarado por su feature RGB correspondiente.
Cada escala concatena RGB y eventos, reduce a 32 canales y vuelve a C=256.

La convolución de salida de cada residual se inicializa a cero. Por ello, antes
de cualquier update, F01/F11 producen igualdad bit a bit con sus entradas RGB,
aunque exista un voxel no vacío. La variante es residual y no sustituye ni
renormaliza las features de `ProjectedFeatureBackend`.

El modelo default contiene **295.040 parámetros entrenables**. Para batch uno y
FP32, las formas teóricas anteriores a cualquier interpolación son:

| Nivel | Encoder | Proyección C=256 | Memoria de la proyección |
|---|---:|---:|---:|
| 0 | `[16,96,160]` | `[256,96,160]` | 15,00 MiB |
| 1 | `[32,48,80]` | `[256,48,80]` | 3,75 MiB |
| 2 | `[64,24,40]` | `[256,24,40]` | 0,94 MiB |
| 3 | `[64,12,20]` | `[256,12,20]` | 0,23 MiB |

El voxel de entrada ocupa 9,38 MiB y las cuatro proyecciones suman 19,92 MiB
por muestra FP32. En la escala mayor, la concatenación RGB–evento contiene
30,00 MiB. Estas cifras son productos de shape y dtype, no una medición de
pico de RAM/VRAM: autograd, temporales, interpolación y el allocator añaden
memoria. No se ha ejecutado GPU ni se ha fijado una semilla de entrenamiento.

## Verificación CPU

`tests/unit/test_detection_event_fusion.py` comprueba:

- identidad exacta de F00/F10 y del residual inicial F01/F11;
- fallback por ausencia, máscara falsa y ventana válida vacía;
- cuatro escalas C=256 con shapes compatibles con el backend;
- bloqueo del gradiente del voxel para muestras ausentes;
- gradiente del voxel para muestras presentes una vez habilitado el residual;
- rechazo de combinaciones ambiguas y observables inconsistentes;
- ausencia de normalizadores aprendidos.

La conexión al trainer queda deliberadamente pendiente hasta congelar el
baseline RGB seleccionado. Después deben compararse F00/F10/F01/F11 con el
mismo presupuesto y añadir controles de eventos cero, permutados y desfasados.
Una compuerta de calidad es otro candidato explícito, no parte de esta base.
