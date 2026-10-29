# Productores TRAIN40 — continuación 27/10/2026

## Alcance

Los ensayos `benchmark_train40_producer_variants.py` miden productores, PAIR,
PHASE17, copias de salida y guards sobre sensores ya residentes en GPU. Excluyen
preprocesamiento, H2D, contexto y TTC final del reloj. La paridad de los tres TTC
se evalúa separadamente para todos los objetos. El oracle conserva batch cuatro
event y una observación RGB por forward, por cada objeto. El objeto cero no se
altera y su oracle se coteja con la predicción original preservada.

Los objetos adicionales son carga artificial declarada: los historiales reales
se escalan con factores distintos. Esto no acredita veinte tracks físicos.

## Diagnóstico numérico

- `index137_reuse_v1`: reproduce E_CTX `abs=0.00014495849609375` frente a límite
  `0.00010398821830749512`. El oracle preparado más ContextRuntime coincide
  bit a bit con los tres TTC originales. La variante serial y paralela coincide
  bit a bit consigo misma. El desvío comienza ya en las features de eventos:
  máximo absoluto `0.00020599365234375`; seis coordenadas exceden la tolerancia
  usada para el diagnóstico de features. La fase experta cambia hasta `1.26e-7`.
- `index137_no_reuse_v1`: conservando grafo completo y fase float64 GPU, pero
  desactivando reutilización de endpoints, todas las features event y E_CTX
  vuelven a igualdad exacta. Los tres TTC pasan. Esto aísla el cambio de batch
  del encoder como causa, sin culpar al contexto ni a la concurrencia.
- `index488_no_reuse_v1`: E_CTX también vuelve a ser exacto; sin embargo R_CTX
  falla (`abs=0.00246810913`) al cambiar su batch de siete endpoints únicos a
  quince endpoints. Por ello no se debe desactivar la reutilización de ambas
  modalidades indiscriminadamente.
- Se añadió control independiente `reuse_event_encodings` y
  `reuse_rgb_encodings`, ambos `None` por defecto: heredan la configuración
  histórica `reuse_encodings`.
- Los cinco recibos `index{137,488,505,514,533}_event_no_reuse_v1` prueban
  **E OFF / R ON**: las tres salidas pasan en los cinco casos. Features event,
  expertos event y E_CTX son bit-exactos frente al oracle original en todos.
  PHASE_ONLY difiere como máximo `9.5367431640625e-7`; R_CTX del caso 488
  difiere `0.00031280517578125`, dentro de su límite `0.000373525505065918`.
  Las otras cuatro diferencias R_CTX son como máximo `4.76837158203125e-7`.
  Esto resuelve los fallos observados, pero falta la admisión completa de la
  nueva ruta en las 554 consultas. Todos los pesos y 595 archivos ligados se
  conservaron en cada ensayo. Las dos repeticiones de tiempo por caso son solo
  diagnóstico, no estimación de colas ni throughput sostenido.

## Capacidad y perfil diagnóstico

`index0_grouped_baseline_v1` (seis repeticiones por ruta):

| Objetos | Mediana serial ms | Mediana paralela ms | TTC contra oracle por objeto |
|---:|---:|---:|---|
| 1 | 15.82635 | 11.1611 | pasa |
| 2 | 23.11225 | 18.41195 | pasa |
| 20 | 223.70095 | 206.06735 | pasa |

La discrepancia histórica serial/paralelo de dos objetos no se reprodujo en
estas seis repeticiones: features y TTC fueron exactos entre ambas rutas.
Esto no borra la evidencia negativa anterior ni demuestra universalidad.

El perfil de los mismos cores en ejecución eager atribuyó tiempo CUDA propio:

| Familia | Trabajo CUDA ms |
|---|---:|
| Convoluciones | 75.41937 |
| Otras operaciones elementales/runtime | 71.543681 |
| Reducciones y normalización | 23.903387 |
| Matmul/lineales | 15.327978 |
| Copias | 18.655156 |
| Muestreo/indexación | 1.74281 |

Estos tiempos son atribución de kernels, no latencia de servicio. El perfil
eager permite asignar operadores; una segunda traza registra replay del grafo.
Las trazas completas permanecen locales y no son necesarias para publicar las
tablas compactas. Las condiciones no permiten presentar la diferencia respecto
a los ~180 ms históricos como una regresión causal ni como una optimización.

## Implementación y siguiente turno

- Helper `clone_producer_models_for_layout` crea copias independientes de los
  productores y opcionalmente convierte solo sus encoders a channels-last.
  Devuelve sus outputs a almacenamiento contiguo para limitar cambios de layout
  en reducciones posteriores. Es experimental y aún no medido en GPU.
- Corregida la composición de vistas: `reuse_endpoint_encoders` rebindea un
  `_forward_impl` instalado como método de instancia. Sin ello, copiar una vista
  de transporte podía invocar el encoder del objeto anterior y saltarse la
  reutilización solicitada. Un test comprueba seis endpoints en vez de doce.
- CuDNN autotuning y el barrido de chunks 2/5/10 siguen pendientes. El ensayo
  acotado de channels-last y chunk1 se documenta al final.
- Se añadió `--object-chunk-size` al driver. Incluye todos los chunks, guards y
  concatenación en el reloj hasta la última salida; nunca divide el lote entre
  objetos para atribuir latencia. Un test CPU comprueba orden y cobertura de los
  veinte objetos, incluido el último chunk parcial.
- Una hipótesis para diferencias entre streams es la selección de workspace
  cuBLAS. NVIDIA documenta `CUBLAS_WORKSPACE_CONFIG=:4096:8`; no se aplicó en los
  ensayos descritos y no se atribuye aquí la discrepancia a esa causa.

## Captura concurrente: corrección candidata posterior

El piloto de servicio multiobjeto observó una discrepancia grande e intermitente
en la primera actualización real con dos objetos después de primar uno y dos
objetos con ceros. `_block_entry` calentaba un stream propio, pero capturaba en
el stream global por defecto de `torch.cuda.graph`. Esto podía ligar ambos
grafos concurrentes a scratch/workspace común.

La corrección captura ahora explícitamente en el stream privado de
warmup y mantiene ese stream vivo dentro de `_GraphEntry`. También se aplica a
la ruta por productor. Los diagnósticos publican IDs de captura para verificar
separación. No se ha cambiado precisión ni aritmética del modelo. El agente del
servicio informa de un ensayo posterior con N=1/2, consultas 0/137 y cinco
repeticiones: **20 actualizaciones pasan los tres TTC**, con 610 archivos
preservados. No se observó de nuevo el error grande en ese ensayo. La muestra
apoya la corrección, pero no prueba ausencia universal de carreras.

La extensión del servicio a veinte ROI de la consulta 553 identificó diferencias
pequeñas de batch RGB (cinco R_CTX fuera del gate, un PHASE_ONLY), mientras E_CTX
pasó. Es un problema numérico distinto del error transitorio grande; se debe
preservar la cobertura y resolver el batch RGB antes de admitir el servicio.

El test CUDA opt-in
`test_first_changed_parallel_update_uses_independent_capture_workspaces` se
ejecutó y **pasó** (1 passed, 46 deselected; 9,42 s). Conserva capturas N1 antes
de crear N2, prima con ceros y suministra cuatro conjuntos nuevos por cantidad.
El paralelo recibe cada conjunto antes que el serial; luego se comparan las
features con tolerancia fija. También exige cuatro IDs distintos de streams de
captura. No sustituye la prueba de TTC con pesos reales.

Se conservaron fuentes exactas de cada versión en `source_versions/`. Las
versiones previas se reconstruyeron invirtiendo los patches y cada byte fue
ligado al SHA ya publicado por el recibo de ejecución correspondiente. Un SHA
distinto habría rechazado la reconstrucción. Los ensayos futuros guardan copias
de las fuentes automáticamente en su directorio `sources/` antes de ejecutar.

Documentación primaria consultada:

- https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html
- https://docs.pytorch.org/docs/2.11/notes/cuda.html
- https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#results-reproducibility

Los recibos comprueban pesos antes/después y 595 archivos originales/fuentes
explícitas sin modificaciones. Este inventario excluye módulos experimentales
ajenos que el driver no ejecuta. No hay entrenamiento ni cambio de tolerancias.

## Channels-last: resultado negativo (28/10, continuación de esta campaña)

Consulta 553, veinte historiales artificiales distintos, E OFF / R ON, chunks de
un objeto, cuatro repeticiones por ruta. El reloj conserva todos los objetos:

| Encoder | Mediana serial ms | Mediana paralela ms | Paridad TTC |
|---|---:|---:|---|
| Contiguo | 370.7394 | 272.86285 | 20/20 pasan |
| Channels-last, outputs contiguos | 500.72195 | 405.97345 | R_CTX falla IDs 8 y 13 |

PHASE_ONLY y E_CTX pasan en la variante channels-last. Su R_CTX presenta máximo
absoluto `9.441375732421875e-5` y relativo `1.1810405766770612e-5`, fuera del gate
en esos dos IDs. Ambas ejecuciones preservan 595 archivos y valores de pesos.
**No se admite esta variante.** Son ejecuciones sucesivas, no un ensayo térmico
contrabalanceado; no atribuir un porcentaje causal preciso a la diferencia de
tiempos. El resultado práctico es que no mostró ventaja en este protocolo y
además falló paridad.

Telemetría fuera del reloj: contiguo 112–115 W, SM 2625–2655 MHz, 83–84 °C y
94–96% de utilización. Channels-last comenzó con 120 W, SM 2647 MHz, 86 °C y 95%;
la lectura final fue 94,7 W / 2520 MHz / 82 °C / 23%. No se modificaron límites de
potencia, relojes ni ajustes globales de GPU.

Recibos: `index553_chunk1_contiguous_v1/summary.json` y
`index553_chunk1_channels_last_v1/summary.json`; cada uno incluye
`source_snapshots.json` y copias exactas bajo `sources/`.
