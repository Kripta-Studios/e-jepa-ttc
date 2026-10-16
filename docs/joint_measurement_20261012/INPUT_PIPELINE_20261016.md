# Lectura de sensores: diagnóstico y optimización del 16 de octubre de 2026

La ejecución activa es la evaluación de los endpoints B1 ya entrenados. Los 24.000
updates de B1 están completos. Optimizar esta lectura no requiere reiniciar ni
reentrenar un modelo. Tampoco demuestra una mejora científica de TTC.

## Qué costaba el batch

La prueba del graph de inferencia separó aproximadamente 158 ms de forward eager,
9 ms de forward capturado y 170–179 ms de carga más transferencia por batch de
cuatro consultas. La captura fue admitida con igualdad exacta de fase y TTC en
132 predicciones P de los dos endpoints. Conserva FP32, no TF32 y tamaños B4/B2.

El perfil CPU posterior, sobre las primeras 128 consultas P mientras la evaluación
seguía activa, midió:

| Operación | Mediana equivalente a B4 |
|---|---:|
| Lectura y descompresión del miembro ZIP | 110,627 ms |
| Apertura y extracción de todos los arrays del NPZ interno | 50,857 ms |
| Batch del lector original, medido por separado | 180,288 ms |

Las medianas por componente no tienen por qué sumar la mediana del batch.
El primer pack ocupa 116.296.353 bytes comprimidos; verificarlo por SHA-256
tardó 0,661 s. Esa verificación se hace por pack, no por batch.
No se ha aislado aún el tiempo físico del disco del tiempo de inflate: la
primera fila incluye ambos. La caché del sistema operativo no se vació.

Leer sólo events/delta/token con lectores ZIP privados dio 78,96 ms/B4 con
cuatro hilos y 71,80 ms/B4 con ocho. Una segunda exploración leyó y verificó
una vez los bytes comprimidos del pack, luego decodificó franjas independientes
en memoria: 78,22 ms/B4 con cuatro y 54,82 ms/B4 con ocho, incluido arranque.
Estas exploraciones coexistieron con la evaluación; son diagnóstico, no el
benchmark aislado de admisión. La implementación final añade una copia pequeña
de B4 para que el consumidor no retenga accidentalmente un pack anterior entero.

Recibos originales: `runtime/loader_profile_v1/RESULT.json` y
`PREFETCH_EXPLORATION.json`, bajo el directorio de artefactos de la campaña.

## Revisión de técnicas disponibles a 16-10-2026

No existe un resultado SOTA único y transferible entre loaders, codecs,
dispositivos y patrones de acceso. Se consultaron fuentes primarias actuales;
las cifras de otros sistemas no se atribuyen a este equipo.

| Técnica / fuente | Aplicabilidad aquí |
|---|---|
| [FFCV: orden secuencial, campos selectivos y batches adelantados](https://docs.ffcv.io/making_dataloaders.html) | Aplicamos esos principios al formato existente. Cambiar a orden quasi-random alteraría la evaluación registrada. |
| [PyTorch: optimización del loader](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html) | Solapar producción CPU y consumo GPU; medir cuántos trabajadores ayudan. |
| [PyTorch: memoria fijada y transferencias no bloqueantes](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html) | Fijar memoria en el hilo consumidor puede añadir una copia y empeorar el tiempo. No se habilita sin medir; primero se elimina el mayor coste de ZIP/NPZ. |
| [NVIDIA DALI: requisitos](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html) | Su instalación documentada exige Linux x64; este proceso es Windows nativo. No se migra el sistema durante la campaña. |
| [nvCOMP: instalación](https://docs.nvidia.com/cuda/nvcomp/installation.html) | Sí documenta Windows >=10 y compresión sin pérdida en GPU. Requeriría adaptar el contenedor y verificar codec/transferencias; no es una aceleración automática del ZIP/NPZ de Python. No se ha medido ni instalado aquí. |
| [GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html) | Evita copias intermedias CPU en configuraciones compatibles. No elimina por sí solo el parsing del NPZ ni demuestra compatibilidad de esta ruta Windows. No aplicado. |
| [ZipFlow, febrero de 2026](https://arxiv.org/abs/2602.08190) | Investiga movimiento comprimido CPU→GPU y reporta mejoras en TPC-H. Es evidencia de bases de datos; no una medición de entrenamiento TTC ni una receta validada para este host. |

La decisión local es conservar los ZIP lossless existentes, amortizar apertura y
hash, decodificar ocho franjas con cursores privados, extraer sólo inputs y
precargar un pack. No se cuantizan píxeles, no se cambian tiempos y no se crea
otra copia descomprimida de decenas de GB en disco.

## Implementación y admisión

`prefetched_sensor.py` es un lector sólo de inferencia. Mantiene como máximo dos
packs decodificados propios, aproximadamente 805 MB, más un pack comprimido,
buffers de trabajadores y el pequeño batch del consumidor. Usa un productor
y ocho decodificadores; no instala un supervisor adicional. Cada archivo se
lee una vez y se calcula el hash de los mismos bytes consumidos. Se comprueban
CRC, unicidad de miembros, tokens, tipos, tamaños y relojes válidos. Los errores
de trabajadores se propagan al consumidor. La memoria se libera al cerrar.

`correspondence_prefetch.py` compone este lector con el graph admitido, sin editar
los productores, las recetas, las identidades de predicciones o el evaluador
histórico. Guarda otra identidad explícita de runtime y hashes de todos los
fragmentos previos. Mantiene la contabilidad conservadora si una inferencia se
interrumpe; no descuenta reservas ya registradas.

Antes de usarlo en la evaluación completa, su registro exige:

- Igualdad exacta de píxeles y tiempos en P[0:512], las últimas dos P y las
  últimas dos H, incluyendo fronteras de pack y poblaciones parciales.
- Igualdad exacta de fase y TTC en P[0:64] y últimas dos P para ambos endpoints.
- Pesos intactos y cero updates de optimizador.
- Benchmark emparejado aislado sobre 512 consultas P por modelo, alternando
  orden de lectores, con arranque, transferencia, forward, salida y drenaje.
- Tiempo total del nuevo lector ≤80% del original en ambos modelos.

Las pruebas unitarias rechazan corrupción, tokens permutados, miembros
duplicados, relojes inválidos y cambios de orden/tamaño del batch.
La comparación de coste científico del modelo conserva su ruta original: la
aceleración del proceso de evaluación no sustituye silenciosamente ese benchmark.

## Resultado v1 y revisión v2

V1 pasó todas las comprobaciones exactas y conservó los pesos, pero se rechazó
por velocidad. Sobre 512 consultas por modelo, original/prefetch tardaron
4,324/3,587 s y 4,315/3,489 s: razones 0,8294 y 0,8086, ambas superiores a 0,8.
No se rebaja el gate para aceptar ese resultado.

En esta medición aislada el original ya cargaba B4 en 28,3–28,6 ms de media;
prefetch bajó la espera a 11,3–11,6 ms. Sin embargo, el tiempo medido de
forward/decodificación/salida subió por la concurrencia del productor CPU.
La diferencia grande con los 170–180 ms iniciales muestra que el estado del
host y de sus cachés importa; no se puede atribuir toda ella a este cambio.
El benchmark emparejado sustituye esa comparación entre momentos distintos.

La revisión v2 mantiene población, orden, ocho trabajadores y gate. Cambia
únicamente la extracción interna de los arrays: analiza sus pequeñas cabeceras
NPY y usa vistas sobre los bytes descomprimidos en vez de copiar el payload a
través de varias capas NPZ. Comprueba formato almacenado, longitudes y dtype;
rechaza objetos, Fortran, versiones desconocidas y miembros internos duplicados.
El CRC del miembro exterior y el SHA de los bytes consumidos siguen vinculando
todos los bytes. La copia al buffer contiguo conserva exactamente FP32.

La fuente v1 permanece reproducible en el commit `41f398f` y sus recibos se
conservan bajo `runtime/B1_sensor_prefetch_v1`. V2 tiene registro y resultados
separados; el código nuevo por sí solo no acredita su admisión.

## V2 admitido: resultados emparejados

El recibo `runtime/B1_sensor_prefetch_v2/RESULT.json` terminó `ADMITTED`.
Registro SHA-256: `bf8ad4e806bf5e2a1934ac2b746c447e2e5f9b095ef4660cc264b146c5dfaf50`.

| Medida | CORRESPONDENCE original | CORRESPONDENCE v2 | TTC-CONTROL original | TTC-CONTROL v2 |
|---|---:|---:|---:|---:|
| Espera media de carga por B4 | 25,84 ms | 5,59 ms | 25,52 ms | 5,21 ms |
| Tiempo total de 512 consultas, incluido cierre | 3,899 s | 2,173 s | 3,939 s | 2,053 s |
| Tiempo total amortizado por B4 | 30,46 ms | 16,97 ms | 30,78 ms | 16,04 ms |
| Razón nuevo/original | — | 0,5573 | — | 0,5213 |

Esto equivale a 44,3–47,9% menos tiempo total, o aproximadamente 1,79–1,92×
throughput en este benchmark. La espera de carga disminuye 78,4–79,6%.
No se presenta como una reducción causal de 170 a 5 ms: los 170 ms proceden de
otro momento y condiciones del host. Los números de esta tabla son emparejados.
El primer batch todavía espera la decodificación inicial: 494–530 ms en esta
prueba. La ventaja se amortiza durante el consumo secuencial del pack.

Se verificaron 516 inputs exactos (514 P y dos H) y 132 salidas exactas de los
dos modelos, con pesos intactos y cero updates. Las pruebas CPU suman 16 casos
del lector y observador; Ruff y Pyright pasaron. El lector original conserva
la responsabilidad sobre los targets y el scorer a través del evaluador: el
nuevo lector nunca devuelve etiquetas a la red.

El proceso de prueba terminó con RSS 4,37 GB, que incluye modelos, PyTorch,
buffers del benchmark y runtime CUDA; no es sólo memoria del lector ni un pico
medido. No se hicieron cambios globales al host ni al codec de los archivos.
La evaluación se reanuda desde 203 fragmentos durables, 51.968 consultas,
verificados por hash. `Resume-JointMeasurement.ps1 -Stage B1Evaluation` elige
esta ruta sólo si está admitida; el runtime vuelve a comprobar sus bindings.
