# 5. Ingeniería al servicio del entrenamiento

## Reutilizar lo admitido

No volver a implementar como novedades caché acotada, transferencias solapadas,
procesos persistentes, CUDA Graphs A5/C2F o recuperación por cursor. Ya existen y tienen
recibos. No extrapolar un pico101updates/min, un microbenchmark21× o tiempo de workers
sumado como velocidad global. Para el modelo nuevo medir TRAIN real en admisión.

No multiplicar controladores y parches de funciones globales. Una cola V13 con estados
M0/B/D/R/KD/F, identidad por fit, un journal durable por tarea y un supervisor ligero.
Reutilizar bibliotecas de checkpoint/durable_io; parametrizar source_root/output_root.
Las constantes ROOT o output de TRAIN40 son históricas, no destinos V13.

## Recursos

Ruta local, sin Runpod ni recursos de pago. Límite de árbol23GB decimal, reserva dura
host2GiB, reanudar con3GiB y commit headroom admisible. Disco≥10GB decimales tras
reservas propias. Medir antes de crear cache, reservar su tamaño real y dejar espacio
para checkpoints/duplicados atómicos. No borrar datasets/pesos ajenos para hacer sitio.

Un entrenador pesado nuevo a la vez. La autorización concurrente antigua tenía QA de
sus dos encoders concretos: no acredita automáticamente dos ResNet50 nuevos. Preparación
CPU/revisión puede solaparse con límites agregados. Coordinar R1/Stage70 por propietario
vivo y lease, no matar procesos ajenos ni inventar permisos por ausencia momentánea.
No tocar Control Center, paging, voltajes, reloj o política global de energía.

## Cachés

B puede cachear representaciones del RGB genérico congelado. D/R con encoders entrenables
NO puede entrenar únicamente sobre sus features congeladas y llamarlo fine-tuning.
Persistir pares de voxels finales y RGB originales uint8 con transforms reproducibles,
o crops finales FP32 lossless por shards acotados;
si hay augmentaciones, documentar cuáles se aplican antes/después del cache. Un único
mapping normalizado no se recalcula con DEV.

Cache keys: dataset revision, acquisition/group, query/frame ID, timestamps exactos,
ROI transform, modality, representation version, normalizer, dtype; para features,
checkpoint SHA adicional. Nunca mezclar coordenadas latentes de productores distintos
sólo porque tengan128 dimensiones. No conservar RGB normalizado como uint8.

Entrenamiento se alimenta de cachés/shards, no relee cientos de MB de HDF5 por minibatch.
Preparación cronológica y bounded prefetch4workers, hasta8items si cabe. Excepciones
por E: ausente deben mostrar la raíz exacta; pausa recuperable y otras tareas viables.

## Precisión y checkpoints

Autocast sólo en el backbone admitido; operaciones sensibles de fase/correlación FP32.
No exigir equivalencia numérica de un modelo nuevo con A5: es arquitectura distinta.
Sí exigir equivalencia de loaders heredados, conversores y la ruta sin RGB con su E.
Paridad CPU/GPU con tolerancias previamente registradas, no bitexact entre dispositivos.

Full checkpoint cada100updates y al pausar: modelo, optimizador, scheduler/scaler,
RNG python/numpy/torch/cuda, sampler, posición/acumulación y hashes. Preferir pausar
al final de update completo. Si guarda durante acumulación, incluir gradientes y
contadores; no omitirlo afirmando resume exacto. Reanudar mismo device/dtype/recipe.

Un hash completo al admitir contenido y en fronteras/publicación; monitoreo barato de
size/mtime dentro del loop. No recorrer Git y todos los archivos por update/draw.
Una modificación sustantiva/hash inesperado bloquea su tarea, no se ignora. Metadata
accidental sin scores/targets se registra y no exige borrar todos los experimentos.

## Pruebas exigidas

Causalidad/availability de RGB; ties timestamp; no lookahead; crop/control por dataset;
P2 nativoigualGarl; queries/cold-start no filtrados porGT; gradientes en encoderRGB y
encoderE; congelaciónE duranteR; missingRGB byte/paridad conE; no llamada loaderRGB en
infer_E; maskingpadding/NaNs; features no mezcladas entre productores; phase/sign/cap;
correlationborders/forwardreverse; acumulación y masa; resumecontinuo/interrumpido;
fallo discohash/RAM en preparación, training y publicación; export y regeneración.

## Medición de coste

Cuatro ámbitos separados: preparación de medios, backbone(s), fusión/emisión, total
con disponibilidad del ROI e imágenes. Publicar cold y warm, batch1, p50/p95/p99,
memoria, contexto y número de objetos. Diferenciar compilar/capture/prefill del coste
recurrente. No sumar p95 individuales como p95 total ni comparar H8tresheads con Garluno
sin explicarlo. El sistema carece de detector/actuador: no llamar AEB a ese total.

Optimizar el cuello medido sin convertir R1 en una condición para investigar RGB.
No hacer que V13 se pase días cerrando registros históricos en lugar de entrenar.

## Coste de caché que se debe presupuestar

Un par FP32[2,20,128,128] ocupa2.621.440bytes; 88.744 pares sin compresión rondan232,6GB
decimales, más RGB/metadata y contexto. No preasignar ese cache sin comprobar espacio.
Medir compresión lossless en TRAIN por shards, reservar límite y usar staging acotado
si es necesario, con progreso durable. El timevolume nativo no se transforma a uint16
o FP16 silenciosamente: sus interpolaciones/normalización son parte del contrato.
Reutilizar caches exactos existentes si coinciden; no declarar preparada toda la nueva
representación porque existan los 12canales de A5. No ampliar datos para llenar el disco.

No convertir a uint8 un crop bilineal normalizado o interpolado y después afirmar
paridad: se perderían valores fraccionarios. Para entrenamiento con acceso aleatorio
preferir bloques comprimidos independientemente por consulta (por ejemplo, miembros
ZIP/arrays separados con índice) en vez de descomprimir un array de32filas entero por
cada muestra. Un mmap completo sólo se autoriza si su reserva de disco cabe.
La optimización I/O conserva el orden de muestras y la receta; no agrupar por secuencia
silenciosamente para aparentar throughput.
