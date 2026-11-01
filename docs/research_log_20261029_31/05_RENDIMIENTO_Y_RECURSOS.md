# 05. Rendimiento: qué se midió y qué se optimizó

## Separar los relojes de la ejecución

El tiempo de optimizador, el de una cabeza sobre tensores precargados y el de una
consulta desde sensores miden trabajos distintos. Un gate de menos de un
milisegundo puede convivir con una consulta que tarda segundos porque hay que
leer eventos, recortar imágenes, construir representaciones y ejecutar productores.
Los informes de esta campaña identifican el límite temporal de cada medida.

El hardware local verificado es AMD Ryzen 9 9955HX, 16 cores/32 procesadores
lógicos, NVIDIA GeForce RTX 5070 Ti Laptop GPU y 32 GB de RAM. El replay final
registra Python 3.11.15, PyTorch 2.11.0+cu128 y CUDA 12.8. La GPU y software del
envío original constan en su entorno; su CPU se comprobó después sobre el mismo
portátil, porque el recibo antiguo no incluía ese campo.

La A40 de 9 vCPU/50 GB planteada durante la conversación no fue la máquina de
esta campaña. No se ejecutó un benchmark emparejado contra ese pod, por lo que no
hay un factor medido de aceleración A40. Más VRAM por sí sola no resuelve el
coste de decodificación y construcción de inputs identificado aquí.

## Extracción nativa del 30 de octubre: CUDA Graphs sí ejecutados

El commit `dfbd20b7` añadió aceleración de productores con formas estables. El
recibo [PERFORMANCE.json](../evttc_transfer_v2/evidence/phase_adapt_final/PERFORMANCE.json)
registra cuatro capturas y su paridad inicial exacta:

| Productor | Shape capturada | Setup s | Replays registrados |
|---|---|---:|---:|
| E_A5_MATCHED | `[4,3,12,128,128]` | 1,0283 | 2.228 |
| E_C2F_MATCHED | `[4,3,12,128,128]` | 1,3580 | 2.228 |
| R_A5 | `[1,3,3,128,128]` | 0,6548 | 11.140 |
| R_C2F | `[1,3,3,128,128]` | 1,5851 | 11.140 |

La captura evita relanzar desde Python la misma secuencia de operaciones GPU.
Se reutilizan buffers y shapes; no se presupone que una entrada dinámica
cualquiera pueda entrar en el grafo. Los tests de paridad comparan el resultado
de la ruta acelerada con la referencia y conservan el contrato numérico.

La extracción reutilizó 1.377 shards y calculó 2.228 estados nuevos. El tiempo
sensorial acumulado de estos últimos fue 1.180,691 s, con media 0,529933 s,
mediana 0,396728 s y p95 1,301974 s. La preparación representó el 69,66 % del
tiempo medido. Una consulta perfilada previamente durante extracción concurrente
no se utilizó como si fuera un benchmark de throughput emparejado.

## Qué evitaron las cachés

El recibo de extracción registra aproximadamente 42,45 GB leídos de eventos y
205,75 GB de hits en la caché de eventos, además de 13.043 hits de frames. Se
abrieron 22 fuentes en ese tramo y se reutilizaron validaciones/cajas. Los contadores
describen bytes solicitados y reutilizados, no un volumen de datos nuevos que
deba descargarse o guardarse en Git.

Se separaron tres niveles:

1. Datos crudos y frames ya decodificados, para consultas vecinas con historia
   solapada. La selección temporal y ROI se sigue aplicando por consulta.
2. Estados/features de productores, congelados y vinculados a sensor, código y
   checkpoints. Permiten entrenar varias cabezas sin reabrir sensores.
3. Checkpoints y recibos completos por shard/experimento. La recuperación utiliza
   trabajo válido existente y rechaza una mezcla de configuraciones incompatibles.

La reanudación preserva las ausencias y las razones del fallo. Un shard incompleto
no se transforma en una predicción válida solo para mantener la longitud de una
matriz. Las pruebas verifican identidad y paridad, no únicamente que exista un
archivo con el nombre esperado.

## Prefill, preload y coste del gate

En este modelo, «prefill» no significa una KV cache de un LLM. La operación útil
consiste en construir la historia causal y reutilizar estados ya calculados.
El preload de arrays de features permite entrenar gates pequeños en CPU con dos
threads. Añadir CUDA Graphs al optimizador de una cabeza diminuta no elimina el
coste dominante de sensores y puede introducir overhead innecesario.

En el router de arranque se completaron 117 ajustes. Sus temporizadores de
optimizador suman 137,703 s, mediana 1,304862 s y p95 1,481902 s por ajuste.
Esos tiempos excluyen setup, lectura, evaluación y exportación. La mediana del
gate CPU batch-one de esa iteración fue 0,285750 ms; tampoco incluye sensores.

El gate robusto previo registró 1,273 ms de mediana y 9,023 ms para geometría más
gate. Cambiaron arquitectura y ruta de evaluación entre iteraciones; estas cifras
no autorizan a atribuir toda la diferencia a un único cambio de hardware o kernel.

## Compresión y persistencia

La población de arrays del experimento de arranque ocupa 1.638.324 bytes sin
comprimir y 244.897 bytes en la caché comprimida, según el recibo. NPZ reduce
almacenamiento y lectura persistente; una vez cargadas, las matrices vuelven a
ocupar su tamaño normal en RAM. No se afirma que comprimir acelere el forward.

Los ZIP de entrega comprimen código, pesos, features pequeñas y recibos. El
paquete final del envío ocupa 36.023.481 bytes. Los datasets HDF5/bag y los entornos
Python no van dentro. El manifiesto comprueba cada miembro después de comprimir;
un ZIP que abre correctamente todavía puede contener un checkpoint equivocado,
por lo que también se verifica la unión entre pesos, replay y evaluación.

## Replay final: otra ruta, otro alcance de tiempo

La ruta final de competición usa una caché lossless de chunks de eventos de
64 MiB y 16 frames RGB. Mide el recorrido completo por consulta desde lectura/
preparación hasta extracción del resultado a host. Sincroniza CUDA alrededor del
trabajo relevante. Deja fuera la carga inicial de modelos, hashes/indexación y
la escritura del ledger. La geometría y el router sí están dentro.

| Etapa | Media s/consulta | Fracción aproximada |
|---|---:|---:|
| Preparación sensorial | 1,158298 | 66,56 % |
| Grafo neuronal original | 0,580139 | 33,34 % |
| Geometría y nuevo router | 0,001853 | 0,11 % |
| Total medido | 1,740290 | 100 % |

El intervalo registrado del replay va de 10:44:03 a 10:56:50 UTC del 31 de octubre.
El setup de esa sesión mide 7,759 s. La suma de costes de consulta no tiene que
igualar exactamente el tiempo de pared completo: existen escrituras, aperturas,
verificaciones y otros trabajos fuera del límite declarado.

El envío original registraba 3,268250 s por consulta. El cociente observado es
1,88× y la latencia media baja aproximadamente un 46,75 %. No se midió una
ablación idéntica con/sin caché en esta ruta final; el dato es una comparación
descriptiva entre ejecuciones. No se presenta como aceleración atribuible a CUDA
Graphs, porque este replay no usa la misma ruta capturada de la extracción nativa.

## Consecuencia práctica

El modelo final sigue lejos de inferencia en tiempo real a la cadencia de cámaras.
La siguiente mejora de throughput requiere perfilar y reducir preparación:
decodificación repetida, representaciones por ROI, reutilización temporal o
pipeline CPU/GPU. Optimizar los 1,9 ms del router apenas cambia los 1.740 ms totales.
Cualquier implementación nueva debe volver a demostrar causalidad, paridad y
límites de memoria; una caché que cambie las ventanas del modelo sería otro método.
