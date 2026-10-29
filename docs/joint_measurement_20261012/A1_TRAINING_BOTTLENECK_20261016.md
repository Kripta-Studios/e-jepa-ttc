# A1: cuello de botella medido durante el entrenamiento

El 16 de octubre, hacia las 19:06 de Madrid, la espera de datos ocupó el
**54,43 % del tiempo muestreado del hilo principal** de A1-LAST-BLOCK. La GPU
promedió **9,8 % de utilización** en la ventana de recursos. CUDA Graphs estaba
ejecutándose: se observaron llamadas `replay` tanto desde el forward como desde
el hilo de backward. La prioridad inmediata es acelerar el lector.

## Medición y alcance

Se adjuntó py-spy 0.4.2 al proceso científico 28392, sin inyección de código,
pausa, reinicio ni updates adicionales. Se solicitaron 40 segundos a 50 Hz,
con hilos inactivos incluidos y lectura no bloqueante. Se obtuvieron 1.920
muestras por hilo, 5.760 entre los tres hilos Python, y se conservaron los
79 errores de muestreo declarados. La suma ponderada por hilo es 38,4 s.

En paralelo, 20 observaciones psutil/nvidia-smi cubrieron 44,15 s entre primera
y última muestra: **30 updates, 1,472 s/update, 2.446 updates/h**. Son ventanas
solapadas, no idénticas. No se atribuye causalidad a diferencias respecto a
otros tramos del entrenamiento. Las pruebas CPU y Pyright anteriores ya habían
terminado antes de este muestreo.

| Actividad del hilo principal | Porcentaje de muestras |
| --- | ---: |
| Esperar o solicitar el siguiente batch | 54,43 % |
| Forward, pérdida y backward, incluidas sincronizaciones CPU/GPU | 34,58 % |
| Optimizador/scheduler | 3,49 % |
| Journal de updates | 3,07 % |
| Transferencia de batch | 1,77 % |
| Clipping de gradientes | 0,99 % |
| Resto | 1,67 % |

La tabla particiona muestras del hilo principal; **no mide tiempos de kernels**.
El hilo de carga trabaja simultáneamente y sus tiempos no se suman a esta tabla.
Sus hojas más frecuentes fueron descompresión ZIP (`_read1`: 16,34 s), CRC
(`_update_crc`: 7,34 s), lectura ZIP (2,20 s), copia de arrays NumPy (2,00 s)
y ensamblado `stack` (1,88 s). Esto apunta a descompresión, validación y copias
seriales como coste dominante de entrada, no a saturación demostrada del SSD.
Los 7,87 MB/s leídos por el proceso son lecturas lógicas; pueden venir de la
caché del sistema operativo y no constituyen una medida de ancho de banda físico.

La CPU del proceso promedió 125,5 % en la escala donde un núcleo es 100 %.
La GPU osciló entre 2 y 24 % de utilización, con 3.375–3.696 MiB ocupados y
83–87 °C. No se midieron motivos de throttling, por lo que no se atribuye el
cuello de botella a temperatura. La RAM disponible bajó a 4,00 GiB; el RSS
del proceso estuvo entre 8,33 y 9,30 GiB.

## Qué está aplicado y qué falta

| Optimización | Estado en este proceso |
| --- | --- |
| CUDA Graphs de forward/backward | Activo y observado en stacks reales; QA A1 de captura/reanudación aprobada |
| Prefetch de un batch con generador privado | Activo; conserva IDs y sampler autoritativo |
| Caché LRU de filas decodificadas, máximo 8 GiB | Activa; reduce retención cuando quedan menos de 4 GiB disponibles |
| Memoria fijada y transferencias `non_blocking=True` | Activas; microbatches contiguos conservan el almacenamiento fijado |
| Caché sensorial y compresión sin pérdida | ZIP DEFLATE externo, NPZ interno almacenado; no se reconstruyen eventos crudos por update |
| SHA de admisión y fingerprints posteriores | Activos; no se recalculan hashes completos por update |
| Lector paralelo y vistas NumPy sin copia intermedia | Implementados como candidato separado; **todavía no instalados** |
| Caché completa en RAM/VRAM | No activa; sólo los píxeles P sin comprimir ocupan aproximadamente 79 GiB |
| Proceso y caché persistentes entre scopes | No: el coordinador lanza un hijo nuevo cada 900 s |
| BF16/FP16, TF32, cuantización o aproximaciones | No activas; se mantiene la receta FP32 y su precisión registrada |
| Captura completa de pérdida/optimizador | No: el modelo está capturado, la pérdida y Adam siguen fuera |

El entrenamiento reutiliza las optimizaciones de captura y transporte ya
admitidas. **No incorpora automáticamente todas las mejoras de inferencia**:
el prefetch secuencial por packs de B1 se diseñó para evaluar cada fila una vez;
entrenar usa draws aleatorios con reemplazo, duplicados y supervisión adicional.
No es correcto conectarlo sin adaptar y comprobar ese contrato.

El límite de 8 GiB permite almacenar como máximo aproximadamente el 10 % de
los píxeles P. Con muestreo uniforme, aumentar ligeramente esa caché no elimina
la mayor parte de los misses. Además, los hijos nuevos pierden la caché y los
grafos cada 15 minutos. La curva del primer tramo contiene 548 updates: los
últimos cien promediaron 1,176 s/update, frente a 1,53 s/update en toda la curva;
el ciclo completo incluyó además admisión, guardado y lanzamiento. La diferencia
no permite separar exactamente hashing, calentamiento y caché, pero confirma
que el ritmo caliente no es una ETA de extremo a extremo.

La pérdida actual realiza materializaciones escalares por microbatch y varios
guards sincronizados. Constituyen la siguiente optimización a medir después de
la entrada; el 34,58 % anterior no debe describirse íntegramente como cómputo GPU.
El journal supone aproximadamente 3 %, por lo que eliminar sus garantías de
recuperación no ataca el coste principal.

## Acción en curso

`parallel_training_sensor.py` conserva el sampler, el batch efectivo, todos los
targets, el collator y FP32. Usa hasta ocho decodificadores, mantiene una sola
tarea de prefetch y evita extraer cada NPY mediante copias intermedias. Sólo
deduplica lecturas físicas: los draws repetidos permanecen en el batch.
Mantiene SHA de packs, CRC de miembros externos y rechazo de cambios de archivos.

Diez pruebas CPU aprobaron igualdad de todos los campos, orden y duplicados,
prefetch y RNG, rechazo de tokens/relojes/hashes/miembros duplicados y mutación
de packs/recibos incluso en cache hits, cierre de workers y límites de caché.
Ruff y Pyright del lector aprobaron. **Faltan admisión con datos P reales,
benchmark controlado y verificación de transición desde el checkpoint**;
no se declara todavía ninguna aceleración de producción por este candidato.

La transición deberá respetar los updates guardados y las fuentes científicas
registradas, mediante una capa de runtime explícita. Evitar reinicios innecesarios
de proceso y conservar la caché entre scopes es otra mejora prioritaria, pero
no debe perder los límites de memoria ni la contabilidad de cada scope.

## Reproducción

Evidencia: `evidence_A1_live_profile_20261016/{STACKS,RESOURCES,SAMPLING,SUMMARY}.json`.
El resumen contiene hashes de entradas y fuentes científicas. Desde la raíz:

```powershell
$env:PYTHONPATH = 'src;.'
python -m operational.joint_measurement.live_profile_report --directory docs/joint_measurement_20261012/evidence_A1_live_profile_20261016
```

Las recomendaciones se contrastaron con las fuentes primarias:
[guía de rendimiento de PyTorch](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide),
[memoria fijada y copias no bloqueantes](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)
y [muestreo externo de py-spy](https://github.com/benfred/py-spy).
Estas técnicas respaldan el enfoque, pero no prueban una mejora local sin medirla.
