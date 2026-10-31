# 11 — Runtime, recursos y reproducibilidad

## Qué recursos están acreditados

La auditoría histórica del 30 de octubre documentó una RTX 5070 Ti Laptop con **12.227 MiB de memoria total**, 32 HDF5 y sus CSV, y aproximadamente 184,20 GiB de HDF5. Eso no acredita memoria libre actual, que esos datos estén montados en esta sesión ni que el entorno de entrenamiento esté sano. También dejó constancia de un problema de arranque de la `.venv` y de rutas eAP en una unidad que no estaba montada. [Plan y evidencia de procedencia](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/superpowers/plans/2026-10-30-evttc-transfer-diagnosis.md).

El procedimiento debe empezar pequeño en el equipo disponible. No extrapolar una duración o un coste total de GPU a partir del número de parámetros de la cabeza: la extracción de features puede dominar el trabajo.

## 1. Preflight antes de reservar GPU

1. Verificar commit y estado del checkout, sin sobrescribir cambios ajenos.
2. Registrar Python, PyTorch, CUDA, driver, GPU, memoria total/libre y espacio de trabajo disponible.
3. Crear o reparar un entorno aislado conforme al lockfile; comprobar imports y una operación mínima. No mantener como entorno de entrenamiento un bypass de carga de `site` usado para lectura de archivos.
4. Comprobar rutas y permisos de los datos necesarios, manifests, paquete sellado y hashes. No descargar otra copia completa si basta con los medios presentes.
5. Ejecutar una consulta pequeña y un minibatch de prueba en un directorio nuevo.
6. Medir RAM/VRAM máximas, coste de lectura y tamaño esperado de cachés con ese piloto.
7. Establecer límites de memoria y almacenamiento que dejen margen para sistema y procesos existentes; no ocupar por defecto toda la memoria anunciada.

Para F1/F2 entrenar desde cachés de features cuando sus identidades sean válidas. Para P0 usar microbatches y acumulación solo tras el preflight. Reducir batch puede alterar normalización dependiente del batch; documentar y comparar ese cambio en lugar de tratarlo como invisible.

## 2. Límites de tiempo que no deben mezclarse

| Medida | Definición operativa |
|---|---|
| Historia de sensores | Distancia entre el primer dato usado y la consulta; actualmente hasta 650 ms |
| Edad de observación | Consulta menos timestamp efectivo de la observación |
| Retraso de disponibilidad | Momento disponible menos timestamp de sensor |
| Latencia de procesamiento | Desde que están disponibles las entradas necesarias hasta que la salida queda lista |
| Latencia extremo a extremo | Desde el evento/frame de referencia hasta salida disponible, con adquisición/colas incluidas |
| Throughput | Consultas o eventos procesados por segundo en ejecución sostenida |
| Arranque | Costes iniciales de carga, indexación, calibración y preparación |

**650 ms de historia no significa esperar 650 ms en cada consulta.** Una ventana pasada puede mantenerse en un buffer. Por otra parte, publicar solo tiempo de kernel no representa latencia extremo a extremo. Cada tabla debe indicar su frontera de medición.

El adaptador offline existente reconstruye la historia para cada consulta y vuelve a calcular productores. Incluye lectura, preparación, modelos y salida al host; excluye carga, indexación, calibración y hashing iniciales. Sus tiempos no prueban throughput streaming. [Guía del runtime](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/evttc_competition.md).

## 3. R00: perfilado antes de optimizar

Con candidato científico congelado y población de perfilado predefinida:

1. Medir lectura/decodificación, ROI y representación, transferencia al dispositivo, productores, contextos y fusión.
2. Registrar arranque y régimen estable por separado; warm-up no elimina consultas científicas.
3. Sincronizar GPU en fronteras necesarias para la medida que se declara; los temporizadores host de etapas pueden incluir trabajo asíncrono.
4. Ejecutar el benchmark con trazas pesadas desactivadas y medir por separado el sobrecoste de instrumentarlas.
5. Informar p50/p95/p99, media, pico VRAM/RAM, tamaño de entradas y throughput sostenido.
6. Registrar orden de variantes y estado frío/caliente. Repetir orden o equilibrarlo si se pretende compararlas; una sola pasada en orden fijo mezcla ventajas de caché.

No sumar tiempos de etapas con fronteras incompatibles ni extrapolar la suma a otra GPU. Medir energía o coste económico solo si se dispone de medición/facturación; no convertir tiempo de pared en consumo por una potencia nominal.

## 4. Cachés con identidad suficiente

La preparación usa la ROI de la consulta actual para construir su historia. Un mismo timestamp histórico puede corresponder a otro tensor cuando cambia esa ROI. Cachear solo por ancla sería incorrecto. [Preparación por consulta](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/operational/evttc_competition/prepare.py).

Una identidad mínima de caché de features contiene:

```text
hash del medio y sensor
intervalos/framenumbers reales
ROI y transformación espacial
calibración y convención bbox
versión de representación y precisión numérica
política de preparación
pesos y normalizadores
```

La cache de bytes RGB o eventos decodificados puede tener una clave más pequeña porque todavía no depende de ROI/modelo. Separar esos niveles evita recomputación sin suponer equivalencias inexistentes. Cada cache tiene manifest, contador de hits/misses, integridad y política explícita de invalidación. Ninguna cache puede contener futuros ocultos, targets en features o población mezclada para ajustar normalización.

## 5. Optimización con paridad

Orden recomendado tras R00: reutilizar decodificación, reducir lecturas repetidas, cachear entradas idénticas, agrupar operaciones compatibles y evaluar el runtime existente de productores. No empezar por CUDA custom si el cuello de botella es lectura. El repositorio ya contiene optimizaciones de `train40_runtime`; su paridad local no acredita automáticamente sensor→TTC en el nuevo adaptador. [Productores optimizados](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/operational/train40_runtime/producers.py).

Antes de reemplazar la ejecución de referencia:

- Comparar tensores preparados, features y salidas finales sobre fixtures y datos de desarrollo fijados.
- Exigir igualdad exacta donde la operación lo permita y tolerancias predefinidas cuando cambien algoritmos numéricos.
- Comprobar causalidad, resets, ventanas vacías y preservación de todas las consultas.
- Documentar si batch, precisión o compilación altera outputs; una aceleración que cambia el modelo pertenece a otra variante.

No mezclar una mejora científica y una optimización de runtime en el mismo contraste principal. El objetivo es poder atribuir cada cambio.

## 6. Recibo por ejecución

Cada run nuevo debe guardar configuración resuelta, commit, IDs de experimento/variante, semilla, roles, hashes de datos y modelos, normalizadores, entorno, tiempos, máximos de memoria y estado final. Las predicciones conservan la clave exacta de consulta y motivos de ausencia. Las métricas se regeneran desde ellas; no escribir manualmente las tablas.

Guardar un `parent_run_id` cuando la variante deriva de otra. Un resume exige los mismos hashes y configuración; si cambian, abrir un run nuevo. Un fallo no se corrige borrando filas del ledger. Las herramientas existentes ya verifican freezes, CSV y archivo final; reutilizarlas cuando el formato corresponda. [Verificador existente](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/scripts/verify_evttc_competition_run.py).

Los hashes verifican identidad, no ausencia de fuga por sí mismos. La genealogía y el rol de cada grabación siguen gobernados por G1. Conservar pesos y medios fuera del Git público según el contrato del proyecto; publicar manifests portables y evidencia que no incluya rutas personales.

## Condición de cierre

R00 termina con un perfil reproducible y, si se implementó una optimización, su comparación de paridad y recursos. Una mejora de latencia solo se afirma bajo las fronteras y hardware medidos. Las estimaciones de campaña se calculan entonces a partir de tamaño de población, repeticiones y tiempos observados, mostrando qué fases se reutilizan y cuáles se recalculan.
