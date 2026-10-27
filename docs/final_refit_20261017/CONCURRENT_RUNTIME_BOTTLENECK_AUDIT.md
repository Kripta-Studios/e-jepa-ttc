# Auditoría de cuellos de botella del runtime concurrente

## Síntesis con la medición concurrente y térmica

La decisión operativa conserva **ambos fits concurrentes**, como pidió el usuario.
Este diagnóstico no autoriza serializarlos, cambiar la receta ni aplicar cambios de
fuente sin verificar y registrar su equivalencia. El checkpoint permanece cada 100 updates.

La captura externa duró 180,020 s; sus 36 muestras abarcan 175,019 s entre el primer
y último punto. En ese intervalo R_A5 avanzó 5.759→5.847 (30,17 updates/min;
1,989 s/update), y R_C2F 521→575 (18,51 updates/min; 3,241 s/update).
La GPU compartida registró 50,9 % de utilización media, 6.657 MiB de VRAM,
48,9 W y 86,6 °C. Los procesos consumieron aproximadamente 1,15 y 1,09 núcleos
de CPU; el host completo, 38,1 %. Hay memoria disponible, pero no evidencia de
que añadir workers resuelva el cuello: los decoders esperaron en 95–98 % de sus
muestras y el prefetch de A5 sólo bloqueó 2,5 ms en 100 batches entre checkpoints.

Evidencia reproducible: [análisis JSON](../../artifacts/final_refit_20261017_runtime_v2/CONCURRENT_DIAGNOSTIC_20261017/analysis.json),
[informe de captura](../../artifacts/final_refit_20261017_runtime_v2/CONCURRENT_DIAGNOSTIC_20261017/REPORT.md)
y scripts junto a esos archivos. El muestreo Python fue no bloqueante, 10 Hz,
60 s por proceso; no es una traza de kernels y no separa ejecución, despacho y
espera del controlador dentro de backward.

**Limitación térmica confirmada, pérdida de rendimiento no cuantificada.** La
captura adicional de 13 puntos, 17:26:42.711–17:27:46.314 CEST (63,603 s), marcó
`SW Thermal Slowdown=Active` en los 13 puntos, 86–88 °C, y `SW Power Cap=Not Active`.
El contador térmico aumentó 4,543554 s: 7,14 % del intervalo según ese contador.
Ni 13/13 implica actividad térmica continua, ni 7,14 % representa el porcentaje
de rendimiento perdido. No puede atribuirse a temperatura toda la diferencia
respecto a una medición anterior. Se conservan [CSV térmico](../../artifacts/final_refit_20261017_runtime_v2/CONCURRENT_DIAGNOSTIC_20261017/THERMAL_60S.csv)
y [snapshot NVIDIA](../../artifacts/final_refit_20261017_runtime_v2/CONCURRENT_DIAGNOSTIC_20261017/GPU_THERMAL_SNAPSHOT.txt).

Nsight Systems 2025.6.3 está instalado, pero rechazó los contadores GPU por
`ERR_NVGPUCTRPERM` y deshabilitó WDDM por requerir privilegios administrativos.
No se obtuvo una traza CUDA de los procesos activos ni se cambiaron permisos,
drivers, refrigeración o políticas de energía. La opción de captura CUDA global
documentada por el CLI sólo alcanza procesos nuevos tras iniciar la colección.

### Orden de optimización recomendado dentro de la ejecución concurrente

1. **Revisar refrigeración efectiva** para eliminar un freno ya observado. El consumo
   de unos 49 W no demuestra margen térmico; subir el límite de potencia no resuelve
   por sí mismo una limitación de temperatura. No hay una ganancia porcentual demostrada.
2. **Sacar el inventario completo de procesos del camino crítico**, compartiendo un
   snapshot con caducidad, identidad de PID y rechazo ante fallo; mantener RAM,
   commit y disco locales. Los 143 ms/update medidos en A5 equivalen aproximadamente
   al 7,2 % de sus 1,989 s/update, usando ventanas próximas pero distintas. El ahorro
   real será menor que eliminar todo el guard; no explica una aceleración de 2×.
3. **Reducir las sincronizaciones dentro del objetivo**, empezando por validaciones
   de entradas y conteos de targets que pueden prepararse en CPU. Consolidar los 16
   escalares finales es una mejora menor distinta. Las barreras intermedias impiden
   adelantar el envío de operaciones CUDA; hace falta medir cuánto cuesta cada región.
4. **Eliminar cálculo repetido del transporte.** `_summarise_one` vuelve a muestrear
   el mismo campo reverse para global y foreground, aunque ese muestreo no depende
   de `spatial_weight`. Reutilizarlo reduce cuatro samplers a dos por llamada completa
   de modelo legacy. Con cuatro gathers por sampler, la auditoría estática predice
   16→8 gathers por partición. El perfil previo registra 240 kernels de backward de
   indexación en tres updates, pero no atribuye inequívocamente cada kernel al sampler.
   Los 132,1 ms acumulados son **milisegundos en tres updates**, no segundos ni
   una medición del entrenamiento concurrente. Compartir nodos cambia el orden de
   acumulación de gradientes: exige paridad BF16 de pérdida, gradientes, AdamW y RNG.
5. **Optimización más agresiva: menos particiones y lanzamientos**, seguida de
   capturas CUDA parciales por firma fija si resultan viables. Fusionar por T podría
   reducir hasta seis forwards a dos, pero puede cambiar dropout, RNG y reducciones.
   No es un reemplazo transparente ya admitido ni tiene una aceleración medida.
   Capturar el bucle completo con I/O, formas variables y decisiones Python no es
   una opción directa. Un runner compartido entre modelos requiere además aislar
   RNG y garantizar recuperación independiente; alternar updates completos no
   elimina automáticamente las barreras dentro de cada update.

El mayor potencial está en hacer el forward/backward menos fragmentado. El perfil
Python sitúa ahí 90–95 % de la residencia del hilo principal, sin demostrar que todo
ese tiempo sea cómputo CUDA útil. Los cambios locales pequeños tienen un beneficio
acotado; no hay evidencia para prometer 31 updates/min en ambos o un factor 2×.

La [guía de PyTorch](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
recomienda evitar sincronizaciones CPU/GPU innecesarias.
Los [CUDA Graphs de PyTorch 2.11](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs)
exigen controlar formas, memoria y operaciones capturables.
[NVIDIA MPS](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html) puede mejorar
concurrencia entre procesos, pero sólo está soportado en Linux/QNX, no en este
Windows nativo. Según [NVIDIA SMI](https://docs.nvidia.com/deploy/nvidia-smi/index.html),
utilización indica tiempo con kernels activos, no porcentaje de toda la capacidad
aritmética utilizada; 51 % no significa que el otro 49 % sea capacidad aprovechable
automáticamente por otro proceso.

No se ejecutaron nuevos optimizer updates técnicos ni se modificaron fuentes
ligadas al runtime durante este diagnóstico. Los updates científicos de ambos
entrenamientos continuaron con su implementación existente.

Fecha: 17 de octubre de 2026. Alcance: lectura de código y recibos en vivo. No se
detuvo el entrenamiento, no se importó PyTorch, no se ejecutó GPU y no se modificó
ninguna fuente ligada al runtime.

El runtime activo está en `artifacts/final_refit_20261017_runtime_v2`, según
`artifacts/final_refit_20261017/ACTIVE_RUNTIME.json`. Los contadores son acumulativos
y el entrenamiento continúa, por lo que las cifras siguientes identifican el snapshot
inspeccionado y no un endpoint.

## Ruta ejecutada por update

El bucle de `operational/final_refit/producers.py:743-862` ejecuta, en este orden:

1. guard de recursos;
2. obtención del batch efectivo desde el prefetch;
3. transferencia de todas sus particiones a CUDA;
4. validación de cada partición;
5. forward y objetivo de cada partición física;
6. comprobación finita, backward y clipping;
7. lectura de la reserva global de recuperación;
8. journal durable `begin`, optimizer step y journal durable `commit`;
9. extracción de 16 escalares de pérdida;
10. cada 100 updates, curva, checkpoint completo y recibos.

La partición se construye por `(historical_role, T)` en
`operational/final_refit/producers.py:317-335`. Con tres roles históricos y dos
longitudes temporales observadas, un batch B32 puede contener entre una y seis
particiones físicas. `operational/rgb_port_revision/loss.py:94-176` hace un forward
separado por partición y reduce después sus componentes para conservar un único límite
de optimizer.

## Hallazgos cuantificados

### 1. El inventario de procesos vuelve a ser el mayor coste administrativo

Los dos `RGB_RUNTIME.json` activos registraban:

| Fit | Calls | Scans completos | Media scan completo | Media guard barato | Guard total/call | Máximo |
|---|---:|---:|---:|---:|---:|---:|
| R_A5 | 811 | 34 | 3,475 s | 3,238 ms | 148,772 ms | 8,091 s |
| R_C2F | 501 | 20 | 3,557 s | 3,271 ms | 145,130 ms | 8,093 s |

El scan completo ocurre cada 25 updates en
`operational/final_refit/resource_monitor.py:255-317`. Aunque la línea de comando se
cachea, cada scan invalida la caché global de `process_iter`, consulta `create_time`
para todos los procesos y vuelve a leer el RSS de todos los procesos protegidos no
clasificables (`resource_monitor.py:127-231`). Dos entrenadores independientes repiten
este trabajo sobre el mismo host.

La cota observada es mucho más cara que el A/B aislado anterior. El microbenchmark
`RESOURCE_MONITOR_AB_V2.json` sigue siendo correcto para aquel estado del host, pero no
predice el coste bajo dos procesos de entrenamiento y actividad concurrente.

### 2. Hay sincronizaciones CPU/CUDA repetidas por partición

Después de copiar el batch a CUDA, `_validate_batch` comprueba finitud, mínimo y máximo
con tres conversiones `bool(...)` (`operational/rgb_port/train_producers.py:731-747`).
Cada conversión necesita materializar un resultado CUDA en CPU.

Para cada partición física, la pérdida hace además:

- cinco `.item()` en T2: conteo físico, supervisado, extent, width/center y pair ratio;
- un sexto `.item()` en T3 para consistencia temporal;
- un `bool(torch.isfinite(relation_error).all())` en
  `operational/rgb_port_revision/loss.py:142-147`.

Por tanto, antes del backward hay **nueve sincronizaciones explícitas por partición T2**
y **diez por partición T3**, además de la comprobación final de `total` en
`operational/final_refit/producers.py:777`. Un batch mixto con seis particiones puede
alcanzar 54–60 límites explícitos de sincronización antes de contar clipping.

Tras el optimizer se extraen 15 componentes y el total mediante 16 llamadas separadas
`float(tensor.cpu())` (`producers.py:790-796`). La primera espera al trabajo pendiente;
las restantes todavía pagan despacho y transferencia individual.

### 3. El backward de indexación es el hotspot CUDA conocido

`RUNTIME_THREAD_PROBE.json` registró:

- `aten::_index_put_impl_`: 135.624,454 µs CUDA acumulados, 258 llamadas;
- `indexing_backward_kernel_stride_1`: 132.111,549 µs, 240 llamadas.

Las pérdidas usan indexación booleana de forma repetida, por ejemplo en
`src/e_jepa_ttc/losses/causal_scale_ttc.py:77-367`; la pérdida relacional también
selecciona por máscara en `operational/rgb_port_revision/loss.py:142-148`. El perfil
localiza el tipo de operación, pero no atribuye por sí solo todo el kernel a una línea
concreta.

### 4. La preparación CPU está solapada razonablemente

| Fit | Preparación/batch | Espera de prefetch/batch | Espera decoder/batch |
|---|---:|---:|---:|
| R_A5 | 415,010 ms | 4,531 ms | 111,412 ms |
| R_C2F | 425,212 ms | 3,771 ms | 95,151 ms |

El prefetch de profundidad dos oculta casi toda la preparación. El decoder todavía
acumula fallback reads, pero su trabajo también ocurre dentro del worker de prefetch.
Subir hilos o profundidad aumentaría RAM y contención de disco sin evidencia de que el
hilo GPU esté esperando esa preparación.

### 5. La contabilidad durable tiene coste menor, pero ocurre siempre

Cada update hace una lectura de todos los `UPDATE_JOURNAL.json` mediante
`assert_recovery_budget` (`resource_monitor.py:330-358`). Con dos fits activos, cada
proceso lee su journal y el del otro.

`ProducerCheckpoint.begin()` y `commit()` escriben dos JSON atómicos con `fsync` por
update (`operational/rgb_port/train_producers.py:330-355` y
`operational/rgb_port/accounting.py:109-130`). Esta doble barrera sostiene el límite
documentado de recuperación. El checkpoint cada 100 añade serialización, `fsync`, hash,
puntero, alias, journal y recibo (`train_producers.py:357-447`). La medición anterior
atribuyó sólo 0,08 % al checkpoint periódico; no hay evidencia para cambiar su cadencia.

## Cambios recomendados, por prioridad

### 1. Compartir el snapshot de recursos entre los dos entrenadores

Es la intervención con mayor evidencia y no toca el objetivo. Un único monitor de host
puede publicar snapshots autenticados con identidad `(PID, create_time)`, RSS, RAM,
commit y disco. Ambos trainers seguirían haciendo el guard barato local por update y
rechazarían un snapshot ausente, corrupto o demasiado antiguo.

Antes de admitirlo hacen falta pruebas de:

- nacimiento, salida y reutilización de PID;
- muerte o bloqueo del monitor;
- snapshot truncado, stale y con identidad equivocada;
- presión de RAM, commit y disco;
- caída entre `begin` y `commit`;
- igualdad de decisiones con el guard actual sobre una traza capturada;
- A/B concurrente real, no sólo un microbenchmark aislado.

Una alternativa más compleja es una enumeración nativa única de Windows. No conviene
reducir simplemente la frecuencia ni ignorar procesos protegidos: eso debilitaría una
garantía operativa.

### 2. Consolidar las transferencias escalares y mover validaciones al host

Los 16 componentes pueden apilarse en orden fijo y copiarse en una sola transferencia
CPU. Las tres comprobaciones de rango pueden realizarse sobre el batch host dentro del
worker de prefetch, antes de la copia CUDA. Ambas medidas preservan el valor del
objetivo.

La aceptación exige igualdad exacta de las 16 curvas, mismo punto de fallo para inputs
inválidos, y paridad de modelo, optimizer y RNG en T2, T3 y batches mixtos de ambos fits.
Debe medirse el tiempo por update; el perfil previo sitúa estas conversiones por debajo
del inventario de procesos.

### 3. Eliminar `.item()` internos sin cambiar las reducciones

Los conteos derivados sólo de targets pueden calcularse en CPU antes de la transferencia
y pasarse junto al batch. Los conteos dependientes del output pueden mantenerse como
tensores y transferirse juntos al final. Esto permitiría lanzar la pérdida sin barreras
CPU entre subcomponentes.

Es una modificación sensible: los conteos controlan ramas vacías, denominadores y el
`topk` de la cola. La prueba debe cubrir máscaras vacías, TTC inválido, T2/T3, todos los
roles y batches mixtos, con igualdad exacta o una tolerancia prospectiva declarada.

### 4. Fusionar particiones del mismo T después de decodificarlas

Actualmente el rol histórico fuerza forwards separados aunque dos particiones tengan la
misma longitud T. Concatenarlas en el orden canónico después de decodificar podría
reducir hasta seis forwards a dos por update. Tiene un techo mayor que optimizar el
prefetch, pero puede cambiar orden de reducciones, asignación de dropout y kernels.

Sólo es admisible si un probe reproduce exactamente índices, RNG, pérdidas, gradientes,
modelo y optimizer para batches reales de cada combinación rol/T y tras una reanudación.
Reordenar el sampler para crear batches homogéneos no es equivalente: cambia la
composición del optimizer update y constituye un cambio científico.

### 5. Optimizar la indexación sólo con atribución de línea y backward equivalente

El siguiente probe CUDA debe añadir `record_function` alrededor de geometría,
foreground, relación DINO y transporte para atribuir las 240 llamadas. Después puede
compararse indexación booleana con reducciones enmascaradas o gathers precomputados.
Un kernel alternativo sólo puede admitirse con gradientes y estados del optimizer
equivalentes; el nombre del kernel actual no justifica sustituirlo a ciegas.

## Cambios que no deben aplicarse durante estos fits

- **Un único proceso o contexto CUDA para R_A5 y R_C2F.** Hoy cada proceso tiene RNG
  CPU/CUDA y checkpoints independientes. Intercalar ambos modelos compartiría el RNG
  global; restaurar un checkpoint rebobinaría el estado usado por el otro. Harían falta
  generadores explícitos en todo dropout o un checkpoint transaccional conjunto.
- **CUDA Graphs directos.** Hay firmas T2, T3 y batches mixtos con uno a seis forwards,
  I/O y ramas Python. Capturarlos requiere buffers estáticos por firma y una política de
  RNG/reanudación demostrada. Ningún graph está admitido.
- **Eliminar journals, guardas o checkpoints.** Reduciría recuperación o cobertura.
- **Rellenar T2 como T3 o cambiar el sampler.** Modifica activaciones, RNG o composición
  del update y es una intervención científica.
- **Aumentar prefetch, cache o threads sin A/B.** Los contadores muestran pocos
  milisegundos de espera directa y el host ya ejecuta dos pipelines concurrentes.

## Medición necesaria para decidir concurrencia

La ventana posterior de 300 s descendió desde 2,036 updates/s en el primer bloque a
aproximadamente 0,891 en el quinto; el último tramo de 120 s fue 0,913 updates/s. Esa
ventana no aísla el inicio del segundo productor ni el coste de profiling. Para decidir
entre ejecución simultánea y turnos gruesos se necesita un A/B de trabajo total:

- mismos checkpoints clonados y mismo número de updates por fit;
- brazo concurrente con dos procesos y brazo secuencial o por bloques, ambos sin profiler;
- throughput combinado, throughput por fit, GPU, energía, RAM y guard_seconds;
- hashes de curvas, checkpoints, RNG y optimizer después de cada brazo.

La métrica de decisión debe ser tiempo para completar el mismo número total de updates
de R_A5 y R_C2F. Comparar sólo la velocidad de uno mientras el otro usa la GPU produciría
una conclusión incorrecta.
