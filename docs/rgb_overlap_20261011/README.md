# V13: solapamiento de preparación RGB y reutilización de geometría constante

Intervención del 11 de octubre de 2026, posterior a la
[primera optimización del cargador RGB](../rgb_loader_20261011/README.md).
La nueva capa está en `operational/rgb_port_rgb_overlap` y se aplica a `R_A5`
y `R_C2F`. Los resultados disponibles y su estado se generan en
[TABLES.md](TABLES.md) desde [EVIDENCE.json](EVIDENCE.json).

**El entrenamiento se reanudó desde el checkpoint exacto 2.124** con la
implementación `2b38c31`: cuatro lectores, dos batches anticipados y matching
geométrico original. La observación posterior y la ETA acotada a `R_A5` están en
[LIVE.md](LIVE.md), regenerable desde [LIVE_RESULT.json](LIVE_RESULT.json).

## Problema y alcance

La primera intervención redujo trabajo de preparación de cada batch: retención
DINO, apilado de los campos realmente consumidos y consultas selectivas de RSS.
El bucle RGB todavía preparaba las entradas de forma síncrona antes de ejecutar
el modelo. Por eso una mejora de CPU aislada no garantizaba un ritmo estable del
entrenamiento completo. Esta segunda intervención intenta preparar los próximos
batches mientras la GPU trabaja y evitar reconstruir máscaras geométricas
constantes en cada comparación temporal.

Los productores de eventos habían terminado. Para estas comprobaciones se pausó
`R_A5` en el update **2.124**, con **2.124 updates durables**, y se detuvo el
supervisor anterior para evitar una reanudación durante las pruebas. Esto describe
el origen de la intervención, no un monitor de estado actual. La receta científica,
los pesos y los estados del optimizador se conservan. `R_C2F` puede arrancar desde
su checkpoint inicial mediante la misma capa; no se supone que ya haya entrenado.

## Implementación

### Lectura paralela de shards RGB

[decoder.py](../../operational/rgb_port_rgb_overlap/decoder.py) prepara los shards
fríos necesarios para el batch con varios hilos, cada uno con un lector privado.
Se compararon dos y cuatro lectores; la configuración admitida tras el benchmark
CPU usa **cuatro lectores**, con admisión separada de la optimización geométrica.
Cada lector ejecuta `PreparedRGBCache.get` original: SHA256, decodificación NPZ/CRC,
tokens y conversión de la fila siguen pasando por la ruta canónica. Sólo el
coordinador incorpora los arrays validados a la LRU compartida, respetando el orden
de consumo. No se elimina la validación de integridad ni se cambia el sampler.

Se limita el trabajo pendiente por número de lectores y por una **reserva estimada
de 128 MiB**, calculada con los tamaños descomprimidos del ZIP. Los shards que no
caben siguen la ruta síncrona. Esta reserva no equivale al pico RSS real: también
existen la caché RGB histórica, los batches preparados, los buffers de lectura y
la memoria del proceso. Las mediciones publicadas distinguen esos conceptos.

### Dos batches efectivos anticipados

[prefetch.py](../../operational/rgb_port_rgb_overlap/prefetch.py) captura la instancia
original de `EffectiveSource`, después de su construcción. Un único coordinador
CPU prepara hasta **dos batches efectivos pendientes**. Los índices vienen del
orden canónico y el consumidor comprueba orden, posición y solicitud antes de
aceptar cada resultado. El coordinador no modifica el cursor ni genera otro
muestreo; tampoco anticipa el orden de la siguiente época.

Cada batch conserva sus partes reales T2/T3 y las reducciones de pérdida de
`EffectiveSource`. No hay frames de relleno, cambio de población ni nueva frontera
del optimizador. Las salidas tienen almacenamiento propio que no se sobrescribe
cuando avanza la preparación. Las excepciones de un futuro se entregan cuando
el consumidor solicita ese batch; los errores no provocan sustitución o salto
silencioso de muestras.

### Máscaras geométricas constantes

**Experimento conservado, desactivado en la configuración de despliegue.**
La paridad GPU fue exacta, pero las dos comparaciones disponibles no demostraron
beneficio de tiempo; `COMPUTE_ENABLED=False` mantiene el matching original.

[compute.py](../../operational/rgb_port_rgb_overlap/compute.py) conserva la máscara
booleana de candidatos válidos del matching local, indexada por batch, resolución,
radio y dispositivo. Su caché admite **64 MiB y 32 entradas**. Los desplazamientos,
correlaciones y operaciones diferenciables conservan su orden y sus tipos; no se
cuantizan activaciones ni se cambia BF16. Se actualizan temporalmente tanto el
símbolo del módulo original como el alias directo importado por `CausalScaleTTC`.

Las máscaras no tienen gradiente ni dependen de las imágenes. La caché no reutiliza
salidas del modelo, gradientes o activaciones aprendidas. Aun así, su admisión exige
comparar salidas, pérdida, componentes y gradientes GPU, incluidos los gradientes
antes y después del clipping; las pruebas CPU por sí solas no bastan.

El primer ABBA completo coincidió con trabajo CPU de la intervención. Se repitió
sin ese benchmark ni entrenamiento concurrente; la variante cacheada tampoco
mejoró la mediana. La variación entre A1 y A2 impide atribuir causalmente todo el
delta a la caché. Se conservan ambos resultados negativos y no se activa el cambio.
Una ejecución GPU anterior incompleta también se archivó: falló el capturador de
instantáneas por el stride de un tensor singleton, se corrigió la instrumentación
y se añadieron regresiones antes de repetir. No cambió la aritmética del modelo.

## Continuidad y recuperación

[producer.py](../../operational/rgb_port_rgb_overlap/producer.py) se inserta dentro
de los wrappers anteriores, conservando la instancia original de la fuente. La
cadena retiene el cargador RGB previo, continuidad, recuperación de I/O y
aceleración ya admitidas. El nuevo fit captura el `EffectiveSource` que construye
el bucle congelado, y sólo inicia la anticipación después de completar la
restauración original.

En un fit nuevo como `R_C2F`, primero se guarda el checkpoint inicial `READY`;
entonces se vincula su cursor original. Las guardas aceptan el cursor posicional
o por keyword. Los saves siguientes no lo vuelven a vincular. Al salir, se espera
al coordinador antes de cerrar los decodificadores y antes de restaurar los métodos
de collate y DINO del padre. Esto también se comprueba cuando hay una excepción.

La recuperación USB conserva sus reintentos acotados y la pausa con estado completo.
Los errores de corrupción siguen siendo errores. La lectura especulativa puede
haber realizado trabajo que no se use antes de pausar; sus contadores lo indican,
pero no representa updates de entrenamiento ni avance del sampler.

[contracts.py](../../operational/rgb_port_rgb_overlap/contracts.py) añade
`RGB_OVERLAP_FREEZE.json`, ligado al freeze RGB anterior, los hashes de código y
tests y la evidencia QA. No reescribe los contratos científicos anteriores. Exige
paridad exacta de batch, paridad GPU de gradientes y conservación de cursor/RNG.
[queue.py](../../operational/rgb_port_rgb_overlap/queue.py) conserva el supervisor
durable y enruta RGB por la nueva capa; incluye todos los tests de ésta en el bundle.

## Comparaciones y límites de la evidencia

El [benchmark CPU](../../operational/rgb_port_rgb_overlap/benchmark.py) utiliza
los siguientes índices del checkpoint protegido. Compara preparación serial,
dos y cuatro decodificadores, una repetición de la referencia y dos decodificadores
con anticipación. Se añadió después una prueba de paridad con cuatro lectores y
dos batches anticipados. Todos los casos
se contrastan con hashes de todos los campos de los batches y con los estados RNG.
El experimento CPU reutiliza el origen conservado del update **1.500**; la prueba
GPU usa el checkpoint posterior **2.124**. Cada comparación mantiene su propio
origen fijo y sus hashes. Sus tiempos no deben sumarse como si fueran un mismo
batch medido de extremo a extremo.

Hay una ejecución por variante de decodificador, cachés lógicas inicialmente
vacías y caché de archivos del sistema sin vaciar. El caso con anticipación puede
solapar trabajo con el cálculo de hashes del benchmark, aunque éste no forme parte
del tiempo de espera contado. Sus tiempos no son una medida de aceleración del
entrenamiento completo. Las cargas concurrentes del host también pueden variar.

El [benchmark GPU](../../operational/rgb_port_rgb_overlap/compute_benchmark.py)
usa entradas reales ya en GPU y orden ABBA, tras calentamiento. Restaura los pesos
y RNG entre variantes. Mide forward, pérdida, backward y clipping, excluyendo
preparación de entradas, paso del optimizador y copia de las instantáneas de QA.
Incluye batches homogéneos y mixtos T2/T3. **Las comprobaciones de esta intervención
consumen cero updates del optimizador de la campaña.**

Las tablas indican explícitamente si cada evidencia sigue pendiente, pasa o falla.
No se infiere paridad GPU a partir de tests CPU, ni una ETA de campaña a partir de
tiempos de componentes. Se necesita observar entrenamiento real tras la admisión,
incluyendo checkpoints, presión de memoria y posibles desconexiones. Este trabajo
no proporciona nuevas métricas TTC ni demuestra superioridad SOTA.

## Reproducir la documentación

El regenerador usa únicamente la biblioteca estándar de Python. Desde la raíz:

```powershell
python docs/rgb_overlap_20261011/regenerate.py --evidence-root E:/EJEPA_results/rgb_overlap_20261011
python docs/rgb_overlap_20261011/regenerate.py --check
python docs/rgb_overlap_20261011/publish_live.py --evidence-root E:/EJEPA_results/rgb_overlap_20261011 --run artifacts/rgb_port_20261008
python docs/rgb_overlap_20261011/publish_live.py --check
```

La primera orden exporta un resumen mediante lista explícita de campos permitidos
y genera las tablas. La segunda regenera y verifica las tablas usando sólo el JSON
público. Los hashes identifican los archivos locales originales, pero el resumen
no publica rutas privadas, IDs de muestras ni tokens individuales. Los resultados
completos, checkpoints y datos permanecen en el almacenamiento local E:.

Las regresiones están en `tests/test_rgb_port_rgb_overlap_{compute,decoder,prefetch,runtime}.py`.
El recuento de pruebas se obtiene del JUnit, sin sumar dos veces suites solapadas.
