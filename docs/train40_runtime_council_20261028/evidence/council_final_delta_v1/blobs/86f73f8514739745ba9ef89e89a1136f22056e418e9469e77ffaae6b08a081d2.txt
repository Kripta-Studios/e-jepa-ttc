# Backend de ingestión de 5 ms con consultas de 20 ms

El backend optativo `SubpacketTemporalPrefixEventHistory` permite repartir la
ingestión de eventos entre cuatro llegadas de 5 ms, conservando las consultas
TRAIN40 de 20 ms y el algoritmo de prefijos de 600 ms. Está implementado y
validado numéricamente; todavía no existe una integración de este backend en un
servicio con llegadas de 200 Hz y consultas de 50 Hz. Su validación no demuestra
una mejora de latencia ni el cumplimiento del objetivo sostenido.

Fuentes de la implementación:

- [Backend nuevo](../../operational/train40_runtime/temporal_prefix_subpacket.py).
- [Pruebas CPU](../../tests/unit/test_train40_runtime_temporal_prefix_subpacket.py).
- [Backend padre conservado](../../operational/train40_runtime/temporal_prefix.py).
- [Contrato heredado de eventos](../../operational/train40_runtime/integral.py).
- [Diagnóstico GPU y sus límites](SUBPACKET_NUMERICS.md), con
  [recibo v2](../../artifacts/train40_runtime_council_20261028/event/subpacket_backend_sensitive_v2/summary.json).

## Alcance y compatibilidad

La clase nueva hereda el constructor, los buffers, los prefijos, la normalización
y los gráficos de consulta del padre. No cambia pesos, normalizadores, modelos,
batch del encoder E, representación H4, ROIs ni tolerancias. No añade reservas
CUDA persistentes propias. Esto no convierte las operaciones heredadas en una
garantía de ausencia de temporales o de memoria reservada por el allocator.

El backend original conserva su API y sus bytes. Para una llamada de 20 ms a la
clase nueva, `ingest` delega el trabajo al método original del padre. Esta ruta
también sirve para `initialize`, que llena los 600 ms iniciales con 30 paquetes
originales. Sólo la clase nueva admite llamadas de 5 ms; no se cambió el backend
por defecto del servicio o del runner paced.

La diferencia deliberada de gestión de errores es conservadora: cualquier fallo
de ingestión en la clase nueva invalida la historia hasta un `reset` explícito.

## Contrato de reloj y datos

Todos los intervalos son semiabiertos `[start_us, end_us)`, sin redondear el reloj.
El inicio de cada llamada debe coincidir exactamente con `history.end_us`.
`reset(start_us)` establece la fase a partir de ese entero, aunque no sea múltiplo
de 5.000 en el reloj absoluto.

| Operación | Intervalo admitido | Condición adicional |
|---|---|---|
| `initialize(raw, start, end)` | 600.000 µs | Reinicia y usa 30 llamadas originales de 20 ms. |
| `ingest(raw, start, end)` | 20.000 µs | Sólo al inicio de un grupo de cuatro buckets. |
| `ingest(raw, start, end)` | 5.000 µs | Despacha a `ingest_subpacket`. |
| `ingest_subpacket(raw, start, end)` | 5.000 µs | Contiguo, válido y dentro del presupuesto del grupo. |
| `query(roi)` / `query_many(rois)` | Sin avance temporal | Historia completa, fase 0 y sin fallo enclavado. |

Se conserva la validación original de las cuatro columnas `x`, `y`, `t`, `p`:
arrays unidimensionales alineados, dtypes enteros, coordenadas dentro del sensor y
timestamps monótonos no decrecientes dentro del intervalo. Los timestamps
repetidos son válidos. Un evento exactamente en `end_us` pertenece al siguiente
intervalo. La normalización del signo sigue siendo `p > 0` positivo y el resto
negativo, sin truncar primero polaridades grandes a int8.

Un subpaquete vacío sigue avanzando el reloj y el bucket. No se omite, no reduce
la duración de la ventana y no evita una transición de época.

La fase es `_total_buckets % 4`. `ready` exige simultáneamente 120 buckets de
historia, fase 0 y ausencia de fallo. Por tanto, puede pasar de verdadero a falso
al ingerir el primero de los tres buckets previos a una consulta, aunque ya haya
más de 600 ms almacenados. Consultar en fases 1, 2 o 3 es un error explícito.
No se permite completar un grupo parcial insertando un paquete de 20 ms.

## Aritmética y épocas

El camino de 5 ms ejecuta las mismas operaciones de una iteración del bucle de
ingestión original: copia de columnas, índices planos, histogramas enteros,
momentos temporales relativos, sumas acumuladas espaciales y actualización del
prefijo. El almacenamiento continúa usando counts int32 y momentos int64.

La época sigue reiniciándose exactamente cada 120 buckets, es decir, cada 600 ms
desde el origen establecido por `reset`. Se conserva el total de la época previa
y el anillo de 121 fronteras. No hay rebasing adicional, reconstrucción desde
representaciones flotantes ni cambio en las fronteras de las consultas. Los
gráficos reciben la fase temporal dinámica del padre.

La captura sintética de gráficos mantiene su contrato: llamar a
`prime_query_graphs([1])`, con gráficos habilitados en CUDA, antes de cualquier
`reset`, `initialize` o ingestión de sensores. No se permite convertir datos
reales de una consulta en primación excluida de la medición.

## Capacidad, fallo y recuperación

`max_packet_events` conserva su significado de capacidad **por 20 ms**, con
default de 2.000.000 de eventos. La suma de los cuatro subpaquetes se comprueba
antes de copiar el subpaquete siguiente. No son 2.000.000 de eventos por cada
llamada de 5 ms. Esto mantiene la cota de capacidad usada por el constructor para
justificar los prefijos enteros de una época completa.

Si los primeros tres subpaquetes ya se ingirieron y el cuarto excede la capacidad,
se rechaza el cuarto antes de su copia. Los primeros tres no se deshacen: la
historia queda inválida y no puede producir TTC desde ese grupo incompleto.
No se truncan eventos, no se emite una predicción con estado anterior y no se
acepta continuar desde el último timestamp aparentemente válido.

Ambos caminos de ingestión capturan `BaseException` únicamente para establecer
el fallo y relanzar la misma excepción. Incluye `KeyboardInterrupt` después de
una copia o de un prefijo parcialmente comprometido; no se transforma la
interrupción en éxito ni se oculta al llamador. Mientras el fallo esté enclavado,
`query`, `query_many` e ingestiones posteriores fallan con
`SUBPACKET_HISTORY_INVALID_RESET_REQUIRED`.

`reset` restaura el origen de reloj/época, invalida el contenido anterior y limpia
el fallo y los contadores nuevos. Como en el padre, no necesita borrar cada plano
del anillo: no se permite consultar hasta reconstruir una historia completa.
Tampoco cancela trabajo CUDA ni repara un contexto CUDA averiado. El propietario
debe resolver o drenar el trabajo pendiente antes de reutilizar buffers; un fallo
CUDA no recuperable exige descartar la instancia.

## Propiedad de buffers y contadores

Todas las llamadas pertenecen a un único propietario y a su stream CUDA. La
clase no incorpora locks ni crea un consumidor concurrente. No se debe ingerir
mientras una consulta o el servicio todavía consume sus resultados prestados.
`query_many` conserva el contrato de tensor prestado que puede sobrescribirse en
la siguiente consulta. `ready` expresa validez lógica; no es una barrera CUDA.

Los contadores distinguen paquetes completos y llamadas parciales:

| Campo | Significado |
|---|---|
| `ingested_packets` | Total heredado durante la vida de la instancia; cada grupo completo de cuatro subpaquetes suma uno. |
| `queries` | Contador heredado, con la semántica de objetos consultados del padre. |
| `total_buckets` / `epoch_start_us` | Estado temporal reiniciado por `reset`. |
| `subpacket_calls_since_reset` | Llamadas de 5 ms completadas correctamente. |
| `subpacket_completed_updates_since_reset` | Grupos de cuatro llamadas de 5 ms completados. |
| `subpacket_events_since_reset` | Eventos procesados por el camino de 5 ms; no incluye ingestiones bulk. |
| `subpacket_pending_events` | Presupuesto ya consumido en el grupo parcial actual. |
| `bulk_20ms_calls_since_reset` | Llamadas de 20 ms completadas; incluye las 30 de `initialize`. |
| `last_completed_subpacket_update_events` | Eventos del último grupo terminado mediante subpaquetes. |
| `subpacket_phase` / `subpacket_invalid` | Fase 0–3 y fallo enclavado actual. |

Un contador de llamadas completadas no acredita por sí solo finalización de los
kernels CUDA. El futuro medidor deberá distinguir enqueue, finalización GPU y
disponibilidad del TTC en host.

## Validación CPU realizada

La implementación actual pasó **15 casos CPU**, con OMP/MKL limitados a dos
hilos, además de Ruff y Pyright. La prueba parametrizada de épocas compara los
seis buffers persistentes de estado entero y los metadatos después de cada grupo
completo, y compara tensores y hashes de píxeles en los buckets 120, 124 y 240.
Cubre normalización `sort` y `topk`, un origen fraccionario respecto a la cadencia
absoluta y un origen grande próximo a `2**52`.

La suite también comprueba:

- Prefill de 600 + 15 ms con el mismo origen, rechazo de consultas parciales y
  conservación del tensor prestado mientras sólo se ingiere.
- Timestamps repetidos, eventos en fronteras semiabiertas, polaridades grandes y
  subpaquetes vacíos.
- Capacidad acumulada exacta, overflow del cuarto subpaquete antes de mutarlo y
  recuperación mediante reset e historia vacía.
- Gap, rollback, duración incorrecta, timestamp en el extremo excluido y orden
  temporal decreciente.
- Mezcla permitida de modos en una frontera completa y rechazo de un paquete de
  20 ms que interrumpe una fase parcial.
- `KeyboardInterrupt` tras copia/scratch mutado en 5 ms y tras comprometer el
  primer bucket de una llamada de 20 ms. Ambas pruebas verifican propagación de
  la misma excepción, bloqueo posterior y recuperación explícita.

Comandos de reproducción, desde un entorno del proyecto ya instalado:

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
python -m pytest tests/unit/test_train40_runtime_temporal_prefix_subpacket.py -q
ruff check operational/train40_runtime/temporal_prefix_subpacket.py tests/unit/test_train40_runtime_temporal_prefix_subpacket.py
ruff format --check operational/train40_runtime/temporal_prefix_subpacket.py tests/unit/test_train40_runtime_temporal_prefix_subpacket.py
pyright operational/train40_runtime/temporal_prefix_subpacket.py tests/unit/test_train40_runtime_temporal_prefix_subpacket.py
```

Pyright debe apuntar al mismo entorno completo que Python. En esta máquina se
usó el entorno hermano `../e-jepa-ttc/.venv` y `--venvpath ../e-jepa-ttc`; no se
silenciaron errores de tipos para compensar el `.venv` local incompleto.

La evidencia GPU queda separada en [SUBPACKET_NUMERICS.md](SUBPACKET_NUMERICS.md).
Ese diagnóstico usa tres queries reales y compara estados/representaciones a lo
largo de dos épocas, pero evalúa los TTC sólo en el anchor final de cada caso.
Su origen `anchor−1.200ms` no debe confundirse con el prefill del futuro servicio
descrito a continuación. No es un ensayo de llegadas independientes de 200 Hz.

## Integración futura: prefill causal y reloj completo

Para conservar el origen del servicio paced actual, siendo `t0` la primera
consulta, la secuencia propuesta es:

1. Mantener `reset(t0−620ms)` mediante el `initialize` original de 600 ms,
   terminando en `t0−20ms`: bucket 120.
2. Ingerir como startup declarado los intervalos hasta `t0−15ms`, `t0−10ms` y
   `t0−5ms`: buckets 121–123. Sólo datos disponibles hasta esos instantes.
3. Incluir el último intervalo `[t0−5ms,t0)` dentro del reloj medido. Después del
   bucket 124 se permite la primera consulta. No sustituir este orden por un
   `initialize` de 600 ms terminado en `t0−5ms`, que desplazaría el origen.
4. En cada periodo posterior, consumir las primeras tres llegadas con el único
   propietario GPU y entregar la cuarta al camino completo de actualización.
   No ingerir la cuarta dos veces: una integración podría pasarla directamente
   a `service.update`, que ya llama a `history.ingest`.

Un productor CPU independiente deberá publicar intervalos de eventos completos
cada 5 ms, sin depender de cuándo termina una inferencia. Las copias, conversiones
y el procesamiento de un intervalo futuro no pueden adelantarse usando el
archivo RAW privado precargado. Cada payload debe conservar sus bytes hasta que
el consumidor termine; los límites de cola y bytes serán explícitos.

RGB debe tener su calendario real de timestamps. Se puede copiar y sellar un
frame después de su llegada y antes del último evento requerido por una consulta,
si esos tiempos efectivamente lo permiten. Una ROI o metadata nueva puede llegar
más tarde: eso impide adelantar sus crops. Los tokens inmutables permiten retener
frames de consultas previas durante la rotación del ring, pero no autorizan a
leer píxeles futuros ni a reutilizar un crop con otra ROI.

El reloj requerido es desde la última observación RAW necesaria hasta todos los
TTC disponibles en host. Se deberán distinguir al menos:

- Deadline nominal absoluto de cada llegada, calculado una sola vez.
- `raw_released_ns`, antes de copia/sellado/preprocesamiento del RAW ya llegado.
- `immutable_ready_ns` o disponibilidad del payload preparado, sin reiniciar el
  reloj de latencia en ese punto.
- Inicio y finalización observada del trabajo del consumidor, con semántica
  explícita para operaciones CUDA asíncronas.
- Último TTC finito de todos los objetos disponible en host.

La latencia principal será `ttc_complete_ns − max(raw_released_ns requeridos)`.
Debe acompañarse de deadline nominal → TTC, para que una fuente o un productor
tardío no aparenten una respuesta rápida al desplazar la frontera de entrada.
Fallos y acknowledgements sin TTC no pertenecen a la población de latencia de
predicciones válidas.

## Deuda de trabajo y alcance del ahorro posible

El propietario GPU serial no puede procesar una llegada mientras una llamada de
servicio anterior lo mantiene ocupado. Aunque no haya consultas acumuladas,
puede haber uno o varios subpaquetes vencidos en cola. El futuro recibo debe
publicar ambas deudas, además de bytes pendientes, descartes, llegadas omitidas,
retraso de adquisición y trabajo GPU aún no confirmado como completo.

Se deben registrar las dos cadencias: todos los deadlines de subpaquetes a 200 Hz
y todas las consultas a 50 Hz. Si se excluyen los tres primeros subpaquetes como
prefill, `N` consultas desde `t0` hasta `t0+(N−1)20ms` tienen `4N−3` llegadas de
subpaquetes medidas. El recibo debe declarar ese borde y los tres subpaquetes de
startup; no inventar cuatro llegadas medidas para la primera consulta.

Adelantar tres buckets puede acortar la ruta posterior a la última observación
si existe tiempo disponible antes de esa frontera. No elimina su trabajo ni
demuestra por sí mismo mayor capacidad sostenida. En esta implementación, cuatro
llamadas de 5 ms mantienen las operaciones por bucket y aumentan las copias H2D
de cuatro a dieciséis por periodo de 20 ms; las cinco operaciones de índices
planos pasan de un lanzamiento por operación a cuatro. También hay más llamadas
de validación y despacho. No se admite una predicción de ahorro del 75 %.

Los cuatro costes de ingestión, la preparación RGB adelantada y todos los modelos
deben quedar dentro de la contabilidad global de trabajo y throughput. No basta
con cronometrar el cuarto subpaquete mientras se omite una cola creciente.

## Próximos pasos al reanudar

1. Leer el handoff y [SUBPACKET_NUMERICS.md](SUBPACKET_NUMERICS.md), conservar el
   diagnóstico fallido anterior y comprobar las fuentes/artefactos vinculados.
2. Diseñar primero la fuente y el medidor con reloj simulado: causalidad de cada
   payload, igualdad al concatenar cuatro intervalos frente al paquete original,
   RGB y metadata realmente disponibles, deadlines fijos, límites de memoria,
   retrasos, overflow, pérdida de paquetes y cierre completo.
3. Integrar el backend mediante una opción explícita o inyección de historia,
   conservando el camino de 20 ms como control. Mantener un único propietario
   GPU y la primación sintética declarada; no abrir una ingestión concurrente
   sobre buffers prestados.
4. Ejecutar el contraste real de un objeto con cobertura completa, paridad contra
   el oracle original, percentiles de latencia y ambas deudas de trabajo. La
   primera prueba corta valida el protocolo; no sustituye una corrida sostenida.
5. Avanzar a dos objetos sólo después de cerrar el criterio de un objeto. La
   evidencia actual del diagnóstico GPU no demuestra dos ni veinte objetos.

Esta sesión se cierra con el backend y la evidencia numérica. El runner de
200 Hz/50 Hz, su integración con llegadas RGB y la demostración de latencia
sostenida quedan pendientes para una sesión posterior.
