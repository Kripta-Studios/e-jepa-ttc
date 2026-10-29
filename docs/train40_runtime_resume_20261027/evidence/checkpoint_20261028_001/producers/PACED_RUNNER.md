# Runner de cadencia sostenida — 28/10/2026

Implementación nueva: `scripts/benchmark_train40_paced.py`.
Tests: `tests/unit/test_train40_paced_runner.py`.
Esta entrega solo ha ejecutado pruebas CPU. No contiene una nueva medición GPU
ni demuestra 50 Hz. No cambia `producers.py` o el servicio. La ampliación
posterior del scheduler en `paced.py` es opt-in y conserva el default anterior.

Validación inicial: **56 tests CPU pasan** (runner nuevo + contrato paced v1),
Ruff/formato correctos y Pyright sin errores. Recibo:
`pytest_paced_cpu.txt` (56 passed, 7,37 s). Incluye regresiones para resultados
marcados fallidos, drops finales, paridad/recibos incorrectos y gates 1/2/20.

Tras añadir la estrategia de espera: **68 tests CPU pasan** (10,05 s), Ruff y
Pyright sin errores. Los doce casos añadidos verifican deadlines absolutos,
remainder tras despertar tarde, cancelación, esperas de máximo 10 ms,
estrategia inválida, metadata y compatibilidad del default `event`.

## Qué mide

Una única secuencia física del manifiesto causal producido por
`generate_sensor_clock_replay` en `replay_inputs.py`. Cada intervalo conserva
su timestamp de sensor de 20 ms, incluidas las exclusiones explícitas. No se
comprimen huecos, repiten consultas ni concatenan secuencias para alcanzar 60 s.
`--updates` selecciona únicamente un prefijo declarado de la secuencia.

`run_paced_replay` ejecuta adquisición raw en su hilo productor independiente,
con deadlines fijos de 50 Hz y cola acotada. El hilo abre y cierra sus propios
HDF5 en cada adquisición; no comparte handles/cache con inferencia. Esta opción
incluye el coste de abrir archivos y leer disco, por lo que no equivale a una
medición de un sensor físico. Las cajas causales se verifican por SHA y reloj;
la proyección, los cuadrados ROI actuales y las regiones artificiales se
construyen dentro de `infer_all`, antes de `service.update`.

Cada payload contiene únicamente eventos CPU, frames RGB uint8 completos,
calibración y metadatos. Los arrays tienen storage propio y son read-only.
No se precomputan representaciones, crops, features ni TTC futuros. El reloj
principal termina después de todos los TTC host de todos los objetos. Los
recibos distinguen deadline→TTC, disponibilidad raw→TTC, retraso de adquisición,
trabajo pendiente, cobertura, descartes y fallos.

Los objetos son ROI distintas obtenidas trasladando la caja anotada actual.
Se declara explícitamente que esto no demuestra veinte tracks reales. El
servicio usa E reuse OFF/R ON, RGB chunk1 y precisión FP32/TF32 OFF.

## Historia y pérdidas

Antes de medir se cargan los modelos y se hace priming sintético de las formas
estándar (cinco tripletas RGB). También se inicializa una sola vez la historia
de 600 ms `[primer_anchor−620000, primer_anchor−20000)`. Ambos costes se
registran fuera del reloj. Si no existe ese soporte temporal se falla. Las
formas no primadas que aparezcan durante el replay conservan su coste dentro
del reloj.

Cada actualización incorpora solamente `[anchor−20000, anchor)`. Un hueco o
error de inferencia deja el consumidor en fallo permanente para ese ensayo:
los paquetes posteriores no producen TTC con historia incompleta ni provocan
refill invisible. Un descarte final también impide admisión aunque no exista
un siguiente paquete que revele el hueco. Los errores de fuente y exclusiones
siguen presentes en cobertura.

## Memoria

El tamaño raw por payload se limita antes de las principales asignaciones.
Los eventos o frames por encima del límite se rechazan completos, sin truncar.
La cota declarada de payloads en cola es `queue_capacity × max_payload_bytes`.
La cota conservadora de payloads vivos es `(queue_capacity + 3) × límite`: cola,
consumidor, último payload retenido por productor y siguiente en construcción.
Se publican bytes reales de cada actualización procesada.

No se presenta esta cota como RSS total: quedan aparte modelos, ring GPU,
metadatos/resultados Python, HDF5 y scratch de lectura. El reader original
impone un límite de 256 MiB a su lectura aproximada de timestamps. El reloj RGB
y las anotaciones también se acotan; no se afirma haber medido el pico de bytes
que ocupó realmente la cola.

## Gates y nueva prioridad de 1/2 objetos

`paced.py` conserva su criterio v1 de veinte objetos. El runner añade
`train40_paced_milestone_gate_v2` con `target_object_count=--objects`.
Se puede probar primero `--objects 1`, después `--objects 2` y finalmente 20.
Cada hito exige lo mismo:

- al menos 60 segundos ofertados a 50 Hz, en una secuencia continua;
- todos los deadlines→TTC estrictamente por debajo de 20 ms;
- ningún trabajo anterior pendiente a las llegadas programadas;
- cero descartes, fallos de fuente, huecos o pérdida de cobertura;
- todos los tres TTC finitos en host por objeto requerido;
- paridad posterior independiente completa y fuentes/originales intactos.

`milestone_with_parity_passed` corresponde al número de objetos declarado.
`capacity_with_parity_passed` solo puede ser verdadero para veinte objetos.
`physical_goal_demonstrated` y `target_20_objects_50hz_demonstrated` permanecen
false: el protocolo es replay de ROI artificiales, no evidencia de sensores y
tracks físicos. No se infiere frecuencia sostenida del inverso de una media.

## Paridad independiente

`--verify-parity` realiza, después de toda la fase cronometrada, una lectura
nueva y el predictor original por cada consulta programada y cada ROI. No usa
las predicciones originales preservadas como referencias de consultas nuevas.
Compara los tres TTC con el gate fijo `atol=rtol=1e-5`, cobertura y recibos de
historia/ROI/frame exactos. Un update perdido no desaparece del gate. Esta fase
puede ser costosa; sin ella ambos gates de admisión permanecen false.

La validación de entrada de este runner compara recibos físicos y predicciones;
no conserva todos los tensores de representación de un replay largo. La
validación tensor a tensor sigue siendo un gate independiente del servicio.

## Preservación y salida

Antes/después se comprueban SHA de candidato, normalizadores, referencias
originales completas, manifiestos, anotaciones y fuentes ejecutadas. Se guardan
copias exactas de las fuentes. RAW se protege mediante tamaño/mtime y digest de
calibración; no se afirma haber hasheado de nuevo archivos HDF5 completos.

La salida debe ser un directorio nuevo y externo a los directorios protegidos.
Se escriben `binding.json`, `original_snapshot.json`, `sources/`, `startup.json`,
`paced.json`, `service_updates.json`, `parity.json` opcional y `summary.json`.
Las excepciones conservan un resumen de fallo y ejecutan la verificación de
originales. No se escribe dentro del candidato ni de las predicciones originales.

## Comando para una futura reserva GPU

Primero debe existir un manifiesto causal independiente con una secuencia que
cubra el intervalo deseado. Los nombres entre `<...>` son parámetros por sustituir.

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' scripts/benchmark_train40_paced.py `
  --candidate 'D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE' `
  --reference artifacts/final_train40_transfer_20261027/final_train40/own `
  --base-manifest artifacts/final_train40_transfer_20261027/QUERY_MANIFEST.json `
  --manifest '<manifiesto-causal-nuevo.json>' --sequence '<physical_sequence_id>' `
  --output '<directorio-nuevo>' --objects 1 --updates 3000 `
  --backend prefix --event-query-graphs --event-normalization topk `
  --queue-capacity 2 --wait-strategy sleep --verify-parity
```

Una prueba corta sirve para detectar fallos, pero no puede pasar el gate de 60 s.
Para el segundo hito se repite con `--objects 2`, conservando las condiciones y
un directorio de evidencia nuevo. No se ha ejecutado este comando en GPU.

## Scheduler aditivo: Event.wait frente a time.sleep

`run_paced_replay(..., wait_strategy="event")` conserva la espera del contrato
anterior y sigue siendo el default de esa API. `wait_strategy="sleep"` usa
`time.sleep` en tramos de hasta 10 ms, recalculando el tiempo restante frente
al mismo deadline absoluto de `perf_counter_ns` después de cada despertar.
No usa busy-spin ni ajusta el próximo deadline por el retraso del anterior.
Comprueba cancelación entre esperas. El límite de 10 ms corresponde al tramo
solicitado; el planificador del sistema puede retrasar un despertar, por lo
que no se presenta como garantía de cancelación en tiempo real.

El nuevo runner defaults a `--wait-strategy sleep`; `--wait-strategy event`
permite controles A/B. Ambos registran implementación/version de Python,
plataforma, reloj, resolución declarada, estrategia y política de cancelación.
**Nunca se descuenta el jitter de deadline→TTC ni se relaja el gate de 20 ms.**

Python 3.11 documenta un timer de alta resolución en Windows para `time.sleep`,
y también advierte que una suspensión puede durar más por planificación del
sistema. La resolución publicada no equivale a precisión observada:
[documentación oficial de time.sleep](https://docs.python.org/3.11/library/time.html#time.sleep).

El consejo tiene autorizado un A/B CPU con `run_paced_replay` real, 3000 updates
a 50 Hz por estrategia, hilos/cola y payload/inferencia fake. Este protocolo
podrá cuantificar overhead de scheduler y handoff; no valida TTC ni sensores.
Las fuentes runtime quedan estables durante ese ensayo.
