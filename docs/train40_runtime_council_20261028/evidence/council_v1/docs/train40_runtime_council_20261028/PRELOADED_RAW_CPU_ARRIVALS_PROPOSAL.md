# Propuesta de llegadas RAW precargadas en CPU

Estado: protocolo prospectivo separado. Este documento no contiene resultados
de la fuente propuesta ni cambia la admisión vigente del control de disco.

La frontera solicitada comienza cuando está disponible la última observación
necesaria. Un feeder offline con RAW residente en CPU puede representar esa
entrada para medir preparación, transferencias y TTC completos. Su resultado
debe describirse como capacidad del servicio con llegadas CPU simuladas; no
acredita adquisición desde cámara, decodificación, almacenamiento ni transporte
del dispositivo hasta CPU.

Scope propuesto: `PACED_PRELOADED_RAW_CPU_ARRIVALS_NOT_PHYSICAL_SENSOR`.
El control `disk` o `persistent` mantiene su reloj con lecturas HDF5 incluidas.
Sus latencias y las de esta fuente se presentan separadas, con ambas fronteras
explícitas. El nuevo scope no corrige retrospectivamente un fallo del control.

## Precarga y memoria

Antes del primer deadline, un feeder privado puede leer del HDF5 el RAW
necesario para el intervalo declarado: eventos sin representación, frames
RGB uint8 completos y bytes de anotaciones. No prepara ROI, crops, resize,
normalización, features ni predicciones. No copia/sella tokens de frames
futuros ni calcula sus hashes de contenido antes de su llegada programada.

La precarga registra duración, bytes, intervalos de sensor, número de eventos
y frames, identidad de fuentes y conversión de formato efectuada por el lector.
Descompresión/decodificación del archivo pertenecen a esa simulación de fuente,
no a una promesa sobre el tiempo de adquisición física. Cualquier conversión
que se consideraría preprocesamiento del servicio debe ejecutarse después de
la llegada y quedar incluida en el reloj, o declararse fuera del scope.

El consumidor no recibe acceso al buffer completo. Solo el feeder puede
seleccionar observaciones del tramo actual; conserva los timestamps de sensor
originales. Su caché no puede compartirse como historial preparado con el
servicio ni usarse para primar modelos con datos reales futuros.

Antes de reservar los buffers, calcular y comprobar un presupuesto total de
memoria numérica que incluya RAW precargado, historia inicial, ring de tokens,
payloads pendientes/en uso y scratch de copia/sellado. Los buffers compartidos
pueden contarse conservadoramente más de una vez, pero no omitirse por ser
compartidos. Declarar por separado overhead Python, HDF5, modelos y memoria
GPU, junto con los límites o medidas disponibles. Registrar tamaño real y
máximo observado de cada contenedor controlado.

Si la población no cabe, fallar antes de medir. No truncar eventos, reducir
resolución, introducir lecturas lazy no declaradas ni ampliar silenciosamente
el presupuesto. Un piloto de una duración menor debe tener una nueva
configuración y mantener todas sus consultas en cobertura.

## Llegadas y trabajo cronometrado

El hilo productor mantiene deadlines host absolutos a 50 Hz, independientes
de la inferencia. Al llegar cada deadline debe:

1. seleccionar únicamente eventos del paquete causal y frames ya disponibles;
2. copiar/sellar/hashear los frames recién llegados, conservar sus bytes
   inmutables y construir los metadatos necesarios;
3. validar y parsear las anotaciones cuando lo permita su disponibilidad
   declarada, pudiendo reutilizar después una validación causal acreditada;
4. entregar un payload acotado cuya propiedad siga siendo válida hasta que
   el consumidor termine, aunque el ring expulse sus frames.

Los frames antiguos ya sellados pueden reutilizarse. El primer warmup global
solo puede sellar frames hasta `primer_anchor−20000 µs`; el frame nuevo de
la primera consulta permanece dentro del reloj. La historia event inicial
sigue el mismo contrato causal del runner existente.

Después del handoff, la construcción geométrica de ROI, representación,
transferencias, productores, contexto y disponibilidad de todos los TTC host
quedan dentro del mismo deadline→TTC. No reiniciar el reloj después de sellar
frames o después de esperar la cola. Conservar también el reloj secundario
de payload disponible→TTC, identificado como tal.

La primación sintética puede cubrir todas las formas declaradas por el ABI
de inferencia, sin mirar píxeles futuros ni calcular features de la población
real. Debe guardar formas, duración y configuración. Cualquier forma no
primada conserva su coste en el update que la introduzca.

## Gates y límites

Mantener cola/bytes acotados, pérdidas y errores explícitos, rechazo de huecos,
paridad TTC independiente y preservación de pesos/normalizadores. La admisión
tensor a tensor debe cubrir también la fuente nueva y los MISS de caché; una
referencia reconstruida que nunca se compara con los tensores consumidos no
demuestra ese contrato.

Reportar percentiles y máximo de deadline→todos los TTC, cobertura, descartes,
duración ofertada, lateness de adquisición y trabajo pendiente. No restar
jitter, sellado ni costes de handoff. No deducir frecuencia de una media.

La precarga no aporta observaciones nuevas: los 250 anchors disponibles siguen
representando 4,98 segundos entre consultas y cinco segundos de períodos.
Tampoco permite saltarse el requisito de 60 segundos del gate natural. Un
ensayo más largo con repetición requiere el protocolo separado
[STRESS_REPLAY_V1_PROPOSAL.md](STRESS_REPLAY_V1_PROPOSAL.md), con sus transiciones
contabilizadas.

Una eventual conclusión positiva quedaría limitada a este servicio, hardware,
población, perfil numérico y fuente CPU simulada. `physical_goal_demonstrated`
permanece falso. No equivale a certificación de sensores ni de seguridad.
