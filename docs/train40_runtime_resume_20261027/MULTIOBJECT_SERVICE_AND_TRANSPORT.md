# Servicio persistente TRAIN40 y transporte RGB

## Estado al cierre solicitado

Se preservaron los pesos, normalizadores y predicciones originales. El servicio
procesa regiones distintas con contexto independiente y conserva la tolerancia
numérica `atol=rtol=1e-5`. Hay pilotos completos admitidos para 1, 2 y 20 ROI,
pero **no se ha demostrado 50 Hz ni menos de 20 ms por actualización completa**.
La prioridad vigente es alcanzar primero ese objetivo con un objeto y después
con dos; veinte queda para una fase posterior.

El perfil conservador elegido usa productores originales FP32, TF32 desactivado,
event endpoint reuse desactivado, RGB endpoint reuse activado, una historia RGB
por llamada de productor, ramas E/R paralelas, historial temporal prefix con
query graphs y normalización top-k exacta. Conserva los recortes PIL sobre la
ROI, ocho workers RGB y preparación RGB/eventos paralela. La ruta uint8 y la
caché de features quedan como opciones experimentales explícitas; el perfil
por defecto no necesita ninguna de ellas. La raíz coordina el módulo común de
configuración y los defaults del servicio y runners; no se modifica el runner científico congelado.

## Qué mide el servicio

`MultiObjectTrain40Service.update` comienza con el nuevo paquete RAW de eventos
y los frames RAW necesarios disponibles en CPU. Aplica la ROI actual a toda
la historia, ingiere una vez el paquete compartido, prepara las entradas,
transfiere datos, ejecuta productores y contexto, y termina cuando están
disponibles en host los tres TTC de todos los objetos.

La carga desde disco, la construcción inicial de 600 ms de historial y la
primación sintética de shapes se registran fuera de ese reloj. Los pilotos
reconstruyen el historial antes de cada repetición. No son una ejecución
sostenida con llegadas independientes ni una medición física desde sensores.
La primera ROI es la original; las siguientes son traslaciones distintas sobre
el mismo RAW. No se presentan como veinte tracks anotados ni se duplican veinte
veces los mismos tensores de una ROI.

El servicio emite recibos por objeto de ROI, ventanas de eventos, frames,
timestamps, deltas y disponibilidad. El runner compara entradas realmente
consumidas con los adaptadores originales y cada TTC con el predictor original.
La ROI primaria se compara además con la predicción preservada. Se comprueban
IDs, cobertura, finitud, hashes y preservación antes/después. Un error de cierre
no omite el resumen ni la verificación de preservación.

## Comparación final PIL / uint8

Los cuatro procesos siguieron orden **PIL, uint8, uint8, PIL**. Cada uno midió
cinco repeticiones para cantidades 1/2 e índices 0/137/553: diez muestras por
modo/caso y 120 actualizaciones completas en total. Se conservaron todas.
Los cuatro runs terminaron `COMPLETE`, con paridad de los tres TTC, píxeles RGB
y deltas exactos, eventos dentro de tolerancia, recibos exactos y cobertura
completa. Cada run PIL verificó 604 archivos intactos; cada uint8, 605.

Las cifras por caso —media, mediana, p95, p99 y máximo— se generan desde los
JSON originales. Hay copias publicables de la [tabla generada](MULTIOBJECT_PAIRED_GENERATED.md)
y el [agregado JSON](MULTIOBJECT_PAIRED_GENERATED.json). Los archivos locales son:

```text
artifacts/train40_runtime_resume_20261027/multiobject/
  paired_1_2_uint8_v1_pil_a/
  paired_1_2_uint8_v1_uint8_b/
  paired_1_2_uint8_v1_uint8_c/
  paired_1_2_uint8_v1_pil_d/
  paired_1_2_uint8_v1_aggregate/summary.json
  paired_1_2_uint8_v1_aggregate/REPORT.md
```

La tabla generada no admite una mejora consistente de latencia con uint8.
PIL conserva medianas menores en los tres casos de un objeto. En dos objetos,
uint8 mejora ligeramente las medianas de dos casos, pero empeora el caso denso
y su cola. Se elige PIL como default conservador; la paridad de uint8 queda
admitida para este piloto, su supuesta mejora general de velocidad no.

Hubo deriva temporal entre procesos y presión de memoria comprometida del
host. Una muestra de telemetría fuera del reloj informó 7509/12227 MiB de
VRAM usada, unos 6 GB de RAM física libre y solamente unos 4.8 GB de memoria
comprometible libre. Es una observación puntual, no una causa demostrada. La
raíz identificó procesos MLflow preexistentes y ajenos; no se terminaron.
También se registró que hubo muestreo de telemetría durante el experimento.
Por ello no se atribuyen causalmente todas las diferencias a la optimización.
Esta limitación afecta a la interpretación de velocidad, no elimina los
resultados negativos ni invalida la comparación independiente de predicciones.

El agregado mantiene distribuciones de primeras repeticiones y repeticiones
posteriores, las primeras actualizaciones reales de cada cantidad de objetos,
y los recibos de primación sintética. Esas vistas no sustituyen la distribución
completa ni convierten una primera repetición en una medición de cold startup.

Regeneración, sin cargar modelos ni ejecutar GPU, hacia un directorio nuevo:

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' scripts/summarize_train40_multiobject_paired.py --runs artifacts/train40_runtime_resume_20261027/multiobject/paired_1_2_uint8_v1_pil_a artifacts/train40_runtime_resume_20261027/multiobject/paired_1_2_uint8_v1_uint8_b artifacts/train40_runtime_resume_20261027/multiobject/paired_1_2_uint8_v1_uint8_c artifacts/train40_runtime_resume_20261027/multiobject/paired_1_2_uint8_v1_pil_d --output artifacts/train40_runtime_resume_20261027/multiobject/paired_1_2_uint8_regenerated
```

El script registra hashes de los archivos de entrada, detecta cambios de
configuración/fuentes compartidas y falla si la evidencia no supera las
compuertas registradas. Los cuantiles usan interpolación lineal; diez muestras
por caso son insuficientes para estimar una cola de producción estable.

## Cambios implementados y comprobaciones

`batched_context.py` mantiene las observaciones de cada objeto separadas,
incluido el padding, los relojes originales y el fallback sin RGB. Sus tests
comparan cantidades 1/2/5/20 contra contextos originales independientes y
comprueban que los parámetros no cambian.

`multiobject.py` ingiere el RAW compartido una vez y agrupa cuatro observaciones
E por objeto. Puede dividir R por objeto para conservar el batch de endpoints
original. Los tickets de productores se mantienen vivos y sus guards se
validan después de juntar streams. La nueva API de entradas preparadas acepta
píxeles/deltas residentes o features previamente validadas, siempre con
metadata actual. La inyección de un productor externo valida el dispositivo
efectivo y los dispositivos de sus parámetros/buffers. Se rechaza combinar
esa inyección con una caché sin contrato explícito de identidad de backend.

La ruta uint8 ejecuta `pack` en CPU y `assemble` en el stream dueño. Transfiere
cada recorte redimensionado único una vez, usa una LUT FP32 calculada con la
división NumPy original y reconstruye el orden de frames repetidos. El lease
protege buffers pinned y GPU. Normalmente se libera después de los consumidores;
en modo diagnóstico permanece válido hasta el siguiente update o cierre para
permitir hashes después del reloj. Un fallo antes de assemble descarta el pack;
uno posterior sincroniza consumidores y libera el lease. Hay tests de ambos
casos y de recuperación, además de hashes de los deltas realmente consumidos.

Antes de los cambios de defaults coordinados por la raíz, **60 tests propios**
pasaron y Ruff/Pyright quedaron limpios. La raíz ejecuta el QA final sobre el
perfil publicado. Dos revisiones independientes detectaron y se corrigieron:
la admisión de una caché que nunca hacía HIT, y una validación incompleta de
dispositivos CUDA implícitos. No se relajaron las tolerancias.

## Resultados anteriores que deben conservarse

- El primer piloto paralelo de dos ROI tuvo un fallo grande en el primer
  replay. Se corrigió la captura de productores para usar y retener streams
  dedicados. El posterior piloto 1/2 × 0/137 × cinco repeticiones pasó completo.
  Esto no prueba que la causa única fuera el workspace de bibliotecas compartido.
- Agrupar R para veinte ROI falló en algunas predicciones pese a entradas
  correctas. La variante con R por objeto y query graphs pasó en el índice553;
  sus tres actualizaciones completas promediaron unos 384 ms. Las fuentes y
  cifras exactas están en `pilot_20_chunk1_querygraphs_v1`.
- La caché exacta de RGB pasó cinco actualizaciones de ese caso: un MISS y
  cuatro HIT. Su reporte separado se regenera en `cache_breakdown.json` bajo
  `pilot_20_exact_cache_v1`. El MISS costó unos 409 ms y los HIT, unos 189 ms de
  media. Hashear 48.38 MB de RAW en cada consulta costó unos 22–23 ms, incluido
  en el reloj. No es una ruta capaz de cumplir menos de 20 ms tal como está.

Los detalles, fallos, comandos y límites anteriores permanecen en
`artifacts/train40_runtime_resume_20261027/multiobject/IMPLEMENTATION.md`.

## Fuentes ejecutadas y handoff pendiente

`source_snapshot_paired_1_2_v1` guarda las 28 fuentes de la unión de dependencias
de los cuatro runs, comprobadas contra sus bindings. Todos los SHA compartidos
coincidieron. La caché anterior tiene 33 fuentes archivadas y una fuente
experimental no importada listada como no disponible: cambió después de la
verificación final satisfactoria. No se sustituyó por una versión diferente.
El runner actual liga la dependencia importada real y falla si aparece un
módulo runtime nuevo después de crear el binding.

El adapter de identidad rápida quedó **en diseño, sin integrarse ni medirse**
en esta línea antes del cierre. La propuesta recibe únicamente llegadas RAW
nuevas, copia/sella sus bytes dentro del reloj y conserva tokens verdaderamente
inmutables para las historias posteriores. Debe rechazar IDs reciclados,
rollback y fuentes mezcladas. Los arrays usados para recortar deben derivar
de los mismos bytes que acredita el token, no de otro array con flag readonly.

Se prevé un ring CPU de 16 frames: las historias nativas alternan dos cadenas
de frames y diez slots no bastan para conservar ambas. Un prefill separado
solo puede incluir timestamps anteriores o iguales al inicio del paquete
medido; no puede esconder el sellado de la última observación. El primer MISS
puede medir todo el sellado inicial y los HIT posteriores deben usar un query
posterior con contexto recalculado. Los helpers nuevos de la línea RGB tienen
su propio estado/QA, pero no equivalen a este adapter ni a una prueba completa.

La siguiente sesión debe comenzar leyendo el perfil publicado, los resultados
admitidos y negativos, y el estado real de esos helpers. Después de controlar
las condiciones de medición, podrá evaluar identidad inmutable y ejecución
sostenida para 1/2 objetos. No debe atribuir un objetivo de 50 Hz al throughput
inverso de un promedio offline, ni borrar los fallos anteriores.
