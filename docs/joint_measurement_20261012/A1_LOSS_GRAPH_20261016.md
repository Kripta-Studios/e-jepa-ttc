# Captura adicional de la pérdida A1

El perfil pasivo del runtime optimizado identifica operaciones de pérdida fuera
del grafo como siguiente candidato de optimización. La espera expuesta de datos
ya está alrededor de 1,3 ms por batch; no se justifica aumentar compresión para
resolver un coste que ahora está en otra etapa.

La candidata `captured_phase_loss.py` captura el forward y backward de la pérdida
en un grafo separado del modelo. Reutiliza `phase_transport_loss.phase_terms`,
sin cambiar operadores, ponderaciones, reducción, FP32, ocho microbatches ni la
suma final de diagnósticos en Python. Conserva la comprobación de dominio antes
del backward. Los diagnósticos se copian antes del siguiente replay para evitar
que su almacenamiento estático sobrescriba valores de microbatches anteriores.

El grafo es efímero y no añade parámetros ni claves al checkpoint. Se reconstruye
tras un reinicio del proceso y se conserva entre scopes dentro de un proceso.
La captura usa copias de entradas sin alias con el grafo del encoder y restaura
el RNG después del calentamiento. No se modifica la receta científica A1.

## Controles registrados antes de CUDA

Registro: `runtime/A1_loss_graph_v1/REGISTRATION.json`.

- 19 pruebas CPU de pérdida, gradientes, agregación, dominio y almacenamiento:
  aprobadas. Incluyen nueve controles previos del objetivo existente.
- 17 pruebas del lanzador y observador horario: aprobadas.
- Ruff y Pyright de los tres módulos nuevos: aprobados.
- Comparación nativa: siguiente batch P de 32 ejemplos desde checkpoint completo,
  cuatro ensayos alternando orden, cero updates; igualdad exacta de pérdidas,
  gradientes, RNG y pesos. Mediana de cocientes de tiempos candidata/referencia
  <=0,85, excluyendo sólo el primer ensayo de captura de ambos modelos.
- Sólo si pasa: ambos brazos A1, referencia de cuatro updates frente a candidata
  en dos scopes de dos y frente a dos más reanudación en proceso/sesión nueva.
  Máximo 24 updates técnicos; igualdad exacta de modelo, Adam, scheduler, sampler,
  RNG de Torch/CUDA/NumPy/Python y curvas, excluyendo tiempos.
- Una captura del modelo y una de la pérdida entre scopes. Antes de instalar:
  todos los resultados PASS y continuidad verificable desde el estado científico.

El wrapper `adaptation_loss_graph_runtime.py` reutiliza el coordinador y bloqueo
existentes. No crea un supervisor paralelo. `Resume-JointMeasurement.ps1` sólo
selecciona esta ruta si existe la admisión; el módulo vuelve a verificar pruebas
y hashes. En ausencia de admisión se conserva el runtime previamente validado.

## Frontera durable de la prueba

El proceso se detuvo en 6.400 updates completados y durables, cero pendientes y
cero recuperación. Se comprobaron identidad de proceso, journal, receipt,
pointer y SHA-256 del checkpoint antes de liberar el propietario. La copia
completa es `runtime/A1_loss_graph_v1/ORIGIN.pt`; `QUIESCED.json` conserva las
identidades y comprobaciones. La reserva del scope interrumpido se conserva
de forma conservadora, sin devolver presupuesto no demostrado.

Un primer intento de lectura del journal encontró un bloqueo transitorio de
Windows durante su sustitución atómica; el finally reanudó al coordinador y no
se detuvo el hijo. Se conserva `quiesce_attempt1.py`. El segundo intento tolera
esa carrera de lectura y sólo detiene procesos con estado durable consistente.

## Resultado CUDA e instalación

La prueba nativa es PASS. Los cuatro ensayos conservan pérdidas, gradientes de
todos los parámetros y RNG CPU/CUDA exactamente; los pesos no cambian. El
cociente mediano caliente candidata/referencia es **0,592932**, una reducción
de **40,7 %** en el forward/backward del batch preparado. Los tiempos calientes
de referencia son 0,251219 / 0,251104 / 0,271589 s y los de candidata son
0,148956 / 0,161354 / 0,145544 s. Esto no mide el coste completo de entrenamiento,
la lectura de datos ni la inferencia.

LAST-BLOCK y FULL pasan los controles exactos de estado, curvas, continuidad y
reanudación. Se ejecutaron 24 updates técnicos, ninguno científico. Cada sesión
persistente conserva una captura del modelo y una de la pérdida en sus dos
scopes. Se retienen los avisos de PyTorch sobre el stream de AccumulateGrad en
los logs; no se desactivaron avisos ni se relajó la igualdad exigida.

La admisión está en `runtime/A1_loss_graph_v1/ADMISSION.json`. A las 20:47 se
inició el coordinador con esta ruta y se reanudó LAST-BLOCK desde 6.400; FULL
utilizará la misma ruta al llegar su turno. El observador se vinculó al nuevo
PID y mantiene su siguiente revisión a las 21:00. Las cachés y grafos efímeros
se reconstruyen una vez al arrancar este proceso, no al cambiar de scope.

`ACTIVATION_VERIFIED.json` es la comprobación posterior de continuidad en el
primer checkpoint nuevo: prefijo de curva y tiempos, identidad, sampler,
avance de Adam y scheduler, sin recuperación adicional. Los snapshots
`PERSISTENT_RUNTIME.json` y `LOSS_GRAPH_RUNTIME.json` distinguen el ritmo de
producción de la mejora del benchmark preparado.

La verificación durable ya es PASS en el update 6.500: los siete controles de
continuidad pasan y se conserva una captura de cada grafo. El snapshot publicado
terminado en 6.700 mide 0,658958 s/update (5.463 updates/h) sobre los últimos cien
updates. La primera ventana tras arrancar midió 0,702611 s/update y 194,36 ms de
espera expuesta de datos mientras se readmitían los packs. En la ventana posterior
terminada en 6.600, la espera volvió a unos 1,34 ms y el intervalo total fue
0,630862 s/update. Son observaciones sucesivas de producción con cargas y cachés
distintas; no sustituyen un benchmark emparejado del entrenamiento completo.

Registro, admisión, prueba nativa, resultados de ambos brazos, frontera durable,
aviso numérico y snapshots están publicados con hashes en
`evidence_A1_loss_graph_20261016/MANIFEST.json`. El checkpoint completo permanece
local y su hash está incluido en la evidencia. La campaña sigue activa y la
evaluación H permanece cerrada hasta congelar ambos endpoints de A1.
