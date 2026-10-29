# Admisión de evaluación TTC para detección → tracking → PHASE

Fecha: 2026-10-29. Alcance: auditoría CPU de metadatos públicos TRAIN. No se
leyeron etiquetas privadas, eventos completos ni píxeles; no hubo inferencia,
entrenamiento o selección de checkpoint.

## Decisión

Existe una ruta reproducible para un **diagnóstico TTC de desarrollo expuesto a
TRAIN40**, pero no para confirmar generalización independiente.

Los ocho frames del replay temporal contienen ocho referencias de endpoints
GarlTTC, correspondientes a cinco consultas y un track. Cuatro frames son el
endpoint final que porta el target TTC público y los cuatro targets son finitos.
Sin embargo, el frame final en el que el replay actual emitió TTC es solamente
el primer endpoint de otra consulta: tiene **cero targets TTC exactamente
alineados**. Por ello no se puede puntuar directamente el par de TTC ya emitido.

La unión correcta es:

1. Garl input ↔ Garl annotation por
   `(sequence_id, sample_token, track_id, public_track_id, timestamp_us)`.
2. Garl endpoint ↔ eAP frame por `(sequence_id, rgb_member_path)`.

No se deben igualar los relojes. En la secuencia inspeccionada, el timestamp de
anotación Garl y el timestamp sensorial eAP tienen un offset grande. La identidad
del frame procede del miembro RGB exacto; ambos relojes y el offset se conservan
en el recibo.

## Identidad: bloqueo concreto

GarlTTC expone `public_track_id`, mientras que eAP expone `instance_id` y
`track_id`. En los 39 labels eAP públicos de estos ocho frames hubo **cero
igualdades directas** con el track Garl solapado. El tracker causal añade además
IDs propios `track_N:generation`.

Por tanto, falta un campo publicado que proporcione esta correspondencia:

```text
generated track_N:generation -> GarlTTC public_track_id
```

Esto no bloquea un matching geométrico exclusivamente del lado del evaluador,
pero impide tratar el ID generado como GT. El matching de evaluación nunca debe
entrar en inferencia ni poblar caches PHASE.

## Exposición del productor

El `MODEL_SEAL.json` identifica el candidato
`PHASE_ONLY_EVENT_S7_TRAIN40_SCIENTIFIC_OOF` y declara
`blind_validation_claim=false` para sus etapas. La auditoría de despliegue
verificada tiene estado `PASS_ALL_FOUR_DEPLOYMENT_PRODUCERS_FULL_TRAIN40`: los
productores de despliegue fueron ajustados sobre 88.744 filas y 40 secuencias.
La secuencia solapada `qGsgzl4Q8B` está incluida, al igual que las consultas
Garl exactas descritas arriba.

Esto permite medir funcionamiento y degradación sobre TRAIN, pero no presentar
la cifra como evaluación ciega. Los folds OOF históricos no vuelven ciego al
checkpoint final refit sobre TRAIN40.

EvTTC tampoco debe llamarse un test virgen. No se encontró evidencia de que el
candidato TRAIN40 se ajustara con labels EvTTC, pero el proyecto ya ejecutó y
publicó una transferencia descriptiva sobre 554 consultas físicas. Una nueva
medición EvTTC puede describir transferencia del checkpoint congelado; no puede
servir retrospectivamente para seleccionar arquitectura ni confirmar
generalización independiente del programa de investigación.

## Las cuatro condiciones admitidas

Las cuatro condiciones deben usar el mismo manifest de endpoints target, el
mismo cutoff sensorial y denominador. Cada condición se ejecuta con una nueva
instancia del servicio para impedir contaminación de historia o caches.

| Condición | ROI durante inferencia | ID durante inferencia | Qué aísla |
|---|---|---|---|
| R0 referencia | `boxes_xyxy[-1]` público Garl | `public_track_id` Garl, namespaced como oracle | Límite PHASE con ROI/ID de referencia |
| R1 perturbada | Perturbación preregistrada de R0 | Mismo ID de referencia | Sensibilidad a jitter/localización, sin error de identidad |
| R2 detectada/ID referencia | ROI detectada, asignada por matching evaluador | ID Garl de la asignación | Daño de localización, manteniendo identidad conocida |
| R3 predicha completa | ROI detectada | `track_N:generation` causal | Sistema completo; matching con GT solo después de inferencia |

R0–R2 son diagnósticos oracle y deben rotularse como tales. Para R2/R3, el
evaluador debe hacer asignación global frame a frame entre ROIs predichas y ROIs
Garl. El panel fija Hungarian con coste
`(1-IoU)+0,1*(distancia_centro/diagonal_GT)`, IoU mínimo 0,30, distancia máxima
0,50 diagonales y margen de ambigüedad bidireccional 0,05. No usa clase, ID o
TTC. Un objeto sin match es un miss, no una fila eliminada. En R3, switches,
historia insuficiente, abstenciones y falsos positivos permanecen visibles. El
tracker R3 conserva sus gates causales: clase exacta, IoU 0,30, margen 0,05,
dos hits, TTL dos y gap máximo 150 ms.

R1 también queda fijado: desplaza la ROI de referencia `+0,05*ancho` y
`+0,05*alto`, conserva su tamaño y solo la traslada hacia dentro si es necesario
para permanecer en 1280×720. No se elige la perturbación después de puntuar.

Los targets se unen solo después de sellar las predicciones. Se reportan TTC
firmado (MAE, mediana, p95 y error de signo), cobertura, misses, falsos
positivos, abstenciones e historia insuficiente. Los intervalos q10/q90 siguen
siendo coordenadas de fase; no se convierten en intervalos TTC continuos. La
incertidumbre se agrupa por secuencia/track, no por frame como réplica
independiente.

## Implementación siguiente necesaria

Se necesita un runner nuevo, sin modificar el productor histórico, que:

1. construya un manifest de endpoints Garl target mediante las dos uniones
   exactas anteriores;
2. añada suficiente media causal anterior para cumplir 600 ms en cada query;
3. selle detector, ROIs, asociaciones y abstenciones antes de leer `ttc`;
4. ejecute R0–R3 en lifetimes separados;
5. aplique el matching evaluador únicamente al puntuar R2/R3;
6. conserve todos los denominadores y emita resultados por secuencia/track.

El replay actual no basta: solo sus cuatro frames anteriores tienen target
alineado y el prefijo de ocho frames no aporta 600 ms de historia antes de todos
ellos. Debe ampliarse hacia atrás con media pública existente. Si se pretende
una conclusión independiente, hace falta además una población excluida de todos
los productores, normalizadores, teachers y decisiones previas; TRAIN40 y el
EvTTC ya observado no satisfacen ese requisito.

## Resolución del instante de referencia

La auditoría del contrato TRAIN40 resuelve el punto que no debe inferirse por el
nombre del PNG: para las cinco consultas, el final de la ventana de eventos del
endpoint target coincide exactamente con el **inicio** de exposición RGB eAP.
No coincide con el midpoint ni con el final de exposición. El reloj numérico
Garl sigue siendo otro reloj y no se resta directamente del reloj eAP.

El frame RGB target solo está completamente disponible al final de exposición,
entre 1.677 y 1.722 µs después de la referencia. El panel adopta de forma
explícita `availability_lag_no_ttc_shift`: inferencia con cutoff en el final de
exposición y evaluación contra el TTC original del inicio, registrando esa edad
sin extrapolar el target. Para causalidad estricta en el instante GT habría que
excluir el RGB actual y usar el frame anterior completamente disponible; cada
consulta conserva ese token alternativo.

## Panel CPU de cinco consultas

`scripts/build_ttc_development_panel.py` selecciona las cinco consultas TRAIN
en las que **cualquier miembro RGB** de la consulta intersecta el replay de ocho
frames y materializa, dentro de la misma secuencia, un prefijo causal mínimo de
al menos 600 ms. No exige que el miembro target final sea uno de los ocho
frames: en la quinta consulta el target no pertenece al seed replay. El panel
conserva las cinco aunque una condición no sea ejecutable:

- R0 y R1 tienen historia ROI de referencia completa en 2/5 consultas.
- R2 está listo en 0/5: faltan predicciones de prefijo y, en tres consultas,
  alguna ROI GT histórica.
- R3 está listo en 0/5 porque todavía faltan predicciones label-free declaradas
  para uno o más frames de prefijo.

No se extrapolan ROIs GT ni se eliminan consultas. El recibo
`output/s5_replay/ttc_development_panel_v3.json` enumera cada dependencia
ausente. La siguiente acción es producir predicciones detectoras para esos
tokens y someter el panel a revisión antes de cualquier inferencia PHASE.

Las búsquedas de historia y del frame anterior completamente disponible se
restringen a `sequence_id`. Las claves exactas `(sequence_id, rgb_member_path)`
y la unión Garl de cinco campos rechazan duplicados en lugar de sobrescribirlos.

## Arnés de cuatro condiciones

`scripts/evaluate_detected_phase_panel.py` separa tres operaciones:

1. `prepare` valida el panel v3, la procedencia label-free y el SHA del
   checkpoint detector; materializa entradas R3 sin TTC, ROI GT ni ID GT, y
   guarda las entradas oracle R0/R1/R2 en un namespace separado.
2. `run` exige cobertura detectora exacta de todos los prefijos y crea un
   runtime, servicio, tracker, generación e historial nuevos para cada par
   consulta–condición. El arnés construye el `DetectionPhaseService`: R0/R1/R2
   usan un `OracleReferenceTracker` diagnóstico, mientras R3 recibe una vista
   redactada y usa `CausalMultiObjectTracker`. En R2 el Hungarian selecciona la
   ROI detectada antes de asignar el ID de referencia; R3 nunca recibe esa rama.
   Cada paquete, incluso con cero detecciones, se ingiere una sola vez.
3. `score` sella primero el SHA del archivo de predicciones y solo entonces abre
   las cinco claves de anotación. Conserva el target TTC del inicio de exposición
   sin desplazarlo al final de disponibilidad y reporta MAE, mediana, p95, error
   de signo, cobertura, misses, detecciones no emparejadas con la referencia del
   panel y abstenciones por historia.

El panel contiene un único track target por consulta, no GT exhaustivo de toda
la escena ni reglas completas de objetos ignorados. Por ello una detección no
emparejada se contabiliza como `unmatched_to_panel_reference_count`, pero la
tasa o el recuento real de falsos positivos de escena queda explícitamente
`null`/no disponible. Todas las predicciones se conservan.

Los `q10/q90` nativos permanecen límites de fase; no se convierten en un
intervalo TTC continuo cuando cruzan cero. Todas las condiciones conservan el
denominador fijo de cinco consultas y la etiqueta
`TRAIN40_EXPOSED_DIAGNOSTIC_ONLY`.

La preparación CPU actual está en
`output/s5_replay/detected_phase_panel_prepare_v8.json`. Declara 15 tokens de
prefijo únicos, siete predicciones aún ausentes y cero tokens extra. Por tanto,
el estado es `BLOCKED_PREDICTION_COVERAGE`: no se lanzó PHASE ni se inventaron
ROIs o predicciones faltantes.

### Adaptador operativo incorporado

`run` usa por defecto el adaptador concreto
`operational/detection_phase_service/ttc_panel_adapter.py`; ya no exige que el
operador escriba una factoría. Su configuración JSON es estricta e incluye
`candidate_root`, `eap_root`, `device: "cuda"`, `max_objects`,
`logical_max_objects`, `rgb_workers`, `rgb_object_chunk_size`, `phase_bounds`
y `max_frame_cache_bytes`, `event_stream_sha256` (ruta relativa → SHA-256), además del schema
`detected_phase_panel_adapter_config_v1`. No existen rutas personales por
defecto.

La validación de configuración, sello del candidato, manifest y hashes de las
fuentes ocurre en CPU. Los modelos CUDA se construyen únicamente dentro de
`run`. Cada consulta–condición posee runtime, historial y caché de medios
nuevos y se cierra antes de la siguiente. Los eventos se leen en intervalos
half-open únicos, contiguos y causales; se conservan solo los 620 ms necesarios
para materializar los 600 ms sellados y el paquete final de 20 ms. Un intervalo
vacío sigue registrado como paquete ingerido. RGB queda limitado al prefijo:
todos sus hashes de miembro son obligatorios y se verifican antes de construir
CUDA, además de volver a comprobarse al leer cada frame.

```text
python scripts/evaluate_detected_phase_panel.py run \
  --prepared output/s5_replay/detected_phase_panel_prepare.json \
  --predictions output/detection_predictions.json \
  --adapter-config output/s5_replay/ttc_panel_adapter.json \
  --output output/s5_replay/detected_phase_panel_predictions.json
```

`--runtime-adapter module:function` se conserva para inyección explícita de
tests o adaptadores propios y es incompatible con `--adapter-config`.

## Evidencia

- Auditor reproducible: `scripts/audit_ttc_evaluation_admission.py`.
- Recibo machine-readable:
  `output/s5_replay/ttc_evaluation_admission_v3.json`.
- El recibo contiene hashes, claves de alineamiento, counts y metadatos de las
  ocho correspondencias; no copia valores TTC.

Conclusión permitida: **es posible un diagnóstico conjunto sobre TRAIN expuesto,
una vez ejecutado el manifest correctamente alineado**. Conclusión no permitida:
precisión TTC nueva del replay actual, generalización independiente o SOTA.
