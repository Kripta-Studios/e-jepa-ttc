# Auditoría de medición, población y puente causal — notas para el plan SOTA

Base inspeccionada: `24b8f36f6e4aa17c965f9526d8cf486b512a8046`, worktree `phase-ttc-sota-review`. Solo lectura del repositorio. No se ejecutó entrenamiento, inferencia ni la suite GPU. No se modificaron archivos del repo.

## 1. Evidencia verificada en esta revisión

- Los 49 hashes de fuentes de `evidence/final_rgb_verification.json` coinciden con los archivos actuales.
- `final_rgb_pipeline_tests.xml` coincide con su hash y registra 232 casos, cero fallos, errores o skips. Esto verifica el recibo de la ejecución previa, no una nueva ejecución de 232 tests.
- `scripts/summarize_detection_runs.py::render` reproduce exactamente `TRAINING_TABLES.md` desde los tres recibos archivados.
- La curva del piloto contiene 512 updates y 2.048 exposiciones: cada uno de los 256 tokens train aparece ocho veces; ningún token selection aparece.
- Las secuencias efectivamente usadas son train `qGsgzl4Q8B`, `nRcwUlHyAd`; selection `JaGC4Q9pgU`, `2Y46OkK3ff`. No son las cuatro sugeridas en el plan anterior.
- Los pesos y los JSON completos de predicciones/GT no están en Git. Por tanto no se han vuelto a calcular las curvas PR desde cajas; se comprobaron código, agregados y su coherencia. Las tablas sí se regeneraron.
- Se ejecutó un fixture sintético del tracker que muestra su ambigüedad inversa no detectada, descrita abajo.
- Se leyeron las cuarenta etiquetas train fijadas disponibles de la auditoría anterior y se recomputó población por rol, identidad y clase. Se reprodujeron los filtros de los 512 frames del piloto usando sus etiquetas y K/T, sin leer imágenes ni ejecutar Torch; se obtuvieron exactamente 829/938 admitidos.

La recomputación de población está en `pilot_population_diagnostic.json`, fuera del repo. Su alcance es metadata y la aproximación pinhole declarada; no certifica disponibilidad de medios ni el universo del scorer.

## 2. La métrica publicada y sus denominadores

Fuente: `src/e_jepa_ttc/detection/development_metrics.py`, especialmente `_match` (90), `_ap101` (115), `evaluate_detections` (132).

`0.02064070955899264` es AP101 con envolvente de precisión, sin mínimos nuScenes de recall/precision, matching greedy por clase y plano XY, distancia **estrictamente menor** que cada umbral 0,5/1/2/4 m. Se promedian los cuatro umbrales y luego las **cinco clases con GT** en selección. No es AP oficial eAP ni AP de ocho clases demostradas.

La convención secundaria `all_eight_absent_zero` produce `0.0129004434743704`. Asignar cero a clases ausentes no proporciona evidencia sobre su precisión; debe conservarse como convención explícita, no como sustituto de evaluación con soporte.

| Clase selection | GT admitido | Predicciones | AP local | Matches a 2 m |
|---|---:|---:|---:|---:|
| car | 721 | 6317 | 0.08338670 | 342 |
| truck | 205 | 1966 | 0.01973244 | 58 |
| motorcycle | 6 | 2816 | 0.00008441 | 1 |
| pedestrian | 2 | 1605 | 0 | 0 |
| bus | 4 | 84 | 0 | 0 |
| tricycle | 0 | 12 | indefinida | 0 |
| bicycle | 0 | 0 | indefinida | 0 |
| other_vehicle | 0 | 0 | indefinida | 0 |

A dos metros son 401/938 = 42,75% de recall de los targets admitidos y 401/12.800 = 3,13% de precisión sobre todas las propuestas, sin corte de score. Son diagnósticos de un umbral; no son AP. No se pueden dividir los mismos matches por el total raw y llamar al resultado recall de evaluación completa sin repetir el matching.

ATE, ASE y AOE se promedian por clase con matches; una clase sin matches no participa. Z no entra en el matching ni en ATE. Por eso debe registrarse también error Z, corners, IoU 3D/BEV y reproyección como diagnósticos separados, sin llamarlos oficiales si no lo son. ASE de selección es 0,52969 macro, pero car es 0,16515, truck 0,74103 y motorcycle 0,68288; el diagnóstico global oculta problemas de dimensiones en clases con pocos matches.

El score monotónico/calibrado no es una receta para aumentar AP: manteniendo pool, cajas y clases fijos, una transformación monótona por clase conserva el orden y la AP. Cortar la cola por threshold tampoco puede mejorar la envolvente AP de ese mismo ranking; sí cambia cobertura operacional. NMS y un ranking geométrico pueden cambiar AP porque eliminan/reordenan propuestas, pero deben medirse conjuntamente con recall y duplicados. No asumir que todos los 12.399 no-matches son duplicados corregibles.

## 3. El universo GT está acoplado al adaptador de entrenamiento

`monodgp_eap_adapter.py::prepare_sample` filtra etiquetas antes de crear `native_ground_truth`. `train_monodgp_portable.py::evaluate`, línea 133, utiliza esa lista como GT de evaluación.

| Razón | Train (1565 raw) | Selection (1936 raw) |
|---|---:|---:|
| Admitido | 829 | 938 |
| Profundidad centro fuera de [0,1;60) m | 635 | 901 |
| Centro proyectado fuera de imagen | 98 | 92 |
| Alguna corner detrás/cerca de cámara | 1 | 4 |
| Centro fuera de bbox clipped | 2 | 1 |

Train retiene 52,97% y selection 48,45% de sus etiquetas raw. Los tres `other_vehicle` raw del piloto train caen por profundidad; bicycle pasa de tres a un positivo.

`decode_eap` (249) rechaza valores no finitos, dimensiones no positivas y profundidad no positiva, pero no aplica el mismo límite de 60 m/FOV del GT. El evaluador tampoco ignora predicciones en regiones de GT excluido. En consecuencia, una predicción legítima de un objeto fuera del perfil puede convertirse en FP local. No hay justificación para atribuir una dirección universal al sesgo ni para corregirlo simplemente descartando más GT.

**Ticket P0 de medición:** desacoplar `EvaluationProtocol` del `PreparedSample`. Definir la elegibilidad con el contrato de la tarea/scorer, común a todos los modelos, y mantener `legacy_pilot_v1` inmutable. Hasta disponer del scorer, publicar una evaluación de todos los GT raw válidos como diagnóstico y un perfil visible/rango definido independientemente del detector, con lista de exclusiones e ignorados. Separar targets entrenables, GT evaluable e ignorados por benchmark. Las predicciones se tratarán según esa misma política, sin trasladar al GT las limitaciones de una arquitectura concreta.

Aceptación: oracle perfecto sobre la población evaluable, casos de rango/frontera, boxes fuera de imagen, GT ignorado versus ausente, clases sin soporte, FP fuera de dominio, Z errónea y duplicados; todo modelo se evalúa sobre idéntico hash de GT. No sobrescribir las métricas piloto ni compararlas directamente con el nuevo perfil.

## 4. Cobertura real disponible sin Test12

El split completo 24/8/8 tiene las ocho clases en cada rol raw. La lectura de Parquet produjo:

| Clase | Raw train24 | Raw selection8 | Raw calibration8 |
|---|---:|---:|---:|
| car | 304200 | 90884 | 101200 |
| truck | 107686 | 38670 | 33085 |
| motorcycle | 22732 | 39555 | 18710 |
| pedestrian | 7808 | 14415 | 7154 |
| bus | 5137 | 3335 | 1738 |
| tricycle | 2268 | 2668 | 623 |
| bicycle | 834 | 789 | 446 |
| other_vehicle | 237 | 153 | 183 |

Pero contar frames no basta. Aplicando las fórmulas actuales de admisión a las dos clases más escasas, con el tamaño original 1280×720 documentado:

| Rol | bicycle: admitidos / IDs | other_vehicle: admitidos / IDs |
|---|---:|---:|
| train24 | 655 / 69 | 53 / 6 |
| selection8 | 596 / 65 | 60 / 5 |
| calibration8 | 270 / 26 | 51 / 4 |

Los IDs son pares `(sequence_id, track_id)` anotados. No prueban objetos físicos distintos entre secuencias ni adquisiciones independientes. Son pocos casos, especialmente en `other_vehicle`; oversampling no crea diversidad nueva.

En train, ejemplos concretos para planificar ampliación de soporte son `Po6ePjWYAz` (19 other_vehicle, 187 bicycle admitidos), `8SdieOgbva` (12,46), `nRcwUlHyAd` (15,46), `9ITOs4twfk` (4 other_vehicle) y `t79dBxj1WS` (3). Deben fijarse tokens por reglas de cobertura/distancia/identidad **antes** de observar resultados. Son candidatos del rol train, no selección de ejemplos por predicciones.

En calibration, todos los other_vehicle admitidos se concentran en `2JkLL5W891` y `rhYxPcU29n`. Esto importa al reservar una auditoría y evita prometer calibración individual de ocho clases con evidencia insuficiente.

**Ticket P1 de población:** inventario de clases por grupo de adquisición, secuencia, identidad, distancia, visibilidad, condición, raw/evaluable/trainable. El sampler puede equilibrar train con exposición máxima por identidad y mantener negativos reales. La población evaluada conservará su distribución y denominadores congelados. Si una clase no alcanza diversidad suficiente, se informa el límite y se solicita dato nuevo; no se confunde aumento de repeticiones con cobertura.

## 5. Independencia y nuevo protocolo

El detector vio dos secuencias de las24 train. Las dos selection ya usadas son desarrollo expuesto a decisiones. Las ocho calibration no se usaron en el piloto del detector. El mapa de sesiones relacionadas aún falta. PHASE TRAIN40 estuvo expuesto a todas las cuarenta secuencias.

Propuesta concreta, condicionada a grupos y soporte: conservar24train y8selection, y dividir las8calibration nunca usadas por el detector en4calibration y4audit final. Registrar `protocol_v2` y la regla antes del primer run nuevo; no mover las dos selection expuestas a audit. Elegir membresía por grupos, dominio y cobertura con metodología previa, sin resultados de modelo. Si no se puede preservar independencia/cobertura, documentar otra asignación antes de experimentar. Las dos secuencias calibration con other_vehicle no son necesariamente dos adquisiciones independientes; no repartirlas mecánicamente.

La calibración ajusta su método preseleccionado en calibration; el audit se abre una vez tras congelar modelo, NMS, thresholds y calibración. Si audit provoca rediseño, pasa a desarrollo y hace falta otra validación o reconocer el carácter exploratorio. Con pocos grupos, los intervalos serán amplios; más frames del mismo trayecto no son más réplicas.

Alternativa si se mantiene24/8/8: declarar que no existe un conjunto local independiente separado de selección/calibración y reservar el test oficial para la confirmación final. No utilizar calibration simultáneamente como holdout de eficacia.

Un refit final del detector en todo train40 puede ser razonable tras congelar decisiones para una evaluación oficial Test12 autorizada; consume el holdout local y cambia productor/calibración. Sus resultados no heredan automáticamente la calibración ni independencia del checkpoint pre-refit.

Para TTC conjunto, ninguna de estas divisiones vuelve independiente a TRAIN40. Se requiere excluir las secuencias evaluadas de **todos** los productores/normalizadores/teachers/calibradores o usar datos externos apropiados. Adaptar solo el head no soluciona exposición. El progreso en detección no debe esperar a este trabajo más costoso: separar claims de detector, tracking y TTC.

## 6. Velocidad: señal parcial, semántica pendiente

La comparación actual usa los mismos401 matches a2m. L2 medio: cabeza0,594901, cero1,381150 y media train0,759128 m/s. En el segundo componente, la cabeza tiene MAE0,03088 frente a≈0,021 de controles; la mejora vectorial procede del primero. Motorcycle aporta un solo match y no respalda un resultado por clase.

El target representa los dos componentes publicados. Faltan marco, referencia, ego-motion y construcción temporal física completa. No deriva TTC, no es AVE oficial y no permite imponer sin más `TTC=-Z/v` como pérdida. Una etiqueta calculada offline puede servir como target documentado; nunca se introduce el futuro en features online.

Plan: medir cada componente, distribución y tasa de casi-cero, velocidad por objeto/escena/rango, cobertura con y sin match, controles constante por clase y dinámica causal donde haya ego-pose. Documentar semántica antes de regularización física y comparar temporal frente monocular bajo iguales datos y presupuesto.

## 7. El puente es una demostración de una query, no el tracker general

`infer_detected_roi_phase.py` asocia solo con detecciones del frame inmediatamente previo, por clase e IoU2D. Elige la detección actual de mayor score (287), exige dos observaciones (295), construye una ROI común (312) y carga `CompetitionRuntime` (418). No invoca `MultiObjectTrain40Service` ni conserva un lifecycle `lost` para reapariciones. Tras un frame vacío se pierde el soporte de identidad.

Fixture ejecutado: track previo `[0,0,10,10]`; dos detecciones actuales `[0,0,6,10]` y `[4,0,10,10]` con misma clase/score. IoU con previo0,6 ambos e IoU mutuo0,2; NMS no suprime. El algoritmo hereda el ID en la primera por índice y asigna ID nuevo a la segunda, marcando `ambiguous=false` en ambas. Comprueba margen detección→tracks, no la ambigüedad inversa tracks→detecciones. No invalida la demo documentada; debe resolverse antes de afirmar asociación general.

**Ticket P2 tracking:** matching con costes/gates explicables, margen bidireccional, assignment global y generación de identidad, estados tentative/confirmed/lost/closed y ventanas causales. No usar posiciones ego como marco estacionario sin ego-motion. Medir ID switches, fragmentación, HOTA/IDF1 cuando existan identidades GT admisibles, recall/FP de tracks y la caída de cobertura TTC. Probar cruces, paralelos, oclusión, desaparición, reentrada, timestamps tardíos, cambio de clase y escenas vacías.

**Ticket P3 puente multiobjeto:** servicio residente, ingestión una vez por paquete, cero objetos, más objetos que capacidad, buffers prestados, caches por generación y snapshot. Cualquier detección debe sobrevivir a TTC no disponible. Registrar clocks de sensor, exposición, llegada, fin de detector y salida. El replay actual admite corte sensorial pero no mide latencia física de llegada. Tampoco debe interpretarse exit0 del CLI como inferencia exitosa: puede devolver `UNAVAILABLE`.

## 8. Evaluar TTC con ROIs predichas

La única salida conjunta, del conjunto train del detector y expuesta a TRAIN40, no tiene GT TTC alineado. El mapeo eAP–GarlTTC continúa siendo un bloqueo distinto: la igualdad de medios no prueba identidad ni reloj de anotación. No generar nuevos targets mediante vecino más próximo sin validar correspondencia.

Matriz recomendada sobre un protocolo admisible: ROI/ID de referencia solo como diagnóstico de evaluación; ROI perturbada controlada con identidad conocida; detección predicha con identidad de referencia solo para aislar localización; detección+tracking predichos completos. Nunca usar ROIs/IDs Garl test como detecciones reales. Mantener separados carril de desarrollo TRAIN40 expuesto y carril de generalización con productores que excluyeron el grupo.

Medir sobre todos los objetos/queries elegibles: detección/track/TTC coverage, abstenciones y sus causas, errores de signo, MAE/mediana/p95 y curvas error–cobertura. FPs, misses y objetos sin historia no se eliminan de la contabilidad. Comparar calidad condicionada a mismos objetos y resultados integrales, evitando mejorar MAE solo porque se emite en casos fáciles. q10/q90 de fase necesitan calibración propia; intervalos que cruzan cero no se convierten ordenando extremos en un TTC continuo.

## 9. Diagnósticos prioritarios sin entrenar otro modelo

1. Recalcular evaluación bajo universo independiente del adaptador; mismo checkpoint/predicciones y denominadores explícitos.
2. Curvas PR y recall@K por clase/rango, score versus error XY/Z, duplicados, confusión y breakdown de invalidaciones.
3. Sustituciones oracle solo como diagnóstico sobre desarrollo: clasificación, profundidad/centro, dimensiones/yaw y ranking por calidad para localizar el cuello. No reportar estos valores como predicciones ejecutables.
4. NMS versus sinNMS, a igualdad de métricas/población, conservando recall y motivos de supresión. Un threshold operativo no sustituye evaluación del ranking.
5. Controles de imagen repetidos, permutaciones dentro de secuencia/condición y pruebas de sensibilidad con incertidumbre por grupos. Una permutación global y dos secuencias no demuestran robustez amplia.
6. Congelar una escalera de datos y optimización que compare poblaciones, exposiciones y recetas equivalentes; tres seeds finales solo tras decidir la receta mediante desarrollo.

La prioridad científica inmediata es completar población y evaluación comparable, después fortalecer RGB. La integración temporal, eventos y adaptación PHASE deben demostrar ganancias incrementales sobre ese baseline fijo antes de compartir backbone.
