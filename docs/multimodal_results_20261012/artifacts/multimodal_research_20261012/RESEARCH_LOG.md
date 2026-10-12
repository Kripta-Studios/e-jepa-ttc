# Continuación autorizada

La autorización explícita del usuario del 12 de octubre permite hasta 18 horas adicionales y un millón de updates físicos. La primera prueba está limitada a dos brazos de 2.500 updates, EVENT y ZERO, seed7. V13, V14 y V15 permanecen históricos. V15 cerró `COMPLETE_NO_CANDIDATE`, con 22.060 updates físicos; no se reaprovechan sus padres ni se ocultan sus gates fallidos.

Se conserva el bundle V16 anterior. Esta investigación escribe en otro directorio de artefactos. El contrato `PROTOCOL.json` se fijó antes de extraer nuevos estados V o evaluar los endpoints nuevos. V es desarrollo ya expuesto, no test independiente. El entrenamiento recibe exclusivamente H, 18.422 consultas, normalización y padres congelados.

La cabeza recibe dos bloques de 170 coordenadas: hidden160 real del refiner congelado, raw_location, point_phase, expertos3, timing4 y longitud válida/8. La última capa está inicializada a cero. Tanto predicción como cuantiles reproducen exactamente R_CTX en toda H durante la extracción FP32 GPU. ZERO anula las primeras 165 coordenadas del bloque event; conserva sólo los relojes y disponibilidad. Ambas rutas tienen igual número de parámetros, sampler y objetivo nativo; no se añade pérdida de coste auxiliar.

Los estados H proceden de refinadores ya supervisados en H. La etapa nueva no es OOF ni un teacher independiente. No hay nuevo límite residual pequeño ni cambio de signo por heurística.

Los dos tests nuevos verifican la paridad inicial de la predicción y cuantiles contra el refiner real con hooks y la invariancia del control ZERO ante cambios de todas las coordenadas predictivas event, incluso con pesos del readout no nulos. No realizan pasos de optimizador. La extracción H verificó la paridad inicial sobre las 18.422 filas.

Se ejecutaron 250 updates EVENT, se guardó estado completo y se reanudó hasta 300. El guard detectó después una cola viva `operational.representation_audit.run` en V17, que trabaja sobre representación event. El proceso propio se pausó de forma duradera; recuperación repetida: cero. Se conserva la cola ajena. Sus resultados no se utilizarán para alterar los gates del ensayo actual.

El diagnóstico H encuentra 49 raw_locations RGB en la zona plana de conversión ±60 s, 219 signos RGB erróneos y 364 desacuerdos de signo RGB/event. E_CTX tiene menor error de fase en 7.772/18.422 filas. El error de fase medio RGB es 0,00359458; un selector que conociera GT tendría 0,00249040. Ese oráculo es descriptivo, no desplegable ni una demostración de generalización. `H_COMPLEMENTARITY.csv` separa contribuciones por grupo, signo y zona plana.

## Comparabilidad bibliográfica

REACT (preprint de septiembre de 2026) informa 9,59±0,74 % de error relativo en su partición EvTTC, cinco semillas y 4,6 ms según su perímetro. Trabaja sin ROI objetivo y el artículo anuncia el código para después de la publicación. Su MiD_w usa logaritmo de TTC inverso y omite negativos; no es nuestra fase logarítmica de razón de alturas ni nuestra cohorte Dev32/FCWD. Por ello estas cifras son contexto, no filas comparables ni una implementación reproducida aquí. [Artículo primario](https://arxiv.org/html/2609.19204v1).

Garl sigue siendo el comparador disponible ejecutado con checkpoints públicos, entradas nativas y cohortes comunes. Una victoria local sobre Garl no permite inferir una victoria sobre todos los métodos/protocolos, y los costes de red preparada, servicio y acumulación temporal deben permanecer separados. [Release oficial](https://github.com/NAIL-HNU/Garl-TTC).

## Avance recuperable, 2026-10-12T10:02:54.403473+00:00

EVENT está pausado con checkpoint completo en 2.425/2.500 updates, sin recuperación repetida; ZERO sigue pendiente. La continuación serial espera la salida del proceso V17 de precisión. No se han generado nuevos estados ni predicciones V. El análisis prospectivo tiene un hash separado en ANALYSIS_BINDING.json.

El diagnóstico H del checkpoint provisional reduce error medio de fase de 0,00359458 a 0,00310341 y errores de signo de 219 a 168. Permutar los estados event eleva el error a 0,00463054. Estas perturbaciones pueden salir de la distribución de estados: miden dependencia descriptiva y no atribución causal ni generalización.

EXACT_WINDOWS conserva igualdad bit a bit en 128 comparaciones de 64 consultas y ahorra 12,6 %/23,2 % del tiempo total de preparación medido en Dev32/FCWD. SELECTED_MAP conserva todos los arrays y dtypes en otras 128 comparaciones; ahorra 38,1 %/38,5 % de la etapa map sobre arrays ya leídos. Son pilotos CPU de un hilo bajo carga concurrente, no latencia total ni resultados aislados. Cinco tests sintéticos adicionales cubren límites espaciales/temporales, ROI fraccionaria, vacío y soporte de ventanas. El perfil sitúa read_crop en 69,4 %/46,3 % y map en 8,7 %/21,8 % de la preparación. No se suman tiempos de campañas distintas.

El archivo WINDOW_PROFILE_EXECUTED_SOURCE.py conserva el código exacto del perfil publicado: después se corrigieron tipos del wrapper sin reejecutar ni sustituir las mediciones. WINDOW_STAGE_COST.json conserva su hash original.

La extracción externa FCWD avanza en CPU con los adaptadores nativos, cuatro observaciones event y RGB registrado sellado. Es independiente del entrenamiento y no lee TTC externo; se guarda cada consulta con hash, reloj, disponibilidad y comparadores recalculados en la misma ruta. Tres intersecciones iniciales con las trazas E_CTX anteriores tienen delta CPU exactamente cero. Los cambios de precisión respecto al histórico GPU siguen separados.

## Estudios cerrados y continuación, 2026-10-12T11:15:09.976487+00:00

Los cuatro primeros fits terminaron: 10.000 updates científicos/físicos, recuperación cero. Baseline EVENT obtiene V MiD 91,300 y MAE 1,852 s; no pasa los gates y su contraste EVENT–ZERO incluye cero. PHASE_UNITS EVENT obtiene MiD 89,882 y MAE 1,846 s, con IC95 EVENT–ZERO [−5,390; −0,561]. Falla exclusivamente el gate del bucket large: +3,602 % respecto a R_CTX, por encima del 2 % prospectivo. El usuario preguntó por un 4 %: USER_SENSITIVITY_LARGE_4PCT.json demuestra que pasaría bajo ese margen, conservando el fallo original y sin declararlo SOTA.

La transferencia CPU FCWD de PHASE_UNITS conserva las 630 consultas y 597 etiquetas, con fallback E_CTX idéntico cuando falta RGB. MiD 71,134 y MAE 1,596 s frente a F_TRUE de la misma ruta 53,925 y 1,430 s. La mejora V no se generaliza aquí. El contraste con ZERO mejora MAE pero empeora MiD. Los tres grupos externos y sus intervalos son descriptivos, no selección de receta.

La auditoría global CPU/histórico encontró delta E_CTX máximo 0,609592 s; el máximo de la muestra previa 0,013831 s no era una cota global. ROUTE_PARITY.json reproduce E_CTX histórico en diez casos (incluyendo los mayores deltas) usando CUDA matmul TF32 desactivado y cuDNN TF32 activado, con máximo de redondeo CSV <1e-15 s. Queda pendiente la comprobación global de R_CTX/F_TRUE y estados en esa ruta. Los históricos no se reemplazan.

NEXT_STUDIES_PREREGISTRATION.json registró antes de abrir PHASE_UNITS externo dos pares adicionales de semillas 13/23 y un par RELATIVE_PHASE semilla 7. Presupuesto 15.000 updates, mismo objetivo y gates originales. Las réplicas son de cabeza, no de productores completos. Las tres pruebas nuevas preservan ancla/cuantiles iniciales, control ZERO y escala relativa sin límite nuevo. Ruff y Pyright pasan. Commit de implementación c65aec1.

El extractor Dev32 CPU sigue recuperable por consulta, sobre 1.024 identidades físicas. La capacidad compartida registra procesos GPU reales, mantiene al menos 6 GiB libres y limita reserva propia; no se reclama latencia aislada. V17 conserva sus recursos y propietarios.

## Réplicas y transferencia completas, 2026-10-12T11:53:38.708102+00:00

Los diez primeros fits completaron 25.000 updates, recuperación cero. Ninguno es candidato bajo todos los gates originales. PHASE_UNITS S13 obtiene V 90,51790 / 1,843921 s y large +2,4086 %, con IC EVENT–ZERO que incluye cero. S23 obtiene 90,25774 / 1,821172 s, large +4,9726 % e intervalo que incluye cero. El margen secundario del 4 % no produce una réplica individual que pase todos los gates. RELATIVE_PHASE S7 obtiene 90,25027 / 1,884901 s; falla large (+2,7912 %) y MAE frente a R_CTX, aunque su contraste EVENT–ZERO es favorable. Los resultados no justifican modificar retroactivamente los criterios.

Dev32 completo conserva 1.024 consultas/946 etiquetas. El replay reproduce E_CTX hasta redondeo CSV y R_CTX/F_TRUE con diferencias máximas 1,91e-5/2,67e-5 s, sin cambios de signo. Su inferencia histórica era CPU, no GPU. PHASE_UNITS S7 mejora descriptivamente MiD de 131,38062 (R_CTX) a 125,62689 y MAE de 1,257864 a 1,230327 s. Su CI descriptivo MiD frente a R_CTX es favorable, frente a F_TRUE incluye cero. Baseline S7 obtiene 127,71128 / 1,193430 s. Las réplicas y la variante relativa también tienen transferencia congelada publicada; la variación FCWD entre semillas se conserva.

La ruta completa de reutilización CPU conserva todos los outputs en 128 comparaciones y reduce el tiempo total medido 18,58 % en Dev32 y 26,29 % en FCWD. Los pesos son residentes; no incluye ROI/registro/RGB, ni es streaming GPU. EXACT_EVENT_ROUTE_SUMMARY.json contiene los tiempos realmente medidos, sin sumar perfiles separados.

El ensayo FUSION_STATE fue registrado antes de abrir la transferencia de UNIT_REPLICATION y RELATIVE_PHASE. Usa H y la evidencia interna V: en H, F_TRUE tiene error de fase 0,00280565 frente a RGB 0,00359458; el oráculo mínimo conjunto sería 0,00222855, no desplegable. FUSED y ZERO_FUSED retienen E/R y difieren sólo en los 162 estados nuevos de la fusión (hidden160, raw y fase). Esto mide información incremental de la fusión, no ausencia de eventos. La última capa conserva inicialmente R_CTX y los cuantiles. Presupuesto 5.000 updates, gates originales. Un anexo previo al primer update fija CPU FP32, un hilo, para permitir el replay GPU nativo simultáneo. Native ProducerCheckpoint admite CPU y conserva RNG/sampler; no se elude ningún presupuesto ni límite de RAM/disco. No se inicializa CUDA en el entrenamiento CPU.

## Investigación posterior, 2026-10-12T12:40:19.559249+00:00

Los catorce fits previos completaron 35.000 updates físicos/científicos, recuperación cero. FUSION_STATE fue negativo: FUSED MiD V93,118 frente a ZERO_FUSED89,239, con intervalo del contraste completamente positivo. Añadir ese estado a un ancla RGB no mejora esta receta. RGB_REGRET protege los cuatro rangos bajo2%, pero V92,088 no supera F_TRUE90,650. Su transferencia CPU FCWD68,557/1,613s y Dev32130,257/1,232s no establece un candidato.

PHASE_ENSEMBLE combina en fase emitida las tres semillas registradas7/13/23 con pesos iguales, sin optimización ni selección de semillas. V89,86130/1,813344s; contraste EVENT–ZERO IC[-4,93818;-0,34947]. Falla large2% y pasa la sensibilidad4%. FCWD CPU65,20115/1,580199s sigue peor que F_TRUE; Dev32125,50063/1,258268s. Es desarrollo adicional sobre cohortes ya expuestas, no confirmación independiente. Tres tests pasan y comprueban signo, soporte nativo y promedio de fase.

FUSION_PRESERVING registra dos fits adicionales2500 con F_TRUE como ancla, ambos multimodales. REFINED_RGB conserva estados de R_CTX y ZERO_RGB anula sólo esas165 coordenadas predictivas, no RGB dentro de F_TRUE. Conserva masas/pérdida/schedule nativos, sin nuevo límite residual. La inicialización reproduce punto, TTC y cuantiles nativos bit a bit en128H+128V, ambos brazos; tres tests adicionales pasan. Las transferencias de FUSION_STATE todavía no se habían abierto.

El replay GPU nativo se reanudó desde360/630, usando únicamente la preparación CPU ya equivalente. NATIVE_GPU_EXACT_PREPARATION verifica además las dos consultas de frontera sin TTC; las inferencias y los batches GPU no cambian. El replay de estados F_TRUE en Dev32 sigue pausado74/1024, para no ejecutar dos replays sensoriales propios a la vez.


## 2026-10-12T13:39:24.907346+00:00 — phase-only replication and group cross-fitting

PHASE_ONLY seeds 7/13/23 all pass the original 2% V criteria and paired event-vs-zero contrast. All three frozen external evaluations are complete. FCWD crucial point estimates improve over Garl event-only, with additional RGB information; F_TRUE retains better FCWD aggregate/MAE. Dev32 MiD does not consistently improve. These are head replicas with fixed producers, not full-system replicas. Adaptive search and descriptive external intervals remain declared. Native GPU FCWD replay is complete and all parent maximum absolute errors are below 1e-6 seconds. The preregistered 4-fold group cross-fit experiment has started after exact native loss/gradient and guard parity; its fixed 25,000-update plan includes eight context fits and two phase-only readouts. Dev32 extra fusion-state replay continues with per-query recoverability and exact existing event-state parity. No official SOTA claim or submission.


## 2026-10-12T13:59:15.808660+00:00 — completed group cross-fit and test12 preparation

All32fits have fixed2500-update endpoints:80000physical/scientific, recovery0. GROUP_CROSSFIT V MiD90.412912,MAE1.893097 fails original MAE gate; paired EVENT-ZERO CI[-6.766188,-2.267140]. External FCWD MiD65.713557,MAE1.576309,MiDc118.016757;Dev32 137.004210,1.263563,194.930046. No promotion. Dev32 extra F states and both pending fusion-study evaluations complete. All36 required test12 RGB archives downloaded (35521710080bytes) from fixed eAP revision. Native event input preflight gives exact tensors on16hash-selected H controls and12test controls, and exact identity mapping for all18422H histories. Histories use actual same-object observations, not assumed regular4x100ms orH8 histories. First PHASE_ONLY seed7 will be frozen for local official-format preparation; no test labels or submissions.


## 2026-10-12T14:51:57.343544+00:00 — frozen official inputs, exact diagnostics and cost

The first PHASE_ONLY EVENT seed7 is frozen under SHA256 85fad0dfb8bac0fa5a09e6f4090818ea2b534908ef3d018e8f88f0805e825b1a. All36 RGB archives passed full SHA verification against immutable Hub metadata. There are6762 official tokens and5805 physical observations:957 aliases from overlapping clips have identical native windows/ROI/clocks and are expanded back at packaging. The native event stage is complete; RGB inference is active with per-observation recovery. No private labels, submissions, new training or change of frozen model.

Offline metric replay passes46778scalar checks across13V and34external tables, and verifies all32fit endpoints. Full external readout tracing covers3308arm/query outputs (630FCWD+1024Dev32, EVENT/ZERO), all exactly equal to sealed predictions and bitwise invariant to hooks. Selected104traces preserve the original32H hash controls and20previous public failure episodes. In Dev32 all13wrong-sign RGB predictions among946labels retain their sign. The18labelled parent-sign disagreements have candidate-minus-RGB MAE+2.273747s; the928same-sign cases have-0.086230s. This is posthoc localization, not a fitted routing policy.

The4000whole-sequence bootstrap vsGarl event on FCWD gives MiD delta-2.312403 CI[-4.298607,+0.515151], MiDc-1.752577 CI[-5.759133,+2.279379]. Against Garl full registered with event fallback, all three aggregate contrasts favor the candidate; F_TRUE still has better FCWD aggregate/MAE. All intervals remain descriptive, not corrected for adaptive search.

The128-run sensor cost was repeated to include the evaluator's final emitted-phase-to-TTC conversion; the original raw-head API profile and executed source are preserved in phase_only_cost_v1. All new states and outputs are exact. CPU one-thread p50 sensor-to-candidate: FCWD1630.349ms, Dev322022.164ms. Registered transforms are precomputed; detector, registration search and initialization are excluded. Concurrent load prevents an isolated GPU latency claim.

Read-only V17 coordination confirms reported NO_CANDIDATE and106680physical updates (106500scientific+180technical), pending0. No foreign files, owners or parents were modified. Our32fits remain80000physical/scientific updates and0recovery. Public benchmark page retrieval did not expose a usable live leaderboard; the GitHub release confirms private test TTC labels and official scoring route, without proving the current best ranking.


## 2026-10-12T15:05:25.379239+00:00 — causal audit and inherited sign errors

All6762test histories pass same-object, disjoint P/H/V/test sequence groups, availability and physical-alias equality checks. Actual maximum support is600006us event and600000usRGB, within the fixed650000us bound. All pinned parents/source hashes and runtime sources remain unchanged since execution commitc62a610. The diagnostic/report files alone changed. In the frozen S7 Dev32 diagnostics, all13negative R_CTX predictions remain negative; six had positive RGB expert-median phase before the context residual. Positive expert sign alone does not imply an accurate TTC. Full detail and table CSV bytes are unchanged by adding this decomposition; initial source/receipt preserved. Benchmark access review found no connected browser and no usable live leaderboard response. No submission or account action was attempted.


## Frozen inference recovery 2026-10-12T15:41:09.698998+00:00

The prior process was absent and its exec session no longer existed. Last saved prediction count: 890; both 5805-observation sensor stages complete. Restarted the unchanged frozen entrypoint, which verifies and reuses each receipt. Original execution log preserved; new log INFERENCE_RESUME_1530_RUN.log. No optimizer updates or recipe changes.


## Shared-admission recovery 2026-10-12T15:51:06.498643+00:00

The native shared exception handler scanned non-Python process metadata. Live parity retained all 13 stable Python metadata records exactly (native scan 15484.84 ms; filtered scan 19.30 ms). Three existing guard tests passed. New test12_guarded_resume.py extends the previously admitted scan context to that handler without changing frozen models, arithmetic, thresholds or per-item checks. Stopped only our slow recovery process. The next attempt correctly rejected concurrent V18 CUDA QA; its STOPPED receipt and log are preserved. Resumed again after that process exited. V18 sources/status were observed read-only.


## Cache-only recovery 2026-10-12T15:58:29.190940+00:00

The per-item full guards remained expensive even after process filtering. Stopped only our recovery PID 29840 while rereading completed event caches (301 logged). New cache-only entrypoint requires every sensor cache and its exact receipt; it cannot regenerate missing sensor observations. Model initialization and the entire head forward AST match the frozen original exactly. Full resource checks now precede bounded batches of 128 cached observations or 32 heads, with deadline checks every item, unchanged memory thresholds and 1 GiB CUDA allocator cap, and final full admission. This is an explicit cadence change, not a claim of unchanged per-item supervision. Six guard tests and Ruff/Pyright pass. Sixteen hash-selected pre-existing query outputs must reproduce every saved array bit-for-bit before the final seal. Original frozen sources remain unchanged. Execution commit 65d1b92; original logs and interrupted receipts retained.


## Prediction cache integrity incident 2026-10-12T16:18:21.185528+00:00

Both 5805-observation sensor caches restored with exact hashes. Frozen head recovery stopped when a saved prediction did not match its receipt. Full audit found eight mismatches among890 saved predictions, clustered at the prior interruption; cause remains undetermined. All eight NPZ/receipt pairs preserved in quarantine, 882 valid predictions retained. Restarted the same cache-only runner to recompute affected and missing predictions. No expected hash was weakened or replaced with an observed corrupt hash. No optimizer updates. The stopped execution and original parity selection remain archived.


## Idle supervisor coordination 2026-10-12T16:25:43.506584+00:00

V18 run retained a CUDA context while WAIT_P_SENSOR; this is not training. Added a frozen-inference-only admission layer retaining native budget/deadline/RAM/disk guards and >=6GiB free / <=1GiB own CUDA. Only the exact V18 run module may coexist, with source-bound ownership code, state <=120s old, WAIT_P_SENSOR or GPU_OWNER_WAIT, and all GPU reservations settled. Actual workers, stale state, source drift or reservations stop execution. No V18 file/process/lock was changed. Eleven guard tests pass; live admission passes after the transient foreign process exited. Old failed recovery receipts are preserved. GPU profiling uses the same coordination and explicitly reports the idle context and concurrent CPU-load scope.


## Completed frozen inference and local evaluation 2026-10-12T16:39:33.284887+00:00

All6762 test12 tokens sealed; all have native RGB inputs. Official public parser PASS and exact CSV contents. Sixteen hash-selected prior query arrays reproduced bit-for-bit. All eight corrupted NPZ files regenerated to the exact original expected SHA-256, with valid CRC; quarantine preserved. No private labels or submission. FCWD GPU profile completed64 calls with fresh sealed-equivalent registration, exact states and predictions: p50 1213.117499999953ms, p95 3060.5612000000747ms, head p50 0.5232500000147411ms. Scope excludes detector/startup/persistent cross-query state and admits concurrent CPU preparation. Training remains32 complete fits /80000 updates /zero recovery. Official access retry returned no recognizable leaderboard; no score claimed. Immutable bundle verification follows.
