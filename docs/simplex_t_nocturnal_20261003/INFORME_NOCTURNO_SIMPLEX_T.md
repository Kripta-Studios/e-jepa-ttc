# Autoridad operativa vigente de esta entrega

Continuación autorizada: 2026-10-03T15:44:42.346562+00:00 → 2026-10-04T01:44:42.346562+00:00; presupuesto propio 10 GB decimales. La ventana histórica y el protocolo científico conservan sus bytes. El push posterior a la entrega está autorizado.

Inventario físico: 24 endpoints completos, 60000 updates científicos guardados; 0 updates pendientes. Los resultados numéricos y criterios permanecen los del protocolo.

El ZIP incluye el bundle H16 independiente previamente regenerado, con sus seis cabezas y cachés; conserva su SHA-256 original. Las cabezas C0 publicadas se regeneran de nuevo desde la extracción del bundle actual. No se reconstruyen datos raw/TRAIN ni expertos.

NEXT_DECISION_NOCTURNA.json contiene la autoridad vigente; el detalle del publicador canónico que sigue conserva también metadatos históricos.

---

# Informe nocturno SIMPLEX-T

Estado de terminación: **QUEUE_COMPLETE**.

24/24 endpoints completos; 60000/60.000 updates científicos guardados. Trabajo perdido/repetido/incierto: cota superior 100 updates.
Ventana: 2026-10-03T17:44:42.346562+02:00 a 2026-10-04T03:44:42.346562+02:00; Europe/Madrid.

T6 queda cerrado e intacto. El candidato histórico continúa siendo TPR-D1-H8-C160. No se han abierto holdouts ni realizado push.

## Resultados y alcance

- N1: {"status": "COMPLETE", "pending_comparators": [], "result_path": "C:\\Users\\Álvaro Schwiedop\\Desktop\\KriptaStudios\\EVOCON_JEPA_Codex_Handoff\\e-jepa-ttc-v10-simplex-t-companion\\artifacts\\simplex_t\\h16_replication_20261003\\analysis\\RESULTS.json", "sha256": "46f876a11f7508b2c65855b25f60fc6a55b0b2c8a1e5275f2e85e5f1cc6f0d45", "conclusion": "POSITIVE_LOCAL_OLD_DEV"}

| Contraste | Candidato MiD | Referencia MiD | Delta de pérdidas | CI95 jerárquico | CI95 secuencias |
|---|---:|---:|---:|---|---|
| 7 | 118.865738910 | 121.646332258 | -2.780593347 | [-5.8793891043486735, 0.3014086898835507] | [-5.04072752615747, -0.297714448636225] |
| 13 | 119.696801787 | 121.840952046 | -2.144150259 | [-5.3172817792701785, 0.7117871784090213] | [-4.254618845589939, 0.0019846953258013914] |
| 23 | 119.161526342 | 121.864780635 | -2.703254293 | [-5.514432149695562, -0.07054196680497006] | [-4.731992735328027, -0.5523821280150378] |
| new13_23 | 119.429164065 | 121.852866341 | -2.423702276 | [-5.244259454753143, 0.04109148518456987] | [-4.3576685199511545, -0.4869114246067176] |
| all7_13_23 | 119.241355680 | 121.784021646 | -2.542665966 | [-5.2133978893639155, -0.11124053108043853] | [-4.364446908410385, -0.6226868591014174] |

Seed7 es exploratoria ya observada; seeds13/23 son nuevas. Los resúmenes new13_23 y all7_13_23 promedian pérdidas emparejadas, nunca predicciones TTC.
- N2: {"status": "COMPLETE", "pending_comparators": [], "result_path": "C:\\Users\\Álvaro Schwiedop\\Desktop\\KriptaStudios\\EVOCON_JEPA_Codex_Handoff\\e-jepa-ttc-v10-simplex-t-companion\\artifacts\\simplex_t\\nocturnal_20261003\\execution\\analysis\\N2\\RESULTS.json", "sha256": "500707893a0f17f5f512b51e0cf69d53e8696885f3a3f9109cf555767b91fdef", "conclusion": "EXPLORATORY"}

| Contraste | Candidato MiD | Referencia MiD | Delta de pérdidas | CI95 jerárquico | CI95 secuencias |
|---|---:|---:|---:|---|---|
| FULL_C0_minus_H8@7 | 121.404706047 | 121.646332258 | -0.241626210 | [-0.6210695932243308, 0.14280577379537274] | [-0.5289332203496362, -0.00043319786180844185] |
| A5_ONLY_C0_minus_FULL_C0 | 127.785376351 | 121.404706047 | 6.380670303 | [-0.09726888418760105, 13.804899200544122] | [1.0923189972123077, 12.37371718445006] |
| C2F_ONLY_C0_minus_FULL_C0 | 126.704974218 | 121.404706047 | 5.300268171 | [-0.1472445951258283, 11.033927080586398] | [0.8239553422182134, 10.042429088992053] |
| A5_PAIR_C0_minus_FULL_C0 | 124.668731524 | 121.404706047 | 3.264025477 | [-2.4680534514039114, 8.889672735990562] | [-1.491152819380906, 8.157581163610807] |

MiD de todos los brazos disponibles, sin selección por score:

- A5_ONLY_C0: 127.785376351.
- A5_PAIR_C0: 124.668731524.
- C2F_ONLY_C0: 126.704974218.
- FULL_C0: 121.404706047.
- H8@7: 121.646332258.
- N3: {"status": "COMPLETE", "pending_comparators": [], "result_path": "C:\\Users\\Álvaro Schwiedop\\Desktop\\KriptaStudios\\EVOCON_JEPA_Codex_Handoff\\e-jepa-ttc-v10-simplex-t-companion\\artifacts\\simplex_t\\nocturnal_20261003\\execution\\analysis\\N3\\RESULTS.json", "sha256": "36a3eb253b477f2c12aab1afc12bf9c5cf95b02d341e6e3abec30989f7ee61ed", "conclusion": "EXPLORATORY"}

| Contraste | Candidato MiD | Referencia MiD | Delta de pérdidas | CI95 jerárquico | CI95 secuencias |
|---|---:|---:|---:|---|---|
| SET_AGE_C0_minus_FULL_C0 | 122.442604226 | 121.404706047 | 1.037898179 | [-3.0355556542166293, 4.6344481546170835] | [-2.248812177406003, 3.418927051935405] |
| SET_NOTIME_C0_minus_SET_AGE_C0 | 123.686599858 | 122.442604226 | 1.243995632 | [-0.6957483995635467, 2.6766788018183627] | [0.0927967215074051, 2.2638898416205233] |

MiD de todos los brazos disponibles, sin selección por score:

- FULL_C0: 121.404706047.
- H8@7: 121.646332258.
- SET_AGE_C0: 122.442604226.
- SET_NOTIME_C0: 123.686599858.

Las cifras nuevas se publican únicamente desde predicciones de endpoints2500 congelados. Las familias incompletas y métricas ausentes no se completan con estimaciones. Las comparaciones son exploratorias sobre OLD_DEV reutilizado, con nueve secuencias independientes; las seeds son réplicas de cabezas, no de todos los expertos.

El inventario acotado BASELINE_REGISTRY.csv registra 22 comparadores y distingue contrato, información disponible y evidencia insuficiente. Una mejor cifra de otra cohorte no establece superioridad comparable.

## Precisión y coste

N4: {"prepared_head": "COMPLETE", "models": 9, "missing_head_models": [], "full_route": "NOT_MEASURED", "missing_dependency": "Explicit exclusive GPU/heavy-I/O slot from the shared resource owner; no such lease is present in the acknowledged interface.", "full_route_admission_sha256": "8876d1ed3e853310dc0c518ef325681e45545b0ce1e8e71ea1e6dfcae85ffbe1", "route_cost_does_not_follow_from_head_latency": true, "profile_receipts_reconciled_without_new_measurements": true}

HEAD_COST.csv, cuando existe, mide sólo la cabeza con entradas preparadas, batch1 y batch128 por separado. No mide el coste del contexto del ROI actual ni permite afirmar ahorro del sistema, reacción AEB más rápida, tracking persistente o incertidumbre calibrada.

## Contabilidad y recuperación

ENDPOINT_INVENTORY.json conserva cada checkpoint y su SHA-256. Los journals PHYSICAL_WORK.json distinguen updates guardados y trabajo incierto. La prueba real H16 restaura el checkpoint100 y continúa desde101 sin entrenamiento duplicado. Las pruebas sintéticas del estudio nuevo tienen contabilidad separada y no son resultados de eAP.

Se conserva CPU FP32/batch128/cuatro threads/interop2. RAM disponible mínima2GiB, RSS máximo4GiB y margen10GB tras reserva1GiB; techo de artefactos propios10GB decimales. La revisión operativa pre-update preservó v1 y no cambió la receta H16.

## Bundle y regeneración

El manifiesto relaciona bytes y SHA-256 de todos los miembros; el ZIP se verifica por CRC y extracción independiente. Los inputs normalizados y pesos incluidos permiten regenerar las cabezas incluidas. No contiene todos los datos raw/TRAIN ni pesos de expertos, y no promete reconstrucción desde eventos.

Las pruebas/recibos declaran el alcance exacto comprobado. Si no hay cabezas completas/publicadas, la comprobación del bundle es de integridad de archivos y checkpoints recuperables, sin scores científicos nuevos.

## Pendiente


No quedan fits autorizados pendientes. No se debe relanzar el trainer.

## Latencia medida con entradas preparadas

| Modelo | p50 ms | p95 ms | Medidas batch1 | Parámetros |
|---|---:|---:|---:|---:|
| A5_ONLY_C0 | 1.738050 | 23.474165 | 500 | 313765 |
| A5_PAIR_C0 | 15.841700 | 36.169620 | 500 | 313765 |
| C2F_ONLY_C0 | 1.787950 | 2.502355 | 500 | 313765 |
| FULL_C0 | 18.375500 | 40.654565 | 500 | 313765 |
| H16_SEED7 | 25.210700 | 48.494380 | 500 | 313765 |
| H1_SEED7 | 8.038950 | 14.644925 | 500 | 313765 |
| H8_SEED7 | 15.858200 | 32.898410 | 500 | 313765 |
| SET_AGE_C0 | 8.813350 | 17.920275 | 500 | 289765 |
| SET_NOTIME_C0 | 8.784200 | 15.534660 | 500 | 289765 |

Estas medidas incluyen la emisión TTC de la cabeza; excluyen generar contexto y ejecutar expertos. COST_ACCURACY.csv aplica el cribado de precisión/p95 con el mismo alcance y conserva system_substitution_supported=false.


# Lectura de los resultados cerrados

N1: H16 obtiene 119.241355680 MiD en la media de pérdidas de tres seeds, frente a 121.784021646 de H8; delta -2.542665966, CI95 jerárquico [-5.2133978893639155, -0.11124053108043853]. Las tres diferencias por seed son favorables. Seed7 era exploratoria y ya observada. El resumen de las dos seeds nuevas tiene delta -2.423702276 y CI95 jerárquico [-5.244259454753143, 0.04109148518456987], que incluye cero: estabilidad local de la optimización y incertidumbre entre escenas deben mantenerse separadas. No es confirmación fresca ni promoción del H16.

N2: FULL_C0 obtiene 121.404706047 MiD. A5_ONLY_C0, C2F_ONLY_C0 y A5_PAIR_C0 obtienen, respectivamente, 127.785376351, 126.704974218 y 124.668731524. Las tres reducciones fallan el cribado prospectivo de precisión. Se conserva también el contraste FULL_C0 frente a H8: no se atribuye a quitar expertos el cambio del objetivo auxiliar.

N3: SET_AGE_C0 obtiene 122.442604226 MiD y SET_NOTIME_C0 123.686599858. SET_AGE menos FULL_C0 tiene delta 1.037898179, CI95 jerárquico [-3.0355556542166293, 4.6344481546170835]. SET_NOTIME menos SET_AGE tiene delta 1.243995632, CI95 jerárquico [-0.6957483995635467, 2.6766788018183627]. Ninguno pasa el límite superior CI95 <+2 MiD fijado para cribado ingenieril. Esto no demuestra equivalencia ni ausencia de efecto; no se habilitan más seeds o updates para rescatarlo.

N4: hay nueve perfiles completos, 500 mediciones batch1 cada uno, con TTC canónico incluido. PROFILE_VARIABILITY.csv conserva dispersión y extremos de los tiempos crudos. Se reutilizaron perfiles válidos tomados en distintos estados de memoria/carga del host. Incluso las cabezas GRU con idéntica capacidad muestran diferencias grandes: no se atribuye causalmente esa variación a quitar un experto. Los ratios publicados describen estas mediciones de cabeza; no son coste del sistema ni ahorro energético. La falta de slot exclusivo GPU/lectura pesada impide medir la ruta integral. No se decide sustitución del sistema. Las 64 entradas TRAIN normalizadas del perfilado y los pesos están exportados sin targets para repetir la medición desde el bundle.

T6 permanece cerrado y H8 conserva su registro. H8 mejora OLD_DEV; las réplicas corresponden a cabezas, no a todos los expertos. El contexto pasado aporta información, pero el mecanismo cronológico no queda demostrado: los expertos pueden conservar información temporal incluso en SET_NOTIME. No se acredita AEB más rápida, tracking persistente o incertidumbre calibrada. Hay nueve secuencias independientes, no27 escenas por combinar seeds. Todos los contrastes nuevos son exploratorios sobre OLD_DEV reutilizado.

La única propuesta posterior está en NEXT_EXPERIMENT_NOCTURNA.md; no se ha ejecutado.
