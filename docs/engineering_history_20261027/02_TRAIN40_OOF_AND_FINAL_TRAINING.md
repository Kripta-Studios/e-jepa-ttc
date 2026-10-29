# TRAIN40, rotaciones OOF y entrenamiento del candidato final

Fecha de corte: 27 de octubre de 2026. Este capítulo describe el sistema realmente sellado y distingue sus pesos finales, los modelos auxiliares usados para producir entradas de entrenamiento y los planes históricos que lo precedieron. Para los resultados de cada etapa del proyecto, véase [la cronología científica](01_SCIENTIFIC_CHRONOLOGY.md).

## 1. Familia elegida y población autorizada

El candidato final pertenece a la familia PHASE_ONLY derivada de V16, semilla 7, RGB más eventos, sin ensemble. Se eligió una interfaz de fases y tiempos con semántica común entre folds. Esta decisión no demuestra que V16 sea la mejor arquitectura posible ni altera el `NO_CANDIDATE` de V18. [FAMILY_COMPARISON.md](../final_refit_20261017/FAMILY_COMPARISON.md) conserva la comparación con C1 + R_CTX y sus limitaciones.

La población de ajuste es el TRAIN40 público disponible: **88.744 consultas de 40 secuencias**. Las antiguas P/H/V tienen 53.890/18.422/16.432 consultas y 24/8/8 secuencias. Después del refit sus métricas son diagnósticos sobre datos de entrenamiento, no validación independiente del modelo final. Las secuencias nominales adicionales de publicaciones anteriores no se consideran utilizadas sin medios/anotaciones públicos y trazabilidad efectivos.

EvTTC, FCWD y test12 no forman parte de esta población de ajuste. El manifest nativo tiene SHA-256 `0e59ce2629f512f68dff6ab77beef68090816adac7adf21994cb29fa6954859c`. La [auditoría de leakage](../final_refit_20261017/LEAKAGE_AUDIT.md) verificó identidad de filas y grupos, 1.812 tracks asociados cada uno a una secuencia y ausencia de intersección de tokens/secuencias TRAIN40–test12. Esto comprueba los contratos concretos inspeccionados; no transforma el desarrollo histórico ya observado en un test ciego.

## 2. Grafo desplegable: nueve componentes entrenables

```text
Eventos                                      RGB
  ├── E_A5 ── PAIR_E                          ├── R_A5 ── PAIR_R
  └── E_C2F                                  └── R_C2F
       │                                          │
  PHASE17 e historial                        PHASE17 e historial
  normalizador eventos                       normalizador RGB
       │                                          │
     E_CTX                                      R_CTX
       └──────── fases y tiempos físicos ──────────┘
                          │
                     PHASE_ONLY
                          │
                     TTC firmado
```

Hay cuatro productores, dos PAIR, dos refinadores de contexto y una cabeza final. Los normalizadores son artefactos adicionales con identidad propia. A5 y C2F son productores diferentes; C2F designa la ruta coarse-to-fine. Las letras E/R distinguen eventos y RGB. PAIR se entrena sobre las representaciones de su A5 correspondiente; no debe recibir arbitrariamente los tokens de otro encoder.

La cabeza PHASE_ONLY conserva las veinte coordenadas físicas/temporales de V16 y su ancla R_CTX. Los 1.955 parámetros de la cabeza histórica no representan la cadena completa. El sistema precisa sus productores y contextos para calcular esas coordenadas. En la interfaz heredada de 340 dimensiones, las coordenadas ocultas no consumidas se rellenan de forma declarada; no se pretende alinear latentes independientes mediante ese relleno.

## 3. Qué problema resuelve el OOF por grupos

Entrenar una cabeza sobre salidas de productores que ya vieron los targets de todas sus consultas genera stacking in-sample. Puede declararse como una receta, pero no puede llamarse fuera de muestra. En este refit se eligieron cinco rotaciones por grupos para reducir ese riesgo en las entradas superiores.

Los bloques son de ocho secuencias cada uno. B0 conserva V; B1 conserva H; B2/B3/B4 particionan P de forma determinista equilibrando números de consultas, sin consultar valores TTC ni resultados de modelos. Los IDs concretos están en [TRAIN40_GROUPS_AND_FOLDS.json](../v16_train40_refit_20261017/TRAIN40_GROUPS_AND_FOLDS.json). Se preserva la unidad secuencia, no se reparten ventanas de una misma secuencia entre roles.

En cada rotación:

1. Veinticuatro grupos entrenan los cuatro productores y los dos PAIR.
2. Ocho grupos distintos entrenan E_CTX/R_CTX y sus normalizadores auxiliares.
3. Los ocho grupos restantes generan las entradas OOF. Ningún ancestro supervisado de esas entradas ha entrenado sobre sus grupos.

La exclusión concierne a toda la genealogía, no sólo a la última cabeza. Una caché con un nombre de fold correcto pero generada por el productor TRAIN40 completo no sería OOF. Los adaptadores recalculan las columnas aprendidas con los padres de la rotación y admiten artefactos únicamente con bindings de población y modelos coincidentes.

El material de entrada puede contener targets de TRAIN en el archivo contenedor; no se declara que el archivo carezca de ellos. Se controla qué columnas consumen los modelos, y las etiquetas se unen explícitamente por ordinal, token y grupo para el ajuste autorizado. La barrera de test12 es diferente: allí no se leen TTC privados.

## 4. Por qué se unen fases físicas y no tokens de encoders independientes

Los encoders de dos rotaciones pueden aprender bases latentes distintas. Concatenar token128 de folds independientes y ajustar una única cabeza como si cada coordenada tuviera el mismo significado sería una hipótesis adicional sin verificar.

La frontera elegida es **PHASE17**, cuyos campos tienen significado físico/diagnóstico común. Para los refinadores finales se unen las features OOF sin normalizar y se ajusta un normalizador global sobre esas entradas TRAIN40. Cada normalizador auxiliar sólo utiliza sus ocho grupos de ajuste. PHASE_ONLY se entrena sobre las veinte coordenadas de fases/tiempos producidas fuera de muestra por los refinadores auxiliares.

El grafo de generación de entradas de entrenamiento es distinto del grafo de despliegue. Al inferir se ejecutan los cuatro productores TRAIN40 finales y las cinco cabezas finales; no se ejecutan cinco rotaciones ni se elige la que puntúa mejor. Los modelos auxiliares son herramientas de construcción del entrenamiento, no un ensemble oculto.

Existe un cambio de distribución explícito: los productores auxiliares vieron 24 grupos y los finales 40. El OOF reduce un riesgo del stacking, pero no garantiza que el refit mejore, ni demuestra causalmente que cualquier diferencia se deba sólo a «más datos». Además, las decisiones históricas sobre arquitectura ya utilizaron desarrollo. [CRITICAL_REVIEW.md](../final_refit_20261017/CRITICAL_REVIEW.md) registra estos límites antes de la ejecución final.

## 5. Exposición científica registrada

Los productores usan semilla 7, inicialización de la receta admitida sin cargar un checkpoint TTC arbitrario, 18 épocas y batch efectivo 32. Para los productores finales:

\[
U=18\left\lceil\frac{88.744}{32}\right\rceil=49.932.
\]

El número 49.932 es una consecuencia de exposición y tamaño de batch, no una interrupción a mitad de entrenamiento. Los checkpoints reales finales registran 49.932 updates, scheduler en época 18 y cursor preparado para época 19. Se selecciona el endpoint final de la receta, sin buscar un checkpoint según test12.

El plan de las rotaciones se conserva en [SCIENTIFIC_EXPOSURE.json](../../artifacts/final_refit_20261019_scientific_oof/SCIENTIFIC_EXPOSURE.json):

| Rotación | Consultas productores/PAIR | Consultas refinadores | Consultas generación | Updates por productor | Updates por PAIR | Updates por contexto | Total por rotación |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 | 53.890 | 18.422 | 16.432 | 30.330 | 6.840 | 2.500 | 140.000 |
| R1 | 52.393 | 17.929 | 18.422 | 29.484 | 6.650 | 2.434 | 136.104 |
| R2 | 52.551 | 18.264 | 17.929 | 29.574 | 6.671 | 2.479 | 136.596 |
| R3 | 52.783 | 17.697 | 18.264 | 29.700 | 6.700 | 2.402 | 137.004 |
| R4 | 54.615 | 16.432 | 17.697 | 30.726 | 6.933 | 2.230 | 141.230 |

Cada total suma cuatro productores, dos PAIR y dos contextos. Los cinco bloques de generación cubren exactamente las 88.744 consultas. No se entrena una cabeza PHASE_ONLY temporal en cada fold: se generan sus entradas.

Las cabezas finales escalan la exposición original con la población admitida:

\[
U_{PAIR}=\left\lceil6840\frac{88744}{53890}\right\rceil=11264,
\qquad
U_{CTX/PHASE}=\left\lceil2500\frac{88744}{18422}\right\rceil=12044.
\]

| Componentes finales | Cantidad | Updates cada uno | Batch efectivo | Estado final |
|---|---:|---:|---:|---|
| E_A5, E_C2F, R_A5, R_C2F | 4 | 49.932 | 32 | COMPLETE, reutilizados como padres TRAIN40 admitidos |
| PAIR_E, PAIR_R | 2 | 11.264 | 32 | COMPLETE |
| E_CTX, R_CTX, PHASE_ONLY | 3 | 12.044 | 128 | COMPLETE |

Las cinco cabezas finales suman 58.660 updates. Los endpoints auxiliares suman 690.934; el plan de cola del 19 de octubre fija **749.594 updates nuevos en 45 ajustes**, además de los cuatro productores finales ya admitidos o en terminación. Esta cifra es la exposición científica planificada por endpoints, no una medición de operaciones físicas, horas de GPU o trabajo repetido tras fallos. No se convierte automáticamente en un contador observado de GPU sumando recibos duplicados.

El plan conceptual anterior consideraba 49 ajustes contando todos los componentes y una posible reutilización P/H/V histórica; con otras reutilizaciones calculaba 39 nuevos/709.458 updates. La alternativa secuencial calculaba 158.524 updates nuevos incluyendo dos RGB. Son versiones de planificación con distintos supuestos. La cola científica posterior es la referencia operativa para las cinco rotaciones; no se suman las tres cifras ni se atribuye su diferencia a pérdida de entrenamiento.

## 6. Pérdidas, scheduler y procedencia

Se conservaron las recetas de origen por etapa. No se trasladó la ponderación LW de V19 a los productores V16 ni se usó la función de ranking 0,5/0,3/0,1/0,1 como sustituto universal de la pérdida de entrenamiento.

La metadata inspeccionada de los productores event declara supervisión foreground `bbox_geometry`, representación `dinov3_local_relational`, peso de destilación 8,0 y teacher ligado por hash. Incluye tres épocas de foreground warmup, learning rate inicial 0,0003, mínimo 0,00003, weight decay 0,0001, clipping de gradiente 1,0 y BF16. El checkpoint registra selección `last_epoch`; aunque conserve campos genéricos de paciencia, el endpoint verificado es de 18 épocas completas.

Esto no es preentrenamiento exclusivamente autosupervisado sin TTC. Se conserva la supervisión genérica de DINO y la de la tarea según la receta. El teacher utilizado en entrenamiento no debe confundirse con una entrada requerida del sistema desplegado.

Los PAIR finales mantienen weight decay 0,0001; los contextos y PHASE_ONLY, 0,001 según la receta sellada. Las cabezas se entrenan en FP32. En contextos/PHASE_ONLY, el warmup se escala a 482 updates y el scheduler termina en 12.044. La exposición y el scheduler se ligan al número de consultas de cada etapa; no se mantiene silenciosamente un scheduler corto sobre una población mayor.

La identidad canónica de receta es `78c83c09fc176846e1135df3b625a2455b71336becb9305589f55269058915c3`. El SHA del archivo serializado OOF_RECIPE es distinto de esa identidad canónica: ambos identifican objetos diferentes y no deben intercambiarse al verificar un bundle.

## 7. Reutilización comprobada de los cuatro productores finales

La auditoría [DEPLOYMENT_TRAIN40_SCOPE.json](../../artifacts/final_diagnostics_20261027/DEPLOYMENT_TRAIN40_SCOPE.json) abrió los checkpoints finales en CPU y contrastó identidad, exposición y hashes. No dedujo el alcance TRAIN40 del nombre de una carpeta o de la última cabeza.

Los event originales proceden de `../e-jepa-ttc-v12-efficient-context/artifacts/train40_system_20261005/fits/a5_seed7/checkpoint_last.pt` y `c2f_seed7/checkpoint_last.pt`. Su identidad declara `public_TRAIN40_all_train`; el manifest ligado contiene 88.744 filas/40 secuencias y coincide con el parquet nativo final. Sus registros de inicio son del 6 de octubre, no del día 27.

Los RGB finales proceden de `artifacts/final_refit_20261018_runtime_v6/fits/R_A5/checkpoint_last.pt` y `R_C2F/checkpoint_last.pt`. Su identidad liga las poblaciones P/H/V completas y disjuntas al mismo TRAIN40. El entrenamiento científico RGB comenzó el 17 de octubre; las fechas de inicio del runtime_v6 del 18 son fechas de una ejecución recuperada/migrada, no un reinicio científico desde cero. Los endpoints se alcanzaron el 19 de octubre.

No se reutilizaron PAIR/H8 de V12 como si sus conversiones y normalizaciones fueran V16. Se reconstruyeron las dependencias superiores compatibles. Del mismo modo, los productores completos de despliegue no se utilizaron para fabricar entradas supuestamente OOF de sus propios grupos.

## 8. Identidad final de los nueve pesos

Los archivos portables están en `D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE/checkpoints`. El sello y los recibos conservan también sus rutas de origen remotas. Los hashes completos permiten reconocerlos tras cualquier traslado de Windows a Linux o entre Pods.

| Componente | SHA-256 del checkpoint |
|---|---|
| E_A5 | `a82e5b65ec9c3d302981438527f2cfccabbbf25c75f574946bd1e9dc97795758` |
| E_C2F | `2b337dd76bd405c63ac9f96023a82196ee32e8180b86fe0015bfd67f0e2aeb57` |
| R_A5 | `3f09a2e06d4ff6aebaa64d8c1a8ccc0ade54eaa25b54bc11700fd70bd9701f29` |
| R_C2F | `d5fa8fd9ed80df6141c3df8bd2ed8ac013c87e5e5bde7367a3b588c5b09e5a35` |
| PAIR_E | `ccdd694f5d812a458c396fdb19bbffe439eff78fe38d57a7b288d4a4fd232680` |
| PAIR_R | `dba94f1ae807d8e84ef68f50c93882a766a92cfc0b9a08da53c30fedbf509bc1` |
| E_CTX_MATCHED | `fec0ca9f2433a913ae1047cd0b47715a1dce39c4c2e4eecb9539f796434d137e` |
| R_CTX | `a19f8188b859dcac7fb0cd4c68d0cd5c5c9a839ad84c0ac7ee1b3d61181d1852` |
| PHASE_ONLY_EVENT_S7_TRAIN40 | `a24bfc72ca7f53a93d53db806b6866c6a5a4b6abde074b5dfec50436b7005408` |

Normalizador eventos: `19f9bf704de53a938e955ef1919e51881ec1f9c85c1d3e2cfd54aa6d76045f9f`. Normalizador RGB: `d902dba58dc9738ed268e56fe57291fda1cf0dfb2e03be4da1171668e689bd59`. MODEL_SEAL.json: `cf5f4eac22e572679c77cd0f6a16425f978600597d4be77f0b1e576c226165a4`.

El estado de identidad `SEALED_NOT_EVALUATED` del sello se conserva: se creó antes de la puntuación oficial y no debe reescribirse como «evaluado oficialmente» porque se haya ejecutado inferencia o una transferencia EvTTC. Los recibos posteriores describen esas operaciones por separado.

## 9. Ejecución, recuperación y comprobaciones científicas

La [cola OOF](../final_refit_20261017/SCIENTIFIC_OOF_QUEUE.md) se activó el 19 de octubre y posteriormente se repartió entre máquina local y Pods. El lugar de ejecución no cambia los roles científicos. R0/R1 locales y las etapas trasladadas deben reconocerse por sus manifests y modelos, no sólo por el nombre del directorio. La migración y la concurrencia se detallan en [06_RUNPOD_MIGRATION_AND_QUEUES.md](06_RUNPOD_MIGRATION_AND_QUEUES.md).

El checkpoint cada 100 updates se mantuvo por instrucción del usuario. El estado durable incluye pesos, optimizador, scheduler, RNG, sampler y cursor. Una pausa limpia guarda el estado; un apagado abrupto puede perder lo posterior al último checkpoint. La recuperación se contabiliza aparte de la exposición científica admitida. Los fragmentos de features y predicciones se reutilizan sólo después de verificar población, fuente y bindings de padres.

Las optimizaciones técnicas se admitieron con pruebas de valores, gradientes y reanudación; una prueba de forward/backward sin optimizer.step no se contabiliza como entrenamiento científico. El preflight de cuatro productores incluyó ocho forward/backward BF16 y gradientes finitos, sin updates científicos. No se repitieron todas las campañas anteriores para documentar esta entrega.

Los registros de entrenamiento contienen snapshots y copias de RUN_PROVENANCE. Por ejemplo, `HISTORICAL_TRAINING_LEDGER.csv` conserva varias observaciones de un mismo fit. Contar sus filas como entrenamientos distintos o sumar tiempos de runs concurrentes como duración de calendario daría cifras falsas. Tampoco el ritmo del último fragmento de 44/64 updates representa automáticamente la media de todo un entrenamiento.

## 10. Test12: predicción completa, aún sin puntuación oficial

El [recibo de revisión del paquete](../../artifacts/final_diagnostics_20261027/FINAL_PACKAGE_REVIEW.json) verifica:

- Exactamente 6.762 tokens del manifest oficial, sin extras ni ausencias.
- 5.805 consultas físicas, 957 alias y doce secuencias.
- Un único `submission.json` en la raíz, esquema `garlttc_prediction_v1`, valores TTC finitos y sin claves duplicadas.
- Cero fallback, nueve checkpoints y dos normalizadores coincidentes con el sello.
- Soporte temporal observado máximo de 600.006 microsegundos dentro del contrato causal de hasta 650 ms.
- ZIP final SHA-256 `5eb69749de7aff682eea640d843ee635379702337d929c86f50472264f1d88bc`.

El nombre del archivo es `PHASE_ONLY_S7_TRAIN40_SCIENTIFIC_OOF.zip`. No es el ZIP V16 histórico ni una regeneración escogiendo la semilla más favorable. Las predicciones se conservaron exactamente. Al releer el CSV de diagnóstico en float64, 589 valores difieren del JSON hasta 7,105427357601002e-15 s por roundtrip decimal; todos son idénticos en float32. Esa diferencia se documentó sin modificar el ZIP.

La revisión del formato no certifica la precisión ni resuelve por sí sola la admisibilidad del historial temporal. El texto público aportado no imponía explícitamente 200 ms como límite, pero tampoco constituía una confirmación explícita de todo el contrato de ROI/contexto de este sistema. No hubo acceso a etiquetas privadas ni envío a CodaBench en esta ejecución.

## 11. Qué preserva el cierre y qué falta para otras afirmaciones

El bundle local conserva checkpoints, normalizadores, receta, seal, entradas y salidas necesarias, fragmentos y recibos de inferencia, curvas y diagnósticos, junto con manifests de los archivos archivados. La validez científica procede de las identidades y contratos; la posibilidad práctica de reanudar procede además de conservar los estados completos y los datos requeridos. No se borra una fuente remota basándose sólo en que el ZIP existe.

La comparación posterior en [EvTTC](../final_train40_transfer_20261027/REPORT.md) no ajustó pesos ni eligió checkpoints. Confirma resultados de transferencia de la cadena sellada sobre una cohorte registrada, con cobertura y límites explícitos. R_CTX obtiene menor error medio agrupado que la fusión en esas cohortes; ese resultado permanece visible y no cambia retrospectivamente la receta.

Para afirmar SOTA oficial falta una puntuación oficial bajo las reglas aplicables y una comparación del alcance correspondiente. Para afirmar tiempo real de producto falta medir el pipeline desplegado con detección/tracking, entradas, hardware y precisión definidos. Para afirmar incertidumbre calibrada o cumplimiento de todos los objetivos originales de AGENTS.md hacen falta sus evaluaciones específicas. El cierre TRAIN40/OOF acredita entrenamiento, genealogía, inferencia y preservación de esta versión concreta; no sustituye esas evidencias.
