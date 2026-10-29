# Experimento primario: medición espacial de pares con representación entrenable

**Propuesta nueva, no ejecución.** Cuatro fits de 12.000 updates, seed7. No se exige que un componente congelado pase los gates finales para ejecutar su alternativa entrenable.

## Entradas y roles

Reutilizar P/H/V de V13, sus parent bindings y fuentes sensoriales sin introducir pesos TRAIN40 que hayan visto H/V. P=53.890, H=18.422 y V=16.432 en los manifiestos revisados; el adaptador debe confirmar la identidad, no corregir números a mano.

Los modelos principales de esta matriz reciben las DOS últimas ventanas event de la consulta nativa, en el mismo ROI común y con sus relojes reales. No redimensionar cada caja independientemente ni fabricar trayectorias. Mantener 12 canales/ventana, 128×128 y dt medido. Si la fuente contiene T3 se selecciona el sufijo T2 y sus targets de geometría válidos cuando correspondan. Los targets TTC siguen en el anchor final auditado. Fuente de datos: `EventProducerSource` y ruta lossless ya validada, NO la caché de features de V17.

La selección P2 es una intervención declarada respecto al A5/H8 T3/contexto históricos. Crear predicciones de A5 original P2 sin entrenamiento para referencia. No comparar P2 contra T3 y atribuir todo a los pesos. Garl usa sus ventanas/canales nativos: compartir información física no significa igual representación numérica.

## Matriz cerrada

| ID | Encoder | Salida | Entrenamiento |
|---|---|---|---|
| NATIVE-P2-FT-S7 | A5 matched P inicial, todos sus módulos entrenables | Ruta nativa de altura/residual/TTC | 12.000 updates con objetivo nativo de productor y supervisión P nativa |
| PAIR-DIRECT-S7 | Clon de `A5.encoder.features`, entrenable | Predicción de fase desde mapas de dos instantes; correlación anulada | 12.000 updates de objetivo de pares |
| PAIR-CORR-S7 | Mismo clon, entrenable | Mismo predictor, con correlación espacial | 12.000, mismos draws/objetivo/estado inicial que DIRECT |
| PAIR-CORR-FROZEN-S7 | Mismo clon, congelado/eval | Mismo predictor CORR | 12.000 updates del readout, sin tocar encoder |

Comparación primaria: CORR−DIRECT. Segunda: CORR−CORR-FROZEN. NATIVE-P2-FT es un control de continuación nativa con pérdidas diferentes; no aislar la arquitectura frente a él. El A5 original sin updates y los sistemas E_CTX/F_TRUE vigentes conservan identidad.

## Arquitectura compacta propuesta

Reutilizar por copia privada la CNN densa nativa, no sus geometry_tokens, PAIR o foreground como entradas ocultas. Mapas por imagen [64,32,32]. Torre de par recibe F_prev, F_curr, diferencia, producto, coordenadas normalizadas del ROI, dt, y un bloque explícito de correlaciones y máscaras.

Correlación por coseno, radio2 (25 offsets) en 32×32 y en 16×16; ambos sentidos. Conservar las 100 similitudes y sus máscaras antes del resumen. No sustituirlas por flow medio o por un softmax que oculte la falta de coincidencia absoluta. Las coordenadas son relativas al ROI, no posición/altura anotada del objeto.

DIRECT tiene la misma anchura, módulos y máscaras de soporte, pero similitudes cero. No se da RGB, box3d, depth, TTC, IDs ni gate derivado de etiquetas como entrada. Las máscaras de límites dependen de la geometría de la malla, no de targets.

Proyección1×1→96, dos bloques residuales espaciales96, pooling medio/máximo, MLP→fase. No se materializa correlación all-pairs 4D sobre todos los píxeles de toda la historia. La referencia es deliberadamente pequeña y prueba el mecanismo, no un reemplazo de RAFT ni optical flow completo.

Todas las características entrenables deben tener gradientes no nulos en un fixture adecuado; verificar cambios reales de `encoder.features` tras updates. Un forward cacheado con `detach()` o `inference_mode()` en CORR/DIRECT es un fallo de implementación. CORR-FROZEN conserva el encoder en eval aun cuando se llama model.train(). DIRECT/CORR entrenables comparten dropout/RNG; congelado difiere además en política train/eval del encoder, declarado.

## Salida y pérdida nuevas

Predecir fase canónica phi_raw, no segundos ni log-alturas independientes. Evaluar la salida realmente emitida con el decoder firmado nativo (incluidos cap y dominios), sin buscar otro clipping en H/V.

Para evitar que la pequeña región de derivada nula del decoder elimine toda señal, la nueva pérdida de par es:
`L_i = abs(phi_emit−phi_gt)/0.03 + 0.1*abs(phi_raw−phi_gt)/0.03`.
Es una pérdida NUEVA y explícita, no paridad con H8 ni NLL nativa. Publicar ambos términos y la discrepancia raw/emitted. DIRECT/CORR/FROZEN utilizan exactamente la misma. No se añade coste auxiliar, cuantiles ni incertidumbre proclamada calibrada.

Inicialización del readout idéntica entre brazos: pesos de última Linear N(0,1e−3), bias0.03. No iniciar todo el predictor en una fase cero con gradiente emitido nulo. Parámetros de inicialización y todos los RNG se congelan antes del primer update.

Masa del nuevo objetivo: igual masa por grupo P y pesos históricos c/s/l/n renormalizados sólo entre bandas TRAIN presentes, uniforme dentro de banda. Sampler P uniforme con reemplazo, mismos IDs por seed, estimador N*w SIN renormalización por batch. La ruta nativa mantiene su objetivo propio. Contar exposición por query/grupo; no presentar repetición como datos nuevos. Si una fila P no cabe en el contrato, detener y diagnosticar antes de alterar población o soporte.

## Receta inicial fija

GPU FP32, TF32off para el cribado; sin cambios de precisión a mitad. Effective batch32; microbatch4 con acumulación, misma lista de IDs. Para la pérdida nativa usar reducción por elementos válidos y top-k del batch efectivo, no media de microbatches ni top-k locales.

AdamW: backbone1e−4, módulos nuevos3e−4, wd1e−4, betas0.9/0.999, eps1e−8. Warmup200 y cosine hasta 10% de LR a update12.000. Clipping norma1 y registro antes/después. En NATIVE, usar grupos backbone/resto equivalentes y los pesos de pérdida nativa auditados; no cambiar teacher o supervision para solventar una dependencia silenciosamente.

Sin augmentaciones nuevas en la matriz inicial: mantiene los datos y evita mezclar hipótesis. Guardado cada100 updates y en pausa; milestones2k/6k/12k sólo para diagnóstico P y recuperación. El endpoint primario es12k. No elegir el mejor H/V retrospectivamente.

## Evaluación e integración

Congelar los cuatro endpoints antes de su primera evaluación H. H puede seleccionar entre recetas de esta campaña, por lo que pasa a ser desarrollo de selección de productores; no usar sus targets para gradient ni volver a llamarlo independiente. No ha entrenado estos productores.

Evaluar cada productor completo en H, no sólo la pérdida geométrica. Después congelar la lista de hasta DOS candidatos de integración. Un candidato con cobertura completa puede continuar por una de estas evidencias prospectivas: ≥3% mejor overall; ≥5% mejor MiDc sin >2% peor overall; o ≥30% menos coste de forward emparejado con overall dentro de+2%, y sin regresiones de signo >0,5pp ni banda >10%. Son reglas de EXPLORACIÓN, no promoción final ni cambio de gates V17.

Si ningún candidato cumple, pasar a la continuación adaptativa del documento03, no afirmar SOTA ni cerrar todo el programa al primer NO_CANDIDATE. La alternativa entrenable nunca depende del éxito de CORR-FROZEN.

Antes de integrar, medir predicciones standalone en V de la familia congelada y comparadores admitidos. Una cabeza temporal adicional no es obligatoria: el objetivo también incluye un modelo más simple. Si se integra corrección, usar sólo features de padres P y entrenar nuevos readouts/normalizadores en H, con control igual y sin reutilizar las viejas features dependientes de A5.

## Medición de la referencia incluida

Al instanciar la CNN nativa de 12 canales, 64 features y dos bloques residuales, la referencia DIRECT/CORR tiene 578.337 parámetros. En CORR-FROZEN se entrenan 402.145; el encoder congelado conserva 176.192. Es una cuenta del código de referencia, no del pipeline completo histórico ni del trainer final integrado. El probe CPU con tensores sintéticos 128×128 verifica el forward, gradientes hacia el encoder entrenable y ausencia de cambios en el padre original; no hay pesos entrenados A5, datos eAP ni updates en esa comprobación.
