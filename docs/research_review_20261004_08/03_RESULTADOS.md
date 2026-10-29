# Resultados científicos y límites de interpretación

Los resultados pertenecen a tres poblaciones distintas: OLD_DEV histórico, TRAIN40 y transferencia EvTTC. Sus métricas no se mezclan. OLD_DEV fue reutilizado durante desarrollo; TRAIN40 contiene los datos de entrenamiento; EvTTC no se usó para entrenar este modelo final, pero las secuencias tienen historial de uso en desarrollo. Ninguna de estas circunstancias permite llamar a todo el análisis un test ciego nuevo.

## E0: control analítico

El control EWMA/transport no entrenó pesos. Sobre 8.192 consultas y nueve secuencias OLD_DEV obtuvo score 155,442 frente a 121,646 de H8; delta +33,796, con intervalo jerárquico 95% [17,090; 51,042]. No ganó en ninguna de las nueve secuencias. Es un resultado desfavorable observado del control en esa población adaptativa, no una dependencia bloqueada. Los valores son score MiD del contrato local, no segundos de error TTC. Véase [evidencia E0](evidence/e0_e3/EWMA_TRANSPORT_CV_RESULTS.json).

## E1 y E2: preparación congelada y WIDE

E1 conservó pesos para separar preparación de aprendizaje. Las primeras mediciones R0/R2 y su paridad no prueban por sí solas aceleración del sistema completo con ingesta causal. R1 se añadió para resolver esa limitación. Está pendiente en el cierre y no modifica los resultados de precisión ya calculados.

E2 completó nueve fits WIDE de 2.500 updates: 22.500 en total. Las tres semillas dan score medio WIDE **118,954**, H8 **121,784** y H16 **119,241**. El contraste WIDE−H8 es −2,830, IC jerárquico [−6,070; 0,149]; WIDE−H16 es −0,287, IC [−2,061; 1,432]. Ambos intervalos contienen cero. Los análisis de las dos semillas nuevas tampoco convierten el resultado en superioridad confirmada.

Los contratos guardan no inferioridad con margen MiD de 2, pero `candidate_promoted=false` y `confirmation=false`. El mejor valor puntual no autoriza elegir WIDE como ganador definitivo. El bootstrap sólo por secuencia es más favorable frente a H8 que el jerárquico; mostrar únicamente aquel ocultaría incertidumbre entre semillas. Las [tablas WIDE](tables/WIDE.md) y [resultados replicados completos](evidence/e0_e3/WIDE_REPLICATION_RESULTS.json) preservan ambas lecturas y análisis por secuencia.

## E3: productores Garl pausados

Se conservaron 17.948 updates de productores nativos; no se entrenó ninguna de las cabezas Garl H1/H8 previstas originalmente. La estrategia cambió por orden del usuario a comparar con pesos públicos ya terminados. Esa rama queda pausada y supersedida para la comparación actual, no derrotada científicamente. No se comparan checkpoints parciales como si fueran los modelos finales publicados.

## TRAIN40: sistema entrenado y diagnóstico de ajuste

A5 y C2F alcanzaron 49.932 updates cada uno; PAIR, 6.840; las cabezas H8 con semillas 7, 13 y 23, 2.500 cada una. Hay **una semilla del sistema de productores y tres semillas de cabeza**, no tres entrenamientos independientes de todo el sistema.

Las 88.744 predicciones TRAIN40 de cada modelo son finitas. Los MAE H8 son 1,898, 1,816 y 1,763 segundos; el checkpoint público Garl event-only da 3,577. Son resultados de ajuste sobre entrenamiento, no estimaciones independientes de generalización. Los [MAE y RMSE regenerados](tables/TRAIN_METRICS.md) proceden del recibo de campaña.

La conversión nativa de Garl basada en cambio de escala puede generar predicciones muy grandes cuando el denominador se acerca a cero. En TRAIN40 su RMSE ronda 123,235 s; un único caso aporta aproximadamente 83,58% del error cuadrático y los diez mayores, 96,62%. Se conservaron, sin recortar por error. H8 tiene soporte nativo ±60 s y Garl salida no acotada: parte de la diferencia de colas está condicionada por esos contratos. El análisis por signo, TTC absoluto y altura de bbox es descriptivo; no seleccionó otro modelo.

## EvTTC: qué se comparó

Se congelaron 1.024 consultas, 32 por cada una de 32 secuencias de ocho familias. De ellas, 946 tienen target TTC válido en su intervalo temporal; las otras 78 no se puntuaron y no se extrapoló ground truth. Los cinco modelos tienen predicción finita en las 946 y no hubo fallos de disponibilidad de entrada. Las métricas principales usan soporte común idéntico.

H8 recibe eventos más metadata de ROI, con ocho lags entre 350 y 0 ms, representaciones nativas de tres ventanas de 100 ms y 12 canales de 128×128, con padding a los slots canónicos. No recibe píxeles RGB. Garl event-only utiliza dos ventanas de 100 ms y 40 canales de 128×128. Garl full recibe esos mismos 40 canales y dos imágenes RGB, seis canales adicionales. Son contextos nativos distintos; no se cambió arquitectura, ventana o hiperparámetro para favorecer un resultado tras ver etiquetas.

Las cajas se tratan como información oracle y su disponibilidad es causal, con retraso declarado. Se proyectan de RGB a eventos mediante calibración de rotación, sin corrección de profundidad/paralaje. EvTTC usa offset de eventos cero en ese contrato, no el ajuste fijo de cinco píxeles de eAP. Estos modelos no incluyen un detector end-to-end, por lo que no se deriva una latencia o precisión de un sistema de seguridad completo.

Se sellaron predicciones event-only antes del scoring. Para RGB+eventos se reutilizan exactamente los mismos IDs, targets y cuatro predicciones anteriores. Los tensores de eventos se verifican por SHA-256 contra la primera evaluación para cada consulta. La extensión full fue solicitada tras observar event-only: no es un experimento completamente ciego, aunque no se ajustaran pesos.

## Sincronización y checkpoint RGB+eventos

El checkpoint público `paper_ours_full.pth` está fijado a la revisión `b676fcdaf26c04bcf896cdb2b208c9c424e8462a` de [NAIL-HNU/GarlTTC-model](https://huggingface.co/NAIL-HNU/GarlTTC-model/tree/b676fcdaf26c04bcf896cdb2b208c9c424e8462a). Se cargó estrictamente, sin claves ausentes o inesperadas. Las rutas de inicialización de backbones se retiraron porque el estado final ya los contiene; no se reemplazaron los pesos entrenados.

Las imágenes RGB se seleccionan por proximidad a los endpoints de eventos, `sync: front`, con tolerancia de 1 ms y separación de 100±1 ms. La selección inicial basada en timestamp de cajas era incorrecta y se corrigió **antes de inferir y puntuar**. La versión final admite las 1.024 consultas, con desfase máximo absoluto de 453 µs y ninguna espera adicional después del anchor final. Se aplicó normalización ImageNet antes del crop común, conforme al código upstream.

La paridad real CPU/GPU en una consulta dio TTC 5,7414732 frente a 5,74145365 s, dentro de las tolerancias declaradas. No es una afirmación de igualdad bit a bit entre todos los dispositivos. Véanse [paridad](evidence/rgb_event/CPU_GPU_PARITY.json), [freeze](evidence/rgb_event/INFERENCE_FREEZE.json) y [coste/sincronización](evidence/rgb_event/COST_AND_SYNC.json).

## Resultados EvTTC e incertidumbre

La [tabla completa regenerada](tables/EVTTC_METRICS.md) contiene MAE, mediana, RMSE, sesgo e intervalos. H8 da MAE 1,321–1,409 s y RMSE 3,027–3,392 s. Garl full da MAE 1,868, mediana 0,692 y RMSE 7,875 s. Garl event-only da MAE 8,048, mediana 0,997 y RMSE 159,247 s.

H8 presenta menores estimaciones puntuales de MAE/RMSE; Garl full presenta menor mediana frente a las tres cabezas. Eso indica diferencias de distribución, no superioridad uniforme. Los contrastes macro por secuencia H8−Garl full son −0,548, −0,515 y −0,463 s. Sus IC 95% son respectivamente [−1,284; −0,034], [−1,210; −0,006] y [−1,137; 0,044]. El tercero contiene cero. Son bootstrap de 2.000 remuestreos de secuencias, no de ventanas supuestamente independientes, y no incorporan tres réplicas completas del encoder ni una corrección por todas las comparaciones exploratorias. No se calculó un intervalo de la diferencia de RMSE.

El peor caso event-only predice 4.891,317 s frente a GT 2,936 s, con alturas estimadas casi iguales; aporta 64,21% del error absoluto y 99,61% del cuadrático. El peor caso full predice −181,965 s frente a GT 6,009 s; aporta 10,64% del absoluto y 60,23% del cuadrático. Se publican estos casos como diagnóstico de estabilidad de la conversión, no como razón para excluirlos. Las medianas evitan que el lector sólo vea el efecto de los extremos.

Los [CSV por consulta](evidence/rgb_event/SCORED_PREDICTIONS.csv) y [por secuencia](evidence/rgb_event/PER_SEQUENCE.csv) permiten examinar heterogeneidad. Los diagnósticos conservados de [event-only](evidence/event_only/POST_SCORING_DIAGNOSTICS.json) y [full](evidence/rgb_event/POST_SCORING_DIAGNOSTICS.json) documentan las colas.

## Coste de evaluación y significado de “externa”

En event-only, las fases medidas sumaron 1.270,05 s de preparación, 120,63 s para nuestro sistema con tres cabezas e historia larga y 7,94 s de Garl: 1.398,62 s en total, sin inicialización. La extensión full sumó 558,91 s de preparación y 46,04 s de inferencia: 604,95 s. Sus medianas son 507,66 ms de preparación y 43,22 ms de modelo por consulta. No son un ensayo controlado de velocidad entre arquitecturas con trabajo idéntico.

“Externa” significa transferencia entre datasets desde eAP/TRAIN40 hacia EvTTC, sin fine-tuning aquí. eAP y EvTTC son datasets distintos, pero no hay que describir sus targets como conceptos completamente incompatibles: ambos requieren respetar definiciones de TTC, geometría y sincronización. Se corrigió la interpretación previa demasiado fuerte de esa incompatibilidad. Tampoco se confunde EvTTC con el dataset de nombre parecido EV-TTC.

El [artículo eAP/Garl](https://arxiv.org/html/2603.16303v1), el [artículo EvTTC](https://arxiv.org/html/2412.05053v2), la [documentación de formato](https://nail-hnu.github.io/EvTTC/download/data_format/) y el [código Garl](https://github.com/NAIL-HNU/Garl-TTC) sirven de referencia de contratos. La tabla oficial usa su propia selección/protocolo y métricas como RTE; estos MAE muestreados no pueden sustituirla. La genealogía exacta de entrenamiento/selección del checkpoint Garl público tampoco está certificada independientemente.

La conclusión defendible es que nuestras cabezas congeladas presentan menores valores puntuales de MAE y RMSE en esta comparación local de transferencia, con peor mediana frente a Garl full y la incertidumbre declarada. No se ha demostrado SOTA, generalización universal ni utilidad de seguridad AEB.
