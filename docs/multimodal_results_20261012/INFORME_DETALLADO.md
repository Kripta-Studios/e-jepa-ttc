# Investigación multimodal V16: trabajo ejecutado y resultados

Fecha de la investigación: 12 de octubre de 2026. Base revisada: `62cf4dbe914ea077c68e42983f869cd5422ab71c`. Cierre científico local: `b3efac4`. Rama: `scientific-recovery-v16-multimodal-validation`.

Completé la comparación pendiente con Garl multimodal en FCWD, instrumenté las etapas del predictor, investigué varias correcciones sobre RGB, entrené 32 fits con 80.000 updates y congelé un candidato con 6.762 predicciones test12. El candidato PHASE_ONLY cumple los criterios internos originales en tres semillas de cabeza. **No he demostrado SOTA oficial**: falta una puntuación oficial comparable y los resultados externos muestran compromisos entre métricas.

Este documento explica las decisiones y su alcance. Los informes numéricos completos conservan todas las filas, controles y resultados negativos:

- [Etapa de inferencia congelada, sin entrenamiento](artifacts/multimodal_validation_20261012/MULTIMODAL_REVIEW_FINAL.md).
- [Investigación posterior: tablas completas](artifacts/multimodal_research_20261012/MULTIMODAL_REVIEW_FINAL.md).
- [Diario de investigación](artifacts/multimodal_research_20261012/RESEARCH_LOG.md).
- [Inventario y contabilidad](artifacts/multimodal_research_20261012/ACCOUNTING.json).

## 1. Punto de partida y autorización

Verifiqué el ZIP de continuación `E_JEPA_TTC_MULTIMODAL_NEXT_20261012.zip`, su CRC y sus 26 archivos de contenido. Su tamaño es 230.440 bytes y su SHA-256 es `051eaba78cf5ffe58dca720d07c02701d1f1b30a190fe8da028e0acb3fa1f2c4`. Los 22.483.682 bytes citados en la revisión correspondían a otro paquete histórico. Conservé esa distinción para no atribuir al ZIP nuevo una verificación de otro archivo.

Preparé un worktree separado y preservé los pesos, normalizadores, fuentes y resultados V13. V13 ya había terminado sus doce fits: 150.000 updates científicos históricos. No reinicié RGB-PORT ni la matriz V14, que había terminado con `NO_CANDIDATE`. Consulté la coordinación de V15 y después la de V17/V18 sin asumir la propiedad de sus procesos ni incorporar sus resultados como entrenamientos de esta campaña.

La primera extensión autorizaba cero updates. Ejecuté esa fase con inferencias congeladas, diagnósticos y mediciones. Después autorizaste otras 18 horas de GPU y hasta un millón de updates. Bajo esa autorización ejecuté una búsqueda de cabezas y un estudio de contextos por grupos. El consumo final de esta investigación fue de 80.000 updates científicos y físicos, cero de recuperación y cero pendientes. Los 150.000 históricos V13 y los presupuestos V15/V17/V18 no se suman a ese consumo como si fueran trabajo nuevo V16.

Tu petición actual autoriza publicar este trabajo mediante push a GitHub y empaquetarlo. Esa publicación no equivale a enviar predicciones al benchmark. Los recibos históricos que dicen «no push» describen la autorización vigente cuando se produjeron; conservo sus bytes y registro la ampliación por separado.

## 2. Cohortes, identidades y métricas

La validación interna V contiene 16.432 consultas de ocho grupos. La cohorte Dev32 usada aquí contiene 1.024 consultas y 946 etiquetas; la ampliación histórica contiene 3.947 consultas y 3.640 etiquetas. No mezclé sus resultados. Comprobé las identidades por secuencia y timestamp, con unicidad y contenido: las 1.024 consultas estaban presentes en la ampliación, pero sólo 39 mantenían el mismo `query_id`. Una unión por posición o por ese identificador habría producido comparaciones erróneas.

FCWD contiene 630 consultas, 597 con etiqueta. El registro sellado admite RGB en 495 consultas, 472 con etiqueta. Las 135 consultas restantes, 125 etiquetadas, requieren fallback. Mantengo tanto la vista de soporte común como la de sistema completo; una mejora sobre consultas seleccionadas no sustituye la evaluación de todas las consultas.

En las tablas, un menor error es mejor. MAE mide error absoluto en segundos. `group_macro_bucket_MiD` es el agregado interno por grupos y buckets. `mean_MiD` externo y MiDc describen otros agregados; sus valores no se intercambian con el ranking oficial. El código del benchmark define:

```text
overall_MiD = 0.5 MiDc + 0.3 MiDs + 0.1 MiDl + 0.1 MiDn
```

Las cohortes externas aquí evaluadas carecen de las etiquetas negativas necesarias para completar ese score. Mantengo el componente desconocido como desconocido. No lo sustituyo por cero. Los intervalos externos proceden de bootstrap de secuencias completas y son descriptivos: FCWD sólo tiene tres secuencias. V, Dev32 y FCWD ya estaban expuestos durante el desarrollo; tampoco presento estos intervalos como una corrección por toda la búsqueda adaptativa.

## 3. Comparación Garl visual/full en FCWD

Ejecuté los checkpoints públicos visual-only y RGB+eventos con carga estricta, FP32 CPU y preprocesado nativo. Comprobé la pareja temporal de Garl por separado de las historias V13: endpoints dentro de ±1 ms y separación de 100 ms ±1 ms. Las 630 consultas pasaron ese control; los offsets observados fueron de −105 a 0 microsegundos. Después apliqué el filtro geométrico sellado: quedaron 495 consultas y 472 etiquetas. No ajusté tolerancias para obtener esa cobertura.

La variante full recibe los seis canales RGB y los cuarenta canales event de su ruta nativa. Conservé normalización, crop, padding e interpolación. El registro usa intrínsecos y una transformación local estimada con eventos; no usa TTC ni profundidad GT. Esto sigue siendo registro aproximado, no calibración extrínseca oficial. R_CTX consume RGB en el estimador, pero esta ruta FCWD también necesita eventos para registrar su ROI.

En las 472 etiquetas comunes, F_TRUE obtuvo MAE 0,700261 s y mean_MiD 55,969252, frente a 1,252557 s y 76,388385 de Garl full registrado. Garl visual obtuvo 1,311225 s y 111,867397. La tabla completa incluye R_CTX y Garl event-only, sin ocultar su mejor comportamiento crucial en esta comparación.

En las 597 etiquetas del sistema completo, F_TRUE con fallback E_CTX obtuvo MAE 1,429754 s y mean_MiD 53,929182. Garl full registrado con fallback event-only obtuvo 2,067345 s y 72,723138. El fallback depende de disponibilidad de RGB; no se activa mirando el error, el GT o el signo del TTC. El contraste F_TRUE menos full-fallback fue −0,637591 s de MAE, con IC95 % descriptivo [−0,849468; −0,503200], y −18,793956 de mean_MiD [−21,231750; −14,595444].

Esto cierra la comparación exploratoria pendiente y favorece a F_TRUE bajo ese registro. No establece una victoria en el benchmark oficial ni una comparación entre sistemas con idéntica historia temporal. Conservé las predicciones nativas y la sensibilidad con límites de ±60 s por separado.

## 4. Aporte incremental de eventos y localización del error

Calculé el contraste que faltaba F_TRUE−R_CTX en V usando las pérdidas de los ocho grupos. La diferencia de agregado es −2,855638, con IC95 % posterior [−7,694064; +1,008157]. F_TRUE tiene mejor agregado, pero peor MAE. El intervalo cruza cero: no acredita el gate ni demuestra equivalencia. En Dev32 se repite la dificultad de establecer una ventaja clara de la fusión sobre el RGB propio.

Instrumenté productores A5/C2F, PAIR, refinadores E_CTX/R_CTX y F_TRUE. Las trazas conservan ROI, tiempos, soporte, expertos, anchor, residual y fase antes y después de la conversión. Elegí 32 controles H por hash y episodios externos ya publicados. Las 1.229 llamadas instrumentadas de la primera fase reproducen bit a bit su llamada CPU sin instrumentar. En cambio, las comparaciones con predicciones históricas de otra ruta/dispositivo alcanzan diferencias de hasta 0,013831 s. Son dos comprobaciones distintas; no presento la ruta CPU como idéntica a la histórica GPU.

En un episodio Dev32 con GT 6,520134 s, R_CTX predice −8,321332 s y F_TRUE llega a aproximadamente −60 s: el residual de fusión cruza cero y la conversión de fase amplifica el error. En otro, GT 0,943878 s, R_CTX predice 1,726628 s y F_TRUE 22,649269 s. Esto justificó probar una corrección que partiera de R_CTX, preservando su predicción inicial.

Después tracé el candidato congelado: 104 trazas de casos seleccionados y 3.308 trazas de las dos cabezas sobre las 630+1.024 consultas externas. Los hooks y salidas coinciden con las referencias selladas. En Dev32, las 18 etiquetas con desacuerdo de signo entre padres concentran un empeoramiento frente a R_CTX: +11,345398 MiD y +2,273747 s MAE. En las otras 928 etiquetas, la diferencia es +0,709197 MiD y −0,086230 s MAE.

Las trece predicciones negativas de R_CTX sobre etiquetas Dev32 positivas siguen negativas después de PHASE_ONLY. En seis, la mediana de expertos RGB era positiva antes del residual de contexto. Esto localiza fallos anteriores a la cabeza nueva. No demuestra que baste con elegir una fase positiva ni autoriza una regla ajustada sobre esos casos. En los dos episodios anteriores PHASE_ONLY reduce los extremos de F_TRUE, pero empeora respecto a R_CTX.

## 5. Entrenamientos ejecutados y resultados negativos

Cada fit completó 2.500 updates. Registré identidad, semilla, fuentes, pérdidas, estados de recuperación y endpoint. Usé los estados congelados de refinadores ya supervisados en H como una nueva etapa supervisada sobre H; esos estados no son OOF. El estudio GROUP_CROSSFIT analiza esa limitación por separado.

Los criterios originales exigían, entre otras condiciones, al menos un 3 % de mejora del agregado sobre R_CTX, no empeorar su MAE, no empeorar F_TRUE en agregado/MAE, límites por bucket del 2 %, control del signo y un límite superior del intervalo EVENT−ZERO no positivo. Las recetas sucesivas se registraron antes de sus nuevas predicciones, pero la elección de qué estudiar fue adaptativa. Una tabla completa de V, controles e intervalos figura en el informe numérico enlazado al inicio.

- **BASELINE:** dos fits con estados RGB/event y anchor R_CTX. EVENT alcanzó agregado 91,300140 y MAE 1,852270 s. No pasó todos los criterios; el intervalo frente a ZERO cruza cero.
- **PHASE_UNITS:** dos fits para corregir el tratamiento de unidades/condicionamiento de fase. EVENT S7 llegó a 89,882376 y 1,846064 s, pero degradó el bucket largo un 3,601818 %. Fracasó bajo el 2 % original.
- **UNIT_REPLICATION:** cuatro fits adicionales, semillas 13 y 23, con controles. No confirmaron todos los criterios, ni siquiera con la sensibilidad del 4 %. Una réplica mantuvo un intervalo que cruza cero y otra degradó el bucket largo un 4,972559 %.
- **RELATIVE_PHASE:** dos fits con corrección escalada por la magnitud de la fase RGB. Llegó a 90,250266 y 1,884901 s. Falló MAE frente a RGB y el bucket largo.
- **FUSION_STATE:** dos fits que añadían estados de F_TRUE a un anchor RGB. FUSED alcanzó 93,117998 y 2,005650 s; el control alcanzó 89,239073 y 1,837367 s. El intervalo de diferencia favorece al control. Mejor ajuste H junto a peor V es compatible con sobreajuste, sin establecer una causa única.
- **RGB_REGRET:** dos fits con penalización adicional por empeorar el error de RGB en H. Obtuvo 92,088216 y 1,848408 s; falló la mejora agregada requerida.
- **FUSION_PRESERVING:** dos fits con F_TRUE como anchor y estados RGB adicionales. La paridad inicial con F_TRUE quedó verificada, pero el brazo refinado terminó en 92,506456 y 2,003201 s. No pasó los criterios; el intervalo frente al control cruza cero.
- **PHASE_ONLY:** dos fits iniciales y cuatro de réplica, semillas 7/13/23. Las tres cabezas EVENT pasan los criterios originales del 2 % y sus contrastes frente a ZERO. Es el candidato conservado.
- **GROUP_CROSSFIT:** ocho fits de contexto más dos de readout, 25.000 updates. Usé cuatro folds de grupos H, normalizadores calculados sólo en grupos de entrenamiento y predicciones excluyendo el grupo objetivo. Comprobé límites de identidad y paridad con la pérdida nativa. EVENT llegó a 90,412912 y 1,893097 s; falló MAE frente a RGB. Resultado `NO_CANDIDATE`.
- **PHASE_ENSEMBLE:** cero updates. Promedié con pesos iguales las fases emitidas de PHASE_UNITS 7/13/23, con control emparejado. Obtuvo 89,861298 y 1,813344 s; degradación larga 3,325271 %. Pasa esa sensibilidad al 4 %, pero no el criterio original. No constituye una confirmación independiente.

La suma es 32 fits y 80.000 updates. Los controles FUSION_STATE y FUSION_PRESERVING siguen siendo sistemas multimodales; anular los estados adicionales no elimina una modalidad ya presente en su anchor. Esa distinción evita interpretar sus resultados como una ablation RGB/event completa.

## 6. Qué cambia PHASE_ONLY

La cabeza nueva tiene 1.955 parámetros. Ese número excluye encoders, PAIR y refinadores. Usa veinte coordenadas de fases y relojes de los padres; elimina los 320 estados ocultos que usaban los primeros correctores. Conserva R_CTX como anchor y aprende una corrección de fase sin un límite residual adicional:

```text
fase_raw_nueva = fase_raw_R_CTX + 0.03 * f_theta(fases, relojes)
```

El MLP tiene dos capas ocultas de anchura 32, LayerNorm y SiLU, y tres salidas con última capa inicializada a cero. La inicialización reproduce R_CTX. El control ZERO conserva relojes y recuentos, anulando las cinco coordenadas predictivas event. Conservé productores, PAIR, contextos y normalizadores V13.

Entrené con AdamW, batch 128, weight decay 0,001, clipping de gradiente 1, warmup 100 y descenso coseno de 3e−4 a 3e−5 hasta 2.500 updates. La pérdida es la nativa de fase emitida más 0,1 de pinball, con masa global H. Las tres semillas son réplicas de cabeza sobre los mismos padres congelados; no son tres entrenamientos independientes del sistema completo.

Los agregados V de EVENT fueron 89,222960, 89,198875 y 89,220028. Sus MAE fueron 1,869604, 1,858718 y 1,862337 s. Los intervalos EVENT−ZERO fueron [−5,737922; −1,302219], [−5,734941; −1,350258] y [−5,685388; −1,398840]. Ninguna de las tres cabezas EVENT falló un criterio original. El control ZERO puede tener mejor MAE: la mejora del agregado no implica mejorar toda métrica.

Congelé la semilla 7, primera semilla registrada, como candidato test12. No escogí la mejor semilla después de examinar las cohortes externas.

## 7. Transferencia: mejora parcial y compromisos

En FCWD, PHASE_ONLY S7 obtiene mean_MiD 59,504120, MAE 1,582103 s y MiDc 93,304249. Las semillas 13 y 23 obtienen agregados 59,118803 y 59,879333, MAE 1,594180 y 1,597057 s, y MiDc 91,540742 y 92,791049. Mantienen fallback E_CTX cuando falta RGB.

Frente a Garl event-only, S7 mejora la estimación puntual de agregado, MAE y MiDc. Sin embargo, los intervalos de agregado y MiDc cruzan cero: −2,312403 [−4,298607; +0,515151] y −1,752577 [−5,759133; +2,279379]. El contraste de MAE es −0,564070 s [−1,083943; −0,259610].

Frente a F_TRUE, S7 empeora el agregado FCWD en +5,574937 y MAE en +0,152349 s, aunque mejora MiDc en −9,018913. Los tres intervalos descriptivos mantienen esos signos. La nueva receta ofrece otro compromiso; no reemplaza de forma universal a F_TRUE.

En Dev32, S7 obtiene mean_MiD 132,292198, MAE 1,216539 s y MiDc 185,793232. Mejora frente a Garl full histórico en las tres métricas, pero frente a F_TRUE su agregado empeora +2,134754 con intervalo [−4,824035; +8,951624]. La diferencia MAE es −0,151928 s [−0,446634; +0,097602]. Tampoco las otras semillas establecen una mejora consistente de MiD frente al RGB/fusión propios.

Recalculé además la contribución positiva con los pesos del benchmark, sin completar el término negativo ausente. En FCWD S7 da 64,571859, frente a 62,310988 de F_TRUE y 61,959093 de Garl event-only. Su ventaja crucial no compensa el peor tramo 3–6 s bajo esos pesos. En Dev32 da 113,161003, frente a 111,219980 de F_TRUE. Esto explica por qué una victoria en `mean_MiD` no basta para afirmar una victoria en `overall_MiD`.

## 8. Registro geométrico y cobertura

Probé acumulaciones causales de eventos de 20/40/80 ms sobre doce consultas seleccionadas por hash y disponibilidad, 36 ensayos. La ruta de 20 ms reprodujo el registro sellado. Las aceptaciones fueron 6/12, 9/12 y 10/12. Más aceptaciones no acreditan menor error geométrico.

Preparé doce paneles y 72 filas de correspondencias para anotación independiente. No había anotaciones independientes completas con las que medir el error de las alternativas. También examiné un prior causal de la última transformación aceptada, limitado por antigüedad, soporte y deriva. Sus filas elegibles no equivalen a recuperaciones nuevas. No promoví un registro alternativo ni bajé NCC en función del error TTC.

La validación geométrica sigue pendiente. No realineé cada frame histórico de forma independiente, ya que eso podría alterar la expansión aparente que utiliza el predictor.

## 9. Coste: qué medí y qué significa

Verifiqué reutilización exacta de ventanas y cálculos event. En 128 comparaciones CPU, un hilo y carga concurrente, conservé bit a bit inputs, features, expertos y salidas. La mediana de la ruta event bajó de 1.284,593 a 1.045,866 ms en Dev32, y de 1.116,826 a 823,243 ms en FCWD: reducciones del 18,58 % y 26,29 %. El perímetro va de HDF5 a E_CTX con pesos residentes; excluye RGB, registro, detector e inicialización.

El perfil CPU del candidato contiene 128 llamadas. Su mediana sensor→candidato fue 1.630,349 ms en FCWD y 2.022,164 ms en Dev32, con registro FCWD precalculado. La cabeza aislada costó aproximadamente 0,718 y 0,778 ms. Que la cabeza sea pequeña no elimina lectura, voxels, RGB y productores.

Completé otras 64 llamadas FCWD con padres GPU y registro recomputado. Cada registro, estado y predicción coincidió con su referencia sellada. Medí p50 total 1.213,1175 ms y p95 3.060,5612 ms. Las medianas de etapas incluyeron registro 148,107 ms, lectura/voxels event 293,177 ms, productores event/PAIR 105,179 ms, lectura/crop RGB 160,774 ms y productores RGB/PAIR 475,126 ms. La corrección CPU final costó p50 0,523 ms.

No sumo medianas de etapas para reconstruir la mediana total. La medición usa pesos residentes, un hilo CPU, sincronizaciones explícitas y un estado de recursos documentado. Excluye detector, carga inicial, comprobaciones del runner, lecturas de referencia y escrituras de evidencia. No es un servicio streaming con estado persistente entre consultas ni una medida aislada de disco frío. Por eso no la comparo como equivalente con los históricos de Garl 37 ms o V12 82 ms.

## 10. Test12: inferencia completa, todavía sin score

Congelé `PHASE_ONLY_EVENT_S7` a las 14:05:52 UTC del 12 de octubre. Fijé fuentes, doce pesos V13, normalizadores, pesos nuevos e inputs antes de terminar la inferencia. Verifiqué 36 TAR RGB contra sus SHA de la revisión fijada del dataset. Los datos sensoriales y la caché event histórica están identificados por hashes; de esa caché sólo utilicé la observación event actual, comprobada con el adaptador nativo.

Los 6.762 tokens oficiales incluyen 957 aliases de observaciones físicas repetidas por clips solapados. Comprobé igualdad de ROI, ventanas y relojes y construí 5.805 observaciones físicas únicas. Cada observación aporta evidencia una vez al contexto y cada token oficial recibe una predicción. No descarté tokens duplicados del archivo de entrega.

Audité separación de grupos P/H/V/test, identidad del objeto y causalidad. La historia permitida era de hasta 650 ms; el soporte sensorial máximo observado fue 600.006 microsegundos event y 600.000 RGB. La auditoría usa las identidades publicadas de secuencia; no demuestra independencia visual más allá de esa partición. El sistema conserva los padres V13 y no constituye un refit nuevo sobre TRAIN40.

La inferencia terminó con 6.762 predicciones, todas con RGB disponible y cero fallback. El parser público acepta los 6.762 tokens y recupera exactamente los valores del CSV sellado. Sólo ejecuté lectura y validación del formato, sin scoring ni etiquetas privadas.

El ZIP de formato oficial es `PHASE_ONLY_S7_GarlTTC_submission.zip`, SHA-256 `60b5a620848659633b875772430cddd7d7532b0fe180ec59904d04b81a9d3ca1`. Está preparado y no enviado. La comprobación de formato no permite estimar su precisión test12. Tampoco he verificado una clasificación oficial actual que permita afirmar SOTA.

## 11. Recuperación, integridad y coordinación de recursos

Una interrupción dejó archivos previos incompletos. Audité las 890 predicciones existentes y detecté ocho NPZ con SHA y CRC incorrectos. Guardé los bytes y recibos originales en cuarentena. No cambié los hashes esperados para aceptar el daño. Al recalcular esos ocho archivos con el modelo congelado, recuperé exactamente los ocho SHA originales y CRC válidos. La causa de la corrupción no quedó establecida.

La recuperación final exigió cachés sensoriales presentes y verificadas, el mismo bloque de inicialización/cálculo de cabeza que el original y controles acotados de recursos. Dieciséis consultas anteriores seleccionadas por hash reprodujeron todos sus arrays bit a bit. La recuperación añadió cero updates de optimizador.

El control de recursos distingue un supervisor V18 en espera de un trabajador GPU activo mediante identidad, fuentes verificadas, estado reciente y ausencia de reservas abiertas. Los estados ambiguos o trabajadores activos detienen la admisión. Conservé límites de memoria y plazo; no modifiqué procesos, locks, receta o fuentes de V18. Los recibos de coordinación son observaciones en el momento de la ejecución, no una afirmación sobre el estado futuro de esa campaña.

## 12. Verificación y archivos entregados

El paquete completo anterior tiene 461.398.299 bytes y SHA-256 `a59bac60a1a97c4c5e0b279247c98b96b1b4ce1ff3dc8db7781b4cf01372bab9`. Contiene 47.842 archivos de contenido, además del manifiesto. Verifiqué todos los hashes y CRC, lo extraje en otro directorio y repetí el cálculo de métricas: 46.778 comprobaciones escalares, trece tablas V, 34 tablas externas y 32 endpoints. La tabla de componentes positivos también se regeneró byte a byte.

La extracción independiente reutilizó el entorno Python existente. No fue una instalación desde cero ni una repetición de inferencia desde sensores. Los HDF5, TAR RGB, parte de los checkpoints públicos Garl y la caché sensorial histórica siguen siendo dependencias externas. El paquete conserva sus identidades y rutas originales; trasladarlo no garantiza ejecutar inferencia sin preparar esas dependencias.

Esta publicación añade un índice legible, este informe, los informes completos, las tablas y recibos seleccionados para GitHub y un inventario SHA-256. El ZIP nuevo conserva también el paquete completo anterior sin modificar sus bytes. Los scripts originales permanecen en `operational/multimodal_validation` y `operational/multimodal_research`; los tests acompañan a la rama. El manifiesto de publicación enumera qué se copia a GitHub y qué sólo se conserva en el paquete completo.

## 13. Por qué permitir un 4 % no demuestra SOTA y qué falta

El 4 % permite una degradación mayor del bucket largo que el 2 %. No reduce el error de ninguna predicción. PHASE_UNITS S7 cambia de decisión bajo esa sensibilidad, pero sus réplicas no cumplen todos los criterios. PHASE_ONLY ya cumple el 2 % en tres semillas; ese éxito es interno y no sustituye una evaluación oficial.

Para afirmar SOTA en una métrica concreta hace falta una comparación reproducible bajo el mismo protocolo, con modalidades, disponibilidad, historia temporal y métrica declaradas, frente a las referencias pertinentes. No hace falta ganar cada bucket para ganar un ranking agregado, pero sí obtener el score que define ese ranking. Aquí faltan ese score oficial y una referencia actual verificada. Los compromisos de transferencia, la búsqueda adaptativa y las réplicas limitadas a cabezas restringen afirmaciones más amplias.

Los siguientes pasos con fundamento son completar la evaluación oficial del candidato congelado cuando esté autorizada y operativa, anotar correspondencias independientes para geometría y diseñar una investigación específica de los cambios de signo y el tramo 3–6 s. Una futura mejora de productores requeriría reconstruir sus features dependientes, normalizador y fusión; no se pueden intercambiar padres en silencio. Un refit TRAIN40 y réplicas completas siguen pendientes. No inicié esos trabajos ni otro entrenamiento durante esta preparación documental.
