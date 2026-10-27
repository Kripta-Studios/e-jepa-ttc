# Historia científica: de JEPA al candidato multimodal TRAIN40

Fecha de corte: 27 de octubre de 2026. Este capítulo reconstruye decisiones y resultados a partir de commits, informes y recibos conservados. El capítulo [02](02_TRAIN40_OOF_AND_FINAL_TRAINING.md) detalla la genealogía del sistema final. Las optimizaciones de ejecución se documentan por separado en los capítulos 03–06.

## Cómo interpretar esta historia

El proyecto cambió de alcance durante su desarrollo. Comenzó como investigación sobre representaciones predictivas de eventos y terminó esta etapa con un sistema multimodal de estimación de TTC, apilado mediante entradas fuera de muestra por grupos. El nombre E-JEPA-TTC describe el proyecto; no significa que todos sus modelos posteriores tengan exactamente el objetivo JEPA inicial ni que una comparación final aísle el efecto causal de JEPA.

Se distinguen cuatro tipos de evidencia: implementación en un commit; experimento ejecutado con resultados; propuesta aún no ejecutada en la fecha del documento; artefacto final verificado. Un commit que incorpora un runner no demuestra que todos sus experimentos se ejecutaran. Un informe antiguo que dice «pendiente» describe su fecha de corte y puede quedar superado por recibos posteriores sin que haya que reescribirlo.

Al comenzar esta reconstrucción, el HEAD de esta rama seguía en `771c96a23784e1f3955cec21ba1f4d9a063a9145`, del 17 de octubre. La ejecución del 18–27 de octubre estaba documentada en archivos locales, snapshots de fuentes y artefactos con SHA-256. No se atribuye ese trabajo posterior al contenido original de aquel commit. Esta entrega incorpora su historia y referencias para el siguiente commit.

## Junio y julio: infraestructura real, baselines y JEPA inicial

El historial comienza el 30 de junio con `df4ecaea`, infraestructura del pipeline, y `4faea6b4`, manifest local EvTTC y lector indexado. Los commits `4081c9e7`, `f25356cb` y `81380f2b` incorporaron referencias triviales, geometría/event rate, caché voxel y entrenamiento TinyCNN. `c3587a54` añadió preentrenamiento JEPA y `e145062b` regularización de varianza. El 1 de julio siguieron la variante temporal multihorizonte (`6ac3aa5e`), el probe con encoder congelado (`8dfe3e86`) y el experimento de pocas etiquetas (`e346729c`, resultados `524a7336`). Son hitos de implementación y de experimentación distintos.

El snapshot `524a7336:docs/local_results.md`, recuperable mediante `git show`, registra un smoke test local con tres velocidades de CCRs-1: baja para entrenamiento, media para validación y alta para test. Había 1.230 ventanas indexadas: 335/418/477. Esta partición fue observada repetidamente durante el desarrollo; sus cifras no equivalen a una evaluación final ciega de EvTTC.

| Experimento local de aquel snapshot | MAE validación, s | MAE test, s |
|---|---:|---:|
| JEPA masked, preentrenamiento sólo train, seguido de TinyCNN | 3,297 | 3,290 |
| JEPA temporal, preentrenamiento sólo train, seguido de TinyCNN | 1,518 | 3,183 |
| Probe con encoder congelado | 1,916 | 2,911 |
| Scratch, 5 % de etiquetas, media de tres semillas | 2,909 ± 0,743 | 3,107 ± 0,277 |
| JEPA temporal y fine-tuning, 5 % de etiquetas | 1,548 ± 0,176 | 2,986 ± 0,106 |

El 5 % suponía 17 ventanas de entrenamiento y semillas 7, 13 y 21. El resultado justificaba seguir estudiando representación y exposición a etiquetas, pero el tamaño y la reutilización de desarrollo impedían una afirmación general de superioridad.

También se corrigió la interpretación de la geometría: `e7feb92a` y `171fafd2` separaron el baseline causal de los cálculos centrados que usan información posterior. El baseline causal asistido por anotaciones de objetos obtuvo 0,439 s sobre 143 consultas de validación y 0,188 s sobre 96 de test en aquel snapshot; no compartía necesariamente la cobertura del modelo neuronal. Las variantes con eventos de todos los splits y la geometría no causal se conservaron como diagnósticos, sin presentarlas como entrenamiento exclusivamente train o inferencia causal.

La distinción metodológica ya aparece aquí: utilizar eventos futuros como target de una pérdida predictiva durante entrenamiento no autoriza acceder a eventos futuros al inferir. Tampoco permite cruzar grupos entre train y evaluación. Los objetivos TTC no debían entrar en el preentrenamiento que se declarase autosupervisado.

## Julio y agosto: eAP, contratos Garl y representación A5

El 30 de julio, `cd42603c` incorporó la comparación emparejada eAP→EvTTC. El 2 de agosto, `1b7279bb` fijó subconjuntos comparables de eAP LevelDynamics. En agosto se dio más peso al contrato real del benchmark: preprocesamiento, consultas, modalidad, fase firmada, teacher y cobertura.

Los commits `95318fad`, `24e40389` y `6e880888`, del 10 de agosto, documentan el avance hacia una comparación Garl con preprocesamiento emparejado. `e3db79fa` incorporó la destilación relacional DINOv3 de A4; `6b453975` y `f7c6c326` reforzaron su procedencia. `92cf688e` preregistró el transporte local temporal A5 y `4609e069` registró la confirmación en filas train disjuntas. `8b6f1a44`, del día 11, añadió la política de réplica posterior al gate y la referencia A4.

Estos hitos explican por qué los productores finales ya no se describen simplemente como el TinyCNN o el JEPA del smoke inicial. A5 incorpora supervisión relacional de un teacher genérico y transporte local; las recetas posteriores combinan esa representación con supervisión geométrica/TTC. Las contribuciones deben atribuirse al sistema y al contraste realmente ejecutado, sin convertir el nombre del repositorio en una ablation retrospectiva.

El 13–14 de agosto aparecen folds por grupos y contratos firmados: `91bf6115` y `e5e76bde` para entrenamiento Garl local a cada fold; `825c1f00` para V6; `f48f2c8c`, `e3f5a610`, `c27dc5e2` y `f9331b29` para el protocolo y cierre V7. A finales de agosto, V8 reforzó el rechazo ante identidades incorrectas (`df7afaee`), la admisión de cachés teacher (`4387404b`, `2d5a7a41`, `54bc9911`) y la declaración de salidas sin soporte como NaN con causa (`81eda4a0`). No se deduce del número de versiones un número equivalente de entrenamientos completos.

## Septiembre y primeros días de octubre: SIMPLEX-T, contexto y coste real

Durante septiembre creció la infraestructura de recuperación, contratos de ejecución y consumidores congelados. Ejemplos son `1ac4fbfe` y `80c190d7` para estado/orquestación, `bca6e39e` para enlazar H16 sin inferencia nueva, `ce594775` para reanudación y QA emparejado, y `b6f32f0d` para genealogía y precisión antes del freeze. La auditoría `0c1a7285` conserva una interrupción técnica sin convertirla en trabajo científico completado.

El [cierre SIMPLEX-T](../simplex_t_results_20261004_complete/README.md) registra 24 cabezas de 2.500 updates: 60.000 updates científicos, más contabilidad técnica y recuperación separadas. Los productores y ajustes anteriores reutilizados no son réplicas nuevas de toda la cadena.

En OLD_DEV —8.192 consultas de nueve secuencias ya observadas—, H8 seed7 tenía MiD 121,646332 y H16 118,865739. Las medias de tres semillas de cabeza fueron 121,784022 y 119,241356, una reducción aproximada del 2,09 %. El intervalo jerárquico publicado favorecía H16, pero el contraste restringido a las dos semillas nuevas incluía cero. Esto distingue una señal agregada del alcance efectivo de las réplicas. No había un ensemble desplegado de tres modelos.

A5_ONLY, C2F_ONLY, A5_PAIR y los controles de orden/edad no superaron todos los criterios registrados. También se aprendió que reducir la cabeza no resolvía el coste total: en la medición desde sensores, la preparación de datos representaba aproximadamente el 85,5 % de H8 y el 90,4 % de H16, mientras la cabeza suponía menos del 1 %. Esas cifras tenían el alcance del ensayo, GPU compartida y ROI anotadas; no constituían latencia de un sistema de frenado desplegado.

## V12 y la revisión de TTC de octubre

La [revisión del 4–8 de octubre](../research_review_20261004_08/README.md) documenta productores TRAIN40 y transferencia eAP→EvTTC. En su comparación de 946 consultas válidas, las variantes H8 presentaban MAE aproximadamente 1,321–1,409 s frente a 1,868 s de Garl full y 8,048 s de Garl event-only. Garl conservaba ventajas en mediana. La cobertura, los adaptadores y la población eran los de esa reconstrucción; no era una reproducción exacta de la tabla del artículo ni una puntuación oficial.

El [estudio TTC del 9 de octubre](../ttc_revision_20261009/README.md) entrenó tres cabezas Direct de 2.500 updates. En DEV32, la mediana entre semillas de MAE pasó de 1,3766 a 1,0729 s, pero la tasa de fallos urgentes a un segundo pasó de 98,32 % a 100 %. En FCWD el MAE empeoró de 1,6249 a 1,7711 s. Se conservaron ambos resultados: una reducción media no demostraba mejora del régimen crucial ni transferencia universal.

Esta etapa hizo explícito el problema de las transformaciones de fase cerca de cero: diferencias pequeñas en la variable interna pueden producir grandes errores en segundos; el clipping y la saturación deben medirse por separado. Las cabezas Direct no se convirtieron automáticamente en los padres del sistema final.

## V13 y V16: separar representación, contexto y corrección multimodal

V13 estableció una ruta RGB paralela y contratos de contexto compatibles con eventos. Los [informes tempranos del port RGB](../rgb_port_audit_20261010/README.md) son snapshots intermedios: sus checkpoints parciales o pendientes de aquel momento no describen el estado final de octubre. La [comparación de familias](../final_refit_20261017/FAMILY_COMPARISON.md) distingue el A5 histórico V13 entrenado sobre P —53.890 consultas y 30.330 updates— del A5 V12 posterior/admitido con todo TRAIN40 —88.744 consultas y 49.932 updates—. Compartir nombres A5 no los convierte en pesos intercambiables.

El candidato histórico V16 se fijó el 12 de octubre. Su ficha se conserva en el worktree hermano `../e-jepa-ttc-v16-multimodal-validation/docs/continuidad_v16_v18_20261012/submission/MODEL_CARD.md`, con referencia publicada `71ec9d900189a120b06c3956029496e6fd704f6f`.

**PHASE_ONLY_EVENT_S7 es multimodal.** Consume fases y tiempos derivados de productores RGB y eventos, ancla su corrección en R_CTX y usa una cabeza de 1.955 parámetros sobre veinte coordenadas. Los encoders, PAIR y contextos continúan siendo necesarios. Los 1.955 parámetros no son el tamaño completo del sistema y «EVENT» no significa inferencia sin RGB.

Los productores y PAIR históricos usaron P; refinadores y cabeza usaron H; V fue desarrollo observado. Las tres semillas eran réplicas de cabeza con padres compartidos. El candidato pasó sus criterios originales, conservó 6.762 predicciones test12 y quedó empaquetado. El ZIP histórico tiene SHA-256 `60b5a620848659633b875772430cddd7d7532b0fe180ec59904d04b81a9d3ca1`; no era un refit completo TRAIN40 ni acreditaba puntuación oficial.

La ficha también conserva resultados desfavorables: PHASE_ONLY mejora la banda crucial de FCWD respecto a F_TRUE, pero no su MAE y mean_MiD. No se eliminó F_TRUE como referencia de transferencia por ser anterior. Los resultados de Dev32 tampoco establecían una mejora uniforme.

## V18: investigación terminada, resultado NO_CANDIDATE

El [informe científico final V18](../joint_measurement_20261012/FINAL_SCIENTIFIC_REPORT.md) y los [resultados de réplica de fusión](../joint_measurement_20261012/FUSION_REPLICATION_RESULTS.md) constituyen el cierre. `NO_CANDIDATE` significa que ninguna variante cumplió conjuntamente las condiciones de promoción registradas, no que todas sus métricas empeorasen.

| Rama | Resultado que se conserva | Límite decisivo |
|---|---|---|
| C1 | La lectura geométrica mejora bandas MiD de H frente al readout libre PAIR-CORR | MAE/colas y otros requisitos impiden promoverla como predictor autónomo |
| D1 | Se ejecutó el aumento de anchura | No justificó ampliar indiscriminadamente el modelo |
| B1 | La supervisión auxiliar de correspondencias mejora frente a un control de igual número de updates TTC | La banda negativa impide promoción frente a A5; igualar updates no iguala FLOPs |
| A1-FULL | Mejora fuerte en agregado y crucial de H | Regresión negativa, pese a pasar la protección global de signo |
| C1 + R_CTX | Mejora interna parcial multimodal, replicada con padres compartidos | No alcanza la mejora crucial mínima del 10 % exigida por V18 |

En H, [A1-FULL](../joint_measurement_20261012/A1_RESULTS.md) obtuvo agregado 78,920, MiDc 95,846, MiDn 87,087 y MAE 5,446 s; A5-P2 tenía 112,285, 157,267, 65,520 y 5,637 s. Por ello, «mejoró el predictor» requería especificar la banda: aproximadamente −29,7 % de agregado y −39,1 % de crucial junto a +32,9 % de error negativo. Una BCE de signo no quedaba demostrada como solución al error de magnitud.

En la misma cohorte V, las medias de métricas entre tres semillas fueron:

| Sistema | Agregado MiD | MiDc | MiDs | MiDl | MiDn | MAE, s |
|---|---:|---:|---:|---:|---:|---:|
| V16 PHASE_ONLY | 89,214 | 131,704 | 48,714 | 40,449 | 47,028 | 1,864 |
| V18 C1 + R_CTX | 87,287 | 126,569 | 50,008 | 41,719 | 48,285 | 1,814 |

La diferencia agregada V18−V16 tenía IC95 % por grupos aproximadamente [−3,686; +0,141]. No demuestra superioridad ni equivalencia. Las réplicas repetían adaptación C1 y cabezas, compartiendo A5 inicial y R_CTX. Las medias de métricas no son predicciones de un ensemble. Los números H y V tampoco pueden ordenarse como si fueran la misma población.

El cierre contabilizó 183.200 updates científicos, 858 técnicos y 273 de recuperación superior: 184.331 físicos como cota contabilizada, sin pendientes de campaña. Las 54,65 horas incluían scopes, carga, guardado y reservas; no eran tiempo puro de kernels. La medición de coste de la fusión conservaba aritmética y batches históricos; sus medianas de 3,012 s para FUSION-TRUE-S7, 6,588 s para F_TRUE y 1,965 s para R_CTX no certifican el mínimo coste batch1 de un servicio rediseñado.

Los commits `fb416812`, `c3f14408` y `a811cc3a` separan cierre científico, código empaquetado y verificación documental posterior. [La decisión post-V18](../post_v18_20261017/DECISION_POST_V18.md), incorporada por `771c96a2`, mantiene el cierre. Su refit de una cabeza sobre H+V, 34.854 consultas y 4.730 updates, era una propuesta distinta; no se debe presentar como el entrenamiento completo ejecutado después.

## V19: una reparación parcial del objetivo, no una arquitectura nueva

La evidencia se conserva en el worktree hermano `../e-jepa-ttc-v19-decision-recovery/docs/decision_recovery_20261017/LOSS_SCREEN_RESULTS.md`. L0 y LW parten del mismo B1, usan adaptación completa, semilla 7, sampler uniforme y 12.000 updates. LW cambia la masa negativa del objetivo; no añade otro encoder.

| Modelo en H | Agregado | Crucial | Negativo | MAE, s | Error de signo |
|---|---:|---:|---:|---:|---:|
| L0 / A1-FULL | 78,920 | 95,846 | 87,087 | 5,446 | 5,607 % |
| LW | 79,867 | 100,418 | 70,775 | 4,613 | 4,055 % |

L0 reprodujo las métricas de A1-FULL. LW pasó de masas 50/30/10/10 a aproximadamente 44,44/26,67/8,89/20, conservando la relación positiva 5:3:1. Mejoró negativos, MAE y signo, pero empeoró agregado y crucial. Su elegibilidad numérica de desarrollo frente a A5 no equivalía a pasar las salvaguardas más estrictas del sistema ni al disparador de réplica frente a L0. El estado quedó `REPLICATION_NOT_ENABLED`.

Los selectores posteriores tampoco justificaron reemplazar la cadena. R0 NATIVE/FULL alcanzó 77,629 de agregado H, pero ejecutaba dos expertos y no cumplía salvaguardas. No era un MoE disperso entrenado conjuntamente, ni una refutación general de cualquier MoE posible. Los diagnósticos de gradiente y signo se conservaron; no autorizaron una nueva campaña por el mero hecho de existir saldo anterior.

## Del 17 al 27 de octubre: una familia final, OOF y cierre de inferencia

La instrucción vigente pasó a ser entrenar el sistema final con todo TRAIN40 público antes del envío oficial. [La decisión](../final_refit_20261017/DECISION.md), [la comparación de familias](../final_refit_20261017/FAMILY_COMPARISON.md) y [la revisión crítica](../final_refit_20261017/CRITICAL_REVIEW.md) separan preparación operativa y superioridad arquitectónica. Se eligió la familia PHASE_ONLY por su interfaz física compatible con apilamiento OOF, conservando V18 como alternativa científica y F_TRUE como referencia, sin entrenar ambas familias por defecto.

La propuesta inicial de stacking secuencial y el cálculo preliminar de reutilizar la primera rotación quedaron superados por [la cola científica de cinco rotaciones](../final_refit_20261017/SCIENTIFIC_OOF_QUEUE.md), activada el 19 de octubre. El capítulo 02 explica los roles, exposición y diferencia entre las entradas auxiliares de entrenamiento y los padres de despliegue.

El 27 de octubre se verificaron los cuatro productores de despliegue sobre 88.744 consultas/40 secuencias y 49.932 updates cada uno. Los productores event provenían de la ejecución TRAIN40 de V12; RGB comenzó el 17 de octubre y pasó por varios runtimes recuperables. No se afirma que los nueve componentes se entrenaran desde cero el día 27. Ese día se completaron/admitieron las etapas superiores finales, el sello, la exportación de cabeza y las predicciones oficiales. La evidencia está en [DEPLOYMENT_TRAIN40_SCOPE.json](../../artifacts/final_diagnostics_20261027/DEPLOYMENT_TRAIN40_SCOPE.json) y [FINAL_PACKAGE_REVIEW.json](../../artifacts/final_diagnostics_20261027/FINAL_PACKAGE_REVIEW.json).

El ZIP final `PHASE_ONLY_S7_TRAIN40_SCIENTIFIC_OOF.zip`, SHA-256 `5eb69749de7aff682eea640d843ee635379702337d929c86f50472264f1d88bc`, contiene los 6.762 tokens oficiales exactos: 5.805 consultas físicas y 957 alias. Son predicciones finitas, sin fallback. Esto verifica identidad y cobertura; no proporciona precisión de test12, cuyas etiquetas siguen privadas. No hubo submission implícita.

## Transferencia final del 27 de octubre: resultado y alcance

Después de congelar el modelo se ejecutó la [comparación EvTTC final](../final_train40_transfer_20261027/REPORT.md). El manifest fijó 16 anclas uniformes por cada una de 32 secuencias compatibles, seleccionadas por disponibilidad de sensores/ROI, más las 46 consultas históricas. Cuatro coincidían: 554 consultas físicas únicas. Ambos sistemas terminaron y sellaron todas sus predicciones antes de abrir TTC para puntuar.

Se obtuvieron 514 targets válidos: 46 históricos y 468 nuevos; otras 40 consultas no tenían TTC dentro de la tolerancia temporal de 50 ms. No fueron fallos de inferencia. Las métricas se reportan condicionadas a esa cobertura, y las dos cohortes son disjuntas en el informe.

| Cohorte | PHASE_ONLY final RTE | Garl full RTE | R_CTX final RTE |
|---|---:|---:|---:|
| Historical46 | 7,7478 % | 13,3273 % | 6,3873 % |
| Extended_new, 468 consultas válidas | 41,2267 % | 140,0689 % | 38,5555 % |

La cadena final mejora el RTE por secuencia de Garl en 26 de las 32 secuencias extendidas; R_CTX conserva un promedio agrupado mejor que la fusión en ambas cohortes. No se oculta ese control ni se cambia el candidato después de verlo. El bootstrap por secuencia es exploratorio; los promedios de error relativo pueden ser sensibles a casos extremos y deben leerse junto con distribuciones y métricas por secuencia.

Los targets válidos abarcan aproximadamente 0,496–17,994 s, todos positivos: esta evaluación no valida negativos. EvTTC ya había sido observado en el proyecto. Además, cada sistema conserva su historial nativo y existen diferencias declaradas de ROI y precisión de backend. Es transferencia descriptiva frente al checkpoint oficial, no puntuación oficial de GarlTTC ni reproducción exacta de la tabla VI. La [revisión final](../../artifacts/final_train40_transfer_20261027/FINAL_REVIEW.json) liga manifest, sellos, modelos y tablas.

## Qué no queda demostrado por el cierre

El resultado cerrado es una cadena multimodal entrenada, identificada, recuperable y con predicciones verificadas. No demuestra por sí solo SOTA oficial, equivalencia de protocolos temporales, superioridad universal frente a todas las referencias, ni que más datos sean la causa aislada de una mejora: también cambió la construcción de las entradas superiores.

La lista original de objetivos de AGENTS.md incluía calibración de incertidumbre, riesgo multihorizonte, robustez extensa, demo online y exportación completa. Este cierre no convierte cada uno en una prestación certificada del candidato final. La exportación de la cabeza no sustituye a exportar toda la cadena desde sensores; ROI anotadas no sustituyen a un detector/tracker; una tasa de entrenamiento o latencia de cabeza no certifica tiempo real del producto. Estas diferencias son parte de la evidencia científica que debe acompañar al whitepaper.
