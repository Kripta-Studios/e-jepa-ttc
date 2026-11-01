# Auditoría del artículo LaTeX

El artículo final presenta una conclusión respaldada y acotada: PHASE-TTC obtuvo un resultado oficial competitivo como sistema completo, pero no establece beneficio JEPA, superioridad uniforme de la fusión ni servicio sostenido a 50 Hz. La documentación del 29–31 añade otra adaptación y pilotos que no figuran en ese snapshot.

## Lectura por bloques

| Sección fuente | Afirmación central | Evaluación |
|---|---|---|
| `main.tex`, resumen | Score oficial, transferencia mixta y deadline incumplido | Coherente con los resúmenes verificados |
| `introduction.tex` | Preservar escala y combinar mediciones semánticas | Hipótesis de diseño razonable; no atribución aislada de cada bloque |
| `final_model.tex` | Nueve componentes; fusión de veinte campos | Define el modelo realmente evaluado; 1.955 parámetros corresponden solo a la cabeza |
| `protocol.tex` | OOF por grupos y métricas firmadas | Explica exclusiones y diferencia entre mean MiD y Overall ponderado |
| `official_results.tex` | 35,80 frente a 45,03, snapshot del 28 | Recalculado 35,801451580824; comparación de sistemas con entradas/contexto diferentes |
| `transfer.tex` | 468 positivos, RGB mejor en media | Recalculado; p95 favorable no domina colas extremas |
| `evttc_competition.tex` | 434 consultas sin GT verificado a ese corte | Cobertura/desacuerdo, no precisión; documentación posterior debe ir en addendum separado |
| `design_analysis.tex` | Importancia de escala, transporte, fase y eventos V18 | Atribuciones limitadas; una referencia de evidencia parece incorrecta |
| `runtime.tex` | p95 29,14 ms con 201/250 respuestas ≥20 ms | No certifica 50 Hz sostenidos; explica el reloj de medición |
| `discussion.tex` | Límites de generalización, independencia y calibración | Coincide con resultados negativos y alcance de cada estudio |
| `methods.tex` | Voxel, JEPA, transporte, fase, incertidumbre | Separa el cociente geométrico de la fase canónica; distinción esencial para las propuestas |
| `evaluation_details.tex` | Genealogía, causalidad, bootstrap y cobertura | Contratos adecuados; requieren recibos, no solo una descripción |
| `derived_transfer.tex` | Bootstrap pareado y ejemplos post hoc | Declara selección posterior y dependencia por secuencia |
| `competition_details.tex` | Checkpoints Garl, preparación y huecos | No equipara mismos nombres de escena a mismas consultas |
| `early.tex` | Objetivos predictivos, objeto, geometría, V5–V8 | Conserva fallos y no convierte varianza latente en prueba de física |
| `middle.tex` | X0, Stages61–76, SIMPLEX-T y transferencia | Resultados modestos y gates fallidos delimitan ideas ya probadas |
| `late_research.tex` | V13–V19 | Distingue beneficio de eventos en C1 del linaje V16 desplegado |
| `history.tex` | Correcciones de test reutilizado y parents | Los commits inspeccionados respaldan los hitos críticos |
| `reproducibility.tex`, `evidence_index.tex` | Recálculo público y fuentes con hash | Algunas fuentes solo distribuyen digest o escalares; no permiten reproducir todos los runs |

## Qué queda establecido

El scorer oficial archivado da Overall MiD 35,801451580824 sobre 6.762 tokens y 0% de fallos según su criterio. El comparador publicado da 45,03. La reducción relativa es aproximadamente 20,49%. La afirmación «primero» está fechada y se refiere a tres entradas mostradas. Esta revisión no consulta un ranking actual ni convierte ese resultado en supremacía general sobre TTC.

La cadena final utiliza cuatro productores A5/C2F por modalidad, dos PAIR, dos contextos y una fusión. La fusión recibe veinte campos explícitos seleccionados del contenedor de 340 dimensiones. Las 160 coordenadas ocultas por modalidad no se fusionan directamente. Es incorrecto describirla como una red total de 1.955 parámetros o como event-only por un ID histórico que contenga `EVENT`.

La evidencia V18 TRUE/ZERO muestra utilidad de información de eventos en C1 con capacidad controlada: agregado 87,29 ± 0,87 frente a 92,47 ± 0,16. Las semillas repiten adaptación y cabeza desde parents compartidos. El resultado no aísla la contribución de eventos en el TRAIN40 final, que deriva de V16. Harían falta controles entrenados en ese linaje y soporte temporal comparable.

## Qué no se desprende del artículo

No se demuestra que imponer una fórmula de alturas causara el éxito. El replay V8 modifica rutas durante inferencia; no retrena cada combinación. Tampoco demuestra que la geometría no ayudara al entrenamiento o que el transporte carezca de estructura física. Ambas extrapolaciones exceden el contraste.

No aparece una ablation final que atribuya mejora a constancia de brillo o a una ley fotométrica. Las señales teacher, foreground y TTC no son intercambiables con una likelihood del sensor.

La reducción de error en fase no garantiza menor error en segundos o menos alertas tardías. La transformación comprime TTC grande y presenta singularidades. El régimen negativo es recesión, no una etiqueta de colisión pasada ni una probabilidad de seguridad.

La ausencia de fallos del scorer no prueba observabilidad ni calibración. Una salida saturada y finita puede ser una convención numérica con soporte insuficiente. Las anchuras en fase tampoco equivalen a probabilidades de contacto.

## Hallazgos documentales concretos

**Referencia de evidencia probablemente equivocada.** En [design_analysis.tex](../../paper/sections/design_analysis.tex), el párrafo de agregación por conjuntos con edad, 122,44 frente a 123,69, cita `M_X05`. La sección `middle.tex` atribuye esas cifras a SIMPLEX-T (`M_SIMPLEX`); X0.5 trata de nueve campos de transporte y calibración. Corregir/verificar ese enlace mejoraría trazabilidad. No cambia por sí mismo los números ni invalida el estudio.

**Fecha y modelo vigentes.** El artículo representa TRAIN40 al 28 de octubre. El adaptador `PHASE-TTC-Adapt-20261031` añade un router y otra población de ajuste. Se necesita una tabla de linajes que muestre qué pesos, salidas y métricas pertenecen a cada uno. No se deben sustituir cifras históricas del PDF con el nuevo resultado local.

**Acceso público parcial.** `M_GEOMETRY` y `M_X05` aparecen como digest-only en el registro público. Las cifras Stage68 se corroboran en `docs/stage66_69_results.md`, pero el bundle del paper no contiene su distribución de rechazos por sí solo. El lector externo necesita una proyección de conteos/causas o acceso al recibo original para recalcularlas.

**Cadena score→predicciones.** El artículo reconoce que el JSON del scorer no incluye hash de las predicciones enviadas. El score tiene procedencia archivada, pero ese recibo aislado no establece un enlace criptográfico con un ZIP concreto. No debe describirse como una verificación byte a byte de toda la cadena oficial.

**Tamaño/coste.** Falta un inventario completo de parámetros y memoria de los nueve componentes. El replay de cinco segundos excluye disco, loading e inicialización con historia. La documentación posterior aporta CPU y otros costes, pero pertenecen a mediciones distintas.

## Cambios editoriales que recomendaría

Mantener el paper del 28 como snapshot; añadir un addendum del 31 con el cierre JEPA, el fallo de soporte/augmentation, el diagnóstico PAIR y la nueva entrega adaptada con su evaluador local. Corregir el enlace de evidencia de agregación temporal. Añadir un cuadro de «datos necesarios para reproducir» que diferencie escalares, predicciones, checkpoints y sensores. Ninguna de estas modificaciones requiere reescribir resultados negativos ni presentar hipótesis como resultados.
