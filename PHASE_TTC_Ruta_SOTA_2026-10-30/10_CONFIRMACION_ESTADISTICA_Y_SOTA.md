# 10 — Confirmación estadística y alcance de una afirmación SOTA

## Qué debe quedar demostrado

Una variante merece promoción si mejora una tarea previamente definida, conserva el contrato de entrada y cobertura, supera controles informativos y reproduce el efecto fuera de los datos que inspiraron su diseño. Un primer puesto oficial acredita una posición concreta del leaderboard; una afirmación general de robustez exige más dominios y evaluación independiente.

El [plan original](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/superpowers/plans/2026-10-30-evttc-transfer-diagnosis.md) ya exige tres semillas, grupos independientes y apertura única de la reserva. Esta sección concreta cómo hacerlo sin confundir variación de entrenamiento, cantidad de consultas e independencia de datos.

## 1. Congelar la pregunta antes del experimento

Completar [la plantilla de experimento](plantillas/EXPERIMENTO.md) con hipótesis, modificación única, baseline, datos, métrica primaria, criterios de selección, presupuesto y condiciones de rechazo.

Para EvTTC, la propuesta inicial es error relativo macro por secuencia en desarrollo, acompañado siempre de micro por consulta, MAE, cobertura, errores por régimen TTC y peor secuencia. Si varios archivos pertenecen a un grupo, indicar su peso: la unidad del score y la unidad de independencia estadística pueden ser distintas. Para GarlTTC, conservar su definición oficial de MiD y sus reglas; no crear una media arbitraria con porcentajes EvTTC.

Fijar un margen práctico `δ` antes de observar los resultados de selección. Debe proceder del uso previsto o de una necesidad de ingeniería. Si todavía no existe, informar efecto e incertidumbre sin inventar un umbral universal del 5 % o 10 %. Hacer lo mismo con los límites de regresión por rango TTC, signo, cobertura y runtime.

**Ejemplo de contrato propuesto, pendiente de implementar y completar:**

```yaml
primary_metric: macro_sequence_relative_error
lower_is_better: true
independence_unit: acquisition_group
practical_margin: PENDIENTE_ANTES_DE_SELECCION
mandatory_coverage: all_required_queries
learned_confirmation_seeds: [7, 13, 21]
candidate_limit_per_hypothesis: 2
holdout_access: one_locked_comparison
guardrails: PENDIENTES_SEGUN_USO
```

No elegir la métrica, bins TTC o política de fallos después de ver cuál favorece al candidato.

## 2. Limitar la búsqueda y conservar todos los intentos

La primera campaña de una hipótesis contiene el baseline y **un máximo de dos candidatos motivados**. Un cambio de ventana, filtro, pérdida, inicialización o criterio de checkpoint cuenta como decisión experimental, aunque no añada parámetros.

Si aparece una señal que exige un control de capacidad, puede abrirse una fase posterior declarada, con su presupuesto y pregunta. Ese control debe comprobar si la ganancia procede de la información nueva o simplemente del número de parámetros. No usar nombres de fases distintos para esconder una búsqueda indefinida sobre la misma validación.

Registrar resultados negativos, fallos técnicos, parámetros, coste y qué información estaba disponible cuando se eligió cada configuración. Una interrupción por OOM no demuestra fracaso científico, pero tampoco desaparece del historial. La selección final debe poder reconstruirse a partir del registro.

## 3. Validación agrupada y anidada

La auditoría de [03](03_PROTOCOLO_Y_DATOS.md) define los grupos y su exposición. Cuando el tamaño de datos lo permite, usar:

1. **Reserva exterior:** grupos completos que no participan en ajuste ni selección de esta campaña; además, comprobar si son históricamente nuevos.
2. **Desarrollo interior:** entrenamiento y validación por grupos para elegir receta, hiperparámetros y checkpoint.
3. **Ajuste final:** entrenar la receta elegida con los grupos autorizados y una regla de parada previamente fijada.
4. **Confirmación:** evaluar una vez baseline y candidato cerrado en la reserva exterior.

Si se utiliza validación cruzada anidada, cada fold exterior evalúa una receta elegida únicamente mediante sus folds interiores. Normalizadores, selección de features, calibradores, teachers ajustados y reglas aprendidas se reconstruyen usando solo entrenamiento elegible. Reutilizar una normalización entrenada con todos los grupos rompe ese aislamiento.

Con pocos grupos, no forzar un gran número de folds. Publicar resultados por grupo, verificar que cada entrenamiento tiene soporte de tarea suficiente y reconocer qué dominios quedan fuera. Si los resultados exteriores se utilizan para diseñar otra intervención, pasan a ser evidencia de desarrollo expuesta; no conservan automáticamente el papel de confirmación final.

El [README del proyecto](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/README.md) documenta usos históricos de EvTTC-32. Por eso un reparto nuevo no basta para afirmar que la reserva nunca influyó en la investigación. Cuando toda la colección útil esté expuesta, usarla honestamente para desarrollo y confirmar en adquisiciones externas nuevas.

## 4. Semillas: confirmar una receta, no escoger una ganadora

Para pilotos aprendidos puede usarse semilla 7 según el presupuesto fijado. Después de seleccionar la receta, ejecutar semillas **7, 13 y 21** con los mismos datos, arquitectura, presupuesto y regla de checkpoint. Si el piloto ya constituye una ejecución válida de semilla 7 bajo ese contrato, puede conservarse; no requiere repetirla para inflar el número de runs.

Publicar cada semilla, promedio y dispersión de la receta. Fijar la semilla de despliegue antes de abrir la reserva; no elegir la de menor error reservado. Si se desea un ensemble, definirlo como sistema distinto, con combinación fijada y coste medido antes de confirmar.

Tres semillas no equivalen a tres datasets. Miden sensibilidad del entrenamiento, mientras que la variabilidad entre grupos representa cambio de adquisición. Para un filtro determinista, repetir tres semillas sin modificar nada no añade evidencia: utilizar grupos distintos y perturbaciones previamente definidas.

## 5. Diferencias pareadas e incertidumbre

Evaluar baseline y candidato sobre las mismas consultas y targets compatibles. Calcular diferencias por secuencia y por grupo, preservando el peso definido en la métrica primaria. Por ejemplo, `Δ = error_candidato − error_baseline`; un valor negativo favorece al candidato.

Cuando haya suficientes grupos para que el procedimiento resulte informativo, calcular un intervalo mediante remuestreo de grupos completos. Mantener emparejados ambos modelos y sus consultas. No remuestrear miles de ventanas correlacionadas como si fueran ensayos independientes. Para modelos entrenados, presentar por separado incertidumbre entre adquisiciones y variación de semillas; un promedio global no debe ocultar una semilla sistemáticamente mala.

Con pocos grupos, los intervalos pueden ser muy inestables. Publicar diferencias individuales, cobertura y magnitud del efecto; evitar una conclusión fuerte basada en un intervalo aparentemente estrecho. Las dos Slider conocidas y sus 69 consultas no permiten una inferencia poblacional fuerte por sí solas. [Población documentada](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/superpowers/plans/evidence/2026-10-30-evttc-audit.json).

La decisión combina el efecto, su relevancia práctica, incertidumbre y regresiones. Un intervalo que incluye ausencia de mejora obliga a describir evidencia inconclusa; no transforma automáticamente el candidato en inútil. Un efecto estadísticamente claro pero irrelevante para el uso tampoco justifica por sí solo una arquitectura más costosa.

## 6. Abrir una sola vez la reserva

Antes de abrirla, guardar un recibo con hashes de código, pesos, normalizadores, configuración, datos, scorer, regla de decisión y las tres réplicas. Ejecutar en un único paquete cerrado el baseline y el candidato que se pretende confirmar.

Si la confirmación falla, registrar el resultado. Puede continuarse investigación, pero la reserva consumida ya no decide la siguiente variante manteniendo una etiqueta de prueba ciega. Conseguir otros grupos para una afirmación confirmatoria nueva.

Una corrección demostrada del evaluador puede justificar recalcular ambos modelos con registro del cambio. No se utiliza esta excepción para modificar umbrales del predictor, quitar consultas o ajustar la métrica tras ver errores.

## 7. Qué afirmaciones son legítimas

| Evidencia obtenida | Afirmación apropiada |
|---|---|
| Mejora en datos utilizados para seleccionar | «Mejora de desarrollo bajo este protocolo». |
| Mejora en reserva de esta campaña, históricamente expuesta | «Confirmación en grupos no usados para el ajuste actual, con exposición histórica declarada». |
| Mejora en grupos o sesiones externos nuevos | «Mejor transferencia al dominio externo evaluado», acotada a su población. |
| Mejor score oficial comparable y permitido | «Mejor resultado/posición en este leaderboard, a esta fecha», con modalidades y entrenamiento declarados. |
| Mejoras consistentes frente a competidores fuertes y protocolos comunes en varios dominios | Evidencia para una afirmación SOTA más amplia, limitada a las tareas verificadas. |
| Adaptación supervisada EvTTC | Variante adaptada; no transferencia sin ajuste en EvTTC. |
| Solo simulación o dos Slider expuestas | Evidencia de mecanismo o diagnóstico; no robustez poblacional de Slider. |

Las reglas y comparadores deben verificarse en las fuentes oficiales de [EvTTC](https://nail-hnu.github.io/EvTTC/competition/) y [Garl-TTC](https://github.com/NAIL-HNU/Garl-TTC) al preparar una entrega. Declarar cajas suministradas, modalidades, historia causal, datos de entrenamiento, cobertura, fallos y cómputo. Mantener identidad distinta para una corrección del adaptador, una fusión ajustada y una arquitectura nueva, aunque compartan pesos parciales.

## Entrega final de la fase

Completar [el reporte](plantillas/REPORTE_RESULTADOS.md), generar todas las tablas desde resultados y conservar el candidato original. Adjuntar diferencias por grupo, cada semilla, fallos, coste y límite de la afirmación. Preparar una nueva entrega oficial solo cuando el protocolo la admita y el autor autorice el envío. Un resultado negativo documentado cierra un experimento y evita repetirlo; no debe ocultarse para mantener una narrativa de progreso.

Siguiente documento: [11 — Runtime, recursos y reproducibilidad](11_RUNTIME_RECURSOS_Y_REPRODUCIBILIDAD.md).
