# 08. Límites, trabajo no ejecutado y siguientes pasos

## Estado científico al cierre

La campaña demuestra que existe señal útil para mejorar la combinación de expertos
con calidad causal y geometría de cajas. También demuestra que pequeños cambios
de fusión pueden tener efectos distintos en error relativo, MAE, colas de TTC,
arranque y robustez. La entrega final mejora la comparación local emparejada y
está técnicamente lista para evaluación externa.

No se ha demostrado que sea el mejor modelo posible, que domine todos los
comparadores geométricos, que generalice a adquisiciones nuevas de Slider o que
su score oficial mejore en la misma proporción que el local. El uso de cajas
suministradas es parte del protocolo: no se ha medido el rendimiento con un
detector de cajas imperfecto entrenado en esta campaña.

## Qué falta para una evaluación comparable

La primera dependencia es la correspondencia de targets. Se necesitan los valores
alineados, o suficiente descripción de sus fuentes/relojes, para reconciliar las
tres secuencias cuyo score original no se reproduce. Deben mantenerse las once
consultas locales sin cobertura hasta resolverlo. Extrapolar o cambiar un offset
para acercar el promedio al PDF introduciría una decisión ajustada al benchmark.

La segunda es la preparación sensorial. Las extrínsecas de Slider requieren una
fuente autoritativa o una medida independiente. El paralaje sintético justifica
investigar traslación/profundidad, pero no demuestra cuántos puntos de error real
proceden de ella. Una corrección con profundidad debe usar una señal disponible
en inferencia, nunca la distancia real del target.

La tercera es la independencia. Hay que reconstruir relaciones de adquisición,
teachers y decisiones históricas hasta donde sea posible, y adquirir o identificar
datos nuevos. Reservar hoy grabaciones que ya guiaron decisiones previas no
borra esa exposición. Los intervals bootstrap de cinco familias describen una
población limitada, con poca potencia y estructura temporal correlacionada.

## Qué no se ejecutó

| Línea del plan | Estado real | Razón para no etiquetarla como completada |
|---|---|---|
| J0/J1/J2 con objetivo dinámico | No ejecutado en esta campaña | No se aisló un límite de representación que justifique ese coste |
| Fine-tuning completo de encoders/productores | No ejecutado | Las intervenciones entrenadas actuaron en fusión/cabeza |
| V-JEPA congelado como encoder alternativo | Propuesta de investigación | No hay training/evaluación comparable realizados aquí |
| MotionJEPA trasladado a TTC | Hipótesis | Referencia bibliográfica no equivale a evidencia propia |
| Modelo temporal/spiking tipo REACT | Idea diagnóstica | No se implementó/reprodujo ese sistema completo |
| Detección 3D G2 | No ejecutado | Los documentos del 29 mantienen bloqueos de G0/G1 |
| CMax/STRTTC oficiales completos | No reproducidos bajo mismo protocolo | Hay controles locales acotados, con alcance distinto |
| Confirmación externa nueva de Slider | Pendiente | Slider-500/750/1000 ya forman parte del historial de exposición |
| Certificación de seguridad/calibración | No obtenida | Los intervalos son descriptivos y algunos labels entran en refits |
| Nueva evaluación oficial EvTTC | Pendiente del organizador | Tener un ZIP válido no produce un recibo oficial |

El nombre histórico E-JEPA-TTC no obliga a atribuir a JEPA mejoras de una fusión
entrenada sin ese objetivo. Los IDs y namespace se conservan por trazabilidad;
el método público mantiene el nombre PHASE-TTC.

## Orden de trabajo justificable tras la entrega

1. Enviar la ficha del modelo original y la nueva versión con su disclosure,
   cuando el usuario decida enviar los correos. Obtener el resultado oficial
   completo y los detalles de alineación pedidos.
2. Reproducir el score del modelo original por consulta antes de interpretar la
   diferencia oficial del nuevo. Registrar cualquier corrección del evaluador
   como versión nueva; no sobrescribir las evaluaciones antiguas.
3. Medir correspondencia intersensor en datos no usados para ajustar la solución.
   Evaluar una corrección mínima solo si hay un defecto físico demostrado y
   una señal de inferencia válida para corregirlo.
4. Incorporar adquisición nueva, especialmente Slider con arranque y varios
   regímenes de velocidad. Congelar receta y reglas antes de abrir sus etiquetas.
5. Perfilar preparación de sensores con una comparación emparejada: mismos
   inputs, pesos, precisión y hardware. Cambiar caché, batching o pipeline de
   forma aislada y exigir paridad causal.
6. Si el error sigue apareciendo en un productor sin señal útil, abrir probes y
   ajuste parcial del bloque identificado. El contraste JEPA dinámico debería
   mantener encoder, datos, teacher, presupuesto, cabeza y selección, comparando
   J2 frente a J1, y conservando J0 como control supervisado.

## Criterios de promoción

No basta con reducir loss, obtener un embedding visualmente atractivo o mejorar
una fila agregada. La promoción debe proteger cobertura, TTC corto, signo,
colas y coste, además de una métrica principal fijada antes. Debe indicar cajas,
modalidades, historia temporal, datos de ajuste, exposición y comparadores.

La versión robusta previa mantuvo su papel de referencia porque el router de
arranque no dominó confirmación y robustez. El nuevo envío no modifica esa
conclusión retrospectiva: tiene otra población de fit y otro replay. Su identidad
separada permite que el organizador decida cómo clasificar un método adaptado.

## Costes de las decisiones mantenidas

La receta fija reduce otra ronda de selección por benchmark, pero puede no ser
óptima para el contrato de cámara derecha. El fallback exacto conserva el comportamiento
fuente cuando falta historia, incluidos sus errores. La geometría sin traslación
evita introducir profundidad no permitida, pero puede conservar un error físico
importante. Las salidas firmadas preservan el método y delegan el postprocesado al
scorer, pero dejan estimaciones negativas visibles. Cada coste está documentado
para que una futura mejora pueda compararse con una referencia identificable.
