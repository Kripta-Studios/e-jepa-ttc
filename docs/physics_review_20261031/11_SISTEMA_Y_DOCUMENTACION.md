# Sistema y documentación: mejoras que no requieren otra representación

**Actualización posterior al corte base:** el screen149 terminó durante esta revisión; F0 pasa y el router no. La ampliación de F0 está registrada en curso y la conservación de historias eAP sigue pendiente. Los pasajes que describen el screen como pendiente corresponden al corte inicial. Véase [actualización de cierre](13_ACTUALIZACION_CONCURRENTE.md).

## Runtime

El paper registra p95 de 29,14 ms desde llegada programada a salida completa en un replay de 250 consultas de un objeto durante cinco segundos. 201 respuestas alcanzan o superan 20 ms. El segundo reloj, desde RAW disponible, da 25,25 ms. Cero drops y paridad no equivalen a cumplir deadline. Se excluyeron disco, loading e inicialización con historia. [Runtime del artículo](../../paper/sections/runtime.tex).

La entrega adaptada del 31 mide una ruta distinta: media 1,740290 s por consulta, desglosada en 1,158298 s de preparación, 0,580139 s de grafo nativo y 0,001853 s de geometría/router. No contradice el replay de milisegundos: cambian inputs, caché y frontera. Tampoco la diferencia frente a 3,268250 s del envío previo es una ablation controlada. [Informe adaptado](../evttc_submission_20261031/REPORT.md).

Por ello, añadir un solver sensorial en inferencia exige justificar su coste total. Un auxiliar de entrenamiento que se retire después es atractivo, siempre que conserve el beneficio. Para mejorar el sistema actual puede ser más productivo optimizar extracción, reutilización causal de historia y scheduling que reducir la cabeza de fusión.

Las optimizaciones previas conservaron resultados negativos de BF16, acumulación, batch/RNG y transporte compilado. Algunas rutas CUDA Graph pasaron solo en forwards preparados; otras estuvieron activas en entrenamiento; otras no se admitieron. No se debe extrapolar un speedup aislado a todos los linajes. [Historia de precisión y transporte](../engineering_history_20261027/03_CUDA_PRECISION_AND_TRANSPORT.md).

## Observabilidad e incertidumbre no son lo mismo

Una medida puede tener alta precisión bajo su modelo y estar sesgada porque ese modelo no representa el movimiento. Un predictor puede ser incierto por falta de datos o por extrapolación de pesos. La fusión debe distinguir calidad de medición, incertidumbre de proceso, error de dominio y calibración empírica.

La entrega reciente no lleva un intervalo calibrado después del refit final. La anchura aprendida de fase tampoco permite anunciar una probabilidad de contacto. Un módulo de observabilidad puede aportar una variable útil; sus intervalos necesitan evaluación en grupos reservados, sin usar los mismos residuos de ajuste como garantía de cobertura.

Las pruebas de causalidad deben abarcar ROI, reloj, selección de frames, normalización y cachés, además de máscaras de atención. El futuro usado como objetivo durante entrenamiento no puede modificar el input pasado. Un cambio en productores obliga a reconstruir las features downstream; las antiguas cachés OOF no certifican conservación tras cambiar el modelo.

## Deuda documental encontrada

Los documentos [technical_report.md](../technical_report.md), [model_card.md](../model_card.md), [dataset_card.md](../dataset_card.md) y [EXPERIMENT_LEDGER.md](../EXPERIMENT_LEDGER.md) conservan estados V5–V8. Su contenido histórico es útil, pero sus nombres genéricos pueden hacer que se lean como estado vigente. `EXPERIMENT_LEDGER.md` aún lista partes de V8 como planned, mientras el paper posterior resume resultados completos.

AGENTS.md aclara que la carta JEPA original es histórica y el nombre actual es PHASE-TTC. Hay que mantener esa misma separación en los puntos de entrada. No conviene marcar como satisfechos todos los requisitos originales porque existe un paper: calibración, varios tipos de réplicas, inferencia con localización externa y reproducción de sensores siguen teniendo límites.

Recomendaría un índice de estado con cuatro campos obligatorios: fecha de corte, linaje del modelo, rol de población y estado de ejecución. Cada informe generado debería enlazar su recibo final; cada plan, indicar que no prueba ejecución. Mantener los snapshots antiguos sin sobrescribir sus tablas.

El artículo ofrece 99 registros de evidencia, pero no todos distribuyen predicciones o originales. Una proyección pública adicional de Stage68 con causas de rechazo y soporte permitiría estudiar la hipótesis de observabilidad sin confundir digest con reproducción. Un inventario completo de parámetros/VRAM de la cadena también cerraría una carencia reconocida.

## Acciones razonables y alcance de esta entrega

Esta revisión entrega análisis, registro Git, índices y recálculos en una carpeta nueva. No actualiza los informes históricos, no promueve CanonicalPair y no abre entrenamiento JEPA. Las acciones recomendadas para una futura tarea son terminar el screen registrado, instrumentar causas de falta de soporte y preparar una tabla de linajes/resultados que evite mezclar artículo y adaptador.

El mayor valor inmediato puede venir de descubrir que una operación de preparación o una convención temporal elimina una señal disponible. Si ese diagnóstico falla, se conserva como resultado y se investiga otra hipótesis. La propuesta de un estado físico debe competir con esa alternativa pequeña, con RGB solo y con la receta actual, bajo las mismas consultas.
