# 06 — Intervenciones mínimas y consistencia causal

## Regla de decisión

Una intervención empieza con la hipótesis que superó G3 y un contrato registrado antes de comparar candidatos. El objetivo de G4 es probar esa hipótesis con el menor cambio que la distinga de sus alternativas. No se promete una reducción del error ni se implementan todas las opciones de este documento.

| Evidencia de diagnóstico | Primera intervención | Control obligatorio |
|---|---|---|
| Violación de geometría/reloj/unidad | Corrección del adaptador | Regresión que falla antes y pasa después |
| Dependencia del prefijo disponible | Política causal de arranque motivada o entrenamiento de ese régimen | Mismas consultas, sin excluir warm-up |
| Predicciones útiles pero variabilidad temporal excesiva | T0, transporte/consistencia causal | Baseline y prueba bajo aceleración |
| Entradas correctas, error sistemático aprendido | Piloto de fusión F0/F1 | Misma capacidad y población |
| Sin datos que distingan causas | Adquisición o documentación adicional | No convertir test expuesto en validación |

El plan original ya exige preservar la entrega y cambiar una cosa por experimento. Aquí se concreta cómo hacerlo. [Plan base](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/superpowers/plans/2026-10-30-evttc-transfer-diagnosis.md).

## 1. Corrección de un defecto demostrado

1. Escribir una descripción de la violación en términos de entradas y salida esperada; incluir el contrato externo o derivación física que la establece.
2. Añadir un fixture mínimo que reproduzca el fallo sin acceder al test.
3. Conservar la versión anterior y crear otra política de adaptador con ID, hashes y fecha.
4. Implementar únicamente la corrección necesaria. Evitar añadir a la vez recorte diferente, filtro temporal y reentrenamiento.
5. Ejecutar la regresión afectada y comprobar cobertura, causalidad y formatos de salida.
6. Comparar el baseline y la variante sobre desarrollo. Las ejecuciones sobre competición ya conocida se etiquetan como diagnóstico posterior.
7. Registrar también casos en los que la corrección física empeore la métrica. Puede ser correcta y revelar otro sesgo del modelo.

Un ejemplo válido sería un eje inequívocamente intercambiado bajo la especificación del sensor. Un ejemplo insuficiente sería que ampliar una ROI concreta reduzca el error oficial. Los offsets temporales se determinan con evidencia de reloj, no mediante búsqueda del mínimo error TTC.

La política actual conserva pesos y fase canónica de 100 ms mientras adapta las cadencias reales. No cambiar esa coordenada por 80 ms a partir de un supuesto: primero aplicar el fixture temporal de [05](05_DIAGNOSTICO_DE_SENSORES_Y_MODELO.md). [ADR-0043](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/decisions/ADR-0043-evttc-competition-sensor-adapter.md).

## 2. Arranque: resolver soporte sin ocultar consultas

El adaptador existente ya utiliza solamente prefijos grabados y distingue silencio observado de historia ausente. El rechazo matemático afecta a la observación completa y queda registrado; no equivale a eliminar la consulta. [Preparación](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/operational/evttc_competition/prepare.py), [runtime](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/operational/evttc_competition/runtime.py).

Si A03 confirma que el arranque domina en desarrollo:

- Mantener un indicador causal de duración grabada, evento observado y disponibilidad por modalidad.
- Comparar una única política motivada con la original, usando los reinicios predeclarados.
- Si se entrena, incluir prefijos de distintas longitudes creados solo desde grabaciones de ajuste.
- Si se cambia admisión, informar cobertura y cada motivo. Una variante con abstención no sustituye a la comparación de población completa.
- No duplicar frames, generar eventos anteriores a la grabación ni completar una predicción con una constante elegida a partir del test.

Una política temporal que necesita historia debe definir cómo responde a la primera consulta. Conservar la salida disponible del modelo como baseline inicial es un control razonable; cualquier fallback adicional necesita un contrato y validación propios.

### Dominio representable y TTC muy corto

Auditar la distribución del target de desarrollo antes de asumir que el readout cubre toda la tarea. La conversión experta de fase rechaza `0<T<=0,1 s`; la conversión final tiene además límites numéricos de fase. Si aparecen targets válidos en ese intervalo bajo la definición acordada, existe una limitación estructural para representarlos exactamente. No sabemos por esa observación si ese es el problema de Slider. Conservar esas consultas y cuantificar la masa afectada: excluirlas ocultaría parte de la tarea. [Dominio y límites nativos](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/src/e_jepa_ttc/simplex_t/phase.py).

Revisar también el lado negativo: aunque el encoder de fase acepta TTC negativos finitos, el clamp final inferior limita las salidas negativas más próximas a cero. Con los límites actuales, el TTC emitido positivo mínimo es aproximadamente 0,100034 s y el negativo más cercano a cero aproximadamente −0,1000002 s; la magnitud máxima se satura en 60 s. Son consecuencias algebraicas del código, no una distribución observada de errores. El mapa completo de soporte debe probarse con fixtures antes de afirmar cobertura de una tarea firmada nueva.

Cambiar a una coordenada con otro soporte, por ejemplo inverse TTC con un readout diseñado para el dominio requerido, constituye una variante nueva y puede exigir reentrenar. Debe aislarse en un experimento propio. No resolverlo cambiando automáticamente 0,1 por 0,08 ni recortando las etiquetas o los errores del scorer.

## 3. T0: consistencia temporal con física explícita

Para distancia `Z` y velocidad relativa de cierre constante `v`, `T=Z/v`. Durante un intervalo sin cambio de objeto ni de velocidad:

```text
T(t + dt) = T(t) - dt
q(t + dt) = q(t) / (1 - dt*q(t)),   q = 1/T
```

Estas relaciones son aproximaciones bajo hipótesis concretas. No se aplican sin control a aceleración, giro, oclusión, cambio de objeto o cruce de la singularidad de contacto.

El repositorio ya tiene `blend_current_inverse_ttc`, que transporta la estimación del par anterior al actual y comprueba su dominio. También dispone de contextos temporales. T0 debe aportar un beneficio adicional verificable; no debe anunciar como novedad algo que la cadena ya realiza. [Productor y transporte existente](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/src/e_jepa_ttc/models/causal_scale_ttc.py).

### Diseño mínimo propuesto

1. Registrar si el transporte nativo se activa y con qué configuración real.
2. Elegir el estado que T0 va a transportar: por ejemplo, inverse TTC del contexto; evitar varios filtros en cascada.
3. Inicializar en la primera observación válida y reiniciar al cambiar identidad de track, reiniciar reloj o perder soporte según una regla fijada.
4. Transportar desde el timestamp efectivo de la estimación al de la consulta.
5. Desactivar el transporte cuando `1-dt*q` no sea positivo con margen numérico o haya evidencia causal de transición incompatible.
6. Combinar observación actual y predicción transportada con una regla registrada de antemano. Su parámetro se selecciona en desarrollo, no por la curva de competición.
7. Conservar la estimación original y todos los indicadores de uso del filtro en el ledger.

La máscara de validez puede usar soporte, discontinuidad de track y señales causales de aceleración o inconsistencia de movimiento. No puede usar el error GT para decidir cuándo corregir. Si no se puede inferir aceleración con fiabilidad, evaluar explícitamente ese régimen y limitar la afirmación; no declarar que la máscara resuelve lo que no mide.

### Qué comparar

Baseline sin nuevo filtro frente a T0 bajo cadencias, prefijos y perturbaciones fijadas. Publicar MAE, error relativo conforme al contrato, peor grabación, cobertura, cambios de signo, respuesta a una caída real de TTC y retraso introducido. Informar resultados por aceleración o cambios de objeto si están anotados de forma independiente.

Una media móvil de TTC puede retrasar señales críticas y promediar magnitudes de signos opuestos de forma engañosa. No añadir una EMA general solo porque suaviza la gráfica. Cerca de `q=0`, el TTC diverge; cerca de contacto, el transporte tiene una singularidad. Deben tener estados y métricas explícitos, no clamps elegidos tras ver fallos.

## 4. Consistencia como pérdida, si hay datos de ajuste

Una alternativa posterior es penalizar en pares válidos la diferencia entre `q_pred(t+dt)` y el transporte de `q_pred(t)`. Usar pérdida robusta, máscara causal fijada y un peso registrado. No sustituye la supervisión ni demuestra exactitud: una secuencia incorrecta puede ser perfectamente consistente.

El experimento separa tres cosas: supervisión igual al baseline, regularizador adicional y arquitectura. Mantener la misma arquitectura y datos al probar el regularizador. Los pares proceden de grupos de ajuste; los pares futuros pueden servir como objetivos durante entrenamiento, pero nunca como entradas de inferencia. La selección y evaluación recorren causalmente la secuencia.

## 5. Promoción o cierre negativo

Promover únicamente con el criterio de selección sellado en [09](09_MATRIZ_EXPERIMENTAL_Y_DECISIONES.md) y confirmar según [10](10_CONFIRMACION_ESTADISTICA_Y_SOTA.md). No abrir nuevas variaciones porque la elegida falle en el holdout. Una pérdida de cobertura cambia la tarea; una mejora media con degradación grave en un régimen debe quedar visible.

Si T0 no mejora o empeora respuesta dinámica, conservar el resultado negativo y pasar a la hipótesis siguiente justificada. Si no aparece un defecto y las predicciones muestran sesgo estable, continuar con [07](07_FUSION_Y_FINE_TUNING.md). El paquete original permanece intacto y cada variante tiene identidad separada.
