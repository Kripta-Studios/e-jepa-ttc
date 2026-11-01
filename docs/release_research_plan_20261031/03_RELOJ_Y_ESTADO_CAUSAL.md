# C-CLOCK: aprovechar la dinámica de aproximación en el release

## Qué error intentamos reducir

El contexto actual tiene acceso a observaciones pasadas, pero disponer de una GRU y tiempos no garantiza que transporte correctamente una medición antigua al instante actual. La hipótesis es que una dinámica explícita reduce error bajo cadencia irregular, historial corto o soporte intermitente. Su riesgo principal es retrasar frenadas y mantener una aproximación que ya cambió.

No añadir directamente una loss de suavidad a TTC: una trayectoria suave puede ser incorrecta. El beneficio tiene que superar un filtro causal sencillo y el contexto actual usando los mismos sensores y la misma historia máxima.

## Estado y ecuaciones

Para profundidad positiva en coordenadas de cámara, q=−Zdot/Z. Bajo velocidad axial constante:

```text
FΔ(q) = q / (1 − Δq)
T(t+Δ) = T(t) − Δ
```

La transición vale para q firmado y antes de cruzar contacto. Con aceleración, `qdot=q²−a_rel`, donde `a_rel=Zddot/Z`. El estado físico no se obtiene invirtiendo la salida saturada ±60 s: conservar fase cruda y su conversión anterior al suelo de emisión. No aplicar positividad forzada a la banda de recesión.

Una parametrización alternativa es `[ρ,vρ]`, con `ρ∝Z`, `ρ_next=ρ+Δvρ` y `T=−ρ/vρ`. La escala puede fijarse al inicio del track, pero no normalizar ρ de manera independiente en cada instante. Con aceleración constante de ese estado, añadir `aρ` da una evolución polinómica; no presupone conocer frenadas futuras.

La cinemática proyectiva tiene antecedentes en [Horn, Fang y Masaki](https://people.csail.mit.edu/bkph/articles/Time_To_Contact.pdf). Las ecuaciones y el diseño de intervención aquí son una propuesta para PHASE. Cambios de orientación, de cámara o del punto de mínima profundidad pueden invalidar la correspondencia entre expansión y target.

## Primer contraste: observador sin reentrenar productores

| Brazo | Comportamiento | Pregunta |
|---|---|---|
| D0 | R0 completo sin modificación | Referencia principal |
| D1 | EWMA causal con retraso medido | ¿Basta suavizar? |
| D2 | Persistencia y transporte determinista de la última medida | ¿Basta usar el reloj sin incertidumbre aprendida? |
| D3 | Observador de q con predicción y actualización por innovación | ¿Ayuda distinguir predicción y medición? |

D1–D3 consumen el mismo historial causal, no acumulan una memoria ilimitada frente a los 650 ms de R0. Para la comparación principal, reconstruir el observador dentro de esa ventana; una variante recursiva de memoria mayor es otro experimento y debe igualarse con un control de mayor historia.

En D3, la predicción avanza con FΔ y compara la nueva medida mediante `innovation=q_measured−q_predicted`. Usar una ganancia acotada y ruido de proceso explícito, ajustados solo en desarrollo. Estimar ruido de medición con residuo/soporte si la evidencia lo permite. No asumir independencia: ventanas de eventos y salidas de contexto comparten datos.

Para un filtro gaussiano aproximado, la sensibilidad de transición es `dF/dq=1/(1−Δq)²`; crece cerca del contacto. La propagación debe reflejar esa fragilidad. Si el intervalo atraviesa la singularidad, marcar transición no admitida y usar la medición/fallback registrado; no forzar certeza numérica.

Evitar contar dos veces la misma evidencia al filtrar la salida R0, que ya resume historia. Comparar por separado un observador sobre medidas actuales de expertos y un postfiltro sobre R0. El segundo es un baseline de sistema, no una fusión bayesiana óptima.

## Segundo contraste: contexto aprendido con estado físico

Abrir D4 solo si D3 muestra una ventaja o identifica una carencia concreta. Conservar productores y PAIR; entrenar un reemplazo o adaptador de un contexto, primero E_CTX, que entregue estado `[q,a_rel]` o `[ρ,vρ]` junto a sus diagnósticos. Mantener la interfaz física superior y la regla de emisión. Recalcular entradas de fusión y comparar inserción congelada con adaptación de descendientes bajo control igualado.

Comparadores D4: contexto original reentrenado con presupuesto igual; predictor temporal libre de capacidad similar; dinámica física; dinámica física con futuro permutado. No atribuir a la ecuación una mejora que también logra el contexto reajustado.

La pérdida propuesta combina la tarea TTC actual con predicción de observables futuros. Por ejemplo, bajo escala isotrópica y velocidad constante, el cambio logarítmico futuro esperado es `rΔ=−log(1−Δ*q_t)`. Compararlo con correspondencias/medidas futuras independientes, admitidas por visibilidad. Usar varios Δ reales y normalizar unidades. Los pesos se fijan en desarrollo; preservar la receta principal de fase del release para evitar que el efecto sea simplemente cambiar de métrica.

La consistencia entre dos estados estimados puede ser una pérdida auxiliar, pero no basta: ambos pueden colapsar o compartir sesgo. Incluir supervisión de tarea, anclajes de movimiento conocidos y controles de predicción constante. La composición de una transición exacta consigo misma es una identidad algebraica y no proporciona señal de entrenamiento.

## Aceleración y decisión causal

Un corrector a_rel libre puede explicar cualquier secuencia y vaciar la restricción. Limitar complejidad y cambio temporal con un prior fijado en desarrollo, junto a innovación observable. Comparar también un régimen de ruido de proceso mayor sin cabeza de aceleración: quizá obtiene la misma mejora con menos coste.

El selector de régimen en inferencia utiliza solo el prefijo observado. Las etiquetas de aceleración sirven para análisis o supervisión autorizada, no para escoger retrospectivamente dónde aplicar la ley constante en evaluación. Dos futuros distintos tras un prefijo idéntico deben compartir la misma predicción inicial; el modelo no puede anticipar una maniobra arbitraria.

## Éxito, fallo y coste

Medir MAE/MiD por banda, error futuro por horizonte, signo, excedencias, retraso de aviso y respuesta tras frenada. Fijar antes del ensayo eventos de cambio de régimen en el banco controlado. Una reducción de oscilación con más retraso no cuenta como éxito por sí sola.

Si D3 supera D1/D2 y conserva el dominio nativo, confirmar el observador como mejora de sistema aunque no cambie el encoder. D4 solo se conserva si mejora frente al contexto reentrenado y al predictor libre. El auxiliar futuro puede retirarse; si el estado dinámico permanece en inferencia, medir su memoria por track, reinicios, huecos y coste.
