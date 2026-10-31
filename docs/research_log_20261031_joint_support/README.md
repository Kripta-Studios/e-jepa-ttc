# PHASE-TTC: interacción entre soporte y reinicios de contexto

Esta campaña evalúa una comparación adicional fijada antes de entrenar. Combina los reinicios reales con las entradas que describen sus relojes y recuentos. No se modificaron campañas anteriores ni se usó el benchmark para escoger configuraciones.

## Qué cambia

F2n conserva el router de 868 parámetros y Fcapn el control de 878. Ambos usan mitad de pérdida limpia y mitad con los 447 reinicios reales; no añaden corrupción multiplicativa QA. Los grupos reservados excluyen también todos sus reinicios. Se equilibran familias y consultas, dividiendo el peso entre las variantes admitidas de cada consulta.

Los 33 inputs originales y la escala del MAE se normalizan con train limpio. Las diez entradas nuevas de F2n se normalizan y se habilitan por varianza usando la unión de soportes limpios y aumentados admitidos de train. Validación no interviene. Esto cambia tanto los ejemplos/objetivo como el conjunto que normaliza el soporte; no aísla solo el efecto de la pérdida.

En el fold Slider, F2n puede aprender variación de ambos recuentos y de los intervalos RGB/eventos: máscara [0,0,1,0,1,0,0,1,0,1]. El experimento anterior con QA solo habilitaba el intervalo RGB. Esta diferencia estaba especificada antes de entrenar.

Tres semillas 7,13,21, ocho folds y 300 actualizaciones por ajuste: 48 fits. El control de selección sigue siendo IQ; IQn/F2/Fcap previos son referencias informativas, verificadas por hashes e identidades. No seleccionan modelos.

## Comparación de las condiciones ejecutadas

Todas las cifras siguientes son OOF de desarrollo con exposición histórica. Se agregan las tres semillas y se promedian las métricas por grabación. No son resultados oficiales ni nuevas métricas del modelo entregable.

| Condición | RAE carretera macro (%) | MAE carretera macro (s) | RAE Slider-500 (%) | MAE Slider-500 (s) |
|---|---:|---:|---:|---:|
| IQ | 17.748872 | 0.543418 | 20.851858 | 0.317744 |
| IQn | 17.429901 | 0.547671 | 25.885725 | 0.459413 |
| F2 | 18.023060 | 0.562809 | 20.369949 | 0.311921 |
| Fcap | 17.141145 | 0.533926 | 21.666019 | 0.351925 |
| F2n | 17.650689 | 0.557399 | 20.616663 | 0.331257 |
| Fcapn | 18.420412 | 0.579737 | 29.786266 | 0.395341 |

Selección conjunta: `None`. Tiempo: 79.546 s. Dominancia F2n/Fcapn: `True`.

La selección exige cobertura completa y no empeorar ninguna de las cuatro métricas limpias frente a IQ. Mejorar solo errores relativos no basta si empeora MAE; no se relajan los criterios después de ver los resultados. La dominancia entre candidatos se interpreta como asociación experimental, no como prueba causal de generalización.

## Verificación y consecuencia

Se verificaron los 48 checkpoints, 300 pérdidas por fit, aislamiento de grupos, masas de variantes, normalizadores y máscaras recalculados por fold, referencias inmutables y reconstrucción exacta de predicciones por fold y promedios de semillas. Las predicciones de reinicio son diagnóstico secundario y no intervienen en selección.

La campaña no ejecuta refit ni exportación automáticamente. Los candidatos rechazados no se evalúan en el benchmark para buscar una cifra favorable. eAP TRAIN40 activo seguiría siendo un control de resustitución y Test12 sin etiquetas no certificaría precisión.

Las pruebas de fusión han identificado intercambios entre carretera, Slider, historia corta, error relativo y error en segundos. No demuestran que el modelo sea irreduciblemente óptimo ni que más capacidad garantice mejora. El siguiente diagnóstico debe justificar una intervención en sensores o productores antes de abrir otra búsqueda de cabezas.

Este informe se regenera con `scripts/report_phase_support_context_joint.py`. Pesos, matrices y datos permanecen fuera de Git; se publican código y evidencia verificable. No se afirma SOTA.
