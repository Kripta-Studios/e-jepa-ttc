# JEPA, información física y condiciones de reapertura

## Qué demostró el proyecto

V8 D2/D3 no pasaron la puerta de beneficio frente a controles scratch, encoder aleatorio congelado y futuro permutado. Hubo predicciones finitas sin que eso implicara una representación mejor para TTC. Los experimentos posteriores con supervisión TTC y profesores RGB no reparan retrospectivamente esa atribución. [early.tex](../../paper/sections/early.tex).

El piloto de profesor fijo del 31 de octubre entrenó nueve brazos: tres semillas por B_static, C_one_step y C_shuffle, 1.000 updates por brazo, batch 16. El predictor se retiró para evaluar el encoder mediante ridge de alpha 1. Su objetivo primario fue cambio de log-altura anotada, **no tasa por segundo ni TTC**. Las cajas aportan localización privilegiada.

| Brazo | MAE primario archivado |
|---|---:|
| B_static | 0,037632390 |
| C_one_step | 0,035640042 |
| C_shuffle | 0,035061886 |
| Inicialización aleatoria | 0,037196409 |

C−B: −0,001992348, IC95% [−0,004962095; +0,000818978]. C−shuffle: +0,000578156, IC95% [−0,002170051; +0,002925897]. Ninguno cumple el criterio de límite superior menor que cero. La mejora de estado futuro coexiste con peor predicción del incremento frente a los dos controles. [Análisis original](../jepa_closure_20261031/closure_analysis.json).

Esto es evidencia negativa de una receta y una sonda concreta. No prueba imposibilidad de JEPA, ni que todo latente sea inútil, ni que rollout nunca pueda funcionar. Tampoco autoriza seguir escalando porque «todavía falta probar otra variante». [ADR-0045](../decisions/ADR-0045-close-jepa-fixed-teacher-pilot.md) cierra el piloto y exige hipótesis nueva sustentada en evidencia.

## Qué aporta la literatura citada

El título enviado corresponde a la [v1 de What Can Latent World Models Know?](https://arxiv.org/abs/2607.27017v1). La [v4 de septiembre](https://arxiv.org/html/2607.27017v4) se titula *Physical Information in Multimodal Predictive Representations*. Distingue disponibilidad en observaciones, lectura desde el latente y uso en la predicción; presenta casos donde una lectura débil coexiste con dependencia dinámica y donde buena lectura no garantiza uso correcto. Su resultado no es una ley sobre todos los cocientes físicos ni una explicación causal del fallo de PHASE.

La inferencia útil para este proyecto es comprobar ambas cosas: recuperar q y responder correctamente cuando cambia el movimiento manteniendo apariencia. Un único probe lineal puede infravalorar información no lineal; un probe muy potente puede reconstruirla por sí mismo. Conviene comparar probes lineales y pequeños no lineales con capacidad fija, padres aleatorios, controles sobre datos crudos y evaluación por adquisiciones.

[Semigroup-JEPA](https://arxiv.org/html/2609.10464v1) usa rollout autorregresivo multistep con gravedad conocida como condicionamiento. En su formulación, iterar un predictor compartido compone por construcción sobre historia latente; no equivale a probar la ley continua exacta de q para intervalos arbitrarios. Su entrenamiento conjunto tampoco reproduce el profesor fijo del piloto local. La conexión merece estudio, pero no demuestra beneficio de TTC ni justifica reabrir la receta cerrada sin un diagnóstico adicional.

## Cuatro preguntas para un nuevo piloto

**Disponibilidad:** ¿puede recuperarse q desde los mismos datos permitidos, sin cajas futuras, profundidad GT o timestamps incorrectos? En sintético se conoce la verdad; en real se necesita una referencia de medición con hipótesis y error declarados.

**Retención:** ¿conserva el encoder esa información tras entrenar, frente a random/static/shuffle y supervisión comparable? Normalizar la expansión por el delta real es necesario para preguntar por una tasa, además del log-cambio que evaluó el piloto cerrado.

**Uso:** ¿una intervención sobre dinámica o un intercambio controlado de información física cambia la predicción con signo y magnitud esperados? Una dimensión latente con nombre físico que el predictor ignora no satisface el objetivo.

**Transferencia:** ¿el beneficio persiste en otra adquisición, sensor o iluminación, con el auxiliar retirado y con iguales reglas de emisión? Las mismas secuencias con más ventanas no responden esa pregunta.

## Qué reabriría y qué no

No reabriría «predicción de embedding futuro con más pasos» como campaña amplia a partir del resultado negativo. Sí consideraría un pequeño ensayo si la auditoría detecta que el tiempo físico se pierde en una representación concreta y una intervención restituye una medición útil. Entonces se podría comparar un objetivo anclado en incrementos observables con static y shuffle, sin cambiar a la vez teacher, encoder, dataset y presupuesto.

Tampoco obligaría al modelo a identificar metros y metros por segundo si el objetivo admite un estado relativo. La afirmación «estado mínimo» requiere definir suficiencia para la tarea y los nuisance necesarios para explicar el sensor. La contribución potencial sería demostrar una retención y uso robustos de q bajo hipótesis explícitas, no presentar el nombre JEPA como la explicación del resultado.
