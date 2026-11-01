# Lecciones del historial que delimitan las nuevas ideas

## Representación predictiva y estabilidad

La comparación Global A0/Dense A1 en EvTTC tuvo cinco folds y tres semillas: Dense mejoró algo el MAE pero empeoró error relativo y el score preregistrado, con mayor coste. No respalda añadir tokens o resolución por defecto. Los ensayos sintéticos de nuisance mostraron que un objetivo temporal residual puede conservar dinámica en un régimen y fracasar cuando cambia el nuisance por frame. La varianza o rango efectivo del embedding no certifica que contenga la variable de interés. [early.tex](../../paper/sections/early.tex).

Esto favorece la propuesta de medir mecanismos bajo intervenciones. También impide declarar victoria porque el embedding no colapse, prediga bien apariencia o tenga una dimensión llamada «física». El control debe desacoplar textura, event rate, movimiento, escala e iluminación.

## Foreground y altura

Object-LHR necesitó identificar el objeto; los recortes independientes eliminaron escala; el prior de movimiento de cajas resolvió parte del problema sin usar eventos. Las versiones de foreground aprendieron siluetas y alturas sin recuperar bien cambios entre instantes. En Causal Scale v5–v8, el operador con geometría ideal funcionó mejor que la estimación de sus entradas. La singularidad de TTC amplifica errores pequeños en la diferencia.

Un estado relativo coherente puede ser una alternativa a dividir dos alturas. Pero si su única supervisión es la misma caja ruidosa en cada frame, el cambio de nombre no elimina el problema. Hace falta una observación temporal distinta, correspondencia física y una escala compartida. [causal_scale_eap_screen.md](../causal_scale_eap_screen.md), [methods.tex](../../paper/sections/methods.tex).

## Transporte, residual e historia

V8 aporta una evidencia fuerte de dependencia de transporte dentro de los pesos finales y evidencia débil para la explicación «todo procede de la altura analítica». Su residual puede usar información física de correspondencia y también atajos. El replay por sí solo no decide entre ambos. Las intervenciones de tiempo, fondo y visibilidad propuestas permiten separar esas posibilidades.

X0.5, Stage61 y Stage62 muestran que superar un shuffle destructivo no basta: el método puede no superar una calibración, pooling o experto congelado. Por eso el piloto físico debe incluir un control de capacidad y otro auxiliar genérico, además del shuffle. [middle.tex](../../paper/sections/middle.tex).

SIMPLEX-T obtuvo mejoras con historia, pero H16 compartía productores y su confirmación de dos semillas cruzó cero. La comparación de agregación con edad versus sin edad identifica utilidad temporal parcial, no necesidad de una GRU ni aprendizaje de una ODE. La propuesta de dinámica anclada es distinta, siempre que compare contra historia de igual soporte y presupuesto.

## Stage68 y nuevas medidas físicas

El soporte de normal flow fue insuficiente: 368/8.192 consultas, 583/16.384 ventanas full-affine y cero reduced-isotropic. Mediana de cinco planos frente al requisito de 32. Esa cascada exige una autopsia de rechazos antes de añadir una red. El control reducido ya abordaba una parte del argumento de observabilidad. [stage66_69_results.md](../stage66_69_results.md).

El experto de expansión del 31 de octubre ya ajusta tres parámetros `q,bx,by`. La energía del histograma inducía contracción espuria; la correlación entre mitades pasó los fixtures de contornos. Con 50% de fondo estático aceptó dos respuestas incorrectas; con 75% se abstuvo. En Slider produjo 62/77 estimaciones y su combinación directa no mejoró el adaptador completo. Este resultado conecta mejor con confianza bajo modelos incorrectos y separación de superficies que con añadir una pérdida de expansión. [Informe](../research_log_20261031_event_expansion/README.md).

## Readout, adaptación y bandas

V14–V19 repiten un patrón: mejorar lo crucial puede perjudicar recesión, TTC largo o MAE. El ajuste de pesos de pérdida no garantiza resolverlo. La cabeza actual usa una fase que no pondera como segundos. Un reloj rígido puede mejorar continuidad y empeorar respuesta a frenada, como ya ilustra el diagnóstico de suavizado T0 con retraso de alerta de 100 ms. La comparación nueva necesita error dinámico y tiempo de respuesta, no solo smoothness.

V18 C1 TRUE/ZERO demuestra que ciertos eventos son útiles después de RGB. No demuestra que toda corrección de eventos lo sea ni que la fusión del artículo generalice mejor que RGB. Es compatible con una rama que aporta información en unos casos y amplifica errores en otros.

## Últimos pilotos: calidad, contexto y tiempo

Las campañas F2/Fcap y F2n/Fcapn prueban diez campos de soporte y reinicios nativos. Ninguna supera todas las métricas de IQ. En un fold, nueve campos eran constantes y quedaban enmascarados; esto limita qué mecanismo podía aprenderse. La combinación posterior habilita variación, pero sigue sin pasar. No basta con «añadir confianza» basada en duración y recuento.

El piloto PAIR localiza una transformación física mal reproducida con componentes congelados. Tiene mayor valor diagnóstico inmediato que otra búsqueda amplia de cabezas. Sin embargo, su corrección es casi tautológicamente equivariante en el bloque intervenido y todavía debe demostrar utilidad downstream. El contexto de eventos empeora en un caso, y la auditoría eAP solo cubre PAIR en la observación actual. [PAIR](../research_log_20261031_canonical_pair/README.md), [auditoría eAP](../research_log_20261031_canonical_pair/EAP_CURRENT_QUERY_AUDIT.md).

## Interpretación acumulada

El historial favorece preservar cantidades, relojes y correspondencias medibles, con controles que separen información, capacidad y prior. No prueba que el sistema haya agotado su potencial ni que se pueda resolver con física rígida. La oportunidad científica está en localizar una pérdida concreta de información y corregirla bajo hipótesis verificables; la oportunidad operativa inmediata incluye preparación de datos y calibración de fallos.
