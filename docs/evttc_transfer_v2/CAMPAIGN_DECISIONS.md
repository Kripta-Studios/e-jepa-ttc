# Alcance y decisiones de la campaña adicional

El ZIP propone una matriz condicional, no un producto cartesiano de entrenamientos.
La cabeza F1 de 1.955 parámetros se entrenó con las 92.315 filas disponibles y se
conserva en su paquete verificable. Las comprobaciones posteriores no autorizan
promoción a SOTA ni una afirmación de mejora universal.

| Bloque | Estado y evidencia | Afirmación permitida |
|---|---|---|
| D00 identidad/solapamiento | Auditoría previa de ocho grabaciones y 365 frames; exclusiones aplicadas al nuevo CV | Exclusión del solapamiento identificado, no de relaciones desconocidas |
| D01 genealogía | Nueve componentes y normalizadores identificados; ancestros/teachers parcialmente sin resolver | Candidato sellado identificado; no certificar ausencia de exposición ancestral |
| D02 protocolo | Siete familias, 24 grabaciones elegibles, 2.531 consultas; exposure histórico declarado | Reserva respecto al ajuste actual de la cabeza, no independencia histórica |
| E00 evaluador | Relojes, signos, referencias y CSV auditados; 257 targets locales sin soporte conservados | Diagnóstico local; no equivalencia oficial de las 434 consultas |
| E01 comparadores | Productores/contextos/bbox sobre población declarada; Garl en auditoría previa; CMax y STRTTC lineal en doce consultas externas | Comparación con cobertura y filas compartidas; puertos geométricos no oficiales |
| A00 físico | Fixtures de paralaje/endpoint/cadencia, trazas anteriores, overlays fijos y productores reales con reloj desplazado/futuro añadido | No se ha demostrado un bug físico que justifique cambiar constante de fase o geometría con GT |
| B0 corrección física | Perfil de lectura explícito para Slider-500; contrato original preservado | Compatibilidad de formato, no mejora aprendida ni calibración intersensor certificada |
| B1 arranque | Controles de eliminación de observaciones completas y disponibilidad retenida | Sensibilidad al soporte; no se fabrica historia ni se promueve una política nueva |
| T0 tiempo | Filtro fijo evaluado y prueba de aceleración | Menor error agregado con retardo de alerta: tradeoff, no promoción automática |
| F0/F1 fusión | Tres refits completos y 21 ajustes agrupados; selección de semilla 7 previa | Mejora relativa en grupos actuales, regresión de MAE/intervalos y resultado externo mixto/negativo |
| F2/Fcap calidad | No abiertos: anchura de contexto no explica consistentemente el error | Hipótesis de calidad adicional sin validar; no afirmar que sea inútil |
| P0 productor | No abierto: falta aislamiento concluyente del bloque y soporte independiente suficiente | No se atribuye el fallo a un productor concreto |
| J0/J1/J2 representación | Probes de contextos ejecutados; rama JEPA no abierta | Los contextos contienen señal TTC; no equivalencia con prueba directa de encoder ni evidencia de beneficio JEPA |
| C00 confirmación | Slider-500 fijado antes de leer etiquetas: F1 no mejora el error relativo global; parentesco de adquisición desconocido | Evaluación externa al ajuste actual, resultado negativo para superioridad general |
| R00 eficiencia | Caché cruda acotada, CUDA Graphs de productores, preloading de features, NPZ sin pérdida, paridad sobre 36 consultas/32 grabaciones | Menor coste de extracción con semántica conservada dentro de la frontera medida |
| Incertidumbre | Cobertura nominal 80% no alcanzada en CV; reescala solo train tampoco la alcanza | Intervalos nativos disponibles, calibración insuficiente |
| Robustez | Nueve condiciones crudas en 48 consultas fijas; pesos y targets iguales | Sensibilidad medida sobre desarrollo expuesto, no garantía fuera de distribución |

## Decisiones adoptadas y coste de equivocarse

1. Se mantiene la matriz condicional del ZIP. No se lanza JEPA o fine-tuning de
   productores sin un diagnóstico que permita atribuir su efecto. Si la decisión
   es demasiado conservadora, se pospone una mejora posible; no se pierde un
   resultado demostrado.
2. Las familias de escenario se usan como agrupación conservadora disponible,
   con exposición histórica visible. Si no coinciden con adquisiciones, los
   intervalos no describen independencia de adquisición; no se presentan así.
3. T0 no se incorpora por una mejora media que tiene un coste temporal medido.
   Su utilidad depende de un requisito de retardo todavía no proporcionado.
4. No se declara ganadora F1 por el error relativo de carretera cuando empeoran
   otros regímenes, incertidumbre o Slider externo. El paquete entrenado permanece
   disponible para investigación, sin ocultar estas regresiones.
5. No se usa distancia TTC real para corregir la ROI ni se reajusta el modelo
   usando Slider-500 después de abrir sus etiquetas. Una campaña futura que use
   ese diagnóstico debe reclasificarla como desarrollo expuesto.

## Lo que requiere información nueva

Genealogía faltante, identidad de adquisiciones relacionadas, extrínsecas Slider
autoritativas, equivalencia oficial del evaluador, márgenes prácticos y adquisiciones
nuevas no se resuelven ejecutando más épocas. El informe conserva esos límites y
los resultados negativos. Las ramas condicionales no ejecutadas no se presentan
como implementaciones o entrenamientos terminados.
