# 01 — Estado real, crítica del plan y decisión de continuación

## 1. El plan inicial acierta en el orden

El documento de 318 líneas propone identidad de datos → desarrollo independiente → evaluación contrastada → diagnóstico congelado → intervención mínima → confirmación. Ese orden es adecuado. Ya evita los errores más dañinos: entrenar con grabaciones de competición, rellenar etiquetas ausentes, elegir por score oficial o presentar desacuerdo entre modelos como precisión. La [fuente revisada](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/superpowers/plans/2026-10-30-evttc-transfer-diagnosis.md) distingue correctamente evidencia, hipótesis y trabajo pendiente.

Lo que falta para convertirlo en una ruta hacia SOTA es concretar la competencia real, hacer obligatoria la auditoría de genealogía, resolver la ausencia de Slider independiente, elegir controles de arquitectura informativos y fijar decisiones de continuación que no dependan de seguir mejorando una cifra conocida.

## 2. Evidencia y nivel de verificación

| Hecho | Evidencia disponible | Alcance de esta revisión |
|---|---|---|
| `main` coincide con `30b28b5c…` | Consulta de GitHub y checkout local del mismo SHA | Verificado en esta sesión. |
| 365 frames compartidos, ocho offsets | Anexo público con hashes y comparación por frame | Se leyó el recibo; no se reabrieron aquí los HDF5 originales. |
| 32 grabaciones y 24 candidatas | Inventario público y manifest versionado | Las 24 son candidatas, no 24 unidades independientes certificadas. |
| 354/365 queries de carretera dentro del reloj de etiquetas | Recibo de cobertura temporal | No demuestra igualdad de target con el scorer oficial. |
| Diez filas de errores recibidos | Anexo que transcribe el PDF del organizador | No se recibió ni reejecutó aquí el scorer privado. |
| Arquitectura y políticas del adaptador | Código del commit fijado y ADR-0043 | Inspección estática; no prueba de precisión ni nuevo replay de pesos. |
| Recursos del portátil y ruta eAP no montada | Auditoría histórica aportada | No son una comprobación actual del hardware o disco en esta sesión. |

Fuente: [anexo de auditoría](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/superpowers/plans/evidence/2026-10-30-evttc-audit.json).

## 3. Dónde se concentra el problema

| Secuencia | Consultas | Error relativo medio recibido (%) | Tiempo offline declarado (s/query) |
|---|---:|---:|---:|
| CCRs1-low | 50 | 7,57 | 1,982 |
| CCRs1-medium | 43 | 9,42 | 2,653 |
| CCRs1-high | 30 | 13,12 | 3,586 |
| CCRs2-low | 46 | 15,28 | 2,234 |
| CCRs2-medium | 55 | 9,34 | 2,534 |
| CCRs2-high | 59 | 7,99 | 3,006 |
| CCRm-low | 42 | 21,01 | 2,259 |
| CCRm-medium | 40 | 15,05 | 2,566 |
| Slider-750 | 51 | 132,00 | 6,192 |
| Slider-1000 | 18 | 156,88 | 9,157 |

Recalculando las filas redondeadas del anexo:

| Población | N | Media por consulta (%) | Media de secuencias (%) |
|---|---:|---:|---:|
| Carretera | 365 | 11,916630 | 12,347500 |
| Slider | 69 | 138,490435 | 144,440000 |
| Total | 434 | 32,040115 | 38,766000 |

Se emplean `micro = sum(n_s * error_s) / sum(n_s)` y `macro = mean(error_s)`. Son estadísticas derivadas de valores redondeados, no nuevos resultados oficiales ni MiD. El peso de Slider es 69/434 = 15,90 % de consultas; su contribución a la suma ponderada de error es aproximadamente 68,72 %. Esto justifica priorizar Slider para investigar el fallo global, sin convertir las dos secuencias conocidas en validación de selección.

Slider-750 y Slider-1000 necesitan diagnósticos diferentes. En la primera hay extremos en las ocho consultas iniciales, dos salidas negativas y siete observaciones de eventos rechazadas por dominio matemático. En la segunda, las 18 salidas son positivas y están entre 1,0714 y 1,9406 s, sin esos rechazos. Un único suavizado de arranque no explica ambas. Los siete rechazos son observaciones internas, no siete predicciones finales ausentes. [Fuente del diagnóstico](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/superpowers/plans/evidence/2026-10-30-evttc-audit.json).

## 4. Ocho mejoras necesarias sobre el plan inicial

### 4.1 Genealogía como requisito de entrada

No basta con comprobar que el entrenamiento futuro evita ocho carpetas. Hay que reconstruir qué datos usaron productores, contextos, cabeza, normalizadores, teachers y decisiones de selección. El repositorio contiene muchas campañas anteriores. Un recibo de hashes prueba identidad; no prueba que un checkpoint nunca haya visto una secuencia.

### 4.2 Separar ausencia de solapamiento y ausencia de exposición histórica

El [README revisado](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/README.md) ya describe 468 consultas etiquetadas en 32 secuencias y evaluaciones anteriores. Reservar ahora unas grabaciones puede impedir ajuste futuro sobre ellas, pero no borra las decisiones que ya se tomaron viendo sus resultados. Hay que registrar ambas dimensiones y buscar datos realmente nuevos para una confirmación fuerte.

### 4.3 Tratar Slider como carencia de datos

El inventario no contiene Slider de entrenamiento independiente. Las 24 candidatas no resuelven ese hueco. Las opciones concretas son nuevas grabaciones del testbed, datos adicionales de los organizadores o un conjunto externo compatible. La simulación ayuda a falsar hipótesis físicas, pero no sustituye esa evaluación real.

### 4.4 Validar el instante del TTC

Dos fórmulas legítimas de expansión pueden referirse al inicio y al final del mismo intervalo. La diferencia es exactamente el intervalo bajo velocidad relativa constante. Con TTC corto, un desfase pequeño puede generar un error relativo grande. Se necesita un fixture de expansión con respuesta analítica antes de ajustar escalas.

### 4.5 Usar la arquitectura que realmente se envió

La cabeza final `PhaseOnlyFinal` toma 20 coordenadas de un estado de 340 dimensiones, aplica un MLP `20→32→32→3` y suma un residual al estado RGB. Tiene 1.955 parámetros. No consume explícitamente varios diagnósticos de soporte/fiabilidad disponibles en el adaptador. Por tanto, la adaptación de esa cabeza y una ampliación controlada de sus entradas son experimentos baratos y motivados. No está demostrado que sean la solución. [Código](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/operational/final_refit/training.py).

### 4.6 Ampliar baselines

Además de Garl-TTC, hacen falta referencias geométricas y RGB solas. La página oficial de EvTTC incluye Image's FoE y CMax con resultados fuertes en varias secuencias; ignorarlas haría demasiado fácil una afirmación SOTA limitada a modelos aprendidos. Las cifras publicadas sirven de contexto; la comparación controlada requiere población, cámara, reloj y reglas comunes. [Benchmark oficial](https://nail-hnu.github.io/EvTTC/competition/).

### 4.7 Controlar objetivos incompatibles

GarlTTC emplea MiD y bandas firmadas; EvTTC publica error relativo. Ganar en uno no garantiza ganar en el otro. Los negativos, el horizonte de historia y las reglas de fallo deben permanecer visibles. Puede ser preferible conservar dos variantes con objetivos diferentes antes que presentar un supuesto ganador universal.

### 4.8 Aprovechar SOTA reciente sin convertirlo en una lista de arquitecturas

REACT aporta una idea de estado continuo con tiempos reales; MotionJEPA aporta un diagnóstico de pérdida de movimiento; V-JEPA 2.1 ofrece representaciones densas. Son fuentes de hipótesis. No justifican reemplazar nueve componentes antes de comprobar un posible defecto del adaptador o del target.

## 5. Prioridad técnica propuesta

| Orden | Acción | Motivo | Condición para abandonarla |
|---|---|---|---|
| P0 | Identidad, genealogía, target y soporte | Un contrato erróneo invalida cualquier comparación posterior | Si falta evidencia, entregar diagnóstico limitado y seguir solo con datos elegibles. |
| P1 | Geometría, reloj, referencia temporal y arranque | Pueden explicar un fallo fuerte entre sensores sin cambiar capacidad | Si pasan fixtures y evaluación independiente, no seguir buscando correcciones ad hoc. |
| P2 | Baselines por rama y adaptación pequeña de fusión | La cabeza es pequeña y el anclaje RGB puede fallar fuera de dominio | Si no mejora fuera de los grupos de ajuste, no escalar capacidad por inercia. |
| P3 | Consistencia temporal físicamente condicionada | Puede reducir innovación espuria conservando respuesta a cambios | Si solo suaviza curvas y retrasa señales útiles, rechazar. |
| P4 | Ajuste parcial de productor o representación dinámica | Solo si el error se localiza antes de la fusión | Si controles equivalentes no mejoran, conservar resultado negativo. |
| P5 | JEPA de movimiento / SSM / nuevas modalidades | Investigación con mayor coste y atribución más difícil | Sin ganancia incremental y datos nuevos, no abrir una campaña larga. |

La conclusión práctica es continuar con el sistema actual y reducir incertidumbre de forma secuencial. El trabajo de mayor valor inmediato es un experimento capaz de separar error de protocolo, error geométrico y error aprendido.
