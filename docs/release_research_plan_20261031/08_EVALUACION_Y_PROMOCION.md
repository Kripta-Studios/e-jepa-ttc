# Evaluación frente al release y criterios de promoción

## Pregunta primaria

El objetivo de producto es mejorar transferencia/fiabilidad manteniendo el dominio nativo del release ganador. Un candidato no hereda el primer puesto por conservar el nombre PHASE-TTC. Distinguir tres resultados: diagnóstico útil, candidato de desarrollo y release validado.

Para C-TIME rigen los criterios ya registrados de su screen. Para nuevas familias, este documento propone una regla conservadora v1, que debe sellarse antes de abrir su confirmación. Los márgenes son decisiones de investigación, no estándares de seguridad.

## Métricas obligatorias

| Aspecto | Medición y reducción |
|---|---|
| Dominio nativo | Overall MiD con pesos oficiales y cada banda; scorer fijado por hash |
| Transferencia | MAE y error relativo micro y macro por adquisición; mediana y p95 |
| Colas | Máximo, conteos >1/3/5/10 s y listado de fallos nuevos |
| Signo | Matriz aproximación/recesión, cercana a q=0 separada |
| Cobertura | Consultas requeridas, disponibles, fallback, inválidas y saturadas |
| Urgencia | Falsos negativos y falsas alarmas a umbrales fijados, tamaños de cada banda |
| Dinámica | Retraso tras cambio de régimen y error futuro por horizonte |
| Incertidumbre | Cobertura/anchura en fase o q, error frente a confianza y score propio si hay distribución definida |
| Runtime | Preparación, productores, contexto, fusión, salida host; p50/p95/p99, memoria y backlog |

La ausencia de banda negativa en una cohorte impide calcular el Overall completo; no redistribuir su peso. La salida +60 s por saturación no significa medición de un horizonte seguro. Mantener las reglas oficiales de target, dominio y fallos, junto a métricas adicionales.

Para comparaciones del scorer, reproducir aliases tal como exige el benchmark. Para inferencia, costes y resampling, reconocer consultas físicas y dependencias por secuencia. No contar los aliases como evidencia independiente.

## Regla propuesta para screening aprendido

Fijar como primaria de transferencia **MAE macro por adquisición** en una población y condiciones definidas de antemano. Mantener error relativo macro como guardrail. Un brazo pasa al siguiente nivel si mejora la primaria al menos 2% relativo en desarrollo, no empeora el error relativo, conserva cobertura y no crea una regresión sistemática de signo/urgencia. El 2% es un umbral de utilidad propuesto para acotar coste, no una mejora esperada ni una frontera científica universal.

Publicar deltas y CIs aunque no alcance ese 2%. Una señal pequeña puede ser interesante, pero requiere una decisión nueva registrada antes de otra confirmación; no rebajar el umbral tras ver el mismo test. Comparar siempre contra el control igualado y R0. En el brazo de incertidumbre, declarar por adelantado si la primaria cambia a un score probabilístico; no cambiarla después porque la MAE no mejora.

## Regla conservadora de confirmación v1

Un único candidato fijado pasa si: mejora la primaria al menos 2% relativo; su límite superior de CI pareado del delta es menor que cero; no pierde cobertura; y la evaluación nativa autorizada no muestra degradación en Overall, crucial ni negativa. Para afirmar no inferioridad estadística nativa con margen cero, exigir límite superior ≤0 en esos deltas; si los intervalos cruzan cero, declarar no inferioridad no demostrada, aunque la media parezca igual.

Ese margen cero es deliberadamente exigente. Si el producto acepta otro intercambio, registrar un margen práctico distinto **antes** de evaluar una nueva confirmación. No convertir el ruido estadístico en un permiso para degradar bandas críticas. Para varias afirmaciones simultáneas, fijar intervalos simultáneos o una corrección de comparaciones en el protocolo estadístico; conservar además los intervalos individuales descriptivos.

Los CIs requieren errores por consulta y grupos evaluables. Un recibo oficial que solo entrega agregados permite comparar esos agregados, pero no reconstruir bootstrap ni certificar esta regla estadística. En ese caso documentar conservación oficial descriptiva y obtener la confirmación estadística en una población nativa reservada con targets autorizados; no inventar incertidumbre a partir de cuatro promedios.

Con tres semillas, presentar todas y la media; el bootstrap por adquisición debe mantener juntas las predicciones de cada semilla/consulta. No tratar semillas×ventanas como observaciones independientes. Si alguna semilla cambia la conclusión de forma marcada, investigar estabilidad antes de refit global.

## Cómo preservar el resultado GarlTTC

Hay tres situaciones posibles:

1. **Predicciones Test12 idénticas:** misma población, formato, valores y scorer implican la misma puntuación para esos inputs. Esto puede verificarse sin ver targets, pero requiere identidad de toda la cadena de predicción, no una auditoría parcial PAIR.
2. **Predicciones distintas y targets privados:** no se puede demostrar conservación del score solo por cercanía numérica. Evaluar el candidato ya fijado mediante el procedimiento oficial disponible, sin convertir el leaderboard en búsqueda de hiperparámetros.
3. **Sin nueva evaluación oficial:** conservar R0 como release y llamar a la variante candidato de transferencia. Resultados internos TRAIN40 no acreditan mantenimiento del primer puesto.

El score oficial original sigue siendo un resultado válido de R0. Su evaluación posterior no se reescribe ni se mezcla con una receta adaptada. El primer puesto es un estado fechado del leaderboard, no una garantía de la nueva variante.

## Latencia y producto

El replay publicado tiene p95 29,14 ms en su frontera declarada y no certifica servicio sostenido a 50 Hz. Una mejora física no debe ocultar más historia, espera de frames futuros o retraso de salida. [Runtime del artículo](../../paper/sections/runtime.tex).

Comparar en el mismo hardware, precisión, tamaño de lote y entradas. Incluir warm start y cold start, huecos, reinicio de track y varios objetos. Medir memoria persistente por objeto, backlog y deadline misses. Como presupuesto inicial del candidato, proponer aumento ≤10% en p95 total frente al replay comparable de R0 y ninguna nueva dependencia futura; si no cumple, queda como resultado científico o perfil distinto, no sustitución automática de producción.

Retirar un auxiliar de entrenamiento puede mantener coste de inferencia; un solver, observador o gate retenido consume tiempo y estado. Reportar también coste de preparar y entrenar el candidato, aunque no aparezca en el runtime.

## Decisión y reversibilidad

Cada decisión incluye modelo/población/hash, control, deltas, CIs, excepciones y coste. Conservar R0, el candidato, la configuración y un mecanismo de selección de versión explícito. No mezclar pesos entre versiones por una recuperación parcial. El cambio de release requiere un paquete reproducible y evaluación de la cadena completa; esta tarea solo documenta ese camino.
