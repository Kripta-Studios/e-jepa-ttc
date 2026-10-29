# Revisión independiente del plan: validez científica y protocolo

Fecha: 29 de octubre de 2026. Documento revisado: `SOTA_DEVELOPMENT_PLAN.md`, versión completa y correcciones posteriores SCI-01/02/03. Base de código: `24b8f36f6e4aa17c965f9526d8cf486b512a8046`. Revisor: agente separado del autor del plan, responsable de la auditoría científica y de población.

## Dictamen

El plan es consistente con el código y la evidencia examinados. No encontré un error material en sus cifras, en la interpretación de AP, en las exclusiones del GT, en la exposición TRAIN40, ni en la separación entre integración funcional y generalización TTC. No promete SOTA ni declara superados gates futuros.

**Aprobado, sin hallazgos materiales abiertos en el alcance revisado.** Las aclaraciones **SCI-01/02/03** están incorporadas y verificadas. El registro siguiente conserva sus motivos y cierres; no eran una invalidación de los resultados históricos.

## SCI-01 — Aclarar qué se mantiene fijo en los oráculos

**Prioridad:** media; corrección de especificación antes de implementar T04.

**Estado final:** cerrado. S0.2 especifica ahora el pool/regla para construir el oráculo y el nuevo matching del evaluador para informar AP. La prueba recomendada queda disponible para la futura implementación T04.

**Localización:** S0.2, «Oráculos de desarrollo con propuestas/matching fijados».

Es correcto mantener el mismo conjunto de propuestas y utilizar una correspondencia de diagnóstico estable para construir sustituciones. Sin embargo, no debe reutilizarse la lista de TP/FP del baseline al calcular AP de las predicciones transformadas: cambiar ranking, clase o geometría puede cambiar la asignación greedy. Dos propuestas que compiten por el mismo GT muestran por qué mantener fijo el TP anterior después de reordenar scores puede producir una cifra ajena al evaluador.

**Redacción propuesta:** «Mantener fijo el pool de propuestas. Definir y guardar la correspondencia usada para construir cada oráculo; al medir AP de las predicciones transformadas, volver a ejecutar íntegramente el mismo evaluador y su matching. Los residuos calculados con correspondencia fija se presentan como diagnósticos condicionales, no como AP. No llamar a cada sustitución un límite superior matemático si no se ha demostrado esa propiedad».

**Aceptación:** fixture con dos propuestas para un GT en el que invertir scores cambia cuál se empareja. La AP del oráculo debe coincidir con ejecutar el evaluador desde cero sobre el JSON transformado. Cambiar clase, centro o dimensiones requiere volver a aplicar las mismas reglas de elegibilidad de predicciones del protocolo.

## Observaciones no bloqueantes

### SCI-02 — Guardar explícitamente el gate de flips y velocidad

**Estado final:** cerrado. S1.4 bloquea explícitamente flips que exijan transformar componentes sin contrato validado, o requiere una variante con supervisión enmascarada y control correspondiente.

S1.4 exige correctamente transformar K, centros, yaw y velocidad «según el marco», mientras S2.5/S3 reconocen que la semántica física de la velocidad publicada sigue pendiente. Conviene hacer explícito que el baseline inicial no usa flips que exijan transformar esos componentes hasta disponer de una transformación validada, o hasta definir una variante que enmascare esa supervisión con un control equivalente. Resize/letterbox y augmentation fotométrica pueden avanzar independientemente. No se debe escoger el signo de una componente solo por su nombre.

### SCI-03 — Un resultado negativo de S2 no debe impedir toda temporalidad

**Estado final:** cerrado. El gate S2 y el diagrama permiten investigar S3 con el baseline S1 tras descartar una intervención S2, sin convertirla en un resultado positivo.

El texto explica que una intervención fallida se descarta y se conserva el baseline, pero el gate S2 pide una mejora y el diagrama lo precede a S3. Añadir que una decisión reproducible de descartar las variantes S2 permite avanzar a un experimento temporal controlado sobre el baseline S1 evitaría convertir una ganancia geométrica/ranking en requisito científico arbitrario. El control RGB de un frame sí debe quedar fijado antes de atribuir una ganancia a historia o eventos.

## Comprobaciones realizadas

### 1. Cifras y alcance del piloto

- AP `0.02064070955899264` corresponde al macro de cinco clases con GT admitido y al promedio de umbrales XY estrictos de 0,5/1/2/4 m en AP101 local. El plan no lo presenta como oficial.
- El macro de ocho clases con ceros, `0.0129004434743704`, se etiqueta como convención; no se confunde con precisión medida de ocho clases.
- 401 matches a 2 m, 938 GT admitidos y 12.800 propuestas producen aproximadamente 42,75 % de recall y 3,13 % de precisión en ese punto. El plan no confunde esas razones con AP ni divide 401 por 1.936 como si fuera un nuevo matching.
- Los soportes de selección 721/205/6/2/4 y las AP por clase coinciden con los recibos inspeccionados. No hay soporte medible de AP para las otras tres clases.
- La diferencia entre AP train y selección no se interpreta como un gap puro: cambia el soporte macro y la población.
- Se reconoce que no se han recalculado AP ni predicciones desde checkpoints pesados en este entorno. Los 232 tests se verificaron en un XML con hash, no se describen como una ejecución nueva.

### 2. Elegibilidad del GT

El plan identifica correctamente que `prepare_sample()` filtra targets antes de generar `native_ground_truth`, y que el runner usa ese resultado en evaluación. La recomputación da train 1.565→829 y selección 1.936→938, con los motivos de exclusión citados. El 48,45 % retenido en selección es correcto.

La propuesta de separar targets entrenables, GT evaluable y GT ignorado corrige el problema de contrato. También es correcto advertir que ni descartar más GT ni exigir automáticamente todo el GT raw como visible constituye por sí solo el protocolo oficial. Deben mantenerse un perfil legacy y un perfil nuevo con identificador/hash distintos.

La AP basada en XY puede ocultar Z deficiente. Las pruebas de frontera, GT ignorado, cero soporte y predicción perfecta previstas son pertinentes. Las transformaciones de scores estrictamente monótonas con pool fijo no mejoran por sí mismas la AP por clase; el plan lo recoge correctamente.

### 3. Soporte raro y separación de datos

Contrasté las tablas con la auditoría de los 40 Parquet train fijados por SHA. Todas las cifras raw por rol y clase coinciden. Bajo el perfil antiguo, los conteos admitidos de bicycle y other_vehicle son correctos, incluyendo **53 anotaciones / 6 pares (secuencia, ID)** de other_vehicle en train24. Esos pares no se equiparan indebidamente a seis adquisiciones u objetos físicos independientes.

El plan identifica los dos train y dos selección realmente utilizados por el piloto. El esquema 24/8/4/4 está condicionado a grupos, exposición y soporte; no se presenta como un split ya válido. Incorpora que solo `2JkLL5W891` y `rhYxPcU29n` aportan other_vehicle admitido dentro del rol original de calibración. No propone mover las dos secuencias de selección expuestas a un holdout virgen.

La política de selección, calibración y auditoría está separada. Un resultado de auditoría usado para rediseñar deja de ser confirmatorio. El refit a cuarenta secuencias consume el holdout local y no hereda automáticamente calibración. Las restricciones de Test12 excluyen entrenamiento, pseudoetiquetado, normalización y selección. La expansión train se plantea por diversidad y cobertura, no por repetición de seis IDs.

### 4. Causalidad, asociación y servicio

El plan distingue timestamps sensoriales de disponibilidad real, conserva la llegada desconocida como tal y rechaza eventos/frames futuros. No considera fijo el marco ego entre instantes ni inventa ego-motion a partir de `ego_translation` del objeto.

El alcance del bridge actual se describe correctamente: siete frames, una detección elegida por score, asociación con el frame anterior, `CompetitionRuntime`, sin evidencia de `MultiObjectTrain40Service` completo ni generalización multiobjeto. El fixture de ambigüedad inversa coincide con la lógica inspeccionada. Lifecycle, generación, resets y aislamiento de buffers son pruebas concretas y pertinentes para la siguiente fase.

Es acertado exigir `ttc_status` además del exit code y medir latencia integral, incluido calentamiento/historia y número de objetos. Los tiempos parciales del piloto no se convierten en una garantía de tiempo real.

### 5. TTC, intervalos y exposición

La mejora de velocidad se limita al target publicado y a los matches; no se denomina AVE oficial ni se usa como velocidad física demostrada. No se fuerza `TTC=-Z/v` sin el contrato pendiente.

La matriz ROI/ID de referencia, ROI perturbada, detección con identidad de referencia y sistema autónomo permite localizar degradaciones siempre que se contabilicen los mismos objetos elegibles y las abstenciones. El plan lo exige, conserva separados score de detección y confianza/calibración TTC, y no transforma q10/q90 de fase en un intervalo continuo de TTC cruzando cero.

La exposición de todos los productores TRAIN40 se conserva explícita. Reentrenar solo el head o reservar un holdout para el detector no basta para demostrar generalización TTC. El alineamiento eAP–Garl de identidad, reloj y target permanece gate científico; no se sustituye por vecino espacial o por compartir una imagen.

### 6. Ejecutabilidad y conclusiones

T00–T08 tienen archivos, orden, pruebas y entregables suficientes para que otro agente pueda iniciar la primera iteración. Las interfaces y flags futuros están marcados como propuestas. El acceso al checkpoint local y el contrato oficial son dependencias reales, no se ocultan como tareas ya resueltas. El trainer puede prepararse sin ejecutar nuevas campañas, y T07 queda condicionado a la admisión de sus dependencias.

Las propuestas de warmup, muestreo, resolución, historia y voxelización se etiquetan como decisiones experimentales. Se mantienen controles, semillas y poblaciones fijas, y no se convierten presupuestos de ocho pasadas en garantías de convergencia o plazos de desarrollo. Las referencias externas se usan como ideas o comparadores condicionados; esta revisión no certifica el estado mundial de todos los benchmarks.

## Límites de esta revisión

Inspección de código, recibos, Markdown, JSON y metadatos train. No se ejecutaron modelos, entrenamiento, GPU, scoring oficial ni Test12; no se abrieron medios o pesos locales del usuario. La nueva auditoría de población reproduce fórmulas legacy con la resolución documentada 1280×720, no valida físicamente cámaras ni independencia de adquisiciones. La revisión de bibliografía/leaderboard corresponde al agente de fuentes primarias y al autor final del plan.

El repositorio permaneció sin cambios de archivos tracked. Los scripts, notas y artefactos de auditoría están fuera del repositorio.

## Cierre de revisión

Última versión inspeccionada del plan: SHA-256 `d63065f341594613554aecb48219b4bbc967aed2fd5628d3c6bd599cce7013b7`.

SCI-01/02/03 cerrados tras verificar S0.2, S1.4, el gate S2 y el diagrama. Se releen también los cambios T03/T05/T06: conservan la procedencia del checkpoint histórico, distinguen el observador nuevo y exigen que configuración/modelo/inferencia y profiler coincidan. T01 conecta el GTv2 independiente al criterio de selección del runner; T02 conserva esquemas de roles versionados y mantiene auditoría fuera de selección/train. El comando de tablas usa `render` sin ejecutar el CLI que escribe resultados históricos. Estos cambios no generan nuevas objeciones científicas. No hay una fase experimental nueva declarada terminada por esta aprobación del plan.
