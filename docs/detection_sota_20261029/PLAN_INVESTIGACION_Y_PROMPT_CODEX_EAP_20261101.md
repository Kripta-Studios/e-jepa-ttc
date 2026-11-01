# Plan de investigación para mejorar la detección eAP y continuar con Codex

Fecha: 1 de noviembre de 2026. Base: `e6abc044`, rama
`dev/eap-s0-evaluation-20261029`. Leer primero
[ESTADO_RESULTADOS_Y_ARTEFACTOS_EAP_20261101.md](ESTADO_RESULTADOS_Y_ARTEFACTOS_EAP_20261101.md).
Este plan incorpora el resultado oficial y la reevaluación alineada. Amplía el
[plan SOTA recibido](evidence/intake/SOTA_DEVELOPMENT_PLAN.md); no borra sus
experimentos ni sus límites científicos.

## 1. Objetivo y criterio de éxito

El objetivo es conseguir un detector de ocho clases competitivo en eAP mediante
mejoras verificadas en desarrollo, y congelar después un candidato para confirmar
en el benchmark. No podemos garantizar una puntuación antes de investigar.

La referencia de partida es la mAP local alineada del RGB con ranking, no su AP
histórica v2. Para hablar de un modelo excelente se necesita algo más que subir
la macro: cobertura de clases, recall en distancias exigentes, estabilidad por
secuencia, errores TP, coste de inferencia y comparación con métodos admitidos
bajo el mismo protocolo. Verificar el leaderboard y sus reglas antes de afirmar
liderazgo. No equiparar un resultado Car de otro protocolo con mAP de ocho clases.

Hitos de trabajo **propuestos**, sin garantía de viabilidad ni significado SOTA:

| Hito | Evidencia de desarrollo requerida |
|---|---|
| Corrección del problema | Población y geometría comprobadas; selección de checkpoint con el nuevo perfil |
| Mejora útil | Ganancia pareada en selección completa, coste documentado y degradaciones por clase visibles |
| Primer salto | Intentar superar 0,10 mAP alineada; comprobar que el avance no proviene solo de coches |
| Segundo salto | Investigar si 0,20 mAP alineada es alcanzable; reevaluar método/datos si las curvas no lo sustentan |
| Candidato fuerte | Comparación reproducida con referencia pertinente y confirmación independiente según datos disponibles |
| Resultado oficial | Candidato congelado, envío conforme y puntuación recibida, separada de desarrollo |

Los umbrales 0,10 y 0,20 son aspiraciones de ingeniería. No son cifras esperadas,
líderes verificados ni fronteras de producción. Si la evidencia demuestra un
límite, comunicarlo y presentar opciones; no prolongar gastos para aparentar éxito.

## 2. Primera iteración: diagnóstico sin entrenamiento

### P0. Recuperar estado y fijar el contrato

1. Leer AGENTS.md, los dos documentos de traspaso y los recibos. Verificar rama,
   checkpoint seleccionado, ranker, hashes y estado de los procesos reales.
2. Conservar productor, perfiles históricos y artefactos. Trabajar en una rama
   nueva desde el cierre; no arrastrar cambios ajenos sin revisarlos.
3. Verificar los resultados alineados guardados. Repetir evaluación solo si hay
   cambios, dudas de procedencia o una prueba concreta que lo requiera.
4. Comprobar fuentes oficiales actuales: reglas de modalidades, datos externos,
   preentrenamiento, coordenadas, filtros, límite de propuestas, scorer público,
   versión de release y leaderboard. Guardar URL, fecha y versión/hash disponible.
5. Si el scorer sigue privado, mantener el perfil de referencia y su incertidumbre.
   Continuar los diagnósticos independientes; no intentar recuperar GT test.

Salida propuesta: `INTAKE_AND_PROTOCOL_RECEIPT.json`. Debe declarar expresamente
si existe equivalencia certificada del scorer o solo alineación con el contrato.

### P1. Revisar geometría y admisión de targets

Antes de ampliar una arquitectura, descartar errores de adaptación:

- Verificar clase, orden L/W/H, centro frente a base de caja, unidades, yaw,
  ejes ego/cámara y transformaciones ida/vuelta.
- Comprobar que calibración, RGB utilizado y resolución pertenecen a la misma
  geometría, incluidos resize, padding, registro entre sensores y rectificación.
- Comparar reproyección de cuboides y posiciones conocidas en muestras de varias
  secuencias, rangos y clases. Guardar ejemplos correctos y fallidos con IDs.
- Trazar cada filtro de `prepare_sample`, límites de profundidad, centro proyectado,
  objetos truncados y capacidad de targets. Separar la tarea evaluada de la
  representabilidad actual de la pérdida.
- Revisar normalización, priors de tamaño por clase, coder de profundidad,
  clasificación foreground/background y decoder. Buscar hipótesis concretas,
  sin dar por hecho que exista un bug.
- Verificar índices de clases en checkpoint, labels y exportación; no modificar
  cajas basándose en el test oficial.

Salida propuesta: `GEOMETRY_AND_TARGET_AUDIT.md` y fixtures de invariantes. Una
corrección requiere ejemplo que falle antes y pase después. Si solo cambia el
exportador, reevaluar primero las predicciones públicas; no reentrenar por rutina.

### P2. Descomponer el error con las predicciones guardadas

Usar las 23.958 muestras de selección y todas las anotaciones raw. Mantener una
tabla paralela del dominio histórico para entender qué se excluía, sin sustituir
la métrica principal por esa población más fácil.

Diagnósticos requeridos:

1. AP y recall a 0,5/1/2/4 m por clase, secuencia y rango. Declarar por separado
   distancia radial ego XY y profundidad óptica; no confundir sus cortes.
2. Recall de propuestas con y sin exigir clase. Reportar asignación greedy y,
   como techo diagnóstico distinto, emparejamiento de cardinalidad máxima por
   frame. No llamar límite exacto a un oracle que conserva un greedy subóptimo.
3. Error longitudinal, lateral, altura, tamaños, yaw y reproyección de matches
   definidos de antemano. Contabilizar objetos sin propuesta y falsos positivos.
4. Confusiones entre clases, duplicados, score frente a calidad y curvas PR.
   Separar bajo recall de mala ordenación de propuestas.
5. Estratos de distancia, tamaño proyectado, truncación/visibilidad si existen,
   y condiciones de iluminación verificadas. Definir cortes antes de comparar
   variantes; conservar denominadores y estratos sin soporte.
6. Sustituciones oracle de clase, profundidad, tamaños y ranking, solo en público
   y como diagnóstico con GT. Describir cómo se asignan propuestas a objetos y
   qué cajas quedan sin asignar. No exportar sus predicciones como inferencia.

Para ATE/ASE/AOE, conservar los defaults del benchmark y añadir estadísticas de
errores sobre matches válidos como diagnóstico separado. Un promedio condicionado
a pocos aciertos puede ocultar que se pierden casi todos los objetos.

Salidas propuestas: `ERROR_DIAGNOSTICS.json`, tablas regenerables, ejemplos
visuales y `NEXT_INTERVENTION_DECISION.md`. La decisión debe contestar qué error
limita más y qué resultado refutaría la intervención elegida.

## 3. Cómo decidir la siguiente intervención

| Hallazgo en desarrollo | Primer experimento | Condición para descartarlo o avanzar |
|---|---|---|
| Error de coordenadas/calibración/decoding | Corregir adaptación y comprobar ida/vuelta y paridad de lo no afectado | Resolver el fallo medido antes de atribuir ganancias a arquitectura |
| GT frecuente fuera del dominio entrenable | Receta nueva de rango/targets coherente con coder, pérdidas y decoder | Mejor recall raw por rango sin invalidar geometría ni hundir clases cercanas |
| Propuestas buenas con clase equivocada | Diagnóstico de supervisión/clase; balance o ajuste de pérdida acotado | Mejor clase correcta y macro, no solo recall agnóstico |
| Pocas propuestas útiles para objetos pequeños | Resolución efectiva y features multiescala; un cambio cada vez | Ganancia en objetos pequeños con coste de memoria y tiempo medido |
| Centros 3D imprecisos, especialmente profundidad | Supervisión geométrica, consistencia o hipótesis múltiples | Mejora a 0,5/1 m, no solo aumento de recall a 4 m |
| Buenas cajas mal ordenadas | Ranker entrenado con objetivo alineado a distancia eAP | Ganancia con pool fijo; conservar baseline y control escalar |
| Curvas aún mejorando con cobertura suficiente | Ampliar entrenamiento con validación periódica y calendario declarado | Mejora de validación sostenida; no justificar duración solo por loss train |
| MonoDGP estancado tras corregir datos/receta | Reproducir un competidor completo bajo el mismo protocolo | Ganancia comparable y coste total aceptable |

Ampliar `topk` de 50 a 500 no crea información si el productor solo emite 50
queries útiles. Auditar queries, grupos y decoder antes de variar capacidad.
NMS, umbrales o recortes espaciales pueden quitar TP o cambiar duplicados:
probarlos en desarrollo, no asumir mejora por enviar menos cajas.

## 4. Cobertura de clases y separación de datos

Crear un censo **raw y admitido** por clase, secuencia, rango, tamaño, track y
grupo de adquisición conocido. Dfull añadió frames y vistas, pero no convirtió
seis IDs raros en diversidad suficiente. Revisar ejemplos de clases ambiguas
como truck, bus y other_vehicle antes de escoger una pérdida o un muestreo.

Mantener el split de desarrollo mientras se compara una intervención. Auditar
duplicados y grupos relacionados. La selección ya está expuesta; reservarla de
nuevo no la convierte en validación independiente. La propuesta antigua 24/8/4/4
no queda aprobada por este documento: exige revisar soporte, independencia y
exposición real de esas ocho secuencias restantes.

Un panel enriquecido de clases raras sirve para diagnosticar, pero la selección
principal debe conservar la prevalencia natural. Limitar sobremuestreo por
identidad/secuencia y comparar con R2-rare, que ya mostró un compromiso negativo.
No inventar diversidad con duplicación, ni mejorar una media ocultando clases.

Considerar datos externos, preentrenamiento o teacher de profundidad solo si las
reglas y licencias los permiten. Registrar procedencia, modalidad, exposición y
disponibilidad en inferencia. No presumir que LiDAR/GNSS anunciados en el corpus
estén presentes o permitidos en la entrada pública del benchmark. Un teacher
entrenado con etiquetas debe respetar las mismas separaciones que el estudiante.

Después de fijar arquitectura, receta y duración, puede estudiarse entrenar el
candidato de envío con más secuencias públicas, incluso las 40 si el protocolo
y las reglas lo permiten. Eso consume la independencia de los roles absorbidos.
Registrar el refit por separado, sin usar test para seleccionar checkpoint y sin
atribuirle la validación del modelo anterior como si fueran los mismos pesos.

## 5. Integrar la evaluación en un productor nuevo

El cierre actual añadió un observador/evaluador externo. **No adaptó el trainer
histórico congelado.** Antes de un entrenamiento largo:

1. Crear configuración y versión nueva con `evaluation_profile` explícito,
   identificador y hash. Rechazar opciones ignoradas y mezclas de protocolos.
2. Construir GT de validación desde etiquetas públicas raw, sin reutilizar la
   admisión de targets del entrenamiento como definición de población.
3. Hacer que validación periódica, monitor y selección de checkpoint usen el
   nuevo perfil. Conservar un panel rápido fijo y validar finalistas en selección
   completa. No elegir por AP histórica y reportar después otra métrica.
4. Verificar igualdad entre evaluación integrada y CLI externa sobre fixtures y
   un panel real pequeño. Guardar hashes de predicciones y denominadores.
5. Guardar seleccionado y último por separado, junto con optimizer, scheduler,
   sampler, RNG, exposiciones consumidas, configuración y fuentes ejecutadas.
6. Probar interrupción y restauración. Distinguir resume completo de inicializar
   pesos desde otro checkpoint con estado de entrenamiento nuevo.

Código existente que revisar antes de implementar: `benchmark_evaluation.py`,
`evaluation_protocol.py`, `evaluation_ground_truth.py`, `evaluation_v2.py`,
`monodgp_eap_adapter.py`, `quality_ranking.py` y los runners de campaña en `scripts/`.
El adaptador histórico carga módulos y valida hashes; no relajar esa procedencia
para que una modificación parezca el productor original.

## 6. Investigación de métodos con una pregunta concreta

La comprobación documental de esta entrega encontró las siguientes fuentes
primarias. La nueva sesión debe inspeccionar código, licencia, receta, commit y
requisitos antes de presupuestar un port. Estos métodos no tienen una mejora
eAP demostrada en nuestro entorno.

| Candidato | Pregunta que debe resolver | Alcance inicial |
|---|---|---|
| [SPAN](https://github.com/WYFDUT/SPAN) | ¿La consistencia entre esquinas 3D y proyección reduce los errores geométricos observados? | Una pérdida/adaptación acotada; su repositorio incluye MonoDGP y una receta antigua con ops compiladas |
| [RARE](https://github.com/HyeonjeongPark37/RARE) | ¿Ordenar calidad y recuperar hipótesis diversas aporta algo más que nuestro ranker fijo? | Estudiar el objetivo y separar ranking de generación de hipótesis; el método actual no es una reproducción de RARE |
| [MonoCoP](https://github.com/alanzhangcs/MonoCoP) | ¿Modelar dependencia entre profundidad, tamaño y orientación supera la base corregida? | Un competidor completo, con admisión de entorno y presupuesto de reproducción |

El repositorio SPAN describe alineamiento espacial y de proyección con introducción
progresiva. RARE describe ranking y recuperación de hipótesis. MonoCoP publica una
cadena de predicciones y un selector por incertidumbre. Esas motivaciones ayudan
a formular experimentos; sus resultados en otros datasets no predicen nuestra AP.
Fuentes consultadas el 1 de noviembre de 2026; fijar commits al comenzar un ensayo.

Buscar también referencias oficiales eAP y una receta RGB reproducible compatible
con sus ocho clases. Seleccionar como máximo un competidor inicial según coste,
requisitos y fallo que intenta resolver. No portar tres frameworks a la vez.
Un cambio de backbone puede estudiarse si el diagnóstico y las curvas indican
una limitación de capacidad; no debe ocultar un error de targets o calibración.

## 7. Temporalidad y eventos después del diagnóstico RGB

No volver a lanzar los mismos screenings solo porque el resultado oficial es
bajo. Reevaluar sus predicciones guardadas con el nuevo perfil cuando permitan
recuperar el GT raw y la misma población; generar únicamente lo que falte.

Si los errores justifican historia causal, comparar frame actual, actual repetido,
historia correcta y RGB con capacidad equivalente. Probar historia permutada o
retardada como diagnóstico; distinguir perturbación de inferencia de ablation
entrenada. Un baseline peor por cambio de host o presupuesto no es un control válido.

Para eventos, contrastar RGB de un frame, RGB temporal, RGB+eventos y RGB temporal
+eventos con presupuestos explícitos. Auditar sincronización, corte causal,
resolución, bins, normalización y pérdida por compresión/downsampling. Los cinco
bins y 50 ms históricos son una receta probada, no un óptimo demostrado.

Condicionar una fusión a calidad observable solo después de superar una fusión
simple. Definir estratos HDR, movimiento y baja densidad antes de ver resultados.
Si faltan poses verificadas, no introducir compensación ego BEV suponiendo que
la posición de un objeto sustituye a la del vehículo.

## 8. Política de experimentos y promoción

Cada ensayo debe declarar: hipótesis, control, población, semilla, inicialización,
presupuesto de exposiciones y tiempo, métrica primaria, calendario de validación,
regla de promoción, coste esperado medido y criterio de parada. Cambiar una causa
principal por ensayo; registrar cualquier cambio inevitable de receta.

Como regla inicial propuesta para screening: exigir al menos +0,005 mAP alineada
en selección completa y revisar cualquier caída mayor de 0,01 AP por clase.
Es una regla operacional, no significancia estadística. Ajustarla **antes** de
ejecutar según soporte y ruido de desarrollo, no después de mirar el resultado.
No rechazar o promocionar una clase rara solo por un cambio minúsculo de AP.

Secuencia de gasto:

1. Fixtures, overfit pequeño y smoke de forward/loss/backward/guardado/restore.
2. Ensayo acotado con curvas y una semilla; controlar presupuesto común.
3. Confirmación en selección completa del checkpoint concreto que se promueve.
4. Solo para el ganador y su control, varias semillas si hacen falta para
   sostener la conclusión final. No multiplicar cada rama exploratoria por tres.
5. Intervalos por secuencia o adquisición verificada; nunca tratar frames
   correlacionados como réplicas independientes. Con ocho secuencias, declarar
   la incertidumbre y no presentar un bootstrap como prueba de independencia.

Guardar resultados negativos, monitores, seleccionado y último. No detener un
ensayo porque una curva parcial parezca ganadora si la regla fijada exigía un
presupuesto comparable. Sí detener por NaN, OOM persistente, fuente inválida,
presupuesto agotado o un criterio de futilidad declarado de antemano.

## 9. Rendimiento, automatización y continuidad

Primero medir lectura/descompresión, preparación, transferencia, forward,
matching/loss, backward, optimizer, validación y escritura. Separar calentamiento
de régimen sostenido y medir el ciclo completo, no solo updates de entrenamiento.

Reutilizar índices TAR, cachés verificadas y prefetch acotado donde reduzcan el
cuello medido. Invalidar una caché si cambian pesos, preparación, resolución,
calibración o targets relevantes. No cachear features de capas que se entrenan
como si fueran constantes. Evitar duplicar datasets ya presentes.

Comparar batch, acumulación, precisión y compilación con rendimiento **y** calidad.
Cambiar batch efectivo altera optimización; cambiar precisión exige controles
numéricos y de aprendizaje. CUDA Graphs requieren formas/memoria estables y deben
resolver operaciones incompatibles, como matching dinámico. Aproximar una pérdida
o geometría es otro experimento, no una optimización con paridad presupuesta.

Dos procesos sobre una GPU solo se justifican si mejoran el tiempo total hasta
completar el trabajo, incluyendo validaciones y checkpoints. Medir antes de
competir por VRAM, I/O y cómputo. No abrir 16 transferencias si la red o el disco
ya están saturados. No mantener un pod facturando mientras se escribe documentación.

Preparar una cola persistente de trabajos admitidos, con dependencia, lock de
escritor único, estado legible, salida de error, parada y resume. Automatizar
etapas con reglas explícitas; las decisiones científicas que no cumplan esas
reglas deben dejar un informe y esperar revisión, no encadenar gasto indefinido.

Cada estado debe incluir progreso comprometido, timestamp del último avance,
exposiciones, fase actual, ritmo móvil real y ETA con rango. Si el usuario pide
los últimos diez minutos, calcular diferencias entre muestras verificadas;
si no existen, decirlo y medir, sin inventar progreso.

Hacer backups incrementales de pesos/estados nuevos, recetas, código congelado y
resultados caros de regenerar. Verificar hashes y restauración antes de detener
o migrar una máquina. Un checkpoint `.pt` debe escribirse de forma atómica;
una copia parcial no cuenta como backup. No copiar de nuevo datasets y cachés
regenerables que ya estén aquí. No reutilizar secretos pegados en conversaciones.

## 10. Entregables y final de la campaña

Cada fase debe cerrar con código, tests pertinentes, JSON/CSV, configuración,
entorno, procedencia y decisión explícita. El informe final debe separar:

- Mejora de desarrollo con protocolo y población fijos.
- Confirmación independiente disponible y sus límites.
- Score oficial del candidato congelado, si se envía.
- Calidad por clase/dominio, cobertura, fallos y coste de inferencia.
- Integración con PHASE y, aparte, cualquier nueva precisión TTC medida sin fuga.

El paquete final requiere pesos, runtime, preprocesamiento, metadatos, licencias,
ejemplo de inferencia y paridad comprobada. Validar todas las claves de test y el
ZIP antes de preparar un nuevo envío. No usar la puntuación test como bucle de
hiperparámetros ni subir múltiples variantes para elegir la ganadora.

No declarar el modelo terminado solo porque concluye el entrenamiento. Si no se
alcanza un salto útil dentro del presupuesto, entregar el mejor candidato válido,
la evidencia negativa, el límite identificado y el coste de las opciones siguientes.

## 11. Mensaje completo para pegar en una nueva sesión de Codex

Adjunta ambos Markdown o abre el repositorio donde están. Copia este mensaje:

```text
Continúa la investigación y mejora del detector PHASE-TTC para eAP 3D Detection.
Quiero un salto sustancial y defendible de calidad en ocho clases. No quiero
repetir cómputo guardado ni otra campaña larga sin saber qué limita el modelo.

Lee AGENTS.md y estos dos archivos completos:
docs/detection_sota_20261029/ESTADO_RESULTADOS_Y_ARTEFACTOS_EAP_20261101.md
docs/detection_sota_20261029/PLAN_INVESTIGACION_Y_PROMPT_CODEX_EAP_20261101.md

Repositorio: Kripta-Studios/phase-ttc.
Rama del cierre: dev/eap-s0-evaluation-20261029.
La implementación alineada está en e6abc044; toma también el commit posterior
que añade estos documentos. Verifica HEAD y procedencia antes de actuar.
En este PC el worktree correcto es:
C:/Users/Álvaro Schwiedop/Desktop/KriptaStudios/EVOCON_JEPA_Codex_Handoff/phase-ttc-s0-evaluation
El checkout e-jepa-ttc-v18-joint-measurement puede estar en otra rama.
Conserva modificaciones ajenas; crea una rama/worktree aislado si hace falta.

Primero ejecuta P0-P2 del plan: verifica artefactos y contrato, audita geometría
y targets, y diagnostica recall, clase, profundidad, tamaño, rango, duplicados y
ranking con las predicciones públicas ya guardadas. No entrenes aún por rutina.
Genera resultados reproducibles y decide una intervención a partir de ellos.

El candidato actual es RGB MonoDGP-R50 con ranker TRAIN. Su mAP oficial es
0.05129508; la local alineada es 0.06204229, frente a 0.05701726 sin ranking.
La AP histórica 0.18502078 usa otro protocolo y no es una estimación de test.
El nuevo perfil reproduce la aritmética observada; no certifica matching privado.
Dfull usó 24 secuencias train; no entrenó con todas las 40 públicas.
No asumas que los screenings cortos descartaron temporalidad o eventos.

Investiga fuentes primarias y recetas reproducibles pertinentes. Escoge el
primer cambio según el error dominante. Considera geometría, rango/targets,
clases pequeñas/raras, resolución, ranking e hipótesis, y después un competidor
o temporalidad/eventos si la evidencia lo justifica. No portes todo a la vez.

Antes de otro entrenamiento largo, integra el perfil alineado en validación
periódica y selección de checkpoints de un productor NUEVO. Verifica paridad
con el evaluador externo. No alteres módulos congelados ni relajes sus hashes.
Fija población, control, receta, presupuesto y regla de promoción antes de correr.
No uses etiquetas privadas ni el score test para ajustar hiperparámetros.

Avanza de diagnóstico a implementación, pruebas y experimentos, y adapta el
plan según los resultados. No te quedes solo en una propuesta. Aprovecha la
autorización ya disponible para trabajo local; no pidas confirmación para cada
paso reversible. Para nuevo gasto cloud, verifica el pod y el presupuesto
vigentes: no deduzcas recursos o gasto ilimitado de IPs y acuerdos históricos.
Si falta un límite de gasto, deja lista la campaña y pide solo ese dato.

Mide cuellos de botella y ETA real antes de optimizar. Usa cachés/prefetch/batch
o paralelismo solo con una medición y las comprobaciones correspondientes.
Prepara una cola reanudable para etapas admitidas. Guarda backups verificados
de lo importante que no exista ya localmente. No dejes un pod ocioso facturando.

Trabaja solo por defecto; no abras subagentes para tareas que puedes resolver
directamente. Conserva pruebas negativas, varios checkpoints y recibos. No
lances semillas adicionales hasta que un candidato justifique confirmación.
Mantén updates breves con procesos reales, progreso, ritmo y ETA medidos.

Continúa hasta entregar el mejor candidato defendible dentro de los recursos
acordados, con paquete reproducible, resultados por clase y conclusión honesta.
No prometas SOTA o producción. Si el presupuesto o la evidencia impiden un
modelo excelente, explica qué limita el resultado y qué cambio concreto falta.
No presentes el objetivo como logrado solo por haber terminado las horas de GPU.
```

La sesión nueva debe empezar verificando el estado real. Este mensaje le da
continuidad y criterios de decisión; no certifica que los recursos o servicios
externos sigan disponibles ni garantiza una puntuación futura.
