# Extensión ejecutable: comparadores, atribución y coste (0 updates)

## Organización

Nueva sesión y worktree sobre `62cf4dbe914ea077c68e42983f869cd5422ab71c`, por ejemplo `scientific-recovery-v16-multimodal-validation`. La V13 de publicación se conserva. V14 queda cerrado. V15 no se cancela ni recibe cambios de coeficientes por este plan.

Leer el estado local y descubrir los archivos reales antes de afirmar que otra cola está activa o parada. No tomar propiedad de un supervisor ajeno. Reutilizar verificación ya válida por hashes; no hacer miles de llamadas Git por consulta.

### Q0. Consolidar identidades y resultados

- Verificar este paquete; contrastar la base local y las modificaciones posteriores legítimas, sin reset ni clean.
- Ejecutar una vez el replay público V13 si no existe recibo local de los mismos hashes. Los recálculos incluidos ya documentan los números del ZIP.
- Inventariar el freeze de los doce endpoints, P/H/V, normalizadores, evaluación nativa V13, V14 y el estado V15.
- Mantener la cohorte 1.024/946 y la expandida 3.947/3.640 como manifiestos separados. Crear reconciliación por identidades físicas; no usar query_id solo si fue renumerado.
- Incorporar F_TRUE–R_CTX en V usando pérdidas locales por grupo; no reconstruir ese intervalo desde dos scores globales. Dev32/FCWD son post hoc, ya expuestos.

Entregables: `STATE_BINDING.json`, `COHORT_COMPARABILITY.csv`, `MATCHED_PAIRED_REVIEW.json`. Si faltan predicciones V locales, dejar ese contraste pendiente sin bloquear los externos.

### Q1. Completar Garl RGB/full FCWD con el registro ya sellado

Es una comparación nueva de **sistemas con registro estimado**; no desbloquea ni suplanta `FCWD_RGB_STATUS` oficial. No necesita entrenar.

1. Reutilizar `REGISTRATION_SEALED.json` y las transformaciones por consulta existentes. No repetir búsqueda, bajar NCC o escoger otra transformación según error TTC.
2. Cargar estrictamente los checkpoints públicos Garl event, visual-only y full, con sus configuraciones y hashes. Si falta un peso se permite sólo obtener ese artefacto público concreto, registrando procedencia/licencia; no entrenar un sustituto ni usar un checkpoint parcial.
3. Integrar su preprocesado nativo: canales, polaridad, normalización, interpolación, padding, crops y orden normalize/crop. No darle a Garl los voxels de A5. No usar profundidad o TTC GT para proyectar cajas.
4. Mantener la historia/representación de cada modelo como la define su release. Añadir RGB no autoriza a hacer Garl-H8 ni cambiar su dt.
5. Auditar endpoints y disponibilidad: una tripleta causal válida V13 no prueba que haya una pareja sincronizada válida para Garl. Si Garl exige otra sincronización, registrar por separado su disponibilidad; no forzar tolerancias para igualar 495 filas. Una evaluación alternativa sincronizada usaría otro manifiesto elegido sin TTC y no reemplazaría el de 630.
6. Predecir las consultas admisibles y guardar causa para las demás. Comparar la intersección donde ambos tienen inputs completos; publicar también cada cobertura y el conjunto completo.
7. Como política de sistema adicional previamente fijada, `GarlFull_registered_fallbackE` usa exactamente Garl event-only cuando su ruta RGB falla. Mostrar por separado standalone full, fallback compuesto y F_TRUE con su fallback E_CTX. No comparar un full que descarta filas con F_TRUE total.
8. Sellar predicciones antes de incorporar labels. Mantener resultados nativos y sensibilidad ±60 separados. Publicar MiDc/s/l, mean_MiD, MAE, mediana, RTE, signos, fallos, colas y cobertura por secuencia/rango. No fabricar overall cuando falta n.

Nombres propuestos: `GARL_VISUAL_REGISTERED`, `GARL_FULL_REGISTERED`, `GARL_FULL_REGISTERED_FALLBACK_EVENT`. No son checkpoints nuevos ni resultados oficiales.

Si el full no es admisible, no improvisar geometría: entregar exactamente qué requisito temporal/espacial falla. Eso no bloquea Q2–Q4 ni V15.

### Q2. Trazas de atribución multimodal

No quedarse en la salida final. Usar los IDs publicados de fallos como diagnósticos externos y 32 controles H por hash estratificados con TRAIN, no una muestra escogida por error en V.

Capturar por consulta y modalidad:
- tiempos reales de frames/ventanas, anchor de etiqueta y momento de disponibilidad;
- ROI original, transformación, margen, soporte cubierto y truncado;
- valores de entrada antes/después de normalización, sin cambiar normalizador;
- alturas, r analítico, residual/productor pre-tanh y post-tanh, saturación relativa;
- TTC/fase de A5, C2F y PAIR; medianas de expertos;
- E_CTX y R_CTX: fase base, residual, raw_location y fase emitida;
- F_TRUE: estados ocultos/entrada combinada, anchor event, residual, pre/post clamp, cuantiles;
- registro: soporte de eventos, NCC, diferencias entre mitades y motivos de rechazo.

Usar hooks/copias privadas y adaptadores ya corregidos para known y mediana CUDA. Exigir igualdad de la predicción al instrumentar; no tocar pesos o archivos del run. Si el modelo final no permite una traza con paridad, documentar el campo pendiente.

Auditar especialmente si el signo malo aparece antes de la fusión, si ésta se acerca a fase cero ante desacuerdo, y si el deterioro depende del encuadre. Una correlación no demuestra causa. No convertir el coste auxiliar en peso del router.

Entregables: `STAGE_ATTRIBUTION.csv`, `FAILURE_EPISODES.md`, `TRACE_PARITY.json`. Las filas vecinas no son escenas independientes.

### Q3. Disponibilidad RGB: mejorar geometría sin optimizar TTC

Primero analizar registros existentes y causas. Separar cobertura crucial (195/201) de larga (62/108 y 45/99 >10). No presentar toda mejora de cobertura como solución del TTC corto.

Definir un pequeño conjunto de evaluación geométrica por hash antes de revisar resultados TTC. Crear paneles evento/RGB y solicitar/exportar correspondencias de imagen independientes cuando no existan; no declarar esos clicks disponibles si no se han producido. NCC y consistencia de mitades no sustituyen error geométrico contra correspondencias independientes.

Preparar, sin activar automáticamente en producción, sólo estas dos alternativas de ingeniería:
1. Prior causal de la última transformación aceptada, con soporte y deriva comprobados. No usar transformaciones futuras ni asumir que existe prior antes de la primera aceptación. Un desplazamiento local no es un extrínseco permanente.
2. Acumulación multiescala causal 20/40/80 ms para obtener soporte en escenas escasas. Comparar geométricamente y contabilizar coste/desenfoque de movimiento; más eventos no implica mejor registro.

La elección entre ellas necesita calidad geométrica prospectiva y sus negativos, no qué variante baja MiD en las mismas tres FCWD. Mantener una transformación coherente a través de la historia del objeto: realinear cada frame independientemente puede borrar expansión.

La ausencia de referencias geométricas impide promover un registro, pero no impide publicar alternativas, costes y una lista precisa de anotaciones necesarias. No reiniciar los algoritmos ECC/SIFT/global affine que ya fallaron sin hipótesis nueva identificada.

### Q4. Coste con perímetro comparable

- Cronología real, varios objetos cuando existan, resets, cold start y p50/p95/p99.
- Distinguir HDF5 offline, replay residente e inputs preparados. Incluir búsqueda de registro FCWD como etapa propia.
- Timers de lectura, decodificación RGB, registro, ROI/voxels, H2D, productores, PAIR, cabezas, copias y commit. No sumar medianas ni llamar al coste de `head_and_commit` coste del MLP.
- Medir una cabeza canónica V12 y mediana3 por separado; no dividir tiempo entre tres.
- Comparadores Garl nativo y optimizado, igualdad de carga y exactas mejoras aplicables. No usar el Garl artificialmente lento para declarar victoria.
- Paridad de rutas antes del perfil. Aceptar sólo aproximaciones previamente identificadas y puntuadas; warp no es exacto por definición.
- Plan de coste: 64 consultas por hash equilibradas entre secuencias de las cohortes accesibles, repetición caliente registrada y arranque separado. Luego una pasada cronológica completa de precisión de las rutas de sistema que se propongan promover; los percentiles del piloto no se convierten en garantía de despliegue.
- Si V15 usa GPU/lectura pesada, serializar. No cambiar energía, drivers, paging o prioridades globales para este plan.

## Cierre y criterio de próximo entrenamiento

Entregar lo completo, lo negativo y las dependencias exactas. No esperar a un score privado para cerrar la comparación pública. Cero updates en este encargo. No añadir otro sistema de orquestación si el actual puede guardar fragmentos y recibos.

Outputs mínimos: `MULTIMODAL_REVIEW_FINAL.md`, `NEXT_DECISION.json`, matrices por cohorte/modalidad/cobertura, trazas, paridad, costes, comandos ejecutados, dependencias por hash, ledger con cero updates y ZIP con SHA-256. No incluir sensores, credenciales o todos los checkpoints antiguos para aumentar artificialmente el bundle.
