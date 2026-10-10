# 08 — Plan para investigar y mejorar FCWD

## Objetivo y reglas de selección

El objetivo inmediato es reducir la sobreestimación en TTC corto y mejorar MiD sin perder las ventajas en rangos medios/largos ni empeorar descontroladamente el coste. No se fija como criterio «ganar alguna tabla»: deben conservarse precisión, cobertura, riesgo y latencia bajo el mismo protocolo.

FCWD y Dev32 ya han guiado el desarrollo. Usarlos para localizar fallos es legítimo si se declara; usarlos repetidamente para elegir hiperparámetros y llamarlos test ciego no lo es. Antes de entrenar otra receta, separar secuencias nuevas de validación y congelar endpoint/selección. Si no hay nuevo conjunto, describir el trabajo como estudio de desarrollo y dejar pendiente generalización.

Este plan propone experimentos; **no implica que ya estén ejecutados ni que sus hipótesis estén confirmadas**. No altera los entrenamientos matched V13 activos. Cada nueva receta debe tener identidad, configuración, semilla, hashes, presupuesto y resultados negativos propios.

## Fase A — Auditoría posible solo con GitHub

1. Ejecutar verificación y replay del informe 07. Comprobar 597 consultas ×4 variantes y gap crucial +27,904676.
2. Unir `evidence/comparison/FCWD_PREDICTIONS.csv` por `query_id` con `FCWD_PAIRED_QUERIES.csv`; verificar GT y no crear emparejamientos por orden de filas.
3. Estudiar curvas por secuencia y rango: TTC real/predicho, error firmado, MiD, A5/C2F transport/confidence/log_variance y tasa de eventos. Conservar filas no elegibles en una tabla aparte, no como TTC cero.
4. Revisar las veinte mayores diferencias y ventanas vecinas. Publicar contribución acumulada al gap para distinguir error persistente de picos aislados, sin declarar independientes las consultas vecinas.
5. Contrastar `feature_shift.csv` y `upstream_metrics.csv`. Estas tablas ya muestran cambios de distribución; no concluir que todo fuera del intervalo train sea fallo ni que todo dentro sea in-domain.

**Entregable:** reporte descriptivo regenerable con IDs, columnas utilizadas y limitaciones. Esta fase no necesita modelos ni datos crudos y es la entrada recomendada para el agente externo.

## Fase B — Paridad y atribución con activos reales

| Experimento | Control | Variable | Evidencia buscada |
|---|---|---|---|
| Reconstrucción de 20 consultas y vecinos | Ruta cruda exacta | Lector/preparador alternativo | Mismos eventos, límites temporales, ROI y voxels dentro de tolerancia |
| Comparación de relojes/ROI | Timestamps reales, ROI con disponibilidad registrada | Adaptación histórica | Detectar shift o información disponible fuera de tiempo |
| Expertos aislados | Mismos pesos y normalizador | A5, C2F, PAIR y combinaciones | Localizar cuándo aparece el sesgo corto |
| Cabeza sin residual / residual acotado | Mismos productores y receta | Agregación | Distinguir sesgo del experto y de la combinación |
| Historial exacto frente a warp | Mismo conjunto, sin cambiar pesos | Solo reproyección | Cuantificar error por IoU, edad, soporte nuevo y tasa de eventos |

No comenzar ajustando la loss hasta descartar una discrepancia de entrada. Si aparece un bug, conservar la evaluación previa, publicar prueba de regresión y recalcular todos los comparadores afectados. Una mejora causada por corregir el baseline también debe publicarse.

## Fase C — Ablations de aprendizaje

### Supervisión y pérdida

Separar el efecto de parametrización, weighting y sampler. La revisión Direct existente cambia batch, tamaño oculto y scheduler, además de loss; no sirve para atribuir causalidad a un único término.

Proponer primero una ablation pequeña con cabeza idéntica, productores congelados, mismo batch, scheduler y updates. Registrar loss/gradiente por bandas c/s/l/n y por TTC≤1. Comparar pérdida original con una variante orientada a MiD y una combinación TTC/MiD validada solo en train/validation. No sumar términos de escala distinta sin inspeccionar cuánto gradiente aporta cada uno.

La banda negativa necesita un tratamiento matemático válido; evitar logs de argumentos no positivos y definir política explícita cerca de cero. Reportar inválidos y no hacer que una loss parezca mejor filtrando predicciones difíciles.

### Datos y balance

El grupo positivo≤1 s de TRAIN40 es escaso. Antes de sobremuestrear ventanas correlacionadas, contar secuencias/episodios independientes que contienen esas situaciones. Usar pesos calculados en train, límites de repetición y seguimiento por secuencia. Sintéticos o augmentations temporales necesitan transformar TTC coherentemente y una validación real separada.

### Productores y distilación

Solo después de localizar la fuente del sesgo, considerar ajuste parcial de productores o un estudiante TTC-aware. Mantener la identidad de profesores y no presentar su validación TRAIN40 como holdout del profesor. Comparar estudiante con/sin término de TTC, coste y todos los datasets; el actual gana FCWD pero pierde Dev32.

### Contexto

Entrenar H4/H2 propios permite comprobar si la pérdida observada al truncar H8 se debe a desajuste de distribución o a falta real de información. Igualar tiempo físico total o declararlo como variable independiente. En V13, ocho posiciones no equivalen automáticamente a ocho observaciones dentro del presupuesto de 650 ms.

**Criterio de promoción:** mejora en validación por secuencia, cobertura íntegra, sin aumento no aceptado de errores críticos/negativos, y presupuesto de latencia fijado antes. Los umbrales numéricos de aceptación deben elegirse con la aplicación y validación, no derivarse de FCWD después de mirar resultados.

## Fase D — Optimización sin sacrificar trazabilidad

1. Medir preparación, productor, cabeza, copias, sincronizaciones y commit de estado por separado en consultas reales completas. La columna `head_and_commit` actual agrupa demasiado para atribuir su coste al MLP.
2. Mantener compactación exacta de paquetes y entrada nativa Garl; estudiar blocks solo donde cambien las lecturas de forma útil.
3. Extender reutilización exacta antes de tolerancias más agresivas. Para warp, medir soporte no recuperable y política de reconstrucción cuando se salga de ROI/edad admitidas.
4. Comparar graphs con overhead de captura, cambios de forma y memoria. Incluir frío, resets, p95/p99 y varias ROI, no solo diez tiempos calientes.
5. Cuantizar selectivamente después de medir sensibilidad TTC. La diferencia PHASE17≈0,8606 del probe BF16 V13 impide asumir inocuidad por el simple hecho de que los tensores sean finitos.
6. Repetir Garl con las mismas optimizaciones exactas que le apliquen. Publicar también el estándar nativo, identificado, sin usarlo como único baseline para ocultar un Garl más rápido.

El primer objetivo de ingeniería razonable es determinar cuánto tiempo queda en preparación/estado después de graphs, no prometer ya menos de 36,8 ms. Los pilotos actuales no permiten extrapolar linealmente a muchos objetos o a hardware embarcado.

## Fase E — Cierre científico

Esperar endpoints V13 completos y evaluar sus propios pesos; no rellenar tablas V13 con V12. Publicar por separado eventos, RGB y fusión, historial físico, costes desde entrada cruda y costes del forward. Una comparación de modalidades debe declarar teacher, supervisión, ROI y sensores disponibles.

Para una afirmación SOTA se necesita el protocolo oficial o una comparación reproducible equivalente, puntuación test12 acreditada cuando se reclame ese benchmark, baselines relevantes, validación independiente y variabilidad entre entrenamientos completos. Las tres cabezas actuales no satisfacen por sí solas tres réplicas independientes. Comparar solo con Garl no establece superioridad frente a toda la literatura.

El envío oficial de los ZIP preparados y la obtención de score privado siguen pendientes. No inventar ese score ni usar las etiquetas públicas TRAIN40 para representar test12. Si el reto no es accesible, el artículo puede describir resultados reproducidos en cohortes públicas con sus limitaciones; debe ajustar sus afirmaciones a esa evidencia.
