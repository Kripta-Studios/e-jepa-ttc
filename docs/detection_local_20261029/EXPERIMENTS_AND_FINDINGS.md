# Experimentos ejecutados y hallazgos

Fecha de cierre: 2026-10-29. Este documento describe los experimentos que se
ejecutaron en la máquina local para comprobar si PHASE-TTC puede incorporar un
detector 3D aprendido para eAP. Reúne resultados positivos y negativos. Las
métricas proceden de los recibos versionados; no se sustituyen por cifras de
artículos ni por estimaciones.

Las tablas numéricas principales están en
[TRAINING_TABLES.md](TRAINING_TABLES.md). El script
[`summarize_detection_runs.py`](../../scripts/summarize_detection_runs.py) las
regenera desde `evidence/training_runs/*.json`, sin dataset, checkpoint o GPU.
Los detalles de los controles de imagen y velocidad se regeneran con los scripts
enlazados desde [IMAGE_SIGNAL_CONTROLS.md](IMAGE_SIGNAL_CONTROLS.md). Este informe
explica qué significan esas mediciones y qué factores impiden compararlas de
forma directa.

## Preguntas comprobadas

Los runs responden a cuatro preguntas de ingeniería. Primero, si el detector
completo puede ejecutar forward, pérdidas, backward, actualización y decode en
la RTX 5070 Ti Laptop sin el operador CUDA antiguo de MonoDGP. Segundo, si el
adaptador aprende una señal real de las cajas eAP, en vez de limitarse a producir
salidas finitas. Tercero, si una cabeza adicional aprende los dos componentes de
velocidad publicados. Cuarto, si una detección producida sin ROI de verdad terreno
puede alimentar la asociación causal y el runtime existente de PHASE.

Las comprobaciones no responden todavía si el detector es competitivo en
CodaBench, si generaliza a las ocho clases o si las detecciones conservan la
precisión TTC del sistema con ROIs anotadas. El scorer oficial de detección y las
etiquetas Test12 no estuvieron disponibles localmente y no se usaron.

## Smoke completo y primer overfit

El backend portable reemplaza la extensión deformable por operaciones PyTorch
diferenciables. Antes de entrenar se ejecutó el modelo completo con ocho logits
de clase. El smoke comprobó pérdidas finitas, gradientes, actualización de la
cabeza activa y decode. Una comprobación inicial falló porque inspeccionaba una
cabeza creada por upstream que el decoder no utilizaba. El recibo fallido se
conservó; la corrección pasó a comprobar `class_embed[dec_layers-1]`.

El experimento `overfit16_seed7_v1` utilizó dieciséis frames reales fijados antes
del entrenamiento. Partió de inicialización aleatoria, batch uno, 640×384 y un
límite de 600 segundos. El criterio exigía que la media de pérdida de centro
final no superase la mitad de la inicial, además de parámetros finitos y una
cabeza activa modificada. El límite temporal cortó el run tras 344 updates. La
razón final fue 0,552355, por lo que el run quedó marcado **FAIL**. El código no
renombró el resultado ni relajó el umbral después de observarlo.

Ese fallo fue útil: mostró descenso de profundidad y centro, actualización real
y checkpoint recuperable, pero el presupuesto no bastó para superar el criterio
fijado. La AP101 de desarrollo sobre las seis clases presentes fue 0,103415. Es
una medida sobre los mismos dieciséis frames usados para entrenar y no aporta
evidencia de generalización.

## Continuación del overfit

`overfit16_seed7_v2_resume` recuperó pesos y estado AdamW de v1. Reinició de forma
explícita el RNG y el sampler con semilla 7, mantuvo como denominador la pérdida
inicial del primer run y añadió 456 updates. Tras 800 updates acumulados, la razón
de pérdida de centro quedó en 0,345219 y el subgate pasó. La AP101 de desarrollo
sobre las seis clases presentes subió a 0,201421; ATE fue 1,08117 m, ASE 0,190580
y AOE 0,083536 rad.

V1 y v2 forman una sola trayectoria de optimización. No representan dos semillas
ni una réplica independiente. La mejora entre ambos tampoco mide el valor de
ImageNet, de la cabeza de velocidad o del batch cuatro, porque esas decisiones
pertenecen al piloto posterior. Un proceso nuevo cargó el checkpoint v2 sin
anotaciones y reprodujo sus predicciones con diferencia máxima cero. Los ZIP de
snapshot v1 y v2 conservan las fuentes exactas porque esos runs precedieron al
commit que incorporó su implementación.

## Perfiles de precisión, RAM y batch

El perfil de precisión comparó FP32 `highest`, FP32 `high` y autocast BF16 sobre
un frame real. Las medianas fueron 2,852, 2,806 y 3,025 segundos por paso. BF16
no aceleró esta carga y aumentó la memoria privada del proceso, así que el piloto
conservó FP32 `highest`. La diferencia máxima de profundidad observada en el
primer forward mixto fue 0,2074; no se declaró equivalencia numérica.

Durante ese perfil quedaban aproximadamente 0,4–1,5 GiB de RAM física y el host
paginaba. Torch asignaba alrededor de 1,5 GB de VRAM. Estas observaciones señalan
presión de RAM, sin identificar una causa única de la latencia. Después de liberar RAM, un perfil breve
con cuatro imágenes midió 14,11 imágenes/s, frente a 5,09 con batch uno, y un
pico asignado de 4,53 GB. Eso justificó batch cuatro para el piloto, pero no
predijo su tiempo integral.

El run sostenido invirtió 1.096,590 segundos en entrenamiento. Los tramos medidos
de update sumaron 564,524 segundos; otros 532,065 quedaron fuera de ese intervalo
en preparación, transferencias y escritura. La mediana por update terminó en
1,02226 segundos, bastante por encima de los 0,28341 segundos del perfil breve.
Desde la preregistración hasta el cierre transcurrieron 1.316,055 segundos. Esta
diferencia obliga a presupuestar con tiempo de pared y no solo con kernels o
pasos calientes. La carga terminó dentro de 1.200 segundos de entrenamiento, por
lo que no se utilizó RunPod.

## Piloto con secuencias separadas y velocidad

`pilot256_seed7_velocity_v1` se fijó antes de leer resultados: dos secuencias y
256 frames train, otras dos secuencias y 256 frames de selección, semilla 7,
batch cuatro, 512 updates, LR global 2e-4, clip 1,0 y backbone ResNet-50 ImageNet1K
V1. Las cabezas detectoras se inicializaron al azar. Cada token train apareció
ocho veces; ningún token de selección entró en la curva.

Los filtros de profundidad y visibilidad dejaron 829 objetos train de siete
clases. `other_vehicle` perdió sus tres anotaciones raw y `bicycle` conservó un
solo positivo. Selección mantuvo 938 objetos de cinco clases, dominados por car y
truck. Hubo siete frames train y uno de selección sin targets admitidos; se
conservaron como fondo. Todos los objetos admitidos tenían dos componentes de
velocidad finitos. La cabeza expone ocho clases, pero este muestreo no entrena
positivos de las ocho.

El piloto completó los 512 updates y 2.048 exposiciones. Los parámetros quedaron
finitos, cambió la cabeza de clase, cambiaron las tres cabezas de velocidad del
decoder y la recarga fue estricta. La pérdida media de centro train pasó de
0,0794023 a 0,0281789 y la de profundidad de 27,3725 a 2,23247. La AP101 de
desarrollo fue 0,058375 en train y 0,020641 en selección. Las macros usan siete
y cinco clases con GT, respectivamente, por lo que su diferencia mezcla cambio
de secuencia, distribución y soporte de clases.

La AP absoluta de selección sigue baja. De sus 938 objetos, hubo 401 matches a
menos de dos metros: 342 car, 58 truck y una motorcycle. Bus y pedestrian no
tuvieron matches. El decoder guardó 50 propuestas por imagen; 12.399 de las
12.800 propuestas de selección no se asociaron a dos metros. Ranking, duplicados,
clasificación minoritaria y geometría siguen limitando el detector.

## Controles frente a soluciones triviales

El control de imagen comparó el modelo con una caja constante por clase ajustada
solo en train y con una permutación prefijada de listas completas entre frames.
La AP101 soportada fue 0,02064071 para el modelo, 0,00412554 para el prior y
0,00372793 para la permutación. La permutación conservó las 12.800 propuestas y
no tuvo puntos fijos. La caída indica que las predicciones contienen información
asociada al frame. Una sola permutación no ofrece significación estadística y el
nivel absoluto permanece bajo.

El control de velocidad reutilizó exactamente los mismos 401 matches. El error
L2 medio fue 0,594901 m/s para la cabeza, 1,381150 para velocidad cero y 0,759128
para la media train. La mejora procede del primer componente. En el segundo, la
cabeza obtuvo MAE 0,03088 m/s y quedó por detrás de cero y de la media train,
ambos cerca de 0,021 m/s. El diagnóstico está condicionado a objetos detectados
de tres clases y no establece que los campos sean velocidad relativa compensada
ni que permitan derivar TTC.

## Inferencia independiente y cadena con PHASE

Tres procesos nuevos ejecutaron inferencia sin abrir etiquetas: siete frames
temporales, dieciséis frames train prefijados y los 256 frames de selección. Los
nueve módulos congelados y el checkpoint se verificaron por hash. Train16 y
selección reprodujeron clase, centro, dimensiones, yaw, velocidad y score con
error máximo cero. En selección, el forward mediano fue 47,89 ms y el frame
completo 82,99 ms; estas cifras excluyen construcción del modelo y PHASE.

El puente causal procesó los siete frames del checkpoint piloto y produjo una
salida completa: clase, caja 3D, yaw, velocidad, confianza, TTC firmado, fase y
límites q10/q90. La salida integrada registró TTC −5,23490 s, fase −0,0189224 y
límites de fase [−0,0224637, −0,0155820]. No había GT TTC alineado. El objeto
seleccionado difiere del usado por el checkpoint de overfit, aunque ambos reciban
un ID local parecido; sus valores TTC no se comparan como evolución del mismo
objeto. PHASE ya estuvo expuesto a TRAIN40, y el ejemplo pertenece a train del
detector. El resultado demuestra transporte e inferencia, no precisión TTC.

## Verificación y conclusión operativa

La suite focalizada terminó con 232 tests, cero fallos, cero errores y cero skips.
Ruff comprobó código y formato de 49 archivos; Pyright no encontró errores en ocho módulos
nuevos. La revisión independiente recomputó controles y tablas, contrastó hashes
y confirmó la paridad. También se añadieron guards que validan manifest, split y
calibración antes de futuras ejecuciones. El detalle está en
[final_rgb_verification.json](evidence/final_rgb_verification.json) y
[INDEPENDENT_FINAL_REVIEW.md](INDEPENDENT_FINAL_REVIEW.md).

Los hallazgos ordenan el trabajo siguiente. La prioridad es entrenar con soporte
real de las ocho clases y validar grupos de adquisición, manteniendo un conjunto
de selección que no cambie al observar resultados. Después conviene reducir
duplicados y mejorar ranking/geometría, repetir con varias semillas y medir la
degradación TTC frente a ROIs anotadas. Solo entonces tiene sentido comparar RGB
con RGB+eventos o explorar backbone compartido. La variante portable funciona en
este hardware; el experimento no acredita SOTA, robustez, scorer oficial ni
seguridad para conducción.
