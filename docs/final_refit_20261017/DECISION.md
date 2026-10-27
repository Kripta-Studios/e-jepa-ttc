# Refit final PHASE_ONLY con TRAIN40

**Instrucción vigente del usuario:** preparar y entrenar el modelo final con
todo TRAIN40 público antes de la evaluación oficial. No se enviará primero el
V16 histórico; queda como referencia congelada. La autorización de trabajo
permite continuar los componentes finales comunes mientras se integra el
stacking superior. El envío externo del modelo final sigue siendo separado.

**Revisión de familia:** véase `FAMILY_COMPARISON.md`. V16 sigue siendo la
recomendación provisional por su interfaz física para el stacking OOF, no porque
su ZIP demuestre superioridad arquitectónica. El presupuesto completo y el runner
OOF siguen pendientes. Los productores RGB comunes ya tienen admisión técnica
estricta; R_A5 inició entrenamiento científico el 17 de octubre a las 14:19
(hora local). Véase `BF16_RUNTIME_FIX.md` y los recibos de `fits/R_A5`.

**Medición operativa:** R_A5 se reanudó desde el update durable 169 y continuó
activo tras medir 900,104 s el 17 de octubre, de 14:33 a 14:48 CEST: 547 updates,
36,46 updates/min, checkpoint cada 100. El runtime científico no se cambió.
El desglose y las prioridades de optimización están en `RUNTIME_15MIN_REPORT.md`.

**Runtime actual:** tras esa medición se admitieron un hilo intra-op y el monitor
de recursos cacheado. R_A5 se migró exactamente desde el update 1.407 a
`artifacts/final_refit_20261017_runtime_v2` y se reanudó; ya se verificaron los
checkpoints 1.500 y 1.600. El directorio anterior conserva el origen y permanece
pausado. Checkpoint cada 100 y receta científica intactos. Véase
[`RUNTIME_OPTIMIZATION_APPLIED.md`](RUNTIME_OPTIMIZATION_APPLIED.md) para las
pruebas, genealogía y medición posterior; no usar el ritmo perfilado anterior
como ETA vigente sin consultar ese informe.

**Concurrencia autorizada posteriormente:** el usuario ordenó ejecutar R_A5 y R_C2F
juntos, sin comparativas adicionales. Se cerró limpiamente la cola secuencial y se
guardó R_A5 en 4.990; ambos productores se lanzan ahora independientemente mediante
`RESUME_CONCURRENT_RGB.ps1`, con exclusión mutua por fit y detección de procesos
existentes. Se conservan el plan, la receta, las 28 fuentes ligadas y checkpoint cada
100 updates. El recibo operativo es
`artifacts/final_refit_20261017_runtime_v2/CONCURRENT_LAUNCH.json`. La ETA obtenida
con R_A5 solo no describe esta nueva ejecución concurrente.

**Estado actualizado:** el diseño secuencial descrito abajo queda como alternativa
de coste menor, no como cola autorizada para arrancar. La revisión posterior del
usuario motiva preparar el esquema OOF anidado de cinco bloques. Véanse
`CRITICAL_REVIEW.md` y `NESTED_ROTATION_REVIEW.json`. Cero updates científicos
nuevos al tomar esta decisión; no se ha ejecutado el plan secuencial.

La petición del usuario del 17 de octubre autoriza preparar y ejecutar el
entrenamiento final, optimizar su ejecución y generar un paquete local para
CodaBench. El envío externo sigue pendiente de autorización expresa. V18
permanece cerrado como `NO_CANDIDATE`.

## Decisión previa al entrenamiento

Se conserva la familia PHASE_ONLY de V16, semilla 7, RGB más eventos y sin
ensemble. Sus criterios históricos y su interfaz física de fusión apoyan esta
elección operativa; no establecen superioridad general frente a V18. F_TRUE obtiene el
menor error en la reconstrucción EvTTC de 46 consultas; eso no basta para
seleccionarlo después de observar esa evaluación ni demuestra una superioridad
general. La comparación V18 frente a V16 sigue siendo incierta.

La población final comprende las 88.744 consultas de las 40 secuencias públicas
TRAIN de eAP. P, H y V pasan a ser datos de entrenamiento de este modelo final;
sus resultados posteriores serán descriptivos de entrenamiento, no una nueva
estimación de generalización. No se incorporan EvTTC, Dev32/FCWD ni Test12 al
entrenamiento. Las seis secuencias adicionales del listado histórico de Garl no
se consideran disponibles por el mero hecho de figurar en ese listado.

Se reutilizan únicamente los productores event A5 y C2F de V12 que ya
completaron 18 épocas sobre TRAIN40. Se conservan sus hashes y su genealogía.
Antes de utilizarlos se exige comprobar la compatibilidad numérica de su ruta
event con la implementación usada por el refit. No se reutilizan PAIR/H8 de
V12: su conversión de fase y normalización difieren de la ruta V16.

| Etapa nueva | Updates científicos | Batch efectivo | Origen de la exposición |
|---|---:|---:|---|
| R_A5 | 49.932 | 32 | 18 épocas |
| R_C2F | 49.932 | 32 | 18 épocas |
| PAIR_E | 11.264 | 32 | ceil(6.840 × 88.744 / 53.890) |
| PAIR_R | 11.264 | 32 | ceil(6.840 × 88.744 / 53.890) |
| E_CTX | 12.044 | 128 | ceil(2.500 × 88.744 / 18.422) |
| R_CTX | 12.044 | 128 | ceil(2.500 × 88.744 / 18.422) |
| PHASE_ONLY_EVENT | 12.044 | 128 | ceil(2.500 × 88.744 / 18.422) |
| **Total nuevo** | **158.524** | | |

Se conservan arquitectura, pérdidas, precisión y tasas de aprendizaje de las
recetas de origen. En las cabezas temporales y PHASE_ONLY se escala el warmup
a 482 updates y el final del scheduler a 12.044. La selección es el endpoint
fijo, sin escoger checkpoint, semilla o clipping según EvTTC o Test12.

Las etapas se entrenan secuencialmente. Las entradas de las capas superiores
se generan con padres congelados que ya han visto TRAIN40. Este stacking es
declaradamente in-sample: no es OOF ni proporciona una estimación independiente
del error. Se regeneran las características, normalizadores y cabezas afectados
por cada cambio de padre. El plan anterior de 33 fits y 751.076 updates queda
como alternativa de investigación no ejecutada; no se presenta como requisito
para esta versión final.

## Ejecución y verificaciones

La ruta inicial de aceleración es la caché RGB comprimida sin pérdida, decodificación
paralela acotada, precarga y caché del teacher DINO ya existente. Se mantiene
BF16 en los productores y FP32 en las cabezas. CUDA Graphs requiere una admisión
específica de RGB porque los lotes se dividen en secuencias T2/T3 de tamaños
variables; una admisión de un productor event no cubre esta ruta.

Las pruebas técnicas usan copias desechables y contabilización separada. Deben
comprobar valores, gradientes, avance del optimizador, scheduler, estado aleatorio
y restauración antes de habilitar una optimización. Las formas no admitidas
mantienen la ejecución ordinaria. No se reduce precisión ni se usa compresión
con pérdida para obtener una cifra de velocidad favorable.

Los tiempos históricos de RGB sugieren unas 15,4 horas para los dos productores,
sin incluir la preparación, extracción de características ni cabezas. Es una
estimación condicionada, no una medición de este refit. El preflight, el benchmark
local y los recibos de ejecución conservarán por separado lo comprobado y lo
pendiente. Ninguna tabla de esta decisión implica que los nuevos fits ya existan.

Al completar el entrenamiento se congelará toda la cadena y se verificará la
exportación antes de inferir las 6.762 consultas oficiales. El paquete local
conservará cobertura, hashes, modalidad, historia temporal, versiones y la
distinción entre validación de formato y puntuación oficial. No se afirma SOTA
sin la evaluación correspondiente.

Referencias de las optimizaciones: [CUDA Graphs en PyTorch](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
y [transferencias con memoria fijada](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html).
Las APIs concretas deben corresponder al PyTorch 2.11 instalado.
