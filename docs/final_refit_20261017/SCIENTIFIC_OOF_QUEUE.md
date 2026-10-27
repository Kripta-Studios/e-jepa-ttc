# Cola científica OOF y final TRAIN40

La continuación solicitada el 19 de octubre implementa cinco rotaciones completas
24/8/8 y la cadena final PHASE_ONLY. El antiguo `continuation_queue.py` queda como
observador histórico; su ausencia de adaptadores no describe esta implementación.
La cola nueva es `operational.final_refit.scientific_oof_queue`.

Activada y verificada el 19 de octubre a las 22:11 CEST: controlador vivo,
bloqueo exclusivo adquirido y estado `WAITING_RGB`, con R_C2F aún en ejecución.
`ACTIVATION_VERIFIED.json` conserva esa comprobación; el estado vigente se consulta
en `SCIENTIFIC_OOF_QUEUE_STATE.json`. El preflight fijó 246 archivos de código
y verificó la cobertura de las 88.744 consultas en los cinco bloques de generación.
Las rotaciones OOF todavía no habían comenzado en el momento de la activación.

## Ejecución

```powershell
.\scripts\resume_scientific_oof.ps1
```

El plan local es
`artifacts/final_refit_20261019_scientific_oof/SCIENTIFIC_OOF_QUEUE_PLAN.json`.
El controlador espera los endpoints completos de R_A5 y R_C2F y el cierre de su
supervisor. Después ejecuta una rotación cada vez y, al terminar las cinco, la
cadena final. El estado durable está en `SCIENTIFIC_OOF_QUEUE_STATE.json`.

La ejecución requiere que el equipo permanezca encendido. Tras una interrupción
del controlador se puede repetir el comando; un bloqueo de propietario y la
identidad de proceso evitan lanzar otra copia sobre un entrenamiento activo.
Una pausa manual (`STOP_REQUEST`) permanece vigente y no se elimina automáticamente.
Los fallos científicos o de integridad quedan registrados y detienen la cola.
Las pausas limpias por recursos se reintentan conservando las reservas de 2 GiB
de RAM física y memoria comprometida.

## Trabajo registrado

Cada rotación entrena desde inicialización seed7 cuatro productores durante 18
épocas sobre sus 24 grupos. PAIR_E y PAIR_R se ajustan sobre esos mismos grupos.
Los refinadores E_CTX y R_CTX, incluidos sus normalizadores, usan los ocho grupos
de refiner. Los ocho grupos restantes generan las entradas OOF. Ningún ancestro
supervisado de esas entradas ha entrenado con su grupo de generación.

Se recalculan todas las columnas aprendidas de eventos con los padres de la
rotación. H8 histórico aporta índices verificados, nunca features aprendidas por
un productor que vio TRAIN40 completo. Los archivos que contienen las entradas
también pueden contener targets de TRAIN; se declara su lectura y se impide su
uso como features. Las etiquetas se incorporan por ordinal, token y grupo exactos
cuando se construye la caché de entrenamiento correspondiente.

El stitch admite PHASE17 físico y las veinte coordenadas físicas/temporales de
PHASE_ONLY en orden RGB y eventos. Los estados ocultos de encoders independientes
no se mezclan entre rotaciones. La adaptación a 340 dimensiones rellena con cero
las coordenadas que la cabeza original no consume.

La cadena desplegable reutiliza únicamente los cuatro productores finales
TRAIN40 admitidos, reconstruye ambos PAIR, entrena E_CTX/R_CTX sobre PHASE17 OOF y
PHASE_ONLY sobre PHASE20 OOF. Se registran por separado la genealogía de las
entradas de entrenamiento y los padres utilizados en despliegue. La distribución
de productores entrenados con 24 grupos puede diferir de la de productores
TRAIN40; el refit no constituye una demostración causal de mejora por más datos.

La exposición nueva fija es de **749.594 updates**: 45 ajustes nuevos contando
productores y cabezas de las cinco rotaciones y las cinco cabezas finales. El
detalle por población/etapa está en `SCIENTIFIC_EXPOSURE.json`. No se deducen horas
de latencias de inferencia; se medirá el ritmo real de los entrenamientos OOF.
No se seleccionan semillas, pérdidas o checkpoints usando los grupos de
generación ni test12. No se reutiliza presupuesto de V18 ni se modifica su cierre.

## Durabilidad y verificación

Los productores guardan cada 100 updates y al alcanzar el endpoint o una pausa
limpia. Los checkpoints conservan pesos, optimizador, scheduler, estados RNG,
orden del sampler y cursor. Las cabezas avanzan en fragmentos de hasta 100
updates. Una interrupción abrupta puede perder el trabajo posterior al último
checkpoint durable; la restauración registra ese margen de recuperación.

Las features y predicciones usan fragmentos recuperables. Su reutilización exige
identidad de población, padres y hashes coincidentes. Los archivos huérfanos se
comparan contra el resultado determinista antes de admitirlos.

Las pruebas incluyen reanudación exacta frente a una ejecución continua,
rechazo de roles/padres/cachés cambiados, cobertura OOF, orden RGB/eventos,
pausas y controlador. La prueba técnica real de los cuatro productores comprende
ocho forward/backward con batch32 y BF16, tanto en calentamiento como con pérdida
completa. Sus pérdidas y gradientes fueron finitos; no ejecutó optimizer updates.
El recibo es `REAL_OBJECTIVE_PROBE.json`.

El final automático sella la cadena, verifica la exportación de la cabeza y
genera las 6.762 predicciones y un ZIP local de test12. No sube archivos a
CodaBench, no accede a sus etiquetas privadas y no declara puntuación oficial o
SOTA. La generación del ZIP completo sólo se declara después de ejecutar y
verificar esas etapas; la preparación del código no equivale a haberlas ejecutado.
