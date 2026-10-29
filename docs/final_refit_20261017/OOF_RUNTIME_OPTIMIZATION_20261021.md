# Optimización operativa OOF — 21 de octubre de 2026

La cola científica continúa con precarga CPU acotada y descompresión RGB paralela sin pérdida. La intervención elimina casi toda la espera por preparación de lotes, pero los ensayos cortos no demuestran una aceleración equivalente del entrenamiento sostenido. No se cambia la receta científica.

## Diagnóstico

El proceso original R_A5 de la rotación 0 alcanzaba aproximadamente 26 updates/minuto. Un reinicio del código original alcanzó inicialmente 169 updates/minuto y volvió a aproximadamente 26 tras unos 100 updates. Por tanto, ni el pico inicial ni el aislamiento por proceso demuestran una mejora sostenida.

El muestreo Python de 30 segundos detectó preparación de lotes, validaciones y descompresión zlib en el camino de ejecución, además del forward y backward. Sus muestras no constituyen tiempos CUDA exactos ni una partición exhaustiva del tiempo total. Los perfiles por fases también cambiaron durante la observación; comparar únicamente sus primeras y últimas iteraciones introduciría confusión temporal.

Con la precarga activa, el checkpoint de 4.300 updates registró 133 lotes consumidos, 24,19 segundos acumulados de preparación en segundo plano y solamente 0,284 segundos de espera del consumidor. Se ha retirado esa espera del camino crítico. Un segundo muestreo de 30 segundos sobre el proceso optimizado localizó aproximadamente 4 segundos muestreados en las comprobaciones CPU de `_validate_batch` del hilo principal, además del forward y backward. Esas comprobaciones de finitud y rango se hacen antes de transferir a CUDA; no son sincronizaciones GPU. Moverlas también al trabajador de precarga es una siguiente intervención concreta, todavía no implementada ni admitida en esta versión. No se omiten controles para ganar velocidad. Todavía no hay atribución causal completa del resto entre lanzamientos, sincronizaciones y cálculo GPU. La utilización GPU instantánea baja no prueba por sí sola que CUDA Graphs vaya a resolverlo.

## Implementación activada

- `operational/oof_prefetch.py`: un productor CPU, hasta dos lotes anticipados, orden y particiones originales, errores propagados y cierre con drenaje.
- `operational/oof_runtime_overlay.py`: reutilización del decodificador RGB paralelo existente, cuatro workers y 128 MiB máximos pendientes por rol; integración con el cursor durable original.
- `operational/scientific_oof_isolated.py`: proceso independiente por productor, estados y códigos de salida comprobados; la cola conserva las dependencias posteriores.
- `operational/oof_runtime_binding.py`: autorización técnica adicional mediante hashes, archivo del checkpoint de origen y comprobación de los estados de reanudación. No sustituye la autoridad científica.
- `scripts/run_isolated_oof_stage.ps1`: lanzador de la cola con el contrato de bloqueo y resultados existente.

Se mantienen BF16, semilla 7, datos, pérdidas, arquitectura, particiones de batch, optimizador, scheduler y RNG. Los checkpoints continúan cada 100 updates y los márgenes físico y de memoria comprometida siguen en 2 GiB. Los 246 archivos de la autoridad científica permanecen sin modificar; E_A5 y E_C2F conservan sus admisiones de endpoint.

## Verificación y magnitud de mejora

Las 20 pruebas específicas pasaron. Ruff y Pyright no detectaron errores en los módulos operativos comprobados. La prueba técnica ABBA compara ejecuciones originales y optimizadas desde una copia del mismo checkpoint, incluyendo reanudación 8+8 frente a 16 updates continuos. Coinciden exactamente los hashes de modelo, AdamW, scheduler, cursor y estados aleatorios Python, NumPy, Torch, CUDA y sampler.

La prueba final ejecutó 81 updates técnicos fuera de la cadena científica; comprobó la identidad de los 253 archivos científicos y operativos antes y después. En esa prueba, la media original fue 0,429745 s/update y la optimizada 0,413046 s/update: reducción del 3,89 %. Las repeticiones técnicas anteriores dieron aproximadamente 9–11 %. Son ensayos cortos; no se usan como ETA de la campaña.

La observación sostenida posterior registró **132 updates en 301,023 segundos: 26,31 updates/minuto, 2,2805 s/update**. Incluye 30 segundos de muestreo Python. Está cerca del ritmo original de aproximadamente 26 updates/minuto: **no demuestra una aceleración global sostenida**. Sí confirma continuidad de la cola y la eliminación de la espera de preparación medida por el runtime. El resultado negativo sobre velocidad global se conserva; no se extrapola el 3,89 % técnico al coste total ni se declara resuelto el cuello de botella restante.

La activación recuperó el checkpoint científico 4.167 y produjo checkpoints nuevos con la identidad científica original y el binding operativo adicional. El plan de cola activo tiene hash `8b99924b3d44a504db309fb1f56d17356b7ac967e0d04a2f49b7d89057a695e3`. La transición administrativa terminada está archivada; repetir el script verifica el estado completo sin modificar el controlador en marcha. Una transición administrativa incompleta exige recuperación explícita y falla cerrada.

## Opciones investigadas que no se activaron

CUDA Graphs requiere controlar formas, direcciones de memoria y estado aleatorio; esta ruta mantiene particiones y formas variables y existe evidencia previa de fallo de memoria en la captura. No se ha demostrado una captura parcial compatible con la receta y la reanudación, por lo que no se activa. Véase la [documentación de PyTorch sobre CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/).

Los datos RGB ya usan compresión sin pérdida; añadir compresión no elimina el coste CPU observado. Se reutiliza la caché existente y se solapa la decodificación. No se introducen cuantización, compresión con pérdida ni aproximaciones matemáticas porque cambiarían el entrenamiento que se está completando. La precarga y el solapamiento siguen las recomendaciones de la [guía de rendimiento de PyTorch](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html), con verificación local de beneficio y equivalencia.

## Evidencias locales

Todos los siguientes archivos están bajo `artifacts/final_refit_20261019_scientific_oof/`:

- `OOF_LIVE_PROFILE.json`: muestreo Python del proceso original.
- `OOF_OPTIMIZED_LIVE_PROFILE.json` y `OOF_OPTIMIZED_PROFILE_SUMMARY.json`: muestreo posterior; se tomó durante 30 segundos de la observación de cinco minutos, que por tanto incluye ese pequeño coste de instrumentación.
- `OOF_PREFETCH_PARITY.json`: comparación técnica final y hashes completos.
- `OOF_RUNTIME_TESTS.xml`: 20 pruebas específicas.
- `OOF_RUNTIME_ADMISSION.json`: fuentes, opciones y evidencias admitidas.
- `PREFETCH_CONTROLLER_TRANSITION.json`: migración administrativa completada.
- `OOF_PREFETCH_LIVE_MEASUREMENT.json`: observación posterior de cinco minutos, con muestras y ritmo calculado desde timestamps de progreso.
- `rotations/0/fits/R_A5/OOF_RUNTIME_STATS.json`: contadores acumulados del productor activo, actualizados al guardar.

No se envían resultados a CodaBench ni se declara superioridad científica por esta optimización operativa.
