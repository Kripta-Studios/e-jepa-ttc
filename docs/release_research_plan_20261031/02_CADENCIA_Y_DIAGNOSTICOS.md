# C-TIME: cadencia física y diagnóstico del release

## Hipótesis y beneficio concreto

PAIR recibe características visuales y tiempos. Si su respuesta conserva una fase asociada a la cadencia de entrenamiento cuando cambia el intervalo real, los contextos reciben una tasa de aproximación incorrecta. La corrección propuesta hace explícita esa conversión antes de construir PHASE17. Puede mejorar transferencia entre frecuencias de cámara sin entrenar otra red.

La evidencia actual permite probar esta hipótesis, no darla por resuelta. En el screen149, F0 pasó las cuatro métricas registradas; el router posterior falló en carretera. F0 pasó de 0,748853 a 0,748672 s de MAE macro en carretera y de 1,178156 a 1,120809 s en Slider. La cobertura quedó igual. [Resultado existente](../research_log_20261031_canonical_screen149/README.md).

## Experimentos

**T0: replay exacto de R0.** Verificar los nueve pesos, normalizadores, entradas, lotes, snapshots y salidas. Capturar fases crudas, duraciones, soporte, clipping y dependencias de historia. Si no coincide con el baseline guardado, resolver primero la diferencia. No interpretar un cambio de preparación como efecto de la hipótesis temporal.

**T1: PAIR canónico sin updates.** Reutilizar la implementación [CanonicalPair](../../operational/phase_timewarp/canonical_pair.py) bajo su contrato. Fijar d0=0,1 s; convertir fase cruda a q, aplicar d0/Δ real y volver a fase antes de la emisión. La conversión debe respetar `1-d0*q>0`; contabilizar salidas fuera de dominio, sin ocultarlas mediante clipping temprano. Mantener la identidad exacta cuando Δ=d0.

Recalcular raw17, expertos, desacuerdos, E_CTX/R_CTX y cabeza. Aplicar por separado la intervención de eventos, RGB y ambas **solo como localización de mecanismo** con la regla fijada antes; no escoger retrospectivamente la combinación favorable y llamarla confirmación del screen previo.

**T2: prueba del sistema de selección temporal.** El warp de tensores preparados comprueba unidades. El re-encoding de eventos, selección de frames y ventanas comprueba el pipeline real. Deben producir informes distintos. Registrar cómo se transforma exposición, historia, target y política de muestreo. El cambio de timestamps no simula por sí solo las no idealidades del sensor.

**T3: conservación nativa.** Reconstruir historias eAP con la misma genealogía de despliegue. La auditoría de 88.744 entradas PAIR actuales no prueba identidad de historias completas. Evaluar fases crudas y emitidas, cobertura, signos, cambios de expertos y salida final. Si hay diferencias, medir precisión en datos autorizados: proximidad numérica al baseline no garantiza no inferioridad.

La ampliación de 2.608 consultas figura en [CONTINUATION.md](../research_log_20261031_canonical_screen149/CONTINUATION.md). Consumir sus recibos al completarse y no lanzar otra copia. La ausencia de resultado final no bloquea redactar ni instrumentar los siguientes diagnósticos fuera de las fuentes congeladas del run.

**Actualización al cierre:** la [auditoría de históricos](../research_log_20261031_canonical_screen149/EAP_EVENT_CADENCE.md) registra 88.744 deltas nominales de eventos exactamente float32(0,1), pero intervalos físicos de 99.945 a 100.058 µs. La rama de identidad canónica conserva el reloj nominal existente; no corrige por ello esos microdesajustes. Registrar `dt_feature` y `dt_sensor` separados. Sustituir uno por otro requeriría un experimento nuevo y reconstrucción coherente, no una modificación silenciosa del screen.

El [preflight de despliegue](../research_log_20261031_canonical_screen149/EAP_DEPLOYMENT_PREFLIGHT.md) localiza 5.805 entradas físicas y 36 shards RGB. Sus deltas de eventos cumplen identidad; los 5.805 deltas RGB no. Faltan verificar hashes de medios y capturar PAIR133 para recalcular la intervención RGB. La disponibilidad de los archivos facilita T3, pero no demuestra conservación de contextos/cabeza ni precisión.

## Magnitudes para detectar el fallo

Con tiempos transformados por t'=t0+α(t−t0), el objetivo ideal es q'=q/α y T'=αT. Calcular `|α*q'-q|` y un error relativo con suelo fijo en q, junto con error TTC. Reportar por separado valores saturados o cercanos a q=0. Una salida constante de cero satisface algunas comprobaciones: acompañar la simetría de targets de movimiento y controles de signo.

Para escala espacial s constante, q debería mantenerse dentro de la resolución/soporte disponible. Un resultado distinto al reducir resolución puede proceder de pérdida real de información; no penalizarlo como violación exacta de una simetría ideal. Separar un cambio puro de coordenadas de un resampling que elimina eventos.

Instrumentar por consulta:

```text
identity, track, camera, query_time, endpoint_times, actual_dt
roi_transform, history_ages, valid_observations, event_count
PAIR.raw_phase, raw_q, raw17, context.raw_location, fusion.correction
emitted_ttc, saturation, reject_reason, preprocessing_ms, total_ms
```

Las trazas pueden revelar dónde se pierde la relación con el tiempo. No necesitan targets privados ni modelos nuevos.

## Criterio de decisión

Conservar el criterio original del screen149 y de su ampliación: no nuevas ausencias, ninguna de las cuatro métricas macro MAE/RAE empeorada y alguna mejora estricta, por ruta. Es un criterio exploratorio ya fijado, no una prueba de significación.

Si F0 mantiene beneficio y pasa conservación/evaluación nativa, C-TIME merece confirmación. Si solo mejora Slider, estudiar una variante de dominio con alcance explícito; no cambiar el release global por ese resultado. Si mejora la simetría y empeora TTC, conservar el hallazgo y estudiar distribución de entrenamiento antes de entrenar PAIR de nuevo.

El control de un eventual reentrenamiento será PAIR con su receta original, mismos ejemplos y presupuesto, frente a PAIR con intervalo explícito/equivariancia. La conservación de fase por construcción no reemplaza el contraste con observaciones independientes.
