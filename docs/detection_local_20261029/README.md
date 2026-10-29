# Detección 3D eAP y PHASE-TTC: documentación de la ejecución local

Fecha: 2026-10-29. Rama: `dev/eap-detection-local-20261029`.

Se implementó y entrenó un detector RGB completo con velocidad aprendida, se
reprodujeron sus predicciones en procesos separados y se ejecutó la cadena hasta
TTC firmado y límites de fase. El piloto mostró señal asociada a la imagen y
superó los controles simples, con AP todavía baja. Los informes distinguen esa
viabilidad de ingeniería de la robustez y evaluación oficial pendientes.

## Lectura principal

| Documento | Contenido |
|---|---|
| [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) | Arquitectura, módulos añadidos, contratos, geometría, port, cabeza de velocidad, puente PHASE y correcciones |
| [EXPERIMENTS_AND_FINDINGS.md](EXPERIMENTS_AND_FINDINGS.md) | Experimentos ejecutados, resultados positivos y negativos, controles, rendimiento, confusores y conclusiones |
| [REPRODUCTION_AND_HANDOFF.md](REPRODUCTION_AND_HANDOFF.md) | Entornos, fuentes, datos, hashes, comandos de entrenamiento/inferencia y condiciones para continuar |
| [RGB_TRAINING_RESULTS.md](RGB_TRAINING_RESULTS.md) | Resumen de resultados y estado final de la entrega |
| [TRAINING_TABLES.md](TRAINING_TABLES.md) | Tablas regeneradas desde recibos JSON, incluidos el primer FAIL y su continuación |
| [INDEPENDENT_FINAL_REVIEW.md](INDEPENDENT_FINAL_REVIEW.md) | Revisión de la implementación y las evidencias del cierre experimental |

## Evidencia por componente

- Datos y geometría: [MEDIA_PILOT.md](MEDIA_PILOT.md),
  [GEOMETRY_RESOLUTION.md](GEOMETRY_RESOLUTION.md),
  [GARL_ALIGNMENT.md](GARL_ALIGNMENT.md).
- Detector: [PORTABLE_DETECTOR.md](PORTABLE_DETECTOR.md),
  [NATIVE_VELOCITY_HEAD.md](NATIVE_VELOCITY_HEAD.md),
  [RGB_EXPERIMENT_ADMISSION.md](RGB_EXPERIMENT_ADMISSION.md).
- Evaluación: [DEVELOPMENT_METRICS.md](DEVELOPMENT_METRICS.md),
  [IMAGE_SIGNAL_CONTROLS.md](IMAGE_SIGNAL_CONTROLS.md).
- Inferencia y PHASE: [INFERENCE_VISUALS.md](INFERENCE_VISUALS.md),
  [DETECTOR_PHASE_BRIDGE.md](DETECTOR_PHASE_BRIDGE.md) y
  [ejemplo integrado](evidence/integrated_prediction_pilot_v1.json).

## Material conservado

Los recibos de `evidence/` incluyen poblaciones, configuraciones, entorno,
snapshots de fuentes, pérdidas y métricas agregadas, controles y verificaciones.
El [archivo documental](evidence/documentation_archive.json) añade el manifest
exacto de tokens, los siete tokens temporales y las curvas escalares de los tres
entrenamientos. No modifica sus resultados.

El repositorio conserva hashes de los pesos y predicciones completas; los archivos
pesados permanecen en el ordenador. La guía de reproducción identifica sus rutas.
Las imágenes, eventos y etiquetas de los datasets no forman parte de este commit.

## Estado histórico y alcance

[LOCAL_EXECUTION_REPORT.md](LOCAL_EXECUTION_REPORT.md) y
[G1_LOCAL_PREFLIGHT.md](G1_LOCAL_PREFLIGHT.md) corresponden a la etapa anterior al
entrenamiento. Sus bloqueos explican decisiones posteriores; no describen por sí
solos el estado actual. El [plan recibido](../../DEVELOPMENT_PLAN.md) y el
[handoff original](../detection_g0_g2_20261029/EXECUTION_REPORT.md) también se
conservan.

La implementación ejecutada admite `G0_RGB_BOX_V1` y `G1_PORTABLE_V1`. El operador
CUDA externo, el scorer oficial, la semántica física completa de velocidad y la
validación TTC independiente siguen fuera de lo demostrado. El piloto no utilizó
etiquetas Test12 ni realizó una submission. Esta ampliación documental no ejecutó
nuevos experimentos y mantiene las cifras del cierre registrado.
