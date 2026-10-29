# E-JEPA-TTC: trabajo, resultados y análisis del 4 al 8 de octubre de 2026

Este informe reúne cinco días naturales de trabajo local. Los entrenamientos TRAIN40 y las comparaciones congeladas con Garl event-only LHR y Garl RGB+eventos están terminados. La medición causal de eficiencia R1 sigue abierta en la instantánea de cierre. No se presenta como terminada ni como resultado negativo.

En las 946 consultas EvTTC con etiqueta válida, H8 obtiene MAE de **1,321–1,409 s**, frente a **1,868 s** de Garl RGB+eventos y **8,048 s** de Garl event-only. Garl RGB+eventos tiene mejor error mediano: **0,692 s**, frente a **0,731–0,766 s** de H8. Las diferencias de colas de error son importantes. Esta evaluación de transferencia, con secuencias históricas de desarrollo, **no demuestra SOTA** ni reproduce exactamente la tabla oficial del artículo.

## Informes

1. [Cronología y cambios de alcance](01_CRONOLOGIA.md): trabajos diarios, decisiones, cambios de protocolo, descargas y ramas abandonadas o pendientes.
2. [Ingeniería, rendimiento y recuperación](02_INGENIERIA.md): caché, CPU/GPU, procesos, CUDA Graphs, RAM, fallos y qué aceleraciones se observaron realmente.
3. [Resultados científicos y análisis](03_RESULTADOS.md): E0, WIDE, ajuste TRAIN40 y las dos comparaciones EvTTC, con sus límites.
4. [Reproducibilidad, contabilidad y entrega](04_REPRODUCIBILIDAD.md): contratos, checkpoints, pruebas, hashes, regeneración y alcance del push.
5. [NEXT_DECISION](NEXT_DECISION.md): qué está cerrado y qué falta para una conclusión más fuerte.

Las [tablas regeneradas](tables/EVTTC_METRICS.md), los [CSV de predicciones](evidence/rgb_event/SCORED_PREDICTIONS.csv), las [métricas por secuencia](evidence/rgb_event/PER_SEQUENCE.csv), las [curvas TTC de los cinco modelos](evidence/rgb_event/TTC_CURVES.png) y el [historial de 77 commits](tables/COMMITS.md) acompañan al texto. Las tablas se calculan desde archivos; no se mantienen manualmente.

## Alcance de la evidencia

La base publicada de la campaña original es `fd16d8102914537653622d3abf298b753120ea43`, release `simplex-t-local-results-20261004-complete`. El código de la última comparación está congelado en `f13f018421afe5c99104475400f8ffbbefcc434d`. Los commits posteriores de esta entrega documentan el trabajo; no cambian esos resultados.

[CUTOFF.json](CUTOFF.json) fija el instante UTC de captura, las fechas incluidas y el commit de código. Las fechas narrativas usan Europe/Madrid (UTC+2). [SOURCE_INVENTORY.json](SOURCE_INVENTORY.json) distingue SHA-256 del original local y de la copia publicada. Los JSON se reformatean y sus rutas personales se sustituyen por marcadores; los CSV conservan sus bytes originales. Los recibos históricos pueden describir estados ya superados: la cronología explica cuál es la evidencia posterior aplicable.

La publicación contiene informes y evidencia compacta, junto con el código ya versionado de esta rama. Los datos crudos, checkpoints grandes y bundles completos siguen conservados localmente; este push no los convierte en descargas públicas. No se han realizado submissions ni se ha abierto Stage76 como parte de estos trabajos.

## Regeneración

Desde la raíz del repositorio, con Python y NumPy del entorno del proyecto:

```powershell
python docs/research_review_20261004_08/regenerate.py
```

El script comprueba el manifiesto SHA-256 de entrega, los hashes de evidencia, identidad de las predicciones heredadas, cobertura, MAE, mediana, RMSE, sesgo, cuatro contrastes bootstrap por secuencia y contabilidad. No necesita GPU, pesos ni datos crudos. Véanse las [instrucciones de verificación de entrega](04_REPRODUCIBILIDAD.md).
