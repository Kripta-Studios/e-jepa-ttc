# Handoff público V12/V13 — 10 de octubre de 2026

Este paquete permite estudiar los cambios recientes y el problema de transferencia a FCWD desde el repositorio público, sin acceso al ordenador de desarrollo. La evidencia respalda mejoras de coste y algunos resultados competitivos; **no respalda una afirmación general de SOTA**. H8 gana a Garl event-only en el mean_MiD de Dev32, pero pierde en FCWD. El test oficial eAP test12 sigue sin puntuación privada.

La aportación nueva de este handoff es una reconciliación de informes, un diagnóstico reproducible por consulta y banda, la publicación del cálculo TRAIN40 y un mapa entre código, evidencias y experimentos pendientes. No se ha entrenado ni seleccionado un modelo para redactarlo. Los entrenamientos V13 existentes siguen su curso; su estado se captura como una fotografía temporal, no como un resultado final.

## Orden de lectura

| Informe | Contenido |
|---|---|
| [01 — Cronología y arquitectura](01_CAMBIOS_Y_ARQUITECTURA.md) | Qué cambió, commits, productores, cabezas y diferencias entre versiones |
| [02 — Datos y protocolo](02_DATOS_METRICAS_PROTOCOLO.md) | TRAIN40, Dev32, FCWD, test12, MiD, elegibilidad y límites de comparación |
| [03 — Resultados reconciliados](03_RESULTADOS_COMPARADOS.md) | Precisión, resultados negativos, truncación, destilación, estado oficial |
| [04 — Diagnóstico FCWD](04_DIAGNOSTICO_FCWD.md) | Localización cuantitativa del error, ejemplos, hipótesis y pruebas discriminantes |
| [05 — Latencia y streaming](05_LATENCIA_STREAMING.md) | Cuellos de botella, cachés, reproyección, graphs, precisión, costes omitidos |
| [06 — V13 y continuidad](06_V13_PORT_Y_CONTINUIDAD.md) | Auditoría RGB, port nativo, pruebas, restauración y límites del estado actual |
| [07 — Reproducción y mapa de archivos](07_REPRODUCCION_Y_EVIDENCIA.md) | Comandos sin GPU, acceso público/local, fuentes y rutas de código |
| [08 — Plan de investigación](08_PLAN_FCWD.md) | Experimentos priorizados, controles y criterios antes de afirmar mejoras |
| [Tablas regeneradas](evidence/recomputed/TABLES.md) | Tablas completas derivadas de CSV; resultados por banda y secuencia |

## Estado que debe retener el siguiente agente

- H8 ejecuta tres cabezas (semillas 7/13/23) sobre productores compartidos. La salida comparada es la mediana de sus predicciones TTC. No son tres réplicas independientes del sistema completo.
- En eAP TRAIN40, H8 obtiene overall_MiD **72,5393**, frente a **93,6824** de Garl event-only, sobre datos de entrenamiento. No es una evaluación de generalización.
- En Dev32, el mean_MiD nativo de H8 es **138,3771**, frente a **165,984** de Garl event-only y **136,336** de Garl RGB+eventos. En FCWD, H8 obtiene **79,9518**, frente a **61,8165** de Garl event-only. Dev32/FCWD no tienen GT negativo y su overall_MiD oficial queda **no disponible**.
- El problema FCWD se concentra en TTC positivo corto: H8 sobreestima; para TTC largo subestima. No corregirlo aplicando un único desplazamiento a todas las predicciones.
- El mejor piloto H8 con reproyección y graphs tiene mediana caliente **82,345 ms**; Garl optimizado en ese mismo piloto tiene **36,806 ms**. Son doce consultas de dos secuencias ya conocidas, no un benchmark completo de despliegue.
- FCWD completo sí se ha reproducido cronológicamente desde eventos en CPU: 630 consultas, 597 elegibles. La variante warp+student baja a mean_MiD **74,5607**, todavía peor que Garl; el estudiante también empeora Dev32.
- test12: 6.762 predicciones H8 y 6.762 Garl event-only preparadas y validadas. No hay envío ni puntuación CodaBench acreditados.
- V13 incorpora adaptadores de inferencia y continuidad. Sus cifras de precisión final no pueden heredarse de V12. La receta de entrenamiento BF16 continúa desde checkpoints, sin activar las aproximaciones de inferencia.

Las cifras anteriores se desarrollan en los informes y se enlazan a sus CSV. Los redondeos no sustituyen los archivos de evidencia.

## Procedencia y corte temporal

El código V12 examinado parte de `6834735d3f21a35cad9fb1d7fb15b6c72eb242e0`, rama `scientific-recovery-v12-efficient-context`. Los nuevos commits de esta rama añaden este handoff. V13 se referencia de forma inmutable en [`da068ebdeb2d967ee10b2afa348e1317b8720ad0`](https://github.com/Kripta-Studios/e-jepa-ttc/tree/da068ebdeb2d967ee10b2afa348e1317b8720ad0). No se cambia su HEAD durante esta publicación: la cola de entrenamiento admite una identidad Git concreta.

Los documentos originales se mantienen como registros históricos. Cuando un documento anterior dice «pausado», «pendiente» o presenta un piloto inicial, no significa que describa el estado actual. Para continuidad V13, usar el informe 06 y su snapshot; para precisión y latencia, distinguir la población y el experimento concreto antes de comparar números.

## Verificación mínima desde GitHub

Desde la raíz de un checkout de esta rama, con Python 3.11 o superior:

```bash
python docs/agent_handoff_20261010/replay_public.py --output /tmp/ejepa-fcwd-review
```

En Windows puede usarse `--output E:/EJEPA_results/public_handoff_replay`. El script usa solo la biblioteca estándar, comprueba la identidad del CSV, recalcula MiD por fila, verifica los agregados originales y genera diferencias emparejadas. No descarga datos, no carga checkpoints y no consume GPU. La reproducción de inferencia o entrenamiento requiere activos adicionales detallados en el informe 07.

El [manifiesto del paquete](MANIFEST.json) registra hashes de documentación y evidencias; se verifica mediante `verify_bundle.py`. No incluye datasets crudos, pesos ni credenciales. El agente puede investigar los errores estadísticos con GitHub; no debe afirmar que ha reproducido la red por haber recalculado sus métricas.
