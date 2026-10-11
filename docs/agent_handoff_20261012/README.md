# Handoff V12/V13: resultados, reparaciones y límites científicos

Corte documental: 12 de octubre de 2026. Este paquete consolida el trabajo de los
días 8–12 y actualiza el handoff del día 10 con los endpoints V13 finales, los
costes completos y la transferencia RGB a FCWD. Está pensado para un agente que
solo dispone del GitHub público o del ZIP: cada conclusión remite a código,
predicciones o recibos. Los documentos anteriores se conservan como historia;
sus estados «pausado» o «pendiente» no son el estado actual.

**Hay mejoras medibles, pero la evidencia no permite afirmar SOTA.** La fusión
V13 mejora MAE frente a Garl multimodal en Dev32 y frente a Garl event-only en
FCWD. En FCWD se usa una modalidad adicional y Garl sigue ganando MiD crucial.
El test12 oficial tiene predicciones empaquetadas, sin puntuación privada.

## Orden de lectura

1. [Estado y arquitectura](01_ESTADO_Y_ARQUITECTURA.md): qué terminó, nombres de modelos, raíces y bloqueos reales.
2. [Cambios y recuperación](02_CAMBIOS_Y_RECUPERACION.md): problema, intervención, código, comprobación y límites.
3. [Protocolos y métricas](03_PROTOCOLOS_Y_METRICAS.md): cohortes, MiD, diferencias entre versiones y comparaciones admisibles.
4. [Resultados regenerados](04_RESULTADOS_GENERADOS.md): endpoints, validación, FCWD y Dev32, cobertura, sesgos y costes.
5. [Diagnóstico de fallos](05_DIAGNOSTICO_Y_FCWD_RGB.md): transformación RGB, concentración del error e hipótesis pendientes.
6. [Optimización y latencia](06_OPTIMIZACION_Y_LATENCIA.md): beneficios comprobados y experimentos negativos.
7. [Por qué no es SOTA y qué falta](07_SOTA_Y_PLAN_DE_INVESTIGACION.md): afirmaciones defendibles y experimentos prioritarios.
8. [Reproducción y mapa de archivos](08_REPRODUCCION_Y_BUNDLE.md): comandos, dependencias, artefactos y contenido del ZIP.

## Qué contiene la entrega

La carpeta `evidence/` añade instantáneas de los resultados V, endpoints finales y
estado de tareas; los CSV externos y las auditorías ya publicados permanecen en
sus carpetas originales para evitar versiones numéricas divergentes. El script
`replay.py` recalcula las métricas externas con la biblioteca estándar y regenera
la tabla consolidada. `SOURCE_MANIFEST.json` identifica por SHA256 las entradas
documentales exactas. Verificar métricas desde CSV no reproduce la inferencia.

El ZIP contiene dos árboles separados, `V12/` y `V13/`, con código, tests,
configuraciones, informes y evidencia textual versionada, más un manifiesto de
integridad. Los checkpoints y datos de sensores se localizan en el mapa, pero no
se redistribuyen. El bundle no incluye el binding privado del ordenador.

## Procedencia histórica V12

Revisión V12 consultada: `1a686e338a1145c6794bec0704bdae4cda95f06d`.
Revisión V13 anterior a esta publicación: `133002a81454982bcf801f94dc99441c13ec1f22`.
El commit exacto de la entrega se obtiene de Git; el ZIP lo registra en su manifiesto.

Handoff V12 completo, con tablas y replay propio:

https://github.com/Kripta-Studios/e-jepa-ttc/tree/1a686e338a1145c6794bec0704bdae4cda95f06d/docs/agent_handoff_20261010

Esta publicación es documental. No consume updates, cambia modelos ni reanuda
fases ya completas. V14 es un worktree independiente y queda fuera de esta entrega.
