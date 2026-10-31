# PHASE-TTC: punto de pausa y continuación — 31 de octubre de 2026

Este es el punto de entrada vigente para retomar esta investigación. El usuario
pidió esperar a que terminara lo lanzado, documentar las últimas cuatro horas,
publicarlo en Git y parar. Los pilotos temporal y canónico terminaron, y terminó
la auditoría CPU de eAP. **No se lanzó el siguiente piloto de 149 consultas ni
otro entrenamiento.** El objetivo se pausa por petición del usuario, no por
haber alcanzado SOTA ni por haber demostrado que ya no se puede mejorar.

## Lectura inicial

1. [Trabajo realizado y resultados](01_TRABAJO_Y_RESULTADOS.md): campañas,
   auditorías, optimización, fallos conservados y conclusiones.
2. [Artefactos y reproducción](02_ARTEFACTOS_Y_REPRODUCCION.md): entorno,
   rutas exactas, evidencia local, comandos y límites de portabilidad.
3. [Pendientes y plan de continuidad](03_PENDIENTES_Y_CONTINUIDAD.md): qué
   ejecutar después, qué no repetir y qué exige una promoción.
4. [Prompt para el siguiente agente](04_PROMPT_CONTINUAR.md): instrucciones
   completas listas para copiar, conservando el mismo objetivo.

## Estado que no debe confundirse

- **Modelo entregable:** `output/phase_submission/FINAL_MODEL`, sin sustitución
  por los experimentos recientes. No se ha enviado un nuevo resultado a EvTTC.
- **Fusión:** 312 ajustes completos acumulados; ninguna nueva variante promovida.
  Son ajustes de cabezas pequeñas, no 312 entrenamientos de encoders.
- **Nueva vía:** PAIR con tiempo canónico y conversión física explícita. Existe
  código probado y un piloto de dos consultas ejecutado. Mejora la consistencia
  temporal de la salida final en esos cuatro contrastes, no demuestra accuracy.
- **eAP:** existe una auditoría de deriva de las salidas PAIR actuales; falta
  conservación de contextos y cabeza completa. No confundirla con la evaluación
  TRAIN40 activa anterior ni con generalización en Test12.
- **SOTA:** no alcanzado ni acreditado. Sigue abierta la discrepancia del
  evaluador de Slider-750 y falta comparación equivalente/confirmación externa.

El repositorio es `Kripta-Studios/phase-ttc`, rama `research/evttc-transfer-v2`.
El commit `4926b765` contiene el diagnóstico temporal anterior al candidato
canónico. El commit que contiene este directorio incorpora la intervención,
la auditoría reproducible eAP y este handoff. Consultar `git log -1` y
`git status --short` al retomar; no asumir que este texto congela cambios futuros.

Los documentos históricos conservan sus decisiones de entonces. En particular,
el primer `docs/evttc_transfer_v2/HANDOFF.md` describía un evaluador incompleto:
el inventario vigente ya tiene 434 targets públicos. Este handoff y los informes
fechados posteriores prevalecen para conocer el estado actual.

No hay autorización para reiniciar trabajo automáticamente mientras siga la
pausa. El prompt del documento 04 expresa la instrucción posterior de reanudar.

## Reanudación posterior

El usuario reanudó el objetivo después de este cierre. El screen de 149 consultas
ya terminó: F0 supera su gate de desarrollo y el router falla el de carretera.
La ampliación fija a 2608 consultas está en marcha. Consultar el
[registro de continuación activa](../research_log_20261031_canonical_screen149/CONTINUATION.md)
y comprobar el estado real del proceso antes de lanzar nada. Este documento
conserva arriba el estado histórico de la pausa, no una orden de volver a parar.
