# V18 JOINT-MEASUREMENT

Estado: RESEARCH_ONGOING / PREPARING_P. Sin resultados científicos V18 todavía.

Base: `396abe922537d9dac19d6db3451e59d283636e20`. Worktree y rama propios.
El paquete recibido permanece en `docs/joint_measurement_handoff_20261012`.
El addendum de implementación fija los detalles antes de entrenar.

La fuente P-only A5 y los tres roles han pasado admisión por identidad y grupos.
Se preparan inputs sensoriales P2 lossless, con recibos por pack. La ruta de par
recibe gradiente hasta el encoder. Los tests CPU cubren correlación, soporte,
freeze, masas, anotaciones P2, reducción efectiva nativa y recomputación con RNG.

Estado durable: `artifacts/joint_measurement_20261012/`.
Datos externos nuevos: ruta configurada en `configs/joint_measurement/primary.json`.
Contabilidad actual: `ACCOUNTING.json`; extracción: `PREPARE_P_PROGRESS.json`.
No se afirma SOTA; tampoco se considera completado el programa por tener código.

Actualización 15:55 UTC:

- Integración en commits locales d9181df y 46f76f6, sin push.
- 32 pruebas V18 y 24 regresiones nativas aprobadas; prueba del bootstrap aprobada.
- Reanudación exacta CPU y CUDA en los cuatro brazos. Se conservó el primer
  intento CUDA fallido y se corrigió el muestreo bilineal manteniendo sus gradientes.
- 260 updates técnicos, cero científicos. GPU_BUDGET.json contabiliza los scopes
  reales de QA y diagnóstico; no son tiempo de entrenamiento científico.
- Preparación P: 26.368 / 53.890 filas durables a las 15:55 UTC.
- Runner primario activo esperando P completo. Respeta el proceso V16 test12
  reanudado; no inicia trabajo GPU sobre un propietario Python ajeno.
- El preflight CUDA con batch efectivo32 quedó aplazado por ese propietario.
  Se puede reanudar con python -B -m operational.joint_measurement.preflight.
- RGB P: padre R_A5 final P-only y manifest de frames preparados verificados.
  No se ha seleccionado ni entrenado una receta RGB V18 todavía.

Comando probado para inspeccionar/reanudar desde este worktree:
pwsh -NoProfile -File scripts/Resume-JointMeasurement.ps1 -Once
Sin -Once continúa la matriz primaria hasta dejar P/H completos para interpretar.
No lanzar una segunda instancia mientras la primera siga viva.

Preparación interrumpida se reanuda con:
python -u -B -m operational.joint_measurement.data P
usando el entorno compartido y PYTHONPATH=src;. documentado en el script.

Pendientes del objetivo: completar los cuatro fits, medir y decidir en H,
comparar en V tras congelación, ejecutar las ramas adaptativas/RGB habilitadas,
réplicas/refit cuando corresponda, costes completos, exports y bundle verificable.
