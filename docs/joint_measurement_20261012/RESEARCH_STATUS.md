# V18 JOINT-MEASUREMENT

Estado: RESEARCH_ONGOING / PRIMARY_TRAINING. Entrenamiento científico en curso;
todavía no hay una comparación completa de endpoints ni selección de candidato.

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

Actualización 16:49 UTC del 12 de octubre de 2026:

- Código científico fijado en 61ee840e2655e0cac1b2a714180e821b6200e80b
  (la revisión exacta también está en RUN_PROVENANCE.json); sin push.
- 62 pruebas CPU aprobadas, incluyendo referencia, contratos nativos,
  bootstrap, selección y recálculo independiente de métricas.
- Reanudación exacta CPU y CUDA en los cuatro brazos. Se conservó el primer
  intento CUDA fallido y se corrigió el muestreo bilineal manteniendo sus gradientes.
- 420 updates técnicos contabilizados, conservando intentos fallidos y repeticiones.
- P completo: 53.890 filas, 211 packs lossless, 20.110.803.791 bytes. El recibo
  COMPLETE.json de la caché es la autoridad sobre sus hashes y cobertura.
- NATIVE-P2-FT-S7 en curso: al menos 225 updates ejecutados y checkpoint durable
  en 200. Los otros tres brazos conservan sus endpoints registrados de 12.000.
- Una comparación CPU del checkpoint del update 100 con el padre P-only verifica
  cambios en los 18 tensores inspeccionados de encoder.features. La diferencia
  máxima absoluta es 0,0012035630643367767. Recibo en
  verification/FIRST_SCIENTIFIC_ENCODER_CHANGE.json. No demuestra generalización.
- Preflight real P, batch efectivo32/microbatch4, aprobado en los cuatro brazos.
  No consume updates. La recomputación de activaciones nativa se desactivó antes
  del entrenamiento tras verificar gradientes idénticos y registrar el cambio
  en PRIMARY_RUNTIME_AMENDMENT.json. La receta y el endpoint no cambiaron.
- H se está preparando con un escritor CPU propio; después se prepara V.
  Esto sólo materializa inputs. La evaluación H requiere los cuatro endpoints.
- GPU_BUDGET.json contabiliza carga, entrenamiento y guardado dentro de los scopes;
  no se presenta como tiempo exclusivo de kernels. Su estado cambia mientras se
  entrena, al igual que ACCOUNTING.json y UPDATE_JOURNAL.json de cada fit.
- RGB P: padre R_A5 final P-only y manifest de frames preparados verificados.
  No se ha seleccionado ni entrenado una receta RGB V18 todavía.

Comando probado para inspeccionar/reanudar desde este worktree:
pwsh -NoProfile -File scripts/Resume-JointMeasurement.ps1 -Once
Sin -Once continúa la matriz primaria hasta dejar P/H completos para interpretar.
No lanzar una segunda instancia mientras la primera siga viva.

Después de completar la matriz y sus cohortes P/H:
python -B -m operational.joint_measurement.primary_evidence
Completa el coste emparejado, selección prospectiva, recálculos NumPy,
bootstrap por grupos y exports CPU. No abre V ni elige una receta adaptativa.
La continuación requiere interpretar esos resultados y registrar la rama habilitada.

Preparación interrumpida se reanuda con:
python -u -B -m operational.joint_measurement.data P
usando el entorno compartido y PYTHONPATH=src;. documentado en el script.

Pendientes del objetivo: completar los cuatro fits, medir y decidir en H,
comparar en V tras congelación, ejecutar las ramas adaptativas/RGB habilitadas,
réplicas/refit cuando corresponda, costes completos, exports y bundle verificable.
