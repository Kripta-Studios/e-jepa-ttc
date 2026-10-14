# V18 JOINT-MEASUREMENT

Estado: RESEARCH_ONGOING / RGB_AUTHORIZED; fase primaria NO_SYSTEM_PROMOTION.
Los cuatro entrenamientos primarios están completos en 12.000 updates cada uno.
La matriz P/H, selección H, evaluación V del candidato seleccionado, coste E2E y
comparación con E_CTX están completos. Esto no completa el programa V18.

Actualización 14 de octubre de 2026, 10:51 Europe/Madrid:

- Caché de features de fusión: lectura NPZ sin pickle, hash comprobado antes de
  decodificar y validación del endpoint/extractor, población y orden admitidos.
  Rechaza arrays de targets. La compresión conserva los valores FP32; el lector
  carga una vez, sin volver a calcular hashes en cada batch. 39 pruebas CPU de
  features y modelo de fusión aprobadas. Aún faltan extracción real y trainer.
- Diagnóstico registrado antes de medir en `diagnostics/known_translation_P`:
  una consulta por cada uno de los 24 grupos P, cuatro traslaciones de ocho
  píxeles y control de permutación espacial. CORR-FROZEN y CORR recuperan ambos
  24.576/24.576 correspondencias interiores con margen estricto positivo; el
  control recupera 961 y 971, respectivamente (aproximadamente 3,9 %).
- Siete pruebas verifican el signo del desplazamiento, ausencia de wraparound,
  tratamiento de empates y control permutado. La medición real se ejecutó en CPU,
  sin inicializar CUDA ni ejecutar updates; no usa targets TTC. Código y recibo
  distinguen expresamente traslación de texturas y movimiento temporal real.
- Este resultado no demuestra precisión TTC ni habilita por sí solo la familia
  C. Tampoco aporta evidencia para aumentar capacidad o preentrenar traslaciones
  bajo la premisa de que la correspondencia básica haya fallado. Se conserva
  la continuación RGB registrada y se esperan sus cohortes P/H completas.

Actualización 14 de octubre de 2026, 10:45 Europe/Madrid:

- El reinicio del equipo a las 03:31 interrumpió RGB-FT. Se reanudó a las 10:19
  desde el checkpoint durable 9.900. Los 91 updates ejecutados sin guardar se
  contabilizan como recuperación. Los recibos CPU de
  `verification/reboot_recovery_20261014` conservan el origen y comprueban el
  prefijo de la trayectoria restaurada; no se reinició el fit desde cero.
- El nuevo verificador `operational.joint_measurement.verify_rgb_endpoint`
  comprobará cada endpoint final de 12.000 en CPU: trayectoria de IDs, estados
  Adam, scheduler y cambio/congelación del encoder. No confundir la comprobación
  ya realizada del checkpoint 9.900 con la verificación del endpoint final.
- `fusion_features.py` reutiliza el normalizador existente con un vínculo a los
  pesos, esquema y código extractor. Exige consultas ordenadas, cobertura event
  completa y relojes separados; ajusta estadísticas sólo en observaciones H
  disponibles. No rellena RGB ausente para calcular esas estadísticas.
- 32 pruebas CPU de features/fusión aprobadas y Ruff sin incidencias. No consumen
  updates de optimizador. El loader de caché deberá verificar los bytes contra
  los recibos durables antes de entregar arrays a esta validación en memoria.
  Faltan la extracción real, el trainer integrado y la receta de fusión, que
  dependerá de interpretar P/H de los dos endpoints RGB congelados.

Actualización 14 de octubre de 2026, antes de iniciar los fits RGB:

- El usuario autoriza continuar y ampliar el presupuesto de todo el programa.
  `BUDGET_AUTHORIZATION_20261014.json` aplica el techo existente de 48 horas
  acumuladas y un millón de updates a todas las ramas habilitadas por evidencia.
  Se conserva la autorización primaria anterior y no se reinicia la contabilidad.
- QA CUDA RGB: 40 updates sintéticos, ambos brazos PASS para 10 continuos frente
  a 5 + restauración + 5. Pesos, optimizador, scheduler, RNG, IDs, pérdidas y normas
  coinciden exactamente. Es una prueba de ejecución, no de precisión TTC.
- Se ejecuta el par ya registrado RGB-CORR-FT / RGB-CORR-FROZEN, seed7, 12k cada
  uno. Ambos endpoints deben congelarse antes de H. El driver completa P/H,
  coste preparado, recálculo independiente, bootstrap y exports; no consulta V
  para seleccionar ni registra automáticamente una fusión sin interpretación.
- Reanudación: `pwsh -NoProfile -File scripts/Resume-JointMeasurementRGB.ps1`.
  Un lock exclusivo impide duplicar el driver. La GPU mantiene sus controles
  existentes de propietario y presupuesto. Las revisiones del agente se realizan
  cada hora en punto de Europe/Madrid, sin cambiar la cadencia de checkpoints.
- El ZIP primario `31760d2` permanece como snapshot histórico verificado. Su espera
  de autorización queda superada por este permiso, no por una edición retroactiva.

Actualización tras completar la comparación de sistema, 13 de octubre de 2026:

- NATIVE es candidato de desarrollo frente a A5-P2 congelado en H: mejora del
  agregado del 7,33 % y del MiDc del 7,73 %, con las protecciones satisfechas.
  Los tres brazos PAIR incumplen las protecciones de signo y bandas. Sus
  debilidades están también en P; no se atribuyen sólo a transferencia P/H.
- En las 16.432 consultas V, NATIVE no supera E_CTX: agregado 126,106 frente a
  106,082; MiDc 178,140 frente a 156,557; MAE 5,445 s frente a 2,497 s.
  Los gates originales rechazan la promoción. V sigue siendo desarrollo expuesto.
- Coste E2E mediano de NATIVE: 189,140 ms frente a 991,200 ms de E_CTX sobre
  64 consultas emparejadas. Incluye eventos crudos y preprocesado, con pesos
  residentes y caché del sistema operativo no controlada. P2 usa menos contexto;
  E_CTX conserva su historial y expertos. No se atribuye todo a la arquitectura.
- Once tablas propias: 377.992 filas. Cinco exports CPU y bootstrap por grupos.
  Los resultados se regeneran desde artefactos mediante `primary_analysis`,
  `primary_closure` y `report` en `operational.joint_measurement`.
- Contabilidad: 48.778 updates físicos, cero pendientes; 91.171,922 segundos
  de scopes propios. La ampliación a 172.800 segundos sigue limitada a los
  cuatro brazos primarios y su evaluación.
- RGB-CORR entrenable y congelado se registraron antes de interpretar V, con
  cero updates. Su ejecución espera ampliar el alcance del techo existente.
  Réplica, refit y ramas adaptativas siguen condicionados a evidencia y presupuesto.
- `NEXT_DECISION.json` conserva la decisión prospectiva P/H.
  `NEXT_DECISION_POST_PRIMARY.json` y `CAMPAIGN_CONCLUSION.json` añaden la
  interpretación de V y conservan el trabajo restante. No afirman SOTA.

Informe generado: `artifacts/joint_measurement_20261012/primary_milestone/README.md`.
La entrega de esta fase incluye checkpoints completos y resultados negativos;
no debe presentarse como cierre del objetivo completo.

Las actualizaciones fechadas siguientes son históricas.

Actualización 21:00 UTC del 13 de octubre de 2026:

- Los cuatro endpoints tienen checkpoint completo y durable en 12.000. El sello
  `PRIMARY_ENDPOINTS.json` fija sus hashes y el padre A5-P2-FROZEN.
- Verificación CPU de NATIVE: `verification/NATIVE_ENDPOINT.json`, PASS. Sus
  18 tensores de `encoder.features` cambian frente al padre P-only, los 59 estados
  de Adam llegan a 12.000 y los primeros 6.131 registros/IDs se preservan exactamente
  tras adoptar el runtime optimizado. Cero updates nuevos durante la verificación.
- Checkpoint final nativo SHA-256:
  `7fee4171269785893bdbdd5e04602d01be8552215a59a1ef7a9a26653253ef3d`.
- Contabilidad: 48.000 primarios + 200 diagnósticos reales P + 420 técnicos +
  158 de recuperación histórica = 48.778 físicos, sin updates pendientes.
  La corrección de journals copiados se conserva en
  `verification/EXTRACTION_ACCOUNTING_CORRECTION_20261013.json`.
- Evaluación en curso: cinco tablas H completas y dos P completas de las cinco.
  No se interpreta una matriz parcial ni se abre V para escoger recetas.
- El presupuesto ampliado a 172.800 segundos de scopes propios cubre sólo estos
  cuatro brazos y su evaluación, según `BUDGET_AUTHORIZATION_20261013.json`.
  No amplía automáticamente el cómputo de ramas adaptativas, RGB, réplica o refit.

Las actualizaciones fechadas siguientes preservan el historial anterior.

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

Actualización 17:54 UTC:

- P/H/V sensoriales completos: 53.890 / 18.422 / 16.432 consultas. No se han
  evaluado endpoints en H/V; preparar inputs no constituye evaluación.
- Nativo guardado en 1.358 updates, sin updates pendientes ni de recuperación,
  para el diagnóstico registrado de prefetch. Éste cerró sus 80 updates reales P
  con reanudación y estados exactos en los cuatro brazos.
- Prefetch no alcanza su gate de aceleración nativo: DO_NOT_INSTALL. Se reanuda
  la matriz con código científico y checkpoint originales. Véase
  RUNTIME_PERFORMANCE.md; las mejoras observadas en los otros brazos se conservan.
- Contabilidad al cerrar el diagnóstico: 1.358 updates primarios + 80 científicos
  de diagnóstico + 420 técnicos = 1.858 físicos; cero recuperación y pendientes.
  Estos valores son una instantánea: el entrenamiento sigue avanzando.

Pendientes del objetivo: cerrar las cohortes P/H, medir y decidir en H,
comparar en V tras congelación, ejecutar las ramas adaptativas/RGB habilitadas,
réplicas/refit cuando corresponda, costes completos, exports y bundle verificable.
