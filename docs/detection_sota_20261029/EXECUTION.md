# Continuación S0–S6: registro de ejecución

Base: `24b8f36f6e4aa17c965f9526d8cf486b512a8046`.
Rama: `dev/eap-s0-evaluation-20261029`. Inicio: 2026-10-29.

El usuario autorizó ejecutar el plan completo y continuar la investigación según
los resultados hasta un modelo final. Este registro distingue implementaciones,
experimentos y gates científicos; no declara completa una fase por tener código.
El plan recibido y sus anexos están conservados byte a byte en
[evidence/intake](evidence/intake/SOTA_DEVELOPMENT_PLAN.md).

## Referencia conservada

- Los 12 archivos enumerados en el manifiesto del ZIP verifican SHA-256.
- El plan suelto coincide byte a byte con el incluido en el ZIP.
- El checkpoint local mide 498.531.791 bytes y coincide con
  `dc0ce304ad1c8a791ef66395be85b09f22526bc11a834472dcb18ce16384e338`.
- El piloto, su snapshot, sus predicciones y su worktree se conservan intactos.
- El checkout original permanece en `main`; el trabajo nuevo usa otro worktree.

El [recibo de intake](evidence/intake/receipt.json) fija los archivos recibidos.

## Secuencia de trabajo

| Fase | Trabajo | Estado inicial de esta ejecución |
|---|---|---|
| S0 | Protocolo independiente, ignorados, diagnóstico y observador externo | En implementación y verificación |
| S1 | Auditoría de cobertura, D4096, trainer con resume y curvas, R0/R1 | Preparación; entrenamiento pendiente de admisión |
| S2 | Intervención geométrica o ranking | Se decidirá con S0/S1 |
| S3 | Historia RGB causal y controles | Pendiente del baseline |
| S4 | Fusión con eventos y controles | Pendiente de S3 y alineamiento |
| S5 | Asociación global, generaciones y servicio multiobjeto | Ingeniería en paralelo; precisión TTC requiere gate independiente |
| S6 | Semillas, calibración, confirmación y modelo final | Pendiente de resultados y población admisible |

Se mantiene el split candidato 24/8/8. El 24/8/4/4 no se adopta sin comprobar
adquisiciones relacionadas, exposición y soporte. No se usan etiquetas test.
La evaluación de desarrollo v2 seguirá identificada como local mientras no haya
scorer oficial verificado. TRAIN40 no acredita generalización sobre TRAIN40.

## Verificación inicial

Pasaron 84 pruebas focalizadas históricas con el entorno PHASE existente:
`test_detection_development_metrics`, `test_monodgp_eap_adapter`,
`test_infer_monodgp_eap`, `test_eap_rgb_data` y `test_detection_protocol`.
La primera colección en el entorno detector falló porque no incluye `h5py`,
limitación ya documentada del entorno; se conservó y se usó el entorno PHASE
para esa suite, sin modificar dependencias.

## Corte reproducible `current_v1`

El estado posterior al registro inicial está publicado en
[GENERATED_RESULTS.md](GENERATED_RESULTS.md) y en el bundle
[evidence/current_v1](evidence/current_v1/README.md). La tabla se genera desde
los JSON canónicos; no contiene valores transcritos a mano. El manifiesto del
bundle conserva SHA-256 tanto de cada fuente como del archivo empaquetado.

El corte incluye la reevaluación S0 v2, controles constante y permutado con
bootstrap por secuencia, paridad del observador, cobertura y auditoría D4096,
perfil real de lectura con dos configuraciones de cache, admisión smoke 4→8,
perfil GPU y el replay real detección→tracking→PHASE con límites nativos de
fase. Excluye deliberadamente datos, RGB/eventos crudos, checkpoints y estados
del optimizer.

R0 v2 permanece identificado como `RUNNING_NO_SCIENTIFIC_RESULT_INCLUDED` en
este corte. El productor congelado es `fcad2299`; la revisión de código S2 es
`88d607ba`. El fallo de preflight de R0 v1 se conserva y acredita cero updates.
No se deriva ninguna conclusión científica de los logs vivos de R0.

Regeneración y verificación byte a byte:

```powershell
python scripts/build_detection_sota_evidence.py
python scripts/build_detection_sota_evidence.py --check
python scripts/build_detection_sota_evidence.py --from-bundle --check
```

Los dos primeros comandos requieren los outputs locales originales y verifican
su regeneración exacta. El tercero es la verificación pública autosuficiente:
comprueba los hashes de cada payload empaquetado, descomprime los JSON y deriva
de nuevo los escalares y tablas sin acceder a `output/`. Los SHA-256 de fuentes
originales con rutas redactadas son una atestación del empaquetado local; el modo
público no pretende reconstruir bytes eliminados por esa redacción.

Los gates exactos S0–S6 y las limitaciones pendientes se extraen en el documento
generado. En particular, `current_v1` no acredita scorer oficial, soporte raro
suficiente, historia RGB causal, contribución de eventos, generalización TTC ni
confirmación multisemilla/SOTA.

## Incidencia posterior al corte: RGB incompleto en el release fijado

R0 v2 terminó con `FAILED` el 2026-10-29 a las 16:57:47 UTC, después de 725
updates completos. El checkpoint durable conserva 512 updates; no llegó a la
primera selección programada en 1.024. No se presenta una AP nueva. Los logs,
el checkpoint y `failure.json` permanecen en su directorio original.

La reproducción estricta identificó el token
`OYgB6RGWcq:000000661_000000160`: el PNG tiene 393.216 bytes y Pillow rechaza
su carga completa. El TAR de 1.075.773.440 bytes coincide con el SHA-256 público
`d1f9956ad10e2cbd436625edc734b90de7de0bd7368204bd7e2273208f3d3aa5`.
Por tanto, una descarga idéntica del release no corrige el miembro incompleto.
No se habilita `LOAD_TRUNCATED_IMAGES` ni se omiten ejemplos silenciosamente.

La validación completa terminó: 23.958/23.958 frames de selección y 4.095/4.096 de
train decodifican estrictamente; los 107 TAR coinciden en tamaño y SHA-256 con el
inventario público fijado. La revisión v5 sustituye únicamente el PNG fallido por
el candidato admisible más cercano de la misma secuencia, 100.001 µs anterior.
Conserva ambas selecciones y todas las identidades entrenables por clase. El
manifest v5 tiene SHA-256
`9beeb6a3c5b76bf994808eee0b3f0dfa2f7339f139bd120dbfa6318e414945f1`.
El recibo liga los 28.054 hashes de miembros antes de CUDA y los comprueba en
primera lectura y resume. El reinicio científico será frío, común a R0/R1/R2;
el run fallido se conserva sin migrar ni reinterpretar su checkpoint 512.
Véanse [RGB_FULL_PREFLIGHT.md](RGB_FULL_PREFLIGHT.md) y
[D4096_READABLE_REVISION.md](D4096_READABLE_REVISION.md).

La evaluación completa también requería corregir su escritura: la versión anterior
materializaba un JSON gigante en memoria. La escritura incremental probada conserva
bytes y métricas exactamente en 1.024/4.096 frames. Su perfil revisado queda en
`output/evaluation_memory/streaming_writer_profile_v1.json`, SHA-256
`a892e08d6048332a0f2928a43131ee7d7b3f3375d8d81e52f8562fff1f7f31d4`.
La proyección a 23.958 frames es una estimación conservadora, no una medición a ese
tamaño. No se ha ejecutado ninguna migración de checkpoints.
