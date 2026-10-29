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
