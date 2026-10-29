# V18 — aprender una medición relativa, no recalibrar indefinidamente la misma representación

Propuesta nueva del 12-10-2026. Base de código inspeccionada: `396abe922537d9dac19d6db3451e59d283636e20`.
V17 permanece cerrado `NO_CANDIDATE`; sus negativas no cambian de nombre ni se promueven retrospectivamente.

**Decisión:** entrenar y comparar un predictor espacial de pares con el encoder realmente entrenable, frente a controles de igual entrada/readout, y completar una evaluación de sistema. No hay garantía previa de SOTA. Este paquete es un plan, contratos y referencia ejecutable; NO es un trainer integrado ni un resultado eAP nuevo.

## Lectura

1. `docs/01_REVIEW_V17.md`: evidencia, recálculos y qué queda descartado.
2. `docs/02_PRIMARY_EXPERIMENT.md`: cuatro fits P, contraste principal y condiciones de integración.
3. `docs/03_ADAPTIVE_PROGRAM.md`: cómo continuar tras un resultado negativo sin buscar indefinidamente un número favorable.
4. `docs/04_BASELINES_EVALUATION.md`: evaluación interna, Garl, RGB, transferencia y test oficial.
5. `docs/05_EXECUTION.md`: recursos, persistencia, contabilidad y alcance de permisos.
6. `docs/06_FILE_MAP.md`: integración concreta con el repositorio.
7. `execution_policy.json` y `PROMPT_CODEX_START.md`.

La cola primaria permite **48.000 updates científicos** (4×12.000). El programa completo se limita a **1.000.000 updates físicos NUEVOS y 64.800 s en scopes GPU propios** al enviar el prompt de autorización. No son saldo de V17 ni una obligación de consumo. El alcance adaptativo está limitado a cuatro experimentos emparejados adicionales dentro de las familias descritas, una réplica completa y un refit final. No es permiso para experimentar con cualquier cosa.

## Qué está probado aquí

La revisión verificó el ZIP V17 contra la release, sus CRC y 5.211 hashes. Recalculó 44 tablas/1.360.146 filas, incluidos controles repetidos; los resultados no son bit a bit idénticos en este runtime (ver recibos). Cargó y ejecutó los 23 exports sobre sus ejemplos preparados. No entrenó ni reconstruyó los eventos de P/H/V.

La referencia `reference/relative_measurement.py` recibe un encoder PyTorch inyectado. Las pruebas incluyen el encoder CNN nativo del snapshot, tensores sintéticos, pérdidas/gradientes y contratos matemáticos. Su arquitectura inicial y constantes son propuestas prospectivas; no se han validado en GPU/eAP. El dataset, los runners, el acceso causal y el sistema de resume reales deben integrarse localmente. No presentar estos tests como evidencia científica.

## Inicio

Desde un worktree existente, usar `scripts/Prepare-Worktree.ps1` con una ruta de destino nueva. No hacer reset, clean o merge para preparar la campaña. Iniciar una sesión local nueva de Codex en el destino y pegar `PROMPT_CODEX_START.md`.

Si V18 existe o está activo, usar su estado local y `PROMPT_CODEX_RESUME.md`; no crear un segundo escritor. No ejecutar el runner V17 para intentar consumir su saldo: su cola terminó.
