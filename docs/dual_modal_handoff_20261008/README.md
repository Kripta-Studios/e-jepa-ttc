# E-JEPA-TTC V13 — dos modalidades, evidencia espacial y comparación justa

Preparado el 8 de octubre de 2026. Es un **encargo nuevo**, no resultados nuevos.
Base revisada: `b46d40bf1f38ae3e3ea201d58b79391fa475ca3c`, rama `scientific-recovery-v12-efficient-context`.

## Decisión

Conservar TRAIN40/H8. No reentrenar el sistema anterior ni reiniciar E3 Garl completo.
Implementar una ruta RGB real y una familia nueva con encoder entrenable, evaluación
por adquisición y dos salidas desplegables: eventos solos y RGB+eventos.

La cola tiene dos niveles: B (puente práctico sobre H8, exploratorio) y D (estudio
arquitectónico con productores que excluyen los grupos de desarrollo). El puente
no bloquea D si da negativo. R1 tampoco es un gate de precisión.

## Orden de lectura

1. `SOURCE_PIN.json`, `execution_policy.json` y `docs/01_REVIEW.md`.
2. `docs/02_DATA_AND_EVALUATION.md`: no convertir ocho secuencias TRAIN40 en
   holdout de unos pesos ya entrenados con ellas.
3. `docs/03_ARCHITECTURE.md`, `docs/04_EXPERIMENTS.md` y `docs/05_ENGINEERING.md`.
4. `docs/06_FILE_BY_FILE.md`, `docs/07_SOTA_AND_CLAIMS.md`.
5. Ejecutar `PROMPT_CODEX_START.md`; usar `PROMPT_CODEX_RESUME.md` tras interrupciones.

## Lo entregado

Documentación, presupuestos, comandos, auditor de predicciones archivadas y código
PyTorch de referencia: encoder ResNet50/FPN, correlaciones locales preservadas,
fusión tardía con ausencia RGB explícita y emisión de fase/TTC. Las pruebas son
sintéticas CPU. **No hay entrenamiento eAP/EvTTC, pesos preentrenados descargados,
perfil CUDA ni adaptador de datos integrado ejecutado en este paquete.**

La implementación de referencia no reemplaza el repositorio. Codex debe integrar
sus contratos con el loader, pérdidas, normalización y recuperación reales y demostrar
paridad donde se reutilizan componentes históricos. No copiar el viejo multimodal.py
como si fuese compatible con H8: sus interfaces son las de ObjectCentricEventJEPA.

## Arranque

Usar una sesión nueva y un worktree nuevo. El preparador no hace reset, clean,
merge ni push; el código científico anterior queda intacto. Necesita la ruta real al
repo local y reutiliza datos/pesos mediante bindings de sólo lectura.

```powershell
pwsh -NoProfile -File .\scripts\Prepare-Worktree.ps1 -SourceRepo <repo-v12-local> -Destination <directorio-v13-nuevo> -AllowFetch
codex --cd <directorio-v13-nuevo>
```

La existencia del paquete no ejecuta ni autoriza por sí sola entrenamiento: el usuario
activa su nueva autorización al enviar el prompt. Una vez activada, Codex debe ejecutar
las ramas viables, no detenerse tras escribir documentos.
