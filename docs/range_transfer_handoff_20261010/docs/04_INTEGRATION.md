# 04 — Integración limitada, archivo por archivo

La aplicación del patch añade módulos y tests, pero NO implementa un nuevo trainer ni modifica los actuales. Debe realizarse en V14, nunca en el worktree V13 activo.

## Archivos entregados

| Archivo | Estado |
|---|---|
| `src/e_jepa_ttc/range_transfer/mass.py` | Implementado y probado: M0/M1/M2, roles y conservación |
| `src/e_jepa_ttc/range_transfer/objective.py` | Implementado y probado: wrapper L0 exacto y término L1 |
| `tests/test_range_transfer.py` | Contratos CPU sin datasets ni optimizer |
| `scripts/analyze_public.py` | Análisis reproducido desde FCWD público |
| `scripts/run_contract_tests.py` | Importación aislada de fase/modelo nativos, sin imports pesados del repo |
| `patches/0001-range-transfer-reference.patch` | Altas de módulos/tests; no cambios al entrenamiento V13 |

## Archivos que Codex debe implementar en V14

| Ruta propuesta | Responsabilidad y reutilización |
|---|---|
| `operational/range_transfer/admission.py` | Resolver padres finales y H/V por manifests; validar roles, esquema, normalizador, receta y código |
| `operational/range_transfer/diagnose.py` | Trazas por etapa, masa/exposición/gradiente y rango; resultados H y transferencia claramente separados |
| `operational/range_transfer/cache.py` | Vista en sólo lectura de la caché H nativa; vectores de masas derivados en output propio, sin reescribir NPZ |
| `operational/range_transfer/train_heads.py` | Reutilizar matemática del modelo y checkpoint durable. IDs/seed nuevos, selección de masa/objetivo; no importar por monkeypatch el trainer activo |
| `operational/range_transfer/analyze.py` | Pares por ID, bandas y grupos, gates fijados, bootstrap por clúster, propagación de inválidos |
| `operational/range_transfer/run.py` | Cola pequeña con estados claros y continuación, max_updates_this_call y reentrada idempotente |
| `operational/range_transfer/report.py` | Manifest de evidencias, pesos nuevos, curvas, contabilidad, decisiones y export esencial |
| `tests/test_range_transfer_integration.py` | Continuidad, esquema, sampler, método nativo L0, acumulación y ausencia de lectura V/FCWD por fit |

No duplicar veinte controladores antiguos. Un runner y reutilización explícita de piezas estables basta. Rutas de funciones existentes: `operational/rgb_port/train_heads.py` (`SealedCache`, `_recipe`, `_lr`, `_loss`), `simplex_t/model.py` (`TemporalRefiner`, `training_loss`), `rgb_port/prepare_features.py` (esquemas H/V). La interfaz `_recipe/_lr` es interna: copiar fielmente a una función pequeña versionada o extraer en V14 sin tocar fuentes admitidas V13; probar paridad.

## Aplicación del código de referencia

Desde la raíz del nuevo worktree V14, después de verificar este handoff:

```powershell
git apply --check docs/range_transfer_handoff_20261010/patches/0001-range-transfer-reference.patch
git apply docs/range_transfer_handoff_20261010/patches/0001-range-transfer-reference.patch
```

No ejecutar esto en V13. Si los archivos ya existen por una continuación previa, comprobar sus hashes y conservarlos; no forzar una segunda aplicación. La aplicación no inicia entrenamiento.

## Admisiones técnicas obligatorias

- M0-L0 reproduce pérdida y gradiente nativos en el mismo batch; si se reutiliza un endpoint, toda la identidad y estado coinciden.
- M1/M2 no cambian sampler, IDs ni reloj; M2 conserva masa grupo×banda, negativo y escala global.
- Lectura de H no abre archivos de labels V/FCWD/Dev32. Mismo grupo no aparece en P/H/V fuera del contrato ya registrado.
- Resume de 10 updates continuos frente a5+restore+5 en una prueba sintética compara pesos, AdamW, scheduler, cursor/RNG y draws. Estos updates pertenecen a los500 técnicos; no hacer fits extra de eAP para comprobar el runner.
- Interrupción durante escritura no produce endpoint válido a medias; no relanzar fits completos.
- Uso de target TTC reconstruido y dominio negativo probado. Una salida inválida no puede mejorar una loss omitiendo su fila.

Los comandos nuevos de fit se publican sólo después de implementar el runner y comprobar su `--help`. El handoff no inventa un módulo ejecutable de entrenamiento que todavía no existe.
