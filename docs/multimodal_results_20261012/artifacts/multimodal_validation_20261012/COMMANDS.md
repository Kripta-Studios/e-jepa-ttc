# Comandos y recuperación

Ejecutados desde el worktree V16 sobre `62cf4dbe914ea077c68e42983f869cd5422ab71c`. Usar el entorno local existente; no instalar ni actualizar dependencias globales. El bundle contiene código de la extensión y evidencia, no un clon completo ni sensores/pesos. Necesita los worktrees y paths registrados en `DEPENDENCIES.json` y `STATE_BINDING.json`.

```powershell
$py = '../e-jepa-ttc/.venv/Scripts/python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = 'src;.;E:/EJEPA_results/fcwd_rgb_calibration_20261012/deps'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$out = 'artifacts/multimodal_validation_20261012'

& $py -B docs/multimodal_next_handoff_20261012/scripts/verify_package.py
& $py -B -m unittest discover -s docs/multimodal_next_handoff_20261012/tests -v
& $py -B docs/multimodal_next_handoff_20261012/scripts/analyze_public.py --output "$out/public_recalculation"
& $py -B docs/multimodal_next_handoff_20261012/scripts/paired_bootstrap.py --output "$out/public_bootstrap"
& $py -B -m operational.multimodal_validation.consolidate

# Primera ejecución Q1: fuente original ahora archivada en executed_sources/operational/...
# python -B -m operational.multimodal_validation.garl --kind visual
# python -B -m operational.multimodal_validation.garl --kind full
# Recuperación de esos mismos fragmentos: verifica y usa la fuente exacta del sello.
& $py -B -m operational.multimodal_validation.resume_garl --kind visual
& $py -B -m operational.multimodal_validation.resume_garl --kind full
& $py -B -m operational.multimodal_validation.score

& $py -B -m operational.multimodal_validation.traces --scope h
& $py -B -m operational.multimodal_validation.traces --scope external
& $py -B -m operational.multimodal_validation.h_raw
& $py -B -m operational.multimodal_validation.geometry
& $py -B -m operational.multimodal_validation.head_cost
& $py -B -m operational.multimodal_validation.traces --scope profile
& $py -B -m operational.multimodal_validation.garl_cost

& $py -B -m pytest tests/test_multimodal_validation.py -q
& ../e-jepa-ttc/.venv/Scripts/ruff.exe check operational/multimodal_validation tests/test_multimodal_validation.py
& ../e-jepa-ttc/.venv/Scripts/ruff.exe format --check operational/multimodal_validation tests/test_multimodal_validation.py
& ../e-jepa-ttc/.venv/Scripts/pyright.exe -p operational/multimodal_validation/pyrightconfig.json
& $py -B -m operational.multimodal_validation.delivery
```

No ejecutar toda la lista automáticamente para inspeccionar resultados ya completos. Las inferencias Garl y el piloto compartido tienen fragmentos/checkpoints de progreso; `h_raw` omite controles terminados. Las trazas `h`/`external` añaden llamadas si se repiten; para una réplica independiente usar otra salida revisada, sin mezclar registros. `geometry`, `garl_cost`, scoring y delivery regeneran sus propios archivos. La identidad histórica Q1 queda protegida: el entrypoint formateado `garl` rechaza un binding diferente, mientras `resume_garl` conserva la fuente archivada exacta.

El replay público se ejecutó con `docs/agent_handoff_20261012/replay.py`; su recibo está en `public_replay/REPLAY_RESULT.json` (29 fuentes, 9.471 comprobaciones). Su CLI documentada permanece en el archivo de la base. Se descargaron únicamente el checkpoint visual faltante, su configuración y README, mediante el CLI HF existente, revisión `b676fcdaf26c04bcf896cdb2b208c9c424e8462a`; el resto de pesos se reutilizaron localmente y comprobaron por hash.

Los logs conservan los intentos iniciales corregidos: OpenCV se encontró en el directorio aislado existente; el freeze CRLF/LF se resolvió conservando la validación original y verificando equivalencia de fuentes; los PNG en rutas Unicode usan `imencode` y escritura de bytes. Un proceso H raw propio fue detenido por RSS agregada y reanudado tras Garl, sin intervención sobre V15. `executed_sources/pre_typecheck/` conserva las fuentes anteriores a ajustes de tipos; éstos no modifican pesos ni recetas.

La verificación del ZIP final comprueba CRC y SHA-256 de cada miembro contra `BUNDLE_MANIFEST.json`. `BUNDLE_VERIFICATION.json` queda junto al ZIP como recibo externo con su SHA-256; se excluye del propio ZIP para evitar una dependencia circular. No se incluyen pesos, sensores crudos, credenciales ni datos privados test12.
