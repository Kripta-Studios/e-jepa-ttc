# Entorno, artefactos y reproducción sin empezar de cero

## Workspace y entorno

Raíz local actual:

```text
C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\EVOCON_JEPA_Codex_Handoff\e-jepa-ttc-v18-joint-measurement
```

Rama `research/evttc-transfer-v2`, remoto `Kripta-Studios/phase-ttc`. Usar el
worktree existente, revisar `git status` y no resetear cambios de otro trabajo.
Python está en `.venv-transfer/Scripts/python.exe`; PowerShell:

```powershell
$env:PYTHONPATH='src;.'
$env:PYTHONUTF8='1'
```

Hardware de las ejecuciones locales: Ryzen 9955HX, RTX 5070 Ti Laptop de 12 GB,
32 GB RAM; PyTorch 2.11.0+cu128. Las rutas de datos en D:/ y E:/ se conservan
como ubicaciones de artefactos existentes, no se deben sustituir por descargas
masivas ni por datos sintéticos que aparenten el mismo experimento.

## Modelo entregable y población

| Ruta local | Contenido y uso |
|---|---|
| `output/phase_submission/FINAL_MODEL` | Entrega actual, no sustituida por PAIR canónico |
| `output/phase_submission/FINAL_MODEL/upstream` | Nueve componentes y normalizadores sellados |
| `output/PHASE_TTC_ADAPT_FINAL_2026-10-31.zip` | Paquete entregable anterior |
| `output/phase_adapt_final/data_v3/training.npz` | Datos/estados de desarrollo; contiene targets, mantener separado de inferencia |
| `output/phase_safe_fusion/evaluator_audit` | Targets públicos corregidos y auditoría del scorer |
| `output/phase_safe_fusion/source_audit` | TRAIN40 activo con features OOF; no confundir con el nuevo PAIR133 |
| `output/phase_safe_fusion/test12_audit` | Estados de despliegue Test12 sin targets |

Release existente: `https://github.com/Kripta-Studios/phase-ttc/releases/tag/phase-ttc-research-20261031`.
No se ha actualizado con el candidato canónico. Git contiene código, informes y
recibos pequeños, **no** raw data, todos los checkpoints ni los NPZ de inferencia.
Un agente en esta máquina puede reutilizarlos. Un clon en otra máquina necesita
además transferir los artefactos locales con sus sellos; clonar Git no basta.

## Resultados y cachés que deben reutilizarse

| Ruta | Estado / objeto |
|---|---|
| `output/phase_safe_fusion/campaign_v1b` | 96 ajustes completos |
| `output/phase_safe_fusion/campaign_v2` | 48 ajustes completos |
| `output/phase_safe_fusion/correlated_v1/campaign` | 24 ajustes completos |
| `output/phase_safe_fusion/native_context_v1` | 48 ajustes de contexto; comprobar ruta por recibos si se reubica |
| `output/phase_native_support/campaign_v1` | 48 ajustes F2/Fcap |
| `output/phase_native_support/joint_v1` | 48 ajustes F2n/Fcapn; selección nula |
| `output/phase_observation_cache/pilot/completed` | Piloto original limpio y consultas de referencia |
| `output/phase_observation_cache/expanded149` | 149 consultas, 447 variantes, sellos y referencias nativas |
| `output/phase_observation_cache/expanded149_sourceclosure` | Cierre suplementario de dependencias históricas |
| `output/phase_native_support/cache_v1` | Features de soporte para desarrollo, reinicios, TRAIN40 y Test12 |
| `output/phase_event_expansion/objective_profile/all77_hybrid_v1` | Ruta híbrida CUDA/CPU de expansión, paridad de 77 consultas |
| `output/phase_timewarp/pilot_v1` | Primer intento fallido; conservar íntegro |
| `output/phase_timewarp/pilot_v2` | Piloto temporal completo; sensores capturados para dos consultas |
| `output/phase_timewarp/canonical_v1` | Intervención canónica completa, sin labels ni actualizaciones |
| `output/phase_timewarp/eap_current_pair_audit` | Inputs PAIR133 y primeras auditorías conservadas |
| `output/phase_timewarp/eap_current_pair_audit/verified_run_v2` | Auditoría CPU definitiva vinculada al CLI publicado |

Los informes de cada campaña enumeran sus rutas y fuentes. Verificar los
recibos antes de consumir datos; no repetir un entrenamiento porque un documento
viejo aún diga «pendiente».

## Sellos recientes

| Artefacto | SHA-256 |
|---|---|
| `phase_timewarp/pilot_v2/COMPLETE.json` | `c39af791d18220a6251ea9af77565fd28106795f8ddb134f605e45c6f74e6ef3` |
| `phase_timewarp/pilot_v2/FREEZE.json` | `55d9e847dffad3f1a0865b8f73591e4156d5505e772c345f5d20ca901770532e` |
| `eap_current_pair_audit/INPUTS_COMPLETE.json` | `6b798917b1e70657041adefbd6d38864ac968d99e260ec6b4bddd42749026123` |
| `eap_current_pair_audit/verified_run_v2/COMPLETE.json` | `75b485aa913bdad1510c8ae073f2711513251c9d20216a51f5bbfd12aee5afec` |
| `eap_current_pair_audit/verified_run_v2/FREEZE.json` | `7de46c9157480c2af535081fc0af3471e6cfbf8b1f544df285e769031f772af4` |

El recibo de `canonical_v1` vincula su `FREEZE.json` y 116 artefactos. Los
verificadores comparan los bytes originales; las copias publicadas de JSON
normalizan CRLF a LF. No confundir el hash del archivo ejecutado con el hash de
su copia normalizada en Git, ni modificar archivos sellados para hacerlos pasar.

## Comandos de lectura/regeneración

Estos comandos regeneran documentación, no entrenan ni abren nuevas campañas:

```powershell
.venv-transfer/Scripts/python.exe scripts/report_phase_support_context_joint.py
.venv-transfer/Scripts/python.exe scripts/report_phase_producer_timewarp.py --pilot output/phase_timewarp/pilot_v2 --failed-attempt output/phase_timewarp/pilot_v1
.venv-transfer/Scripts/python.exe scripts/report_phase_canonical_pair.py
```

No hace falta repetir los pilotos para continuar. Si hay una razón concreta para
reproducirlos, sus scripts exigen salidas nuevas:

```powershell
.venv-transfer/Scripts/python.exe scripts/pilot_phase_producer_timewarp.py --output output/phase_timewarp/pilot_reproduction_new
.venv-transfer/Scripts/python.exe scripts/pilot_phase_canonical_pair.py --output output/phase_timewarp/canonical_reproduction_new
.venv-transfer/Scripts/python.exe scripts/audit_phase_canonical_pair_source.py --inputs output/phase_timewarp/eap_current_pair_audit --output output/phase_timewarp/eap_current_pair_audit/reproduction_new --upstream output/phase_submission/FINAL_MODEL/upstream --device cpu
```

Son ejemplos para una reanudación autorizada, **no trabajos pendientes que deban
lanzarse durante la pausa**. No ejecutar comandos sobre carpetas completadas.

## Verificación de cierre

La comprobación conjunta de los cinco módulos de pruebas nuevos dio **40 tests
correctos**. Ruff pasó y Pyright dio cero errores en los módulos/scripts nuevos.
No se presenta esto como una repetición del test suite completo del repositorio.

```powershell
.venv-transfer/Scripts/python.exe -m pytest tests/unit/test_phase_timewarp.py tests/unit/test_phase_timewarp_capture_pilot.py tests/unit/test_phase_canonical_pair.py tests/unit/test_phase_canonical_pilot.py tests/unit/test_phase_canonical_pair_source_audit.py -q
.venv-transfer/Scripts/python.exe -m pyright --project output/phase_event_expansion/objective_profile/pyright_transfer.json operational/phase_timewarp scripts/pilot_phase_producer_timewarp.py scripts/report_phase_producer_timewarp.py scripts/pilot_phase_canonical_pair.py scripts/report_phase_canonical_pair.py scripts/audit_phase_canonical_pair_source.py
```

El pyright raíz apunta a otro entorno. La configuración local citada selecciona
`.venv-transfer`; los falsos errores de resolución de Torch con el entorno
equivocado no se deben corregir cambiando las APIs del modelo.

## Límites de las cachés para el siguiente experimento

En `expanded149` hay 1.416 observaciones aceptadas: 578 de eventos y 838 RGB.
Capturar PAIR133 requiere unos 753.312 bytes FP32 adicionales más identidades;
raw17 no permite reconstruir los 128 tokens. El coste está en volver a ejecutar
la preparación/productores, no en guardar esos features. La estimación 8–10 min
se basa en la extracción anterior; **la extracción nueva no se ha lanzado**.

Los inputs actuales eAP se extrajeron de dos archivos locales sellados en
`D:/E_JEPA_POD2_RETURN_20261027/`: `final_inputs_63e2720817ba6521.tar` y
`final_generated_fd84565b0df06d79.tar`. `INPUTS_COMPLETE.json` identifica miembros,
hashes, padres y campos leídos. Para reconstruir productores también existen
shards en E:/; consultar `docs/research_log_20261031_joint_support/producer_next_steps.md`.
No asumir que esos inputs actuales cubren las ROIs condicionadas por cada
consulta histórica.
