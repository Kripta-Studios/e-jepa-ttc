# Registro de ejecución

Comandos representativos ejecutados realmente en esta sesión, bajo V14 salvo
las lecturas iniciales de V12/ZIP. Los recibos JSON/CSV son la evidencia de resultado.

- Get-FileHash del ZIP en Downloads; inspección de miembros y lectura íntegra
  de readmenow.txt, README, execution_policy.json y PROMPT_CODEX_START.md.
- Preparación del worktree separado desde da068eb y git apply --check seguido
  de git apply del patch sólo en V14.
- scripts/verify_package.py y scripts/analyze_public.py del handoff.
- python -B -m operational.range_transfer.run --help.
- python -B -m operational.range_transfer.run con --diagnostics-only --self-test:
  auditoría real H, diagnósticos públicos y 20 updates sintéticos contabilizados.
- pytest tests/test_range_transfer.py tests/test_range_transfer_integration.py:
  pruebas de contratos, paridad, separación de roles, presupuesto y checkpoint.
- ruff check/format sobre los archivos nuevos; pyright --pythonpath del entorno
  existente, con resultados JSON en qa.
- scripts/range_transfer_v12_trace.py: primer intento rechazado por diferencia
  de fuentes V12/V13; se aisló el import de V12. Intento siguiente rechazado por
  import dependiente del nuevo módulo; corregido. Otra comprobación identificó
  el wrapper decorado de Torch; se verifica inspect.unwrap de la función real.
  El intento admitido por identidad quedó PAUSED_RESOURCE antes de inferencia.
  Ninguno de estos intentos ejecutó updates de optimizador.
- Reentrada real mediante pwsh -NoProfile -File scripts/Resume-RangeTransfer.ps1:
  salida 3 por el gate de memoria comprometida; ningún fit científico iniciado.
- Reintento aislado de la traza histórica: PAUSED_RESOURCE, cero consultas crudas,
  cero updates. Se conserva el recibo y se pausa esta rama.
- Resultado final de la suite nueva: 41 tests y 17 subtests aprobados.
  Tras separar la contabilidad en un módulo ligero: dos tests seleccionados
  adicionales aprobados y recomputación real del ledger sin cambio (20 técnicos).
- Ruff aprobado; Pyright: 18 archivos, cero errores y advertencias del analizador.
  Se inspeccionaron visualmente las tres figuras generadas desde CSV.

El runner ejecuta como máximo 32 trazas crudas diagnósticas, conservando fragmentos
por consulta y el recurso observado. Sus fallos no borran resultados ni impiden
las tareas independientes. La reentrada se registra en qa/RESUME_DELIVERY.log.

No se ejecutaron push, merge, submissions, emails, cambios de configuración
global, nuevas semillas científicas, ni modificaciones a procesos V13/R1.

## Continuación con padres finales — 11 de octubre

- Relectura del estado V13, finales 30330/30330/6840 y caché H; admisión
  registrada en PREFLIGHT_ADMISSION.json y ADMISSION.json.
- Auditoría H final: 18422 filas, dominio íntegro; paridad de modelo, objetivo,
  decodificador, esquema y normalizador. Ninguna escritura en V13.
- trace_h y gradient_audit sobre H final, sin optimizer.step ni acceso a V.
- Ruff, Pyright y 41 tests (17 subtests) antes de los fits; resultados en qa.
- `pwsh -NoProfile -File scripts/Resume-RangeTransfer.ps1`: factorial real
  seed7 con endpoints 2500, captura en qa/FULL_CAMPAIGN_20261011.log.
- `qa/TRACE_MATCHED_H_RAW.py`: evidencia conservada del rechazo de forma known.
- `qa/TRACE_MATCHED_H_PRODUCERS.py`: 118 observaciones A5/C2F de 32 controles H;
  PAIR no se infiere cuando el contrato nativo falla. Cero updates.
- `qa/SUMMARIZE_H_CONTROLS.py`: atribución descriptiva por rango de esos controles;
  no se extrapola a toda H ni se emplea para escoger una receta.
- `qa/TEST_MATCHED_PAIR_ADAPTER.py`: PASS, forma real known por consulta,
  igualdad exacta de las 133 features y rechazo de reloj/intervalos inválidos.
- `qa/TRACE_MATCHED_H_ADAPTED.py`: completadas 118 observaciones A5/C2F/PAIR
  con el adaptador diagnóstico separado, 32 controles H, cero updates.
  Conserva el error original; no modifica V13 ni el código/caché del factorial.
- `qa/VERIFY_SCIENTIFIC_DELIVERY.py`: PASS para seis endpoints finitos de
  2500 updates, hashes de código/checkpoints/predicciones, curvas completas,
  cohortes alineadas y sampler idéntico. Genera pérdidas por consulta y
  subrangos/grupos; verifica el manifiesto SHA256 de cada entrada del ZIP.
- Cola cerrada: NO_CANDIDATE. 15000 científicos + 20 técnicos, 0 recuperación;
  ninguna réplica ni transferencia nueva. Todos los gates conservados.
- Inspección visual de curvas científicas y mapa de gradientes: etiquetas,
  unidades y leyendas legibles; los objetivos H no se confunden con scores V.
