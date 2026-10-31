# eAP: precondición temporal de los históricos de eventos

La auditoría del índice TRAIN40 confirma que sus 88.744 filas, en 40 grupos,
usan exactamente `float32(0.1)`. Sus tokens, grupos y bits de `delta_t_s`
coinciden en orden con las consultas actuales de eventos ya auditadas. Se
verificaron el hash del tar y los recibos de esa auditoría antes de leer arrays.
No se abrieron `ttc_s`, `phase` ni otros targets, ni se ejecutaron productores.

La ruta `operational/final_refit/event_history.py` toma ese delta del índice
en `_query_plan` y lo repite en los 16 slots de `_fresh_pair_phases`. Para
entradas FP32 finitas con soportes válidos, `CanonicalPair` pasa el lote original
sin cambios a PAIR y conserva los bits de su salida cuando todos los deltas son
0.1. También conserva el logaritmo con epsilon del constructor histórico; no
reconstruye esas columnas en la rama de identidad.

Esto demuestra la precondición de identidad temporal para esa ruta indexada.
**No es una ejecución completa de historias, contextos y cabeza**, ni verifica
que las cachés OOF sean entradas del modelo de despliegue. Queda por comprobar
la genealogía efectiva del despliegue, validar las entradas históricas y auditar
RGB/contextos/cabeza con una genealogía consistente.

El reloj usado por esa ruta es nominal. Los intervalos físicos finales del
índice tienen mínimo/mediana/máximo de 99.945 / 100.001 / 100.058 µs; 88.720
filas no son exactamente 100.000 µs. Esta auditoría no sustituye el delta
nominal por el físico ni propone ese cambio como parte del experimento actual.

Evidencia pequeña: [eap_event_cadence.json](evidence/eap_event_cadence.json).
Recibo local: `output/phase_timewarp/eap_event_cadence_v1/REPORT.json`.
Los hashes identifican las fuentes inspeccionadas y los bytes originales.

```powershell
$env:PYTHONPATH='src;.'
.venv-transfer/Scripts/python.exe scripts/audit_phase_eap_event_cadence.py --archive-root D:/E_JEPA_POD2_RETURN_20261027 --current-audit output/phase_timewarp/eap_current_pair_audit --output output/phase_timewarp/eap_event_cadence_new/REPORT.json
```
