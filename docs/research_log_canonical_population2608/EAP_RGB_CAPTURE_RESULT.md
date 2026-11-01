# Resultado: la reproducción histórica RGB no supera el piloto

El intento corregido `eap_rgb_capture_pilot_v2` terminó con exit 0, 24 de 24
observaciones capturadas y **24 fallos de paridad histórica**. El estado COMPLETE
significa que terminó la comprobación; `parity_status=FAIL` impide ampliar esta
ruta a las 5805 observaciones. No se ejecutó PAIR canónico en eAP.

El intento v1 sigue conservado: falló antes de capturar observaciones por omitir
el adaptador ABI histórico, como explica el [protocolo](EAP_RGB_CAPTURE_PROTOCOL.md).
La corrección de ese error del runner permitió medir el siguiente fallo; no se
relajaron las tolerancias ni se eligieron nuevas consultas.

El [informe regenerado](evidence/eap_rgb_capture_pilot_v2_report.json) verifica
los hashes de FREEZE, RESULTS, los 24 NPZ, el inventario y los archivos de
diagnósticos. Recalcula cada veredicto contra el raw17/expert3 histórico y exige
que coincida con el guardado. Conserva máximos por columna, sin promediar fallos.

- Tiempo instrumentado de captura/verificación: 51,111 s; no es latencia de servicio.
- Diferencia absoluta máxima en raw17: 0,0053281784.
- Diferencia absoluta máxima en expert3: 0,0000110175461.
- Las dos primeras columnas de raw17 coinciden exactamente en los 24 casos;
  ya existen diferencias en los diagnósticos de productores anteriores a PAIR.

Estas magnitudes pertenecen a características/ fases, **no son segundos de TTC**.
El resultado localiza una discrepancia original anterior a la intervención;
no prueba su causa ni una regresión de precisión del candidato. Los pesos y
medios están vinculados a la misma genealogía, pero no se ha reproducido la
salida histórica del productor dentro de las tolerancias fijadas.

## Continuidad sin repetir experimentos

El trabajo concurrente contiene un diagnóstico original de las cuatro políticas
TF32 en `output/release_research_20261031/native_backend_diagnosis_v1`. Se
verificaron en lectura sus 52 artefactos contra COMPLETE: repetibilidad local
exacta y ninguna política alcanza paridad histórica en sus dos consultas.
No lo ejecutó este runner ni deben confundirse esas dos consultas con este
piloto de 24. Recibo de la verificación de lectura:
`output/phase_timewarp/eap_rgb_capture_validation_v1/RELATED_BACKEND_CHECK.json`.
No repetir esas cuatro políticas como si fueran una hipótesis aún no probada.

También existe un protocolo concurrente de sensibilidad emparejada en la
plataforma local. Antes de iniciar otra captura, comprobar su estado y respetar
sus archivos. Ese diagnóstico responde a una pregunta distinta y no revoca
este fallo de admisión histórica. Una comparación local no permite afirmar
conservación de precisión Test12 sin targets. El entregable no se modifica.

## Reproducción del informe

```powershell
$env:PYTHONPATH='src;.'
.venv-transfer/Scripts/python.exe scripts/report_phase_eap_rgb_capture.py --capture output/phase_timewarp/eap_rgb_capture_pilot_v2 --inputs artifacts/runpod_event_authority_supplement_20261026/payload/workspace/e-jepa-ttc-v16-multimodal-validation/artifacts/multimodal_research_20261012/test12_phase_only/INPUTS.json --baseline-audit output/phase_safe_fusion/test12_audit --output RUTA_NUEVA.json
```

Código del piloto ejecutado: `f457f764`; fuentes/modelos en FREEZE. Se publican
los recibos compactos; los NPZ y medios quedan locales. Once pruebas del runner
pasan; Ruff check/format y Pyright pasan en el runner, sus pruebas y el generador
del informe. Los fallos de paridad del experimento permanecen visibles aunque
las comprobaciones de software pasen.
