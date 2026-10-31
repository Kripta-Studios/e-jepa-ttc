# eAP: disponibilidad de la ruta de despliegue Test12

Esta comprobación prepara la validación pendiente; no ejecuta un candidato ni
usa targets. El inventario público mantiene su hash admitido
`b78730abe68de7fbe06c6c98b25d5fa025c04d9303720b680ce0c0f575d2e526`.
Sus 6.762 consultas lógicas corresponden a 5.805 consultas físicas mediante
los aliases congelados. Las historias referencian exactamente las observaciones
del inventario.

Se localizaron las 5.805 entradas de eventos en
`E:/EJEPA_GARLTTC_TEST12_20261010/inputs`. Los 36 shards RGB de
`E:/eAP_dataset/data/test` están presentes y tienen los tamaños esperados.
En el preflight solo se comprobó disponibilidad/tamaño. La verificación posterior
`eap_media_verified_v1` completó los hashes de los 5.841 archivos: todos coinciden
con el inventario. Se leyeron 55.130.168.689 bytes en 415,994 s, sin decodificar
arrays ni ejecutar productores. No hicieron falta descargas.

Recibo compacto: [eap_media_complete.json](evidence/eap_media_complete.json).
Se verificaron además los hashes de FREEZE/FILES y la correspondencia ordenada de
las 5.841 identidades del recibo. Los manifiestos completos y medios permanecen
locales en `output/phase_timewarp/eap_media_verified_v1` y sus rutas originales.
Esto permite preparar el replay; no demuestra conservación de las predicciones.

Los hashes de los 12 archivos de diagnósticos de despliegue coinciden con el
recibo de `output/phase_safe_fusion/test12_audit`. Ese recibo pertenece al mismo
`parent` del bundle entregado. Los diagnósticos contienen raw17/expert3 de
observaciones nuevas, entradas/salidas de contexto y cabeza; no contienen PAIR133.
Por tanto, permiten comprobar el original, pero no bastan por sí solos para
recalcular la intervención RGB.

| Comprobación temporal | Eventos | RGB |
|---|---:|---:|
| Observaciones físicas | 5.805 | 5.805 |
| Delta exactamente float32(0.1) | 5.805 | 0 |
| Delta fuera de identidad | 0 | 5.805 |
| Longitudes de historia observadas | 1–4 | 1–5 |

La ruta de despliegue `FinalV16Train40Predictor` pasa el delta del inventario a
eventos y usa `deltas_seconds(producer_sensor_times_us)` para RGB. Los intervalos
RGB finales son 100.001 o 100.002 µs. El hecho de que estén muy cerca de 100 ms
no prueba ausencia de cambios después de contexto/cabeza.

Si la población EvTTC completa supera sus criterios fijados, la comprobación
de conservación puede usar estas entradas y su genealogía de despliegue,
reproduciendo primero el original y capturando los PAIR133 que faltan. Debe
preservar lotes, normalizadores, historias y aliases, y separar diferencias
numéricas de plataforma de los cambios producidos por la intervención. El router
sigue siendo diagnóstico tras fallar el screen previo; esta disponibilidad no
lo promueve. Sin targets Test12, medir deriva tampoco demuestra precisión.

Evidencia: [eap_deployment_preflight.json](evidence/eap_deployment_preflight.json).
Salida local: `output/phase_timewarp/eap_deployment_preflight_v1/REPORT.json`.

```powershell
$env:PYTHONPATH='src;.'
.venv-transfer/Scripts/python.exe scripts/preflight_phase_eap_deployment.py --inputs artifacts/runpod_event_authority_supplement_20261026/payload/workspace/e-jepa-ttc-v16-multimodal-validation/artifacts/multimodal_research_20261012/test12_phase_only/INPUTS.json --baseline-audit output/phase_safe_fusion/test12_audit --upstream output/phase_submission/FINAL_MODEL/upstream --media-root E:/eAP_dataset --output output/phase_timewarp/eap_deployment_preflight_new/REPORT.json
.venv-transfer/Scripts/python.exe scripts/verify_phase_eap_media.py --inputs artifacts/runpod_event_authority_supplement_20261026/payload/workspace/e-jepa-ttc-v16-multimodal-validation/artifacts/multimodal_research_20261012/test12_phase_only/INPUTS.json --media-root E:/eAP_dataset --output output/phase_timewarp/eap_media_verified_new
```
