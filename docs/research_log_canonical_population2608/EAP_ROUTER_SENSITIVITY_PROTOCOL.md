# Sensibilidad local del router entregable: protocolo previo

Esta extensión espera a que finalice el diagnóstico nativo emparejado concurrente.
No ejecuta productores ni modifica sus fuentes. Reutiliza sus estados originales
y canónicos completos para distinguir la deriva de F0 de la del router adaptado
congelado. No promueve el router, que falló el screen149, ni reclasifica el fallo
de paridad histórica de eAP. No sustituye una evaluación de precisión Test12.

Admisión: COMPLETE sin FAILED de captura, replay original y replay emparejado;
inventarios completos verificados por hash; genealogía entre las tres etapas;
5805 consultas físicas únicas que coinciden exactamente con los aliases del
inventario público confiable. Se exige repetición local exacta del original y
contexto de eventos exacto en el resultado nativo previo. No se lee un run parcial.

Por consulta, cargar los dos contextos sellados y convertir fase a TTC con el
mismo `native_vector` y dispositivo CUDA del screen/población. Su componente F0
debe coincidir exactamente con la predicción sellada. Reconstruir las cajas y
su disponibilidad causal con `query_history` y `startup_geometry`, sin usar
cachés OOF ni targets. Esa geometría y calidad son idénticas entre ambos brazos.
Ejecutar en CPU batch1 el mismo ensemble entregable de semillas 7/13/21, cargado
con sus checksums. No entrenar, seleccionar pesos ni ajustar thresholds. Si la
rama de historia no está activa, exigir su retorno exacto a F0.

Publicar conteos de cobertura, nuevas ausencias, signos, igualdad exacta, media,
cuantiles y máximo de cambio absoluto en segundos, global y por secuencia.
Expandir los 6762 aliases solo como vista de entrega; no contarlos como muestras
independientes ni hacer bootstrap de ventanas. Mantener diferencias históricas
y locales separadas. No se declara conservación de precisión porque falten
targets, aunque la deriva resultara pequeña. El modelo entregable queda intacto.

El runner `scripts/audit_phase_eap_router_sensitivity.py` requiere directorio
nuevo. No hay resultado de esta extensión al fijar el protocolo. Su ejecución
debe esperar a que las tres etapas padre terminen y pasen sus guardas.

Actualización operativa previa a ejecutarlo: la captura v1 fue interrumpida.
Consumir su recuperación v2 completa. Esperar también el fin del coordinador
de latencia concurrente para no introducir carga de este diagnóstico en sus
mediciones. No se modifica el control ni el criterio científico del protocolo.

Validación previa: ocho pruebas de cobertura, signos, ausencias, igualdad exacta
de F0 y rechazo de padres incompletos/alterados; Ruff check/format y Pyright
scoped pasan. Esto valida el runner, no constituye un resultado eAP.

```powershell
$env:PYTHONPATH='src;.'
$env:PYTHONUTF8='1'
.venv-transfer/Scripts/python.exe scripts/audit_phase_eap_router_sensitivity.py --capture output/release_research_20261031/native_platform_followup_v2/capture --original output/release_research_20261031/native_platform_followup_v2/original_replay --paired output/release_research_20261031/native_platform_followup_v2/paired_replay --inputs artifacts/runpod_event_authority_supplement_20261026/payload/workspace/e-jepa-ttc-v16-multimodal-validation/artifacts/multimodal_research_20261012/test12_phase_only/INPUTS.json --output output/phase_timewarp/eap_router_sensitivity_v1
```
