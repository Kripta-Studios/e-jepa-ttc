# eAP: reproducción del contexto y cabeza originales

El replay `eap_original_context_replay_v1` reproduce las 5.805 consultas físicas
de Test12 desde sus diagnósticos de despliegue sellados, sin targets, productores
de sensores ni intervención canónica. Ejecuta E_CTX_MATCHED y R_CTX sobre sus
cuatro entradas guardadas, reconstruye los dos bloques de 170 coordenadas y
los logits de anchura RGB, y ejecuta la cabeza PHASE_ONLY sobre esos bloques
recalculados. No se limita a reutilizar el estado de cabeza guardado.

Resultado: **PASS, cero consultas fuera de tolerancia**, en 117,910 s de replay.
Se compararon los 22 campos guardados de contexto/entrada de cabeza/salida en
cada consulta. Las selecciones de experto coinciden exactamente en ambos
contextos. Es paridad numérica del original; no identidad bit a bit ni una
prueba de conservación del candidato.

| Campo | Máxima diferencia absoluta |
|---|---:|
| E_CTX raw_location | 7,6834e-9 |
| R_CTX raw_location | 1,4901e-8 |
| Estado de cabeza de 340 coordenadas | 8,9407e-7 |
| Logits de anchura RGB | 7,1526e-7 |
| TTC final | 0,0001049042 s |

El código se publicó antes de ejecutar (`95bb18b6`). FREEZE fijó CPU, lotes de
tamaño uno, un hilo, algoritmos deterministas, rtol=1e-5/atol=1e-6 para campos
flotantes y rtol=1e-4/atol=1e-4 para TTC; los campos categóricos exigen igualdad
exacta. No se ajustaron esas tolerancias después de ver resultados. La
comparación de TTC combina tolerancia absoluta y relativa, por eso el máximo
absoluto ligeramente superior a 1e-4 s no contradice el PASS.

La diferencia numérica de plataforma puede ser del mismo orden que pequeñas
correcciones PAIR. La futura medición debe comparar original y canónico en la
misma plataforma y con los mismos lotes, además de comprobar el enlace con los
diagnósticos históricos. No atribuir la deriva CPU frente al archivo original
a la intervención. El router adaptado no se ejecutó en esta comprobación.

Recibos pequeños:
[resultado](evidence/eap_original_context_complete.json) y
[protocolo congelado](evidence/eap_original_context_freeze.json).
Se comprobaron los hashes de FREEZE/RESULTS y las 5.805 identidades del resultado.
Los diagnósticos, resultados por consulta y pesos permanecen fuera de Git.

Ruff pasa. Pyright pasa apuntando explícitamente al entorno `.venv-transfer`;
la configuración del repositorio usa `.venv` y produjo errores de resolución
de atributos de PyTorch. La configuración local usada está en
`output/phase_timewarp/eap_context_replay_validation_v1/pyrightconfig.json`.
No se modificó la configuración compartida ni se suprimieron errores de tipos.

```powershell
$env:PYTHONPATH='src;.'
.venv-transfer/Scripts/python.exe scripts/replay_phase_eap_contexts.py --inputs artifacts/runpod_event_authority_supplement_20261026/payload/workspace/e-jepa-ttc-v16-multimodal-validation/artifacts/multimodal_research_20261012/test12_phase_only/INPUTS.json --baseline-audit output/phase_safe_fusion/test12_audit --upstream output/phase_submission/FINAL_MODEL/upstream --output output/phase_timewarp/eap_original_context_replay_new
```
