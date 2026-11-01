# Recaptura original RGB de despliegue eAP

Protocolo fijado después del gate EvTTC de 2608 y antes de ejecutar esta recaptura.
El candidato canónico todavía no se evalúa. El objetivo es recuperar PAIR133 de
la genealogía de despliegue Test12 y comprobar que los productores originales
reproducen los raw17/expert3 archivados.

El piloto selecciona la primera y última consulta física de cada una de las
12 secuencias, ordenadas por tiempo y token; toma su observación RGB actual:
24 identidades fijadas sin leer predicciones ni targets. No elige casos por error.
Se usa CUDA FP32, batch1, dos hilos CPU, semilla 7, algoritmos deterministas y
TF32 desactivado. Se conservan T2/T3, ROI, normalizadores y deltas del inventario.
La referencia tiene el mismo modelo sellado y sus diagnósticos se autentican
contra hashes de archivos, admissions y NPZ. Los shards usados se verifican
contra el inventario confiable antes de decodificarlos.

Paridad de raw17 y expert3: mismas formas/dtypes, todos finitos, rtol=1e-5 y
atol=1e-6. Se guarda también igualdad exacta y máximo cambio absoluto. Un fallo
se conserva y bloquea la expansión; no se aflojan tolerancias para admitirlo.
El piloto no es una evaluación de precisión ni prueba de conservación completa.

Si pasa, la misma fuente ejecuta `--mode full --pilot RUTA_PILOTO` para capturar
las 5805 observaciones RGB de todas las historias, conservando todos los fallos.
Solo después de reproducir el original completo se podrá reconstruir PAIR
canónico, raw17, historias, contextos y cabeza frente a un original emparejado
en la misma plataforma. Las observaciones de eventos tienen delta identidad;
su reutilización exige mantener genealogía y demostrar identidad de la ruta.
El replay anterior de contextos CPU no sustituye este control de productores.

El intento `eap_rgb_capture_pilot_v1` terminó con exit 1 y cero observaciones:
el runner omitía el adaptador `NativeObservation` usado por el despliegue y
falló la validación del known mask agregado frente al soporte temporal.
Se conserva su FAILED.json en la evidencia. El admission histórico registra
explícitamente este adaptador; el archivo local coincide byte a byte con el
miembro correspondiente de `frozen_code_91972eaa97dcf9b0.tar`, cuyo SHA-256 se
verificó. SHA del adaptador:
`af10f369e5c2f4896100808dcd21abd290bbf1c8168ba5a75a00fb11ea271e2c`.

El runner corregido exige ese contrato y hash en cada admission y aplica el
mismo adaptador solo durante el expert block. No modifica máscaras, pesos ni
aritmética de PAIR. Dos pruebas nuevas comprueban el rechazo de un contrato
ausente o alterado (fallaron antes de implementar el guard). El nuevo intento
usa `eap_rgb_capture_pilot_v2`, conservando íntegro v1 y las mismas 24 identidades
y tolerancias. Once pruebas pasan, además de Ruff y Pyright scoped.

No se leen targets Test12, no se entrena, no se cambia el entregable y no se
promueve el router. Sin targets, incluso deriva nula o pequeña no certifica
mejora de precisión. Los resultados del piloto y de la expansión se publicarán
por separado, una vez completados.

```powershell
$env:PYTHONPATH='src;.'
$env:PYTHONUTF8='1'
.venv-transfer/Scripts/python.exe -u scripts/capture_phase_eap_rgb_pair.py --inputs artifacts/runpod_event_authority_supplement_20261026/payload/workspace/e-jepa-ttc-v16-multimodal-validation/artifacts/multimodal_research_20261012/test12_phase_only/INPUTS.json --baseline-audit output/phase_safe_fusion/test12_audit --upstream output/phase_submission/FINAL_MODEL/upstream --population-freeze output/phase_timewarp/population2608_v1/FREEZE.json --media-root E:/eAP_dataset --mode pilot --output output/phase_timewarp/eap_rgb_capture_pilot_v2
```
