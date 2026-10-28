# Auditoría de medición y piloto causal — 28/10/2026

El inventario local permite pilotos causales de 1–5 segundos. No permite cumplir
el protocolo vigente de 60 segundos continuos sobre una única secuencia: las
32 fuentes RAW registradas duran menos de 60 segundos. Esto no impide seguir
optimizando ni medir los pilotos; limita la afirmación que puede hacerse con
el ensayo sostenido actual.

## Evidencia obtenida

El script [audit_and_build_manifest.py](audit_and_build_manifest.py) leyó los
extremos de los datasets de timestamps de las 32 fuentes HDF5 del manifiesto
original. No leyó píxeles RGB, coordenadas/polaridades de eventos, targets TTC
ni checkpoints. Comprobó tamaño y mtime antes y después; no calculó hashes de
los HDF5 completos.

Los resultados están en
[raw_sequence_inventory.json](../../artifacts/train40_runtime_council_20261028/measurement_audit_v1/raw_sequence_inventory.json).
Las duraciones event van de 16,886321 a 31,712052 segundos. La mayor corresponde
a `CPNAO-low`. Los extremos de un archivo solo establecen una cota temporal;
no demuestran continuidad de anotaciones ni cobertura de consultas.

Se generó con la función canónica existente el manifiesto de `CPLA-low`:

- 250 consultas aceptadas de 250 programadas, sin exclusiones;
- anchors `16750664..21730664`, separados exactamente por 20000 µs;
- 5 segundos de períodos ofertados a 50 Hz y 4,98 segundos entre anchors;
- soporte RAW para la inicialización de 600 ms hasta `primer_anchor−20 ms`;
- lectura de cajas causales con el contrato existente; ninguna etiqueta TTC.

El [manifiesto causal](../../artifacts/train40_runtime_council_20261028/measurement_audit_v1/causal_pilot_manifest.json)
y el [recibo de generación](../../artifacts/train40_runtime_council_20261028/measurement_audit_v1/manifest_generation_receipt.json)
conservan parámetros, comando, commit, hashes del manifiesto base y fuentes,
hashes de salidas y verificaciones. La generación pasó Ruff y las nueve
comprobaciones explícitas del script. No acredita todavía latencia ni paridad
TTC del nuevo replay.

Regeneración hacia un directorio nuevo:

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' `
  docs/train40_runtime_council_20261028/audit_and_build_manifest.py `
  --base-manifest artifacts/final_train40_transfer_20261027/QUERY_MANIFEST.json `
  --output artifacts/train40_runtime_council_20261028/measurement_audit_REGEN `
  --sequence CPLA-low --updates 250
```

## Contrato revisado

La revisión inicial corresponde al código publicado en `2521de1a`. El runner
`scripts/benchmark_train40_paced.py` recibe una secuencia RAW real, conserva todas
las posiciones de su grilla y deja las exclusiones presentes en cobertura.
No repite consultas ni une secuencias para completar el mínimo de duración.

`operational/train40_runtime/paced.py` mantiene un productor RAW independiente,
deadlines absolutos y cola acotada. El consumidor devuelve todos los TTC como
escalares finitos de host antes de fijar el final del update. Los descartes y
fallos quedan registrados. Un hueco invalida la historia del consumidor; los
paquetes siguientes no producen predicciones con historia incompleta.

El reloj principal para la admisión es `deadline_response_ms`. Incluye retraso
de adquisición, copias RAW, construcción de ROI, preprocesamiento,
transferencias, modelos, cola y disponibilidad final en host.
`raw_availability_response_ms` comienza después de adquirir/copiar el payload;
su valor aislado omite esos costes previos y no basta para admitir el objetivo.
La fuente persistent solo precarga frames hasta `primer_anchor−20000 µs`.

El gate `train40_paced_milestone_gate_v2` para uno y dos objetos exige todos los
checks de cadencia, al menos 3000 períodos a 50 Hz, todos los resultados bajo
20 ms, cero descartes/fallos/huecos, tres TTC finitos por objeto, oracle original
posterior para todas las consultas y preservación de originales.
`summary.status == COMPLETE` significa que la ejecución y su cobertura fueron
válidas; la admisión temporal está en `milestone_with_parity_passed`.

Dos diferencias de implementación requieren atención al interpretar un piloto:

- La fuente persistent del cierre copiaba todos los frames seleccionados en
  cada payload, aunque leyera de disco únicamente frames nuevos. Esas copias
  cuentan en el reloj principal. La integración de identidad inmutable necesita
  demostrar que cada token y cada array de crop comparten los mismos bytes,
  incluyendo lifetime de tokens retenidos por la cola.
- El runner paced publicado fijaba `shared_rgb_images=True`, mientras el
  perfil PIL admitido usa ROI-first. Ambos pueden conservar píxeles, pero tienen
  costes diferentes. El siguiente piloto debe registrar el perfil efectivo y
  resolver esta desviación antes de atribuir mejoras a otra optimización.

La paridad posterior compara TTC y recibos físicos, no conserva todos los
tensores de cada replay largo. La admisión tensor a tensor del servicio sigue
siendo una comprobación independiente.

## Pilotos siguientes

Primero se ejecuta el prefijo de 50 consultas del manifiesto nuevo:

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' scripts/benchmark_train40_paced.py `
  --candidate 'D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE' `
  --reference artifacts/final_train40_transfer_20261027/final_train40/own `
  --base-manifest artifacts/final_train40_transfer_20261027/QUERY_MANIFEST.json `
  --manifest artifacts/train40_runtime_council_20261028/measurement_audit_v1/causal_pilot_manifest.json `
  --sequence CPLA-low --objects 1 --updates 50 `
  --raw-source persistent --wait-strategy sleep --verify-parity `
  --output artifacts/train40_runtime_council_20261028/paced_CPLA_low_1obj_50_NEW
```

Después, un output nuevo y `--updates 250` permiten medir los cinco segundos
de carga ofertada. Ninguno puede pasar el gate de 60 segundos. El informe debe
mantener media, p50, p95, p99 y máximo, fallos, cobertura, trabajo pendiente y
los tres TTC. Los datos no justifican convertir el inverso de la media en
frecuencia sostenida.

Una alternativa de estrés explícitamente repetido se propone en
[STRESS_REPLAY_V1_PROPOSAL.md](STRESS_REPLAY_V1_PROPOSAL.md). El runner no se
ha implementado ni ejecutado como parte de esta auditoría.
