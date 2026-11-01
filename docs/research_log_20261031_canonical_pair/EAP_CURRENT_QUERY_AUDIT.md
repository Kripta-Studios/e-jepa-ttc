# Auditoría eAP de PAIR en la observación actual

Se usaron features de productores de despliegue TRAIN40, no los estados OOF de la auditoría de fusión activa. Solo se leyeron features, identidades y grupos; no se leyeron targets. Los padres de los productores y los miembros de los archivos locales quedaron vinculados por hashes.

| Modalidad | Filas | Identidad exacta 100 ms | Batches rechazados | Cambios de signo | Cambio medio absoluto TTC (s) | Máximo (s) |
|---|---:|---:|---:|---:|---:|---:|
| event | 88744 | 88744 | 0 | 0 | 0.00000000000 | 0.00000000000 |
| rgb | 88744 | 24 | 0 | 0 | 0.00009167075 | 0.00695800781 |

Tiempo medido después de cargar modelos: 1.666964 s.

La identidad se verifica en bytes para fase cruda, fase emitida y TTC en las filas exactas de 100 ms. La política de suelo de magnitud del inverso permanece sin cambios. Un contador inicial que comprobaba TTC == 60 era incorrecto: el valor FP32 emitido es 59,999996. Se conserva el recibo inicial y la corrección explícita; el CLI definitivo usa el predicado de inverso.

No se han reconstruido todas las observaciones históricas, los contextos ni la cabeza final. Estos cambios de salida no son errores contra ground truth ni certifican conservación de eAP. Test12 sigue sin targets disponibles.

El CLI reproducible es `scripts/audit_phase_canonical_pair_source.py`. Requiere la extracción sellada en `output/phase_timewarp/eap_current_pair_audit` y un directorio de salida nuevo. Regenerar este documento no ejecuta inferencia.
