# EvTTC: evaluación tras la aclaración de los autores de Garl

Este experimento reconstruye los tres intervalos descritos en el correo aportado
por el usuario. No reemplaza Dev32, ni acredita por sí solo reproducción exacta
de la Tabla VI. Las consultas se fijaron antes de leer TTC y se ejecutan pesos
finales congelados, sin fine-tuning ni selección del mejor modelo por resultado.

**Completado:** 43 consultas, todos los modelos disponibles. RTE pooled F_TRUE
12,010% frente a Garl full 13,674%; macro de escenarios 10,237% frente a 15,399%.
F_TRUE mejora dos escenarios y pierde en CCRs2-medium. El intervalo exploratorio
por escenarios incluye cero. Véanse [conclusiones](CONCLUSIONS.md),
[resultados completos](REPORT.md) y `FINAL_QA.json`.

## Fuentes y alcance

- [Aclaraciones y dudas pendientes](AUTHOR_CLARIFICATION.md): separación entre
  lo que confirma el correo, lo que publica el paper y nuestras decisiones.
- [Protocolo ejecutado](PROTOCOL.md): selección de secuencia, cámaras, ROI,
  timestamps, labels, descartes, precisión y límites de comparación.
- `REPORT.md`, `RESULT.json`, `SCORED_PREDICTIONS.csv`: resultados generados
  después del sellado, con disponibilidad propia y consultas comunes.
- `CANDIDATE_AUDIT.csv`: inventario de los instantes seleccionados y rechazos.
- `evidence/`: manifiesto prospectivo, comprobaciones y sellados de ejecución.

Código: `operational/author_evttc/{inputs,run,score}.py`.
Tests: `tests/test_author_evttc.py`. Las implementaciones anteriores de
transferencia V12/V13 se conservan; la adaptación es un módulo adicional.

La campaña local escribe en `E:/EJEPA_results/author_evttc_20261012`.
Los datos EvTTC se leen de su ubicación existente referida por el manifiesto
histórico; no se crea otra copia en el SSD ni se trasladan datasets. Todos los
artefactos nuevos de ejecución quedan en E:.

## Reproducción local

Para verificar solo la evidencia pública, sin datasets, pesos, GPU ni PyTorch:

```powershell
python -B docs/author_evttc_20261012/replay_public.py --output E:/EJEPA_results/author_evttc_public_replay
```

Este replay usa biblioteca estándar: comprueba hashes, 1.260 valores de métricas
y 516 predicciones contra sus sellados. No ejecuta la red.

Desde V13, con el entorno del proyecto y `PYTHONPATH=src;.` en Windows:

```powershell
python -B -m operational.author_evttc.run prepare --output E:/EJEPA_results/author_evttc_20261012 --v12 ../e-jepa-ttc-v12-efficient-context
python -B -m operational.author_evttc.run v13 --output E:/EJEPA_results/author_evttc_20261012 --v12 ../e-jepa-ttc-v12-efficient-context --device cpu
python -B -m operational.author_evttc.run baselines --output E:/EJEPA_results/author_evttc_20261012 --v12 ../e-jepa-ttc-v12-efficient-context --garl-code E:/Garl-TTC --device cpu
python -B -m operational.author_evttc.score --output E:/EJEPA_results/author_evttc_20261012
```

La inferencia exige modelos, datos y contratos locales. Los CSV/JSON públicos
permiten recalcular estadísticas; no contienen los pesos. Si cambia código,
protocolo o checkpoint, conservar la campaña anterior y usar otra ruta.

Los informes históricos del handoff del día 12 describen su corte previo a este
correo. Esta carpeta es un anexo posterior; no se reescriben sus resultados
para hacerlos parecer una reproducción del protocolo recién aclarado.
