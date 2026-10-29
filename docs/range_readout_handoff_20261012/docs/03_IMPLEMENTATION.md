# Integración mínima, fuentes y pruebas

## Arquitectura de código propuesta
Nombres nuevos a implementar, no comandos que ya existan:
- `src/e_jepa_ttc/range_readout/sampling.py`: estimador de importancia y draws.
- `src/e_jepa_ttc/range_readout/residual.py`: operador antisym C05/C20 con pendiente igual.
- `operational/range_readout/admission.py`: bindings propios; no modificar admisiones V13.
- `operational/range_readout/cache_readout.py`: captura de inputs del residual en P/H/V
  sin guardar datasets completos, hashes por shard y reanudación por observación.
- `operational/range_readout/train_readout.py`: scorer-only, full checkpoint/RNG/cursor.
- `operational/range_readout/prepare_families.py`: nuevas features17 y norm H por padre.
- `operational/range_readout/run.py`: cola cerrada, ledger propio y comandos reales.
- `operational/range_readout/report.py`: pérdida por query, grupos, bandas, constraints.
- `tests/test_range_readout_*.py`: contratos; no alterar tests históricos para pasar.

Copiar y adaptar la maquinaria V14 existente antes de inventar un orquestador nuevo.
Para archivos históricos usar rutas readonly. No modificar los originales.

## Pruebas obligatorias
1. E_p[w/p*l] coincide con sum(w*l); uniform equivale a N*w, p=w a loss sin doble peso.
2. IDs de sampling probabilístico y RNG son exactos al reanudar.
3. Operador C05 igual al original; antisym al invertir endpoints; output/derivadas
   iniciales iguales C05/C20 con capa finalzero; tipos y formas.
4. Congelación: hash encoder, transport, geometry, uncertainty y PAIR no cambia tras
   un update sintético del scorer; sólo variables autorizadas reciben gradiente.
5. Caché versus raw: outputs y gradientes scorer y reducciones T2/T3/top-k iguales
   dentro de tolerancia fijada antes; no raw/replay global durante cada step.
6. PAIR token y soportes iguales tras modificar sólo residual. Fase A5 y sus
   diferencias NO se reutilizan del cache viejo. Known mantiene shape porquery[B].
7. No targets/roles/IDs numéricos en forward; P/H/V exclusión verificada.
8. Pérdida y output preservan negativos, límites de dominio, no-finite y cap policy.
9. Checkpoint continuo vsinterrumpido: pesos, optimizador, scheduler, draws, RNG.
10. Ningún modelo o control faltante se sustituye por pesos V12/TRAIN40.

## Recursos
Guardar cache vectorial, no todos los voxels en RAM. CPU un thread numérico y límite
propio4GiB para cabezas; proyecto23GB, hostreserva2GiB/reentrada3GiB, disco10GB tras
reservas. GPU necesaria para extracción se coordina con dueño real; no crear un
owner ficticio ni dos supervisores. No cambiar drivers, paging, energía o afinidades.
Si hardware/entorno difiere, registrar y validar paridad; no endurecer ni suavizar
las tolerancias mirando resultadosV.

## Artefactos finales
FINAL_REPORT.md, NEXT_DECISION.json, ACCOUNTING.json, SOURCE_BINDINGS.json,
PRODUCER_REACH_P_H.csv, SAMPLER_EXPOSURE.csv, CLIPPING.csv, curvas y pesos,
PREDICTIONS.csv completo, GROUP_METRICS, métricas por bandas/subrangos, timers con
scope, todos los fallos, tests y comando resume real. Separar entrenamiento del
readout/cabeza de inferencia total. Mantener negativos aunque no haya candidato.

## Tras esta cola
Si el rango ayuda a A5, estudiar C2F/ajuste encoder en un protocolo nuevo con controles.
Si el encoder contiene señal pero PHASE17 la pierde, probe de tokens/dense correlación
con una sola genealogía de productor y normalización H (no repetir mezclas LATENT).
Si no hay separación útil en features actuales, priorizar datos/adquisiciones y
representación antes de otro selector. RGBV13 y Garlfull se juzgan por sus propios
endpoints, no por estas hipótesis. Teacher RGB que no mejora no se destila a ciegas.

## Pruebas de referencia del paquete
Desde su raíz: `python -m pytest tests/test_reference_operators.py -q`.
Sólo comprueban operadores matemáticos, derivadas y RNG de muestreo. No ejecutan
ningún optimizer.step. La paridad raw/caché, checkpoints reales y la máscara de
parámetros congelados necesitan la integración del agente y cuentan en el presupuesto
técnico definido. PyTorch/Pandas/NumPy son los del entorno existente: no actualizarlo.
