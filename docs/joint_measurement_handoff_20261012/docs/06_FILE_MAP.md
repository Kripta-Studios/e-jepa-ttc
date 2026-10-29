# Integración archivo a archivo (rutas nuevas propuestas)

| Ruta | Trabajo |
|---|---|
| `src/e_jepa_ttc/joint_measurement/model.py` | Integrar referencia de par y factory con CNN nativa privada. No cambiar CausalScaleTTC histórico. |
| `src/e_jepa_ttc/joint_measurement/loss.py` | Pérdida de fase raw/emitted, masas globales y dominios explícitos. |
| `operational/joint_measurement/data.py` | Adaptar EventProducerSource/RGBSource a P2, preservando time/ROI/targets y roles. Sin targets en model inputs. |
| `operational/joint_measurement/train.py` | Bucle efectivo32, controles, checkpoint completo y encoder realmente entrenable. Reusar primitivas de RGB_PORT con rutas propias. |
| `operational/joint_measurement/evaluate.py` | Cohortes exactas, scorer fijado, inválidos, grupos y comparadores nativos. |
| `operational/joint_measurement/run.py` | Cola y estados por hipótesis; no cerrar toda la misión con NO_CANDIDATE de un brazo. |
| `operational/joint_measurement/report.py` | Curvas, arquitectura/params, clipping, cambios de pesos, coste y claims. |
| `configs/joint_measurement/primary.json` | Cuatro IDs, datos, seeds, loss, endpoint y hash de protocolo. |
| `artifacts/joint_measurement_20261012/EXPERIMENT_REGISTRY.jsonl` | Registro prospectivo apend-only de las ramas adaptativas. |
| `scripts/Resume-JointMeasurement.ps1` | Invocación real al runner sólo después de integrarlo y validarlo. |
| `tests/test_joint_measurement*.py` | P2/T2-T3, rangos, masks, parity, gradientes hasta encoder, freeze, update/resume y cohortes. |

Puntos de reutilización: `CausalScaleTTC.encoder.features`; `local_transport` para correlación; `rgb_port_revision.loss.EffectiveBatch` y data loaders admitidos; `simplex_t.phase` para emisión; `range_transfer.analyze` para métricas locales y `garl_comparison` para scorer público; componentes de atomic IO/accounting actuales con raíces nuevas.

Una optimización vectorizada de correlación debe pasar paridad de salidas Y gradientes frente a referencia simple. No modificar tokens cacheados ni usar WEIGHTS.pt V17 como si fueran un encoder actualizado. Los pesos fuente del encoder son los padres matched P declarados, no los exports de readout.

La referencia no selecciona fuentes, no hace inferencia raw, no entrena la campaña y no es un driver que pueda ejecutarse de principio a fin sin integración. El agente debe implementar realmente esas piezas, no inventar comandos que asumen que existen.
