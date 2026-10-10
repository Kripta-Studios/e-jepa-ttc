# 7. Integración archivo por archivo

| Destino propuesto | Responsabilidad |
|---|---|
| `docs/efficient_context_20261004/DECISION.md` | Release de partida, estado Stage70 y tarea. |
| `configs/campaign/efficient_context_v1.json` | IDs, fuentes, criterios, presupuestos y orden. |
| `src/e_jepa_ttc/efficient_context/historical_sources.py` | Raíces absolutas, bindings de sólo lectura y outputs independientes. |
| `src/e_jepa_ttc/efficient_context/window_plan.py` | Intervalos/ROI exactos y diagnóstico de coincidencias. |
| `src/e_jepa_ttc/efficient_context/mapped_union.py` | Filtro y proyección una vez; views por ventana. |
| `src/e_jepa_ttc/efficient_context/sensor_buffer.py` | Búfer acotado, identidad, evicción y cutoffs. |
| `src/e_jepa_ttc/efficient_context/producer_dispatch.py` | Slots válidos, PAIR compartido y transferencias. |
| `src/e_jepa_ttc/efficient_context/sparse_history.py` | WIDE8 fijo, gaps, padding y disponibilidad. |
| `src/e_jepa_ttc/efficient_context/garl_adapter.py` | Encoding nativo, anchors y productores correctos. |
| `operational/efficient_context/run.py` | CLI real y cola con dependencias separadas. |
| `operational/efficient_context/profile.py` | Mediciones emparejadas R0/R1/R2. |
| `operational/efficient_context/analysis.py` | MiD, pérdidas por seed, intervalos y guardrails. |
| `operational/efficient_context/package.py` | Publicación recuperable y verificación. |
| `tests/test_efficient_context_*.py` | Contratos numéricos y fallos previstos. |

Son destinos propuestos, no archivos ya implementados. Adaptar ubicaciones a las
convenciones del repositorio sin cambiar la semántica. Reutilizar loaders,
`training.py`, evaluación y journals canónicos; no copiarlos para luego divergir.
La carga de checkpoints históricos debe permanecer compatible.
