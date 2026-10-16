# V18: conclusión actual y trabajo pendiente

**La campaña sigue abierta; no hay promoción del sistema ni confirmación SOTA.** La fusión presenta una señal controlada en las tres semillas, pero no cumple el gate crucial original del10%. Los negativos y las regresiones se conservan.

Corte UTC: `2026-10-16T06:03:32.715596+00:00`. Fuente: `CAMPAIGN_CONCLUSION.json`.

| Etapa | Evidencia completa | Estado |
|---|---|---|
| primary | True | PRIMARY_EVIDENCE_COMPLETE_RESEARCH_ONGOING |
| RGB | True | RGB_READY_FOR_INTERPRETATION |
| C1 | True | C1_READY_FOR_INTERPRETATION |
| fusion_seed7 | True | FUSION_METRICS_READY_FOR_INTERPRETATION |
| fusion_three_seeds | True | REPLICA_METRICS_READY_FOR_INTERPRETATION |
| D1 | True | D1_READY_FOR_INTERPRETATION |
| B1 | False | PENDING |

| Fusión, media de3 seeds | Agregado V | MiDc V | MAE V, s |
|---|---:|---:|---:|
| TRUE | 87.287 | 126.569 | 1.814 |
| ZERO | 92.469 | 136.601 | 1.838 |

El agregado mejora frente al control ZERO de igual capacidad. Esto no demuestra superioridad frente a V16: el intervalo de la diferencia agregada entre sus medias incluye cero. Véase [el informe de réplica](FUSION_REPLICATION_RESULTS.md).

## Contabilidad del corte

- Updates científicos: 139,300.
- Técnicos: 750; recuperación superior: 263; pendientes superiores: 0.
- Físicos superiores: 140,313.
- Scopes GPU contabilizados/reservados: 155980.3 / 172800 s.

Es una instantánea no atómica mientras hay trabajo vivo. No son kernels GPU puros; las reservas activas pueden cambiar al liquidarse. No se recontaron los controles históricos reutilizados.

## Pendiente

- Completar B1: endpoint, P/H, coste, export e interpretación registrados
- Resolver/documentar el fallo de paridad antes de admitir el coste sensorial de fusión
- Decidir con P/H si procede familia A, datos o refit; sin búsqueda automática
- Congelar cualquier export test admisible; el envío oficial sigue sin autorización
- Publicar informes y verificar independientemente el ZIP final <=100000000 bytes

Las ramas condicionadas se justifican con evidencia; un resultado negativo cierra su experimento. La evaluación oficial requiere autorización separada y no se utiliza como bucle de ajuste.
